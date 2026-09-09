"""生成 DPO rejected 数据。

用变体自己的 SFT 模型对 chosen 池生成"差的"回答，作为 DPO rejected。

数据流（与 build_dpo_chosen / merge_dpo_data 同 variant 口径）：
  build_dpo_chosen.py → dpo_chosen_{single,multi}.jsonl → 本脚本 →
  dpo_rejected_{single,multi}[.N].jsonl（分片）→ merge_dpo_data.py → dpo_data.jsonl

用法:
  # rejected 统一用变体自己的 SFT 模型（sft_best.pt）——rejected 与当前
  # policy 同分布才有区分度；用基座模型生成单轮 rejected（分布外）已弃
  python data_tools/dpo/generate_rejected.py --format single \
      --sft_data data/nano/dpo/dpo_chosen_single.jsonl \
      --model_path checkpoints/nano/sft/sft_best.pt \
      --output data/nano/dpo/dpo_rejected_single.jsonl

  # 多轮同理，上下文只喂对话历史（脚本内剥离尾轮答案）
  python data_tools/dpo/generate_rejected.py --format multi \
      --input data/nano/dpo/dpo_chosen_multi.jsonl \
      --model_path checkpoints/nano/sft/sft_best.pt \
      --output data/nano/dpo/dpo_rejected_multi.jsonl

  # 并行加速: 追加 --shard-total N 并以 --shard-id 0..N-1 分进程跑，分片自动 .N.jsonl
  # 中断续跑: 同 shard 重跑覆盖；.partial 每 50 条落盘，最多丢 49 条
"""

import argparse
import json
import os
import sys

_sys_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _sys_root not in sys.path:
    # 项目根前置：保证 import 本地源码 gleamlm（否则落到 site-packages 旧发行版）
    sys.path.insert(0, _sys_root)

import torch

from gleamlm import load_model_for_inference
from gleamlm.inference.generator import generate_tokens
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.chatml import format_chatml
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def generate_rejected_single(
    model, tokenizer, instruction: str, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9
) -> str:
    device = next(model.parameters()).device
    prompt_text = format_chatml(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

    generated: list[int] = []
    for token_id in generate_tokens(
        model,
        prompt_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        penalty_window=50,
        stop_ids={tokenizer.eos_id, tokenizer.pad_id},
    ):
        generated.append(token_id)

    response = tokenizer.decode(generated, skip_special=True)
    if tokenizer.eos_token and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response


def last_assistant_turn(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]] | None, str | None]:
    """从 messages 中剥离最后一轮 assistant 回复，返回 (context, last_content)。"""
    if not messages or messages[-1]["role"] != "assistant":
        return None, None
    context = messages[:-1]
    target = messages[-1]["content"]
    return context, target


def build_prompt(context: list[dict[str, str]], tokenizer) -> list[int]:
    return tokenizer.encode(
        format_chatml(context, add_generation_prompt=True),
        add_bos=False,
        add_eos=False,
    )


def generate_rejected_multi(
    model,
    tokenizer,
    context: list[dict[str, str]],
    max_new_tokens=256,
    temperature=0.95,
    top_k=50,
    top_p=0.9,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = build_prompt(context, tokenizer)
    im_end_id = tokenizer.im_end_id
    stop_ids: set[int] = {tokenizer.eos_id, tokenizer.pad_id}
    if im_end_id is not None:
        stop_ids.add(im_end_id)

    generated: list[int] = []
    for token_id in generate_tokens(
        model,
        prompt_ids,
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        penalty_window=50,
        stop_ids=stop_ids,
    ):
        generated.append(token_id)

    return tokenizer.decode(generated, skip_special=True)


def main():
    # 分片并行时每进程只留 1 个 CPU 线程。torch 默认按物理核数开线程，
    # 多进程叠满反而互相抢占，实测 4 进程总吞吐低于单进程。
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Generate DPO rejected data")
    parser.add_argument(
        "--format", type=str, choices=["single", "multi"], default="single", help="数据格式"
    )
    # 路径
    parser.add_argument(
        "--sft_data", type=str, default=None, help="单轮 chosen 文件 (build_dpo_chosen 产物)"
    )
    parser.add_argument("--input", type=str, default=None, help="多轮 SFT JSONL (messages 格式)")
    parser.add_argument(
        "--model_path",
        "--model",
        dest="model_path",
        type=str,
        default=None,
        help="模型 checkpoint 路径",
    )
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=str, default="data/dpo_data.jsonl", help="输出文件")
    parser.add_argument("--limit", type=int, default=0, help="最大样本数 (0=all)")
    parser.add_argument(
        "--shard-id", type=int, default=0, help="分片序号 (0-based, 配合 --shard-total 多进程并行)"
    )
    parser.add_argument("--shard-total", type=int, default=1, help="分片总数")
    # 生成参数
    parser.add_argument(
        "--temperature", type=float, default=0.95, help="温度 (单轮 default=0.8, 多轮 default=0.95)"
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    fmt = args.format
    if fmt == "single":
        data_path = args.sft_data
        if not data_path:
            print("Error: --sft_data required for single format", file=sys.stderr)
            sys.exit(1)
    else:
        data_path = args.input
        if not data_path:
            print("Error: --input required for multi format", file=sys.stderr)
            sys.exit(1)

    model_path = args.model_path
    if not model_path:
        print("Error: --model_path required", file=sys.stderr)
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    model, config = load_model_for_inference(model_path, device)
    tokenizer = BBPETokenizer.load(args.tokenizer_path)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params")

    # 加载数据
    samples: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if (
                    fmt == "single"
                    and "instruction" in item
                    and "output" in item
                    or fmt == "multi"
                    and "messages" in item
                ):
                    samples.append(item)
            except json.JSONDecodeError:
                continue
    if args.limit:
        samples = samples[: args.limit]
    if args.shard_total > 1:
        samples = samples[args.shard_id :: args.shard_total]
        if args.shard_id:
            args.output = args.output.replace(".jsonl", f".{args.shard_id}.jsonl")
    print(f"Loaded {len(samples)} samples from {data_path}")

    temp = args.temperature if fmt == "multi" else min(args.temperature, 0.95)

    dpo_data: list[dict] = []
    for i, s in enumerate(samples):
        if fmt == "single":
            instruction = s["instruction"]
            chosen = s["output"]
            rejected = generate_rejected_single(
                model,
                tokenizer,
                instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            dpo_data.append({"instruction": instruction, "chosen": chosen, "rejected": rejected})
        else:
            messages: list[dict] = s["messages"]
            context, target = last_assistant_turn(messages)
            if context is None or target is None:
                continue
            rejected = generate_rejected_multi(
                model,
                tokenizer,
                context,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            dpo_data.append(
                {
                    # messages 只含对话历史（不含尾轮答案），与 DPODataset 消费约定一致：
                    # 训练时 prompt = messages + generation prompt，chosen/rejected 作为尾轮续写。
                    # 历史坑：曾把 target 也拼进 messages，导致答案在 prompt 区重复（被 mask 区吃掉）。
                    "messages": context,
                    "chosen": target,
                    "rejected": rejected,
                }
            )

        if (i + 1) % 20 == 0:
            # 每 20 条一行进度；flush 即时写出（stdout 有块缓冲）
            print(f"  [{i + 1}/{len(samples)}]", flush=True)
        if (i + 1) % 50 == 0:
            # 断点续跑文件：50 条一落盘，最多丢 49 条
            with open(args.output + ".partial", "w", encoding="utf-8") as pf:
                for item in dpo_data:
                    pf.write(json.dumps(item, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Done -> {args.output} ({len(dpo_data)} pairs)")


if __name__ == "__main__":
    main()
