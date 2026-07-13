"""使用训练好的模型生成文本样例，用于人工评估

用法：
    python scripts/generate_samples.py --model checkpoints/best_model.pt --tokenizer checkpoints/
"""

import argparse
import contextlib
import sys

import torch

if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

from gleamlm import load_model_for_inference
from gleamlm.inference.streamer import TextStreamer
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH


def generate_samples(
    model,
    tokenizer,
    prompts,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    device="cuda",
):
    streamer = TextStreamer(tokenizer)
    model.eval()

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(prompts)}] {prompt}")
        print(f"{'=' * 60}")
        print("Generated: ", end="", flush=True)

        generated = ""
        for chunk in streamer.generate_text(
            model, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty
        ):
            generated += chunk
            try:
                print(chunk, end="", flush=True)
            except UnicodeEncodeError:
                print(
                    chunk.encode("utf-8", errors="replace").decode("utf-8", errors="replace"),
                    end="",
                    flush=True,
                )
        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GleamLM 文本样例生成")
    parser.add_argument("--model", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help="分词器路径")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="最大生成 token 数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config = load_model_for_inference(args.model, args.device)
    tokenizer = BBPETokenizer.load(args.tokenizer)
    total, _ = model.get_num_params()
    print(f"Model: {total / 1e6:.2f}M params, device: {args.device}")

    prompts = [
        "你好，请介绍一下你自己。",
        "什么是人工智能？",
        "写一首关于春天的诗。",
        "推荐一道简单好做的家常菜。",
        "用一句话解释深度学习。",
    ]

    generate_samples(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens, device=args.device,
    )


if __name__ == "__main__":
    main()
