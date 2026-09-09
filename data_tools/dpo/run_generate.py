"""DPO 数据生成一键编排（WebUI「DPO 数据生成」任务的入口脚本）。

把 data_tools/dpo 三脚本串成一条幂等流水线，供 GUI（教学用户不碰 CLI）与
手动用户共用。全程打印步骤分段行（===== [n/5] … =====），任一步失败立即
退出非 0 并跳过后续步骤；已产生的分片可重跑覆盖（幂等）。

步骤：
  1. build_dpo_chosen      chosen 池（sft_mix 单源确定性重放 = 复用旧池）
  2. generate_rejected     single 路 rejected（--shard-total N 并行分片）
  3. generate_rejected     multi 路 rejected（同上）
  4. merge_dpo_data        先写临时文件，成功后才原子替换正式产物
                          （旧 dpo_data.jsonl 自动改名 dpo_data_bak_v{N} 保留）
  5. 清理                  chosen/rejected 中间产物（用完即删，不入库）

rejected 模型：各变体自己的 SFT 模型（sft_best.pt，自动探测 checkpoints/
<variant>/sft/sft_best.pt；--model-path 可覆写）——rejected 与当前 policy
同分布（模型自己会犯的错），DPO 信号精细（2026-09-09 拍板；曾建议单轮用
基座 final.pt，分布外已弃）。

用法:
  python data_tools/dpo/run_generate.py --variant lite
  python data_tools/dpo/run_generate.py --variant nano --shards 8  # 显存富余拉满
  python data_tools/dpo/run_generate.py --variant nano --limit 1 \
      --output data/nano/dpo/_tmp_gen.jsonl    # 冒烟（不落正式位）

生成慢的说明：generate_rejected 是逐条自回归（无 batch 接口），小模型
(40M) 在 4070Ti/A10 上算力过剩（GPU 利用率 ~30%），纯单进程慢。默认
--shards 4 多进程并行把 GPU 跑满：每个进程独立加载模型，显存占用
~0.7GB/进程（40M 模型），12GB 以上显卡可开到 8。
"""

import argparse
import contextlib
import glob
import os
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(os.path.dirname(D))  # 项目根


def _script(name: str) -> str:
    return os.path.join(HERE, "data_tools", "dpo", name)


def _run(cmd: list[str], name: str) -> int:
    """跑一个子进程（stdout/stderr 直通面板日志），打印分段标记。"""
    print(f"\n===== [{name}] =====", flush=True)
    return subprocess.run(cmd, cwd=HERE).returncode


def _run_shards(cmd_head: list[str], shard_total: int, name: str) -> int:
    """分片并行：每片一个进程 --shard-id i --shard-total N（0 号片无后缀）。"""
    print(f"\n===== [{name}] x{shard_total} 并行 =====", flush=True)
    if shard_total <= 1:
        return subprocess.run(cmd_head, cwd=HERE).returncode
    procs = [
        subprocess.Popen(
            cmd_head + ["--shard-id", str(i), "--shard-total", str(shard_total)], cwd=HERE
        )
        for i in range(shard_total)
    ]
    codes = [p.wait() for p in procs]
    bad = [i for i, c in enumerate(codes) if c != 0]
    if bad:
        print(f"!! 分片 {bad} 失败 exit={codes}", file=sys.stderr, flush=True)
        return 1
    return 0


def _cleanup(dpo_dir: str) -> None:
    """删除 chosen/rejected 中间产物（可再生，不入库）。"""
    removed = 0
    # .partial 断点文件同属中间产物（merge 已完成，保留只会脏目录）
    for pat in (
        "dpo_chosen_*.jsonl",
        "dpo_rejected_*.jsonl",
        "dpo_rejected_*.[0-9].jsonl",
        "dpo_rejected_*.partial",
    ):
        for p in glob.glob(os.path.join(dpo_dir, pat)):
            os.remove(p)
            removed += 1
    if removed:
        print(f"[cleanup] 已删除 {removed} 个中间产物文件", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO data pipeline: chosen -> rejected -> merge")
    parser.add_argument("--variant", default="nano", help="数据变体（data/<variant>/…）")
    parser.add_argument(
        "--model-path",
        default=None,
        help="SFT 模型（缺省自动探测 checkpoints/<variant>/sft/sft_best.pt）",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="rejected 生成并行分片数（默认 4 进程；单进程时 GPU 利用率 ~30%，多进程跑满）",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="每路最多生成条数（冒烟用，不传=全量）"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="merge 输出（缺省 data/<variant>/dpo/dpo_data.jsonl；冒烟传临时路径避免落正式位）",
    )
    args = parser.parse_args()

    dpo_dir = os.path.join(HERE, "data", args.variant, "dpo")
    out = args.output or os.path.join(dpo_dir, "dpo_data.jsonl")
    out_abs = os.path.abspath(out)

    # 前置校验：sft_mix 源 + SFT 模型必须就绪（GUI 用户先跑 SFT 再点本卡）
    mix = os.path.join(HERE, "data", args.variant, "sft", "sft_mix.jsonl")
    if not os.path.isfile(mix):
        sys.exit(f"Error: 源数据不存在: {mix} (sft_mix 是唯一有效 SFT 集)")
    model_path = args.model_path or os.path.join(
        HERE, "checkpoints", args.variant, "sft", "sft_best.pt"
    )
    if not os.path.isfile(model_path):
        sys.exit(
            f"Error: SFT 模型不存在: {os.path.relpath(model_path, HERE)} — "
            f"请先完成 {args.variant} 的 SFT 训练（或在表单填入 --model-path）"
        )
    os.makedirs(dpo_dir, exist_ok=True)
    py = sys.executable
    chosen_single = os.path.join(dpo_dir, "dpo_chosen_single.jsonl")
    chosen_multi = os.path.join(dpo_dir, "dpo_chosen_multi.jsonl")
    rej_single = os.path.join(dpo_dir, "dpo_rejected_single.jsonl")
    rej_multi = os.path.join(dpo_dir, "dpo_rejected_multi.jsonl")

    # 1. chosen 池（确定性，等价复用）
    if _run([py, _script("build_dpo_chosen.py"), "--variant", args.variant], "1/5 chosen 池") != 0:
        sys.exit(1)

    # 2/3. rejected 生成（single/multi 均用同一 SFT 模型）
    gen_kwargs = ["--shard-total", str(args.shards)] if args.shards > 1 else []
    if args.limit:
        gen_kwargs += ["--limit", str(args.limit)]
    if (
        _run_shards(
            [
                py,
                _script("generate_rejected.py"),
                "--format",
                "single",
                "--sft_data",
                chosen_single,
                "--model_path",
                model_path,
                "--output",
                rej_single,
            ]
            + gen_kwargs,
            args.shards,
            "2/5 single rejected",
        )
        != 0
    ):
        sys.exit(1)
    if (
        _run_shards(
            [
                py,
                _script("generate_rejected.py"),
                "--format",
                "multi",
                "--input",
                chosen_multi,
                "--model_path",
                model_path,
                "--output",
                rej_multi,
            ]
            + gen_kwargs,
            args.shards,
            "3/5 multi rejected",
        )
        != 0
    ):
        sys.exit(1)

    # 4. merge 到临时文件，成功后才原子替换（失败不碰正式产物）
    tmp = out_abs + ".new"
    if (
        _run(
            [py, _script("merge_dpo_data.py"), "--variant", args.variant, "--output", tmp],
            "4/5 合并清洗",
        )
        != 0
    ):
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        sys.exit(1)
    if os.path.isfile(out_abs):
        bak = _next_bak(out_abs)
        os.replace(out_abs, bak)
        print(f"[backup] 旧数据已保留: {os.path.relpath(bak, HERE)}", flush=True)
    os.replace(tmp, out_abs)
    print(f"Done -> {os.path.relpath(out_abs, HERE)}", flush=True)

    # 5. 清理中间产物
    _cleanup(dpo_dir)


def _next_bak(out_abs: str) -> str:
    """自动找下一个备份名 dpo_data_bak_v{N}.jsonl（沿用 bak_v1 先例）。"""
    base, ext = os.path.splitext(out_abs)
    n = 1
    while os.path.isfile(f"{base}_bak_v{n}{ext}"):
        n += 1
    return f"{base}_bak_v{n}{ext}"


if __name__ == "__main__":
    main()
