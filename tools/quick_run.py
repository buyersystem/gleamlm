"""
GleamLM 快速训练 + 验证一体化脚本

三级规模，适合不同验证场景：

  Level 1 冒烟测试  (~30s):  验证代码能跑通，loss 是否下降
  Level 2 小规模跑  (~5min):  看模型是否在学，生成是否开始有语义
  Level 3 全量训练  (~小时): 正式训练获取可用模型

用法:
    python tools/quick_run.py --level 1 --variant nano
    python tools/quick_run.py --level 2 --variant lite
    python tools/quick_run.py --level 3 --variant pro
"""

import argparse
import os
import shutil
import subprocess

TEST_DATA_DIR = "data/smoke_splits"
TEST_CKPT_DIR = "checkpoints_smoke"


def conda_env_usable(name):
    """探测 conda 环境是否可用：conda 未安装 / 环境不存在都算不可用。

    用于 --conda_env 的回退判断——默认值是本机环境名，他人 clone 后通常没有该环境，
    直接 conda run -n 会以 non-zero 退出，脚本开箱即失败。
    """
    try:
        r = subprocess.run(
            ["conda", "run", "-n", name, "python", "-c", ""],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        # OSError 含 FileNotFoundError：conda 本身不在 PATH
        return False
    return r.returncode == 0


def run(cmd, desc="", conda_env="dl2llm"):
    if conda_env:
        cmd = f"conda run -n {conda_env} {cmd}"
    if desc:
        print(f"\n{'=' * 60}")
        print(f"  {desc}")
        print(f"{'=' * 60}")
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if result.returncode != 0:
        print(f"  [!] Command failed (exit code {result.returncode})")
        return False
    return True


def prepare_small_data(n_train=2000, n_valid=500):
    print("\n>>> 准备小数据集...")
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for split, n in [("train", n_train), ("valid", n_valid)]:
        src = f"data/nano/pretrain/{split}.txt"
        dst = f"{TEST_DATA_DIR}/{split}.txt"

        if not os.path.exists(src):
            print(f"  跳过: {src} 不存在")
            continue

        lines = []
        with open(src, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)

        with open(dst, "w", encoding="utf-8") as f:
            f.writelines(lines)

        npy = f"{TEST_DATA_DIR}/{split}_ids.npy"
        if os.path.exists(npy):
            os.remove(npy)

        print(f"  {split}.txt: {len(lines)} 行, {os.path.getsize(dst) / 1024:.0f} KB")

    print("  小数据集准备完成!")


def main():
    parser = argparse.ArgumentParser(description="曜珑GleamLM 快速训练+验证")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1=冒烟测试 2=小规模训练+验证 3=全量训练",
    )
    parser.add_argument(
        "--variant", type=str, choices=["nano", "lite", "pro"], default="nano", help="模型变体"
    )
    parser.add_argument("--verify_only", action="store_true", help="只验证已有模型，不训练")
    parser.add_argument(
        "--conda_env",
        type=str,
        default="dl2llm",
        help="conda 环境名 (默认: dl2llm；该环境不存在时自动回退到当前解释器，传空字符串则不用 conda)",
    )
    args = parser.parse_args()

    # 回退：指定的 conda 环境若不可用（他人 clone 后默认的 "dl2llm" 通常不存在），
    # 改用当前解释器直接运行，避免开箱即失败。显式指定且可用的环境不受影响。
    if args.conda_env and not conda_env_usable(args.conda_env):
        print(
            f"[!] conda 环境 '{args.conda_env}' 不可用，改用当前解释器直接运行。\n"
            f'    （指定其它环境: --conda_env <名称>；完全不使用 conda: --conda_env ""）'
        )
        args.conda_env = ""

    v = args.variant
    ckpt_dir = f"checkpoints/{v}"

    # 验证已有模型
    if args.verify_only:
        print(f"\n>>> 验证已有模型: {ckpt_dir}/final.pt")
        run(
            f"python -m tools.eval_runner --model {ckpt_dir}/final.pt --data_dir data/{v}/pretrain --benchmarks ppl --max_batches 50 --batch_size 4",
            "PPL 评估 (50 batches)",
            conda_env=args.conda_env,
        )
        run(
            f'python -m gleamlm.inference.cli --model {ckpt_dir}/final.pt --prompt "介绍一下你自己"',
            "生成样例",
            conda_env=args.conda_env,
        )
        return

    # Level 1: 冒烟测试
    if args.level == 1:
        print("\n" + "=" * 60)
        print("  Level 1: 冒烟测试 (验证代码能跑通)")
        print("=" * 60)

        prepare_small_data(n_train=2000, n_valid=500)

        ok = run(
            f"python manual/pretrain.py --model manual/configs/{v}.yaml "
            f"--data {TEST_DATA_DIR}/train.txt --val_data {TEST_DATA_DIR}/valid.txt "
            f"--output_dir ./{TEST_CKPT_DIR} "
            f"--epochs 2 --batch_size 8 --accumulate 4 --no-pbar",
            "训练 2 epochs (预计 ~30s)",
            conda_env=args.conda_env,
        )
        if not ok:
            print("\n[!] 训练失败")
            return

        run(
            f"python -m tools.eval_runner --model {TEST_CKPT_DIR}/final.pt "
            f"--data_dir {TEST_DATA_DIR} --dataset valid "
            f"--benchmarks ppl --max_batches 30 --batch_size 4",
            "PPL 评估",
            conda_env=args.conda_env,
        )
        run(
            f'python -m gleamlm.inference.cli --model {TEST_CKPT_DIR}/final.pt --prompt "介绍一下你自己"',
            "生成样例",
            conda_env=args.conda_env,
        )

        if os.path.exists(TEST_CKPT_DIR):
            shutil.rmtree(TEST_CKPT_DIR)
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

        print("\n>>> Level 1 完成!")

    elif args.level == 2:
        print("\n" + "=" * 60)
        print("  Level 2: 小规模训练 + 验证")
        print("=" * 60)

        prepare_small_data(n_train=10000, n_valid=2000)

        ok = run(
            f"python manual/pretrain.py --model manual/configs/{v}.yaml "
            f"--data {TEST_DATA_DIR}/train.txt --val_data {TEST_DATA_DIR}/valid.txt "
            f"--output_dir ./{TEST_CKPT_DIR} "
            f"--epochs 5 --batch_size 8 --accumulate 8 --no-pbar",
            "训练 5 epochs (预计 ~5min)",
            conda_env=args.conda_env,
        )
        if not ok:
            print("\n[!] 训练失败")
            return

        print("\n>>> 开始完整验证...")
        run(
            f"python -m tools.eval_runner --model {TEST_CKPT_DIR}/final.pt "
            f"--data_dir {TEST_DATA_DIR} --dataset valid "
            f"--benchmarks ppl --max_batches 100 --batch_size 4",
            "PPL 评估 (100 batches)",
            conda_env=args.conda_env,
        )
        run(
            f'python -m gleamlm.inference.cli --model {TEST_CKPT_DIR}/final.pt --prompt "介绍一下你自己"',
            "生成样例",
            conda_env=args.conda_env,
        )

        if os.path.exists(TEST_CKPT_DIR):
            shutil.rmtree(TEST_CKPT_DIR)
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

        print("\n>>> Level 2 完成!")

    elif args.level == 3:
        print("\n" + "=" * 60)
        print("  Level 3: 全量正式训练")
        print("=" * 60)

        cmd = f"python manual/pretrain.py --model manual/configs/{v}.yaml"
        ok = run(cmd, f"全量训练 ({v})", conda_env=args.conda_env)
        if not ok:
            print("\n[!] 训练异常退出")
            return

        print("\n>>> 训练完成，开始验证...")
        run(
            f"python -m tools.eval_runner --model {ckpt_dir}/final.pt --data_dir data/{v}/pretrain --benchmarks ppl --batch_size 4",
            "完整 PPL 评估",
            conda_env=args.conda_env,
        )
        run(
            f'python -m gleamlm.inference.cli --model {ckpt_dir}/final.pt --prompt "介绍一下你自己"',
            "生成样例",
            conda_env=args.conda_env,
        )
        print("\n>>> Level 3 完成!")


if __name__ == "__main__":
    main()
