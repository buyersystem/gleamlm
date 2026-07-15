"""GleamLM 模型量化导出。FP32 → FP16，体积减半。

用法：
    python tools/quantize.py --input checkpoints/best_model.pt --output checkpoints/model_fp16.pt
"""

import argparse
import os
import sys

from gleamlm.deploy.quantize import quantize_to_fp16


def main() -> None:
    parser = argparse.ArgumentParser(description="GleamLM FP16 量化导出")
    parser.add_argument("--input", type=str, required=True, help="输入模型路径")
    parser.add_argument("--output", type=str, required=True, help="输出 FP16 模型路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 输入模型不存在: {args.input}")
        sys.exit(1)

    quantize_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
