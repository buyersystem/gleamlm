"""快速 PPL 评估。用法: python tools/eval_ppl.py [--max_batches 100]"""

import argparse
import os

import torch

from gleamlm import load_model_for_inference
from gleamlm.data.dataset import LMDataset, collate_fn
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)  # tools/ 在项目根下
import math

from torch.utils.data import DataLoader

from gleamlm.evaluation.ppl import _compute_raw_loss


def compute_ppl_fast(model, data_loader, device, max_batches=None, pad_token_id=0):
    """Fast PPL computation — delegates to shared _compute_raw_loss."""
    total_loss, total_tokens, _ = _compute_raw_loss(
        model,
        data_loader,
        device,
        pad_token_id,
        max_batches,
    )
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser(description="命令行 PPL 评估")
    parser.add_argument(
        "--variant", choices=["nano", "lite", "pro"], default="nano", help="模型变体"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径 (默认: checkpoints/{variant}/best_model.pt)",
    )
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="数据目录 (默认: data/{variant}_data)"
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(_PROJECT_ROOT, "checkpoints", args.variant, "best_model.pt")
    if args.data_dir is None:
        args.data_dir = os.path.join(_PROJECT_ROOT, "data", f"{args.variant}_data")

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config = load_model_for_inference(args.model, device)
    total, _ = model.get_num_params()
    max_seq_len = config.get("max_seq_len", 1024)
    print(f"Model: {total / 1e6:.2f}M params, max_seq_len={max_seq_len}")

    tokenizer = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
    print(f"Vocab: {len(tokenizer)}")

    print(f"Data dir: {args.data_dir}")

    for split in ["valid", "test"]:
        txt_path = os.path.join(args.data_dir, f"{split}.txt")
        if not os.path.exists(txt_path):
            print(f"  Skip {split}: no data file")
            continue
        print(f"\nEvaluating {split} set...")
        ds = LMDataset(args.data_dir, tokenizer, max_seq_len, split)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
            num_workers=0,
        )
        n_batches = min(args.max_batches, len(dl)) if args.max_batches else len(dl)
        print(f"  Samples: {len(ds)}, Batches: {n_batches}")
        avg = compute_ppl_fast(model, dl, device, args.max_batches, tokenizer.pad_id)
        print(f"  {split}: avg_loss={avg:.4f}, PPL={math.exp(avg):.2f}")


if __name__ == "__main__":
    main()
