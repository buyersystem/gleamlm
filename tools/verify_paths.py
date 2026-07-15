"""Verify all import paths and config constants work correctly."""

import argparse
import os

from gleamlm.dataset.dataset import LMDataset
from gleamlm.models.model import GleamLMModel
from gleamlm.tokenizer.tokenizer import BBPETokenizer
from gleamlm.utils.config import DEFAULT_TOKENIZER_PATH, load_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

parser = argparse.ArgumentParser(description="路径验证")
parser.add_argument("--variant", choices=["nano", "lite", "pro"], default="nano")
args = parser.parse_args()

cfg = load_config(
    os.path.join(_PROJECT_ROOT, "configs", f"{args.variant}.yaml"), model_name=args.variant
)
data_dir = cfg.data.data_dir
checkpoint_dir = cfg.data.checkpoint_dir

print("=== Path Resolution ===")
print(f"Tokenizer: {DEFAULT_TOKENIZER_PATH}")
print(f"  exists: {os.path.exists(DEFAULT_TOKENIZER_PATH)}")
print(f"Checkpoint dir: {checkpoint_dir}")
print(f"Data dir: {data_dir}")
print(f"  exists: {os.path.exists(data_dir)}")

# Test tokenizer loading
tok = BBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
print(f"Tokenizer: vocab={tok.get_vocab_size()} OK")

# Test model creation from config
model_cfg = cfg.model
m = GleamLMModel(
    vocab_size=model_cfg.vocab_size,
    d_model=model_cfg.d_model,
    num_layers=model_cfg.num_layers,
    num_heads=model_cfg.num_heads,
    num_kv_heads=model_cfg.num_kv_heads,
    d_ff=model_cfg.d_ff,
    dropout=model_cfg.dropout,
    max_seq_len=model_cfg.max_seq_len,
    pad_token_id=tok.pad_id,
    use_flash_attn=getattr(model_cfg, "use_flash_attn", False),
)
print(f"Model: {sum(p.numel() for p in m.parameters()):,} params OK")

# Test dataset
ds = LMDataset(data_dir, tok, 128, "valid")
print(f"Dataset: {len(ds)} samples OK")

print("\nAll imports and paths verified from project root CWD.")
