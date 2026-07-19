from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .models.model import GleamLMModel
from .utils.config import extract_checkpoint_config

try:
    __version__ = version("gleamlm")
except PackageNotFoundError:
    __version__ = "0.0.0"


def load_model_for_inference(
    model_path: str, device: str = "cuda", checkpoint: dict | None = None
) -> tuple[GleamLMModel, dict]:
    import torch

    if checkpoint is None:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

    config = extract_checkpoint_config(checkpoint)

    if "args" in checkpoint:
        tokenizer_path = getattr(checkpoint["args"], "tokenizer_path", None)
    elif "config" in checkpoint:
        tokenizer_path = checkpoint["config"].get("tokenizer_path", None)
    else:
        tokenizer_path = None

    config["dropout"] = 0.0

    model = GleamLMModel(**config).to(device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint: {unexpected}")

    model.eval()

    if checkpoint.get("dtype") == "float16":
        model = model.half()

    config["tokenizer_path"] = tokenizer_path

    return model, config
