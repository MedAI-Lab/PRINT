"""Shared PRINT input, fold loader, VPR extraction and diagnostic inference."""
import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from safetensors.torch import load_file

RUNTIME = Path(__file__).resolve().parent / "runtime"
sys.path[:0] = [str(RUNTIME / "scripts"), str(RUNTIME / "third_party/URFM"), str(RUNTIME)]
from factor_model_factory import build_reconstruction_model
from spatial_modules import install_spatial_arm
from print_model.preprocessing import load_views


def load_mae_encoder(path, device="cpu"):
    """Load the ViT-B/16 encoder from a full MAE checkpoint or an encoder state file."""
    from models_vit import vit_base_patch16
    model = vit_base_patch16(num_classes=0, global_pool=False)
    if Path(path).suffix == ".safetensors":
        state = load_file(str(path))
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = {k: v for k, v in checkpoint["model"].items()
                 if not k.startswith("decoder_") and k != "mask_token"}
    model.load_state_dict(state, strict=True)
    return model.to(device)


@torch.inference_mode()
def load_fold(spec, weights, device):
    fold = spec["fold"]
    path = weights / spec["checkpoint"]
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError(f"Weight checksum mismatch in fold {fold}")
    if spec.get("agate") != "no-beta":
        raise ValueError("Unsupported view-gate configuration in the fold manifest")
    model = build_reconstruction_model(None, "CP_F30", spec["config"],
                      4200 + fold, spec["geometry"])
    install_spatial_arm(model, "LOCAL_CAPACITY_CONTROL")
    if path.suffix == ".safetensors":
        for name in ("decoder_embed", "decoder_pos_embed", "decoder_blocks", "decoder_norm", "decoder_pred", "mask_token"):
            delattr(model, name)
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def factor_batch(model, images, mask):
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if images.is_cuda else nullcontext()
    with precision:
        native = model.factor_forward_details(images, mask)["prediction"]
        flipped = model.factor_forward_details(images.flip(-1), mask)["prediction"]
    return ((native.float() + flipped.float()) / 2).cpu().numpy()


def predict_fold(spec, weights, images, *, vpr_only=False):
    fold = spec["fold"]
    model = load_fold(spec, weights, images.device)
    mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
    values = factor_batch(model, images, mask)[0]
    if values.shape != (13,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid VPR in fold {fold}")
    if vpr_only:
        return {"fold": fold, "vpr": values.tolist()}
    head = spec["head"]
    standardized = (values - head["mean"]) / head["sd"]
    score = expit(standardized @ head["coef"] + head["intercept"])
    return {"fold": fold, "vpr": values.tolist(), "risk": float(score)}


