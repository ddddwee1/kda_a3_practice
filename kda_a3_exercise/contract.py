"""Executable input/output contract for the public Kimi-Linear KDA core."""

import json
from pathlib import Path
from typing import Dict, Optional

import torch


CONTRACT_PATH = Path(__file__).with_name("contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
INPUT_NAMES = ("q", "k", "v", "g", "beta", "initial_state")
OUTPUT_NAMES = ("o", "final_state")


def validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                    beta: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> None:
    values = dict(zip(INPUT_NAMES, (q, k, v, g, beta, initial_state)))
    for name, value in values.items():
        if name == "initial_state" and value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + " must be a torch.Tensor")
        if not value.is_contiguous():
            raise ValueError(name + " must be contiguous")
        expected_dtype = torch.bfloat16 if name in ("q", "k", "v") else torch.float32
        if value.dtype != expected_dtype:
            raise TypeError(name + " must have dtype " + str(expected_dtype))
        if value.device != q.device:
            raise ValueError("all inputs must be on the same device")
        if not torch.isfinite(value).all().item():
            raise ValueError(name + " must contain only finite values")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k must share [B,T,H,K]; v must be [B,T,HV,V]")
    b, t, h, key_dim = q.shape
    hv, value_dim = v.shape[2:]
    if min(b, t, h, key_dim, hv, value_dim) <= 0:
        raise ValueError("all tensor dimensions must be positive")
    if v.shape[:2] != q.shape[:2] or hv % h:
        raise ValueError("q/k/v must share B,T and HV must be divisible by H")
    shapes = {"g": (b, t, hv, key_dim), "beta": (b, t, hv),
              "initial_state": (b, hv, key_dim, value_dim)}
    for name, shape in shapes.items():
        if values[name] is not None and tuple(values[name].shape) != shape:
            raise ValueError(name + " must have shape " + str(shape))
    if (g > 0).any().item():
        raise ValueError("g must be nonpositive natural-log decay increments")
    if ((beta < 0) | (beta > 1)).any().item():
        raise ValueError("beta must be in [0,1]")


def compare_output(name: str, actual: torch.Tensor, expected: torch.Tensor) -> Dict:
    """Require both pointwise bounds and relative L2; reject structural errors."""
    if not isinstance(actual, torch.Tensor):
        raise TypeError(name + " must be a torch.Tensor")
    if actual.shape != expected.shape:
        raise ValueError(name + " shape mismatch: " + str(tuple(actual.shape)))
    if actual.dtype != expected.dtype:
        raise TypeError(name + " dtype mismatch: " + str(actual.dtype))
    if not actual.is_contiguous():
        raise ValueError(name + " must be contiguous")
    actual_cpu = actual.detach().cpu().double()
    expected_cpu = expected.detach().cpu().double()
    if not torch.isfinite(expected_cpu).all().item():
        raise ValueError("golden produced nonfinite " + name)
    finite = torch.isfinite(actual_cpu)
    if not finite.all().item():
        return {"passed": False, "nonfinite": int((~finite).sum().item())}
    tolerance = CONTRACT["tolerances"][name]
    error = (actual_cpu - expected_cpu).abs()
    bounds = tolerance["atol"] + tolerance["rtol"] * expected_cpu.abs()
    bad = int((error > bounds).sum().item())
    reference_norm = torch.linalg.vector_norm(expected_cpu).item()
    relative_l2 = (torch.linalg.vector_norm(error).item() / reference_norm
                   if reference_norm > 0 else None)
    l2_passed = relative_l2 is None or relative_l2 <= tolerance["relative_l2"]
    return {"passed": bad == 0 and l2_passed, "bad_elements": bad,
            "max_abs": error.max().item(), "relative_l2": relative_l2,
            "max_tolerance_ratio": (error / bounds).max().item()}
