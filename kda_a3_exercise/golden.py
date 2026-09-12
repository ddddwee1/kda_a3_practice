"""Public-model PyTorch golden; no imports from EasyASC or its KDA projects.

KimiDeltaAttention calls FLA with use_qk_l2norm_in_kernel=True. The wrapper
below translates that normalization to PyTorch and calls the vendored FLA
chunk implementation. The recurrent reference is used only for independent
verification. See upstream/SOURCES.json and upstream/LICENSE.fla.
"""

from typing import Dict, Optional, Tuple

import torch

from .contract import CONTRACT, INPUT_NAMES, validate_inputs
from .upstream.fla_naive import naive_chunk_kda, naive_recurrent_kda


def normalize_qk(x: torch.Tensor) -> torch.Tensor:
    """FLA l2norm_fwd: fp32 reduction, sqrt(sum(x*x)+eps), same-dtype store."""
    xf = x.float()
    inverse_norm = 1.0 / torch.sqrt((xf * xf).sum(dim=-1, keepdim=True)
                                    + CONTRACT["qk_l2norm_eps"])
    return (xf * inverse_norm).to(x.dtype)


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
            beta: torch.Tensor, initial_state: Optional[torch.Tensor] = None,
            *, aux: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Primary golden: chunk prefill, with state carried between chunks."""
    return chunk_prefill(q, k, v, g, beta, initial_state, chunk_size=CONTRACT["golden_chunk_size"])


def chunk_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                  beta: torch.Tensor, initial_state: Optional[torch.Tensor] = None,
                  chunk_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FLA's chunk algebra with a bounded CPU working set and neutral tails.

The vendored chunk function requires aligned lengths. Padding uses zero
q/k/v, zero log decay and zero beta: padded tokens produce zero output and
leave the final state unchanged. Participant inputs are never padded by the
checker. There is no token-recurrent fallback in this path.
"""
    validate_inputs(q, k, v, g, beta, initial_state)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size not in (32, 64):
        raise ValueError("golden chunk_size must be 32 or 64")
    qn, kn = normalize_qk(q), normalize_qk(k)
    sequence_length = q.shape[1]
    outputs = []
    state = initial_state
    for start in range(0, sequence_length, chunk_size):
        valid = min(chunk_size, sequence_length - start)
        chunk_inputs = []
        for value in (qn, kn, v, g, beta):
            part = value[:, start:start + valid]
            if valid < chunk_size:
                padding = value.new_zeros((value.shape[0], chunk_size - valid) + tuple(value.shape[2:]))
                part = torch.cat((part, padding), dim=1)
            chunk_inputs.append(part.contiguous())
        out, state = naive_chunk_kda(
            *chunk_inputs, scale=q.shape[-1] ** -0.5, initial_state=state,
            output_final_state=True, chunk_size=chunk_size)
        outputs.append(out[:, :valid])
    return torch.cat(outputs, dim=1).contiguous(), state.contiguous()


def recurrent_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                        beta: torch.Tensor, initial_state: Optional[torch.Tensor] = None,
                        *, aux: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent token recurrence for verifying the chunk-prefill golden."""
    validate_inputs(q, k, v, g, beta, initial_state)
    return naive_recurrent_kda(
        normalize_qk(q), normalize_qk(k), v, g, beta,
        scale=q.shape[-1] ** -0.5, initial_state=initial_state, output_final_state=True)


def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
             beta: torch.Tensor, do: torch.Tensor, dht: torch.Tensor,
             initial_state: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
    """Optional autograd reference, not a backward submission interface.

Gradients are those of this PyTorch forward (including normalization/casts).
They are not a claim of bitwise parity with optimized FLA backward kernels.
"""
    validate_inputs(q, k, v, g, beta, initial_state)
    expected = {"do": (v.shape, v.dtype),
                "dht": ((q.shape[0], v.shape[2], q.shape[-1], v.shape[-1]), torch.float32)}
    for name, value in (("do", do), ("dht", dht)):
        shape, dtype = expected[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
            raise ValueError(name + " has an invalid shape or dtype")
        if value.device != q.device or not torch.isfinite(value).all().item():
            raise ValueError(name + " must be finite and on the input device")
    with torch.enable_grad():
        leaves = [value.detach().clone().requires_grad_(True) if value is not None else None
                  for value in (q, k, v, g, beta, initial_state)]
        out, state = forward(*leaves)
        loss = (out.float() * do.float()).sum() + (state * dht).sum()
        active = [value for value in leaves if value is not None]
        gradients = iter(torch.autograd.grad(loss, active))
        result = {"d" + name: next(gradients).detach() if value is not None else None
                  for name, value in zip(INPUT_NAMES, leaves)}
    result["dh0"] = result.pop("dinitial_state")
    return result
