"""Input-independent, read-only matrix helpers supplied by the evaluator."""

from typing import Dict

import torch

from .contract import CONTRACT


def make_auxiliary() -> Dict:
    """Build exact bf16 constants; this runs outside the participant guard."""
    result = {}
    for size in CONTRACT["auxiliary_matrices"]["sizes"]:
        ones = torch.ones((size, size), dtype=torch.bfloat16)
        result[size] = {"lower": torch.tril(ones), "upper": torch.triu(ones),
                        "ones": ones, "identity": torch.eye(size, dtype=torch.bfloat16)}
    return result


def clone_auxiliary(matrices: Dict) -> Dict:
    return {size: {name: value.clone() for name, value in group.items()}
            for size, group in matrices.items()}


def assert_auxiliary_unchanged(actual: Dict, expected: Dict) -> None:
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        raise ValueError("submission modified auxiliary matrix sizes")
    for size, group in expected.items():
        current = actual[size]
        if not isinstance(current, dict) or current.keys() != group.keys():
            raise ValueError("submission modified auxiliary matrix names")
        for name, original in group.items():
            value = current[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != original.dtype
                    or value.device != original.device or value.shape != original.shape
                    or not value.is_contiguous() or not torch.equal(value, original)):
                raise ValueError("submission modified auxiliary matrix %s/%s" % (size, name))
