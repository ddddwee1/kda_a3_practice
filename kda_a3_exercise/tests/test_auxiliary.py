"""Matrix orientation, input independence and read-only fixture enforcement."""

import pytest
import torch

from kda_a3_exercise.auxiliary import make_auxiliary, clone_auxiliary, assert_auxiliary_unchanged
from kda_a3_exercise.cases import load_cases
from kda_a3_exercise.check import check_case
from kda_a3_exercise.golden import forward
from kda_a3_exercise.host_guard import HostGuard, HostOperationError


@pytest.mark.parametrize("size", [32, 64, 128])
def test_triangular_constants_implement_both_cumsum_orientations(size):
    matrices = make_auxiliary()[size]
    data = torch.arange(size * 3, dtype=torch.float32).reshape(size, 3) / 32
    prefix = data.cumsum(dim=0)
    torch.testing.assert_close(matrices["lower"].float() @ data, prefix, rtol=0, atol=0)
    torch.testing.assert_close(data.t() @ matrices["upper"].float(), prefix.t(), rtol=0, atol=0)
    torch.testing.assert_close(matrices["lower"].float() + matrices["upper"].float(),
                               matrices["ones"].float() + matrices["identity"].float(), rtol=0, atol=0)
    assert all(value.dtype == torch.bfloat16 and value.is_contiguous() for value in matrices.values())


def test_auxiliary_generation_is_input_independent_and_does_not_change_rng():
    rng = torch.random.get_rng_state().clone()
    matrices = make_auxiliary()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert_auxiliary_unchanged(matrices, make_auxiliary())


def test_readonly_constants_can_be_selected_but_not_computed_with_on_host():
    matrices = make_auxiliary()
    with HostGuard():
        lower = matrices[64]["lower"].view(1, 64, 64)
        assert lower.shape == (1, 64, 64)
    with pytest.raises(HostOperationError):
        with HostGuard():
            matrices[64]["lower"].fill_(0)


@pytest.mark.parametrize("mutation", ["value", "keys", "replacement"])
def test_auxiliary_mutations_are_detected_even_with_correct_outputs(mutation):
    def submitted(q, k, v, g, beta, initial_state=None, *, aux=None):
        if mutation == "value":
            aux[64]["lower"][0, 0] = 0
        elif mutation == "keys":
            aux.clear()
        else:
            aux[32]["ones"] = torch.empty((1,))
        return forward(q, k, v, g, beta, initial_state)
    result = check_case(submitted, load_cases("smoke")[0], audit_host=False)
    assert not result["passed"]
    assert "auxiliary matrix" in result["error"]


def test_auxiliary_copies_do_not_alias_the_evaluator_constants():
    original = make_auxiliary()
    copied = clone_auxiliary(original)
    copied[32]["lower"].zero_()
    assert original[32]["lower"][0, 0].item() == 1
