"""Source checks, independent chunk algebra, state continuity and gradients."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from kda_a3_exercise.cases import load_cases, make_inputs
from kda_a3_exercise.contract import compare_output, validate_inputs
from kda_a3_exercise import golden
from kda_a3_exercise.golden import backward, chunk_prefill, forward, normalize_qk, recurrent_reference
from kda_a3_exercise.upstream.fla_naive import naive_chunk_kda, naive_recurrent_kda


def small_case(t=7, h=2, hv=2, k=8, v=6):
    return dict(load_cases("core")[0], B=1, T=t, H=h, HV=hv, K=k, V=v,
                gate="weak", beta="random", state="nonzero")


def test_upstream_snapshot_hash_and_license():
    root = Path(__file__).resolve().parents[1] / "upstream"
    manifest = json.loads((root / "SOURCES.json").read_text())
    assert hashlib.sha256((root / "fla_naive.py").read_bytes()).hexdigest() == manifest["vendored_sha256"]
    license_record = next(item for item in manifest["fla"]["files"] if item["path"] == "LICENSE")
    assert hashlib.sha256((root / "LICENSE.fla").read_bytes()).hexdigest() == license_record["sha256"]
    assert manifest["mathematical_changes_to_vendored_reference"] == []


def test_normalization_uses_epsilon_inside_sqrt_and_bf16_store():
    x = torch.tensor([[[[3.0, 4.0], [0.0, 0.0], [1e-5, 0.0]]]], dtype=torch.bfloat16)
    actual = normalize_qk(x)
    expected = torch.tensor([0.6, 0.8], dtype=torch.bfloat16)
    torch.testing.assert_close(actual[0, 0, 0], expected, rtol=0, atol=0)
    assert torch.equal(actual[0, 0, 1], torch.zeros(2, dtype=torch.bfloat16))
    assert 0.009 < actual[0, 0, 2, 0].item() < 0.011
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("chunk_size,hv", [(32, 2), (64, 2), (32, 4), (64, 4)])
def test_recurrence_matches_independent_upstream_chunk_algebra(chunk_size, hv):
    data = make_inputs(small_case(t=2 * chunk_size, hv=hv))
    args = [normalize_qk(data["q"]).float(), normalize_qk(data["k"]).float(),
            data["v"].float(), data["g"], data["beta"]]
    kwargs = dict(initial_state=data["initial_state"], output_final_state=True)
    recurrent = naive_recurrent_kda(*args, **kwargs)
    chunked = naive_chunk_kda(*args, chunk_size=chunk_size, **kwargs)
    for actual, expected in zip(chunked, recurrent):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5)
    public = forward(**data)
    # Equivalent chunk/recurrent reductions need not round to identical bf16.
    torch.testing.assert_close(public[0].float(), recurrent[0].bfloat16().float(), rtol=8e-3, atol=2e-6)
    torch.testing.assert_close(public[1], recurrent[1], rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("chunk_size", [32, 64])
def test_chunk_tail_padding_preserves_output_and_final_state(length, chunk_size):
    data = make_inputs(small_case(t=length, hv=4))
    actual = chunk_prefill(**data, chunk_size=chunk_size)
    expected = recurrent_reference(**data)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=8e-3, atol=2e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-4, atol=3e-6)


def test_primary_golden_uses_chunk_algebra_without_recurrent_fallback(monkeypatch):
    calls = []
    source_chunk = golden.naive_chunk_kda

    def observed_chunk(*args, **kwargs):
        calls.append((args[0].shape[1], kwargs["chunk_size"]))
        return source_chunk(*args, **kwargs)

    def forbidden_recurrent(*args, **kwargs):
        raise AssertionError("primary golden must not call token recurrence")

    monkeypatch.setattr(golden, "naive_chunk_kda", observed_chunk)
    monkeypatch.setattr(golden, "naive_recurrent_kda", forbidden_recurrent)
    out, state = golden.forward(**make_inputs(small_case(t=129)))
    assert calls == [(64, 64)] * 3
    assert out.shape[1] == 129
    assert torch.isfinite(state).all()


@pytest.mark.parametrize("value", [0, 16, 33, True])
def test_invalid_reference_chunk_size_is_rejected(value):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_prefill(**make_inputs(small_case()), chunk_size=value)


def test_one_token_closed_form_with_nonzero_state():
    data = make_inputs(small_case(t=1, h=1, hv=1))
    q = normalize_qk(data["q"])[:, 0].float()
    k = normalize_qk(data["k"])[:, 0].float()
    decayed = data["initial_state"] * data["g"][:, 0].exp().unsqueeze(-1)
    recalled = torch.matmul(k.unsqueeze(-2), decayed).squeeze(-2)
    update = torch.matmul(k.unsqueeze(-1), (data["v"][:, 0].float() - recalled).unsqueeze(-2))
    expected_state = decayed + data["beta"][:, 0, :, None, None] * update
    expected_out = torch.matmul((q * q.shape[-1] ** -0.5).unsqueeze(-2), expected_state).squeeze(-2)
    actual_out, actual_state = forward(**data)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual_out[:, 0], expected_out.bfloat16(), rtol=0, atol=0)


def test_state_carry_across_an_arbitrary_tail_boundary():
    data = make_inputs(small_case(t=67))
    whole = forward(**data)
    left = {name: value[:, :33].contiguous() for name, value in data.items() if name != "initial_state"}
    right = {name: value[:, 33:].contiguous() for name, value in data.items() if name != "initial_state"}
    left_out, state = forward(**left, initial_state=data["initial_state"])
    right_out, final = forward(**right, initial_state=state)
    torch.testing.assert_close(torch.cat([left_out, right_out], dim=1).float(), whole[0].float(), rtol=8e-3, atol=2e-6)
    torch.testing.assert_close(final, whole[1], rtol=3e-4, atol=2e-6)


def test_zero_beta_preserves_only_decayed_initial_state():
    data = make_inputs(dict(small_case(), beta="zero"))
    _, final = forward(**data)
    expected = data["initial_state"] * data["g"].sum(dim=1).exp().unsqueeze(-1)
    torch.testing.assert_close(final, expected, rtol=2e-6, atol=2e-7)
    data["initial_state"] = None
    out, final = forward(**data)
    assert torch.count_nonzero(out).item() == torch.count_nonzero(final).item() == 0


def test_input_generation_is_reproducible_and_does_not_change_global_rng():
    before = torch.random.get_rng_state().clone()
    first = make_inputs(small_case())
    second = make_inputs(small_case())
    assert torch.equal(before, torch.random.get_rng_state())
    for name in first:
        assert torch.equal(first[name], second[name])


@pytest.mark.parametrize("suite", ["core", "extended", "performance"])
def test_all_public_inputs_obey_the_contract(suite):
    for case in load_cases(suite):
        assert case["T"] >= 128
        validate_inputs(**make_inputs(case))


@pytest.mark.parametrize("bad", ["dtype", "gate", "beta", "nonfinite", "layout", "state", "empty", "heads"])
def test_invalid_inputs_are_rejected(bad):
    data = make_inputs(small_case())
    if bad == "dtype":
        data["q"] = data["q"].float()
    elif bad == "gate":
        data["g"][0, 0, 0, 0] = 0.1
    elif bad == "beta":
        data["beta"][0, 0, 0] = 1.1
    elif bad == "nonfinite":
        data["k"][0, 0, 0, 0] = float("nan")
    elif bad == "layout":
        data["q"] = data["q"].transpose(-1, -2)
    elif bad == "state":
        data["initial_state"] = data["initial_state"].transpose(-1, -2).contiguous()
    elif bad == "empty":
        data = {name: value[:, :0].contiguous() if name != "initial_state" else value for name, value in data.items()}
    elif bad == "heads":
        data["v"] = data["v"][:, :, :1].contiguous()
    with pytest.raises((TypeError, ValueError)):
        forward(**data)


@pytest.mark.parametrize("name,index", [("g", (0, 0, 0, 0)), ("initial_state", (0, 0, 0, 0))])
def test_backward_final_state_loss_matches_finite_difference(name, index):
    data = make_inputs(small_case(t=2, h=1, hv=1, k=4, v=3))
    do = torch.zeros_like(data["v"])
    dht = torch.ones_like(data["initial_state"])
    grads = backward(**data, do=do, dht=dht)
    assert set(grads) == {"dq", "dk", "dv", "dg", "dbeta", "dh0"}
    plus = {key: value.clone() for key, value in data.items()}
    minus = {key: value.clone() for key, value in data.items()}
    epsilon = 1e-3
    plus[name][index] += epsilon
    minus[name][index] -= epsilon
    numerical = (forward(**plus)[1].sum() - forward(**minus)[1].sum()) / (2 * epsilon)
    grad = grads["dh0" if name == "initial_state" else "d" + name][index]
    torch.testing.assert_close(grad, numerical, rtol=2e-2, atol=2e-3)
    assert grads["dq"].dtype == torch.bfloat16
    assert grads["dg"].dtype == torch.float32


def test_backward_without_initial_state_and_with_output_loss():
    data = make_inputs(dict(small_case(), state="none"))
    do = torch.ones_like(data["v"])
    dht = torch.zeros((1, 2, 8, 6))
    grads = backward(**data, do=do, dht=dht)
    assert grads["dh0"] is None
    assert grads["dq"].abs().sum().item() > 0
    assert all(torch.isfinite(value).all() for value in grads.values() if value is not None)


def test_bf16_inter_chunk_state_storage_fits_the_two_percent_budget():
    case = next(case for case in load_cases("core") if case["id"] == "prefill2048_weak_decay")
    data = make_inputs(case)
    expected = forward(**data)
    q, k = normalize_qk(data["q"]), normalize_qk(data["k"])
    state = data["initial_state"].bfloat16().float()
    outputs = []
    for start in range(0, case["T"], 64):
        args = [value[:, start:start + 64].contiguous() for value in (q, k, data["v"], data["g"], data["beta"])]
        out, state = naive_chunk_kda(*args, initial_state=state, output_final_state=True, chunk_size=64)
        state = state.bfloat16().float()
        outputs.append(out)
    assert compare_output("o", torch.cat(outputs, dim=1).contiguous(), expected[0])["passed"]
    assert compare_output("final_state", state, expected[1])["passed"]
