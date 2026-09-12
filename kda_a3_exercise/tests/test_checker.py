"""The acceptance harness must reject common, numerically wrong submissions."""

import json
from pathlib import Path
import zipfile

import pytest
import torch

from kda_a3_exercise.cases import load_cases
from kda_a3_exercise.build_package import build_archive
from kda_a3_exercise.check import check_case, main
from kda_a3_exercise.contract import compare_output
from kda_a3_exercise.golden import forward, normalize_qk, recurrent_reference
from kda_a3_exercise.upstream.fla_naive import naive_recurrent_kda


@pytest.mark.parametrize("case", load_cases("core") + load_cases("extended"), ids=lambda case: case["id"])
def test_chunk_golden_agrees_with_independent_recurrent_reference(case):
    assert check_case(recurrent_reference, case, audit_host=False)["passed"]


@pytest.mark.parametrize("mistake", ["zero", "state", "dtype", "nan", "tail", "scale", "normalization", "mutate"])
def test_harness_rejects_known_mistakes(mistake):
    def wrong(q, k, v, g, beta, initial_state=None, *, aux=None):
        if mistake == "mutate":
            q.add_(1)
        o, state = forward(q, k, v, g, beta, initial_state)
        if mistake == "zero":
            o = torch.zeros_like(o)
        elif mistake == "state":
            state = torch.zeros_like(state)
        elif mistake == "dtype":
            o = o.float()
        elif mistake == "nan":
            o[0, 0, 0, 0] = float("nan")
        elif mistake == "tail":
            o[:, -1] = 0
        elif mistake in ("scale", "normalization"):
            qn, kn = (normalize_qk(q), normalize_qk(k)) if mistake == "scale" else (q, k)
            o, state = naive_recurrent_kda(qn, kn, v, g, beta, scale=1 if mistake == "scale" else None,
                                           initial_state=initial_state, output_final_state=True)
        return o, state

    result = check_case(wrong, load_cases("smoke")[-1], audit_host=False)
    assert not result["passed"]
    if mistake == "mutate":
        assert "modified input q" in result["error"]


def test_global_error_metric_rejects_zero_under_loose_pointwise_floor():
    expected = torch.full((1, 2, 1, 128), 1e-5, dtype=torch.bfloat16)
    result = compare_output("o", torch.zeros_like(expected), expected)
    assert result["bad_elements"] == 0
    assert result["relative_l2"] == 1
    assert not result["passed"]


@pytest.mark.parametrize("name,dtype", [("o", torch.bfloat16), ("final_state", torch.float32)])
def test_absolute_floor_is_relaxed_but_relative_budget_stays_two_percent(name, dtype):
    expected = torch.linspace(-1, 1, 1024).to(dtype)
    close = (expected.float() * 1.015).to(dtype)
    wrong = (expected.float() * 1.03).to(dtype)
    assert compare_output(name, close, expected)["passed"]
    assert not compare_output(name, wrong, expected)["passed"]


def test_cli_template_fails_and_writes_report(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    assert main(["--submission", str(root / "submission.py"), "--suite", "smoke", "--report", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["passed"]
    assert len(report["results"]) == 3
    assert all("NotImplementedError" in item["error"] for item in report["results"])
    assert report["kernel_timing"] is None
    assert not report["device_execution_verified"]


def test_cli_golden_mode_is_clearly_labeled(tmp_path):
    output = tmp_path / "golden.json"
    assert main(["--check-golden", "--suite", "smoke", "--report", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["run_kind"] == "golden_harness_check"
    assert not report["full_core_suite"]


def test_cli_unknown_case_does_not_silently_skip():
    with pytest.raises(SystemExit) as error:
        main(["--check-golden", "--case", "misspelled_case"])
    assert error.value.code == 2


def test_archive_is_reproducible_and_includes_license_and_template(tmp_path):
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    assert build_archive(first) == build_archive(second)
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert len(names) == 24
        assert "kda_a3_exercise/upstream/LICENSE.fla" in names
        assert b"NotImplementedError" in archive.read("kda_a3_exercise/submission.py")
        assert all("__pycache__" not in name and "results/" not in name for name in names)
