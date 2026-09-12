"""Positive whitelist tests and common host-compute bypass regression cases."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kda_a3_exercise.cases import load_cases
from kda_a3_exercise.check import check_case, load_submission, main
from kda_a3_exercise import host_guard
from kda_a3_exercise.host_guard import HostGuard, HostOperationError, audit_source


def test_views_and_uninitialized_allocation_are_allowed():
    x = torch.randn(2, 3)
    with HostGuard() as audit:
        flat = x.view(6).reshape(3, 2).flatten()
        same = flat.unsqueeze(0).squeeze(0).detach()
        bits = x.view(torch.int32)
        allocations = [torch.empty(2, 3), torch.empty_like(x), x.new_empty((2, 3)),
                       torch.empty_strided((2, 3), (3, 1))]
        assert same.shape == (6,)
        assert bits.dtype == torch.int32
        assert len(allocations) == 4
    assert not audit.violations
    assert audit.host_operations


@pytest.mark.parametrize("operation", [
    lambda x: x + 1,
    lambda x: torch.exp(x),
    lambda x: x.sum(),
    lambda x: x @ x,
    lambda x: torch.nn.functional.normalize(x),
    lambda x: torch.cumsum(x, dim=0),
    lambda x: x.float(),
    lambda x: x.clone(),
    lambda x: x[0],
    lambda x: x.transpose(0, 1),
    lambda x: torch.cat([x, x]),
    lambda x: torch.zeros_like(x),
    lambda x: torch.ones(2, 2),
    lambda x: torch.full((2, 2), 3.0),
    lambda x: torch.tensor([1.0, 2.0]),
    lambda x: x.fill_(0),
    lambda x: torch.ops.aten.add.Tensor(x, x),
])
def test_numeric_and_materializing_host_ops_are_rejected(operation):
    x = torch.randn(2, 2).bfloat16()
    with pytest.raises(HostOperationError):
        with HostGuard():
            operation(x)


@pytest.mark.parametrize("operation", [
    lambda x: x.numpy(),
    lambda x: x.tolist(),
    lambda x: x.item(),
    lambda x: x.data_ptr(),
    lambda x: x.untyped_storage(),
    lambda x: float(x),
])
def test_tensor_data_cannot_escape_to_python_or_numpy(operation):
    x = torch.tensor(0.5)
    with pytest.raises(HostOperationError):
        with HostGuard():
            operation(x)


def test_reshape_that_requires_a_copy_is_rejected():
    noncontiguous = torch.arange(6).reshape(2, 3).t()
    with pytest.raises(HostOperationError, match="clone"):
        with HostGuard():
            noncontiguous.reshape(6)


def test_catching_the_exception_does_not_clear_the_violation():
    x = torch.randn(2, 2)
    with pytest.raises(HostOperationError, match="caught"):
        with HostGuard():
            try:
                x + 1
            except HostOperationError:
                pass


def test_missing_runtime_is_not_accepted_as_a_submission():
    def no_kernel(q, k, v, g, beta, initial_state=None, *, aux=None):
        return torch.empty_like(v), torch.empty((q.shape[0], v.shape[2], q.shape[-1], v.shape[-1]))
    result = check_case(no_kernel, load_cases("smoke")[0])
    assert not result["passed"]
    assert "no trusted EasyASC" in result["error"]
    assert result["host_audit"]["enabled"]


def test_submission_host_compute_is_audited_by_default():
    def host_compute(q, k, v, g, beta, initial_state=None, *, aux=None):
        return q + 1, initial_state
    result = check_case(host_compute, load_cases("smoke")[0])
    assert not result["passed"]
    assert "aten.add" in result["error"]
    assert result["host_audit"]["violations"]


@pytest.mark.parametrize("statement", [
    "import numpy as np", "import ctypes", "import threading",
    "from kda_a3_exercise.golden import forward", "from torch.utils._python_dispatch import _disable_current_modes",
])
def test_static_audit_rejects_direct_bypasses(tmp_path, statement):
    source = tmp_path / "solution.py"
    source.write_text(statement + "\n")
    with pytest.raises(HostOperationError):
        audit_source(source)


def test_static_audit_checks_local_helpers(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("from helper import forward\n")
    (tmp_path / "helper.py").write_text("import numpy\n")
    with pytest.raises(HostOperationError, match="numpy"):
        audit_source(source)


def test_module_level_tensor_computation_is_rejected(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("import torch\ncached = torch.ones(1)\ndef forward(*args, **kwargs):\n    return cached\n")
    with pytest.raises(HostOperationError):
        load_submission(source)


def test_cli_has_no_submission_host_audit_disable_flag():
    with pytest.raises(SystemExit) as error:
        main(["--no-host-audit"])
    assert error.value.code == 2


def test_runtime_root_alone_does_not_trust_an_arbitrary_helper(tmp_path):
    source = tmp_path / "helper.py"
    source.write_text("def compute(x):\n    return x + 1\n")
    namespace = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    x = torch.ones(1)
    with pytest.raises(HostOperationError):
        with HostGuard(runtime_root=tmp_path):
            namespace["compute"](x)


def test_namespace_package_runtime_discovery(tmp_path, monkeypatch):
    (tmp_path / "torchplugin.py").write_text("")
    spec = SimpleNamespace(origin=None, submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(host_guard.importlib.util, "find_spec", lambda name: spec)
    assert host_guard.discover_runtime_root() == tmp_path


def test_trusted_runtime_does_not_exempt_participant_callbacks(tmp_path):
    source = tmp_path / "torchplugin.py"
    source.write_text("class OpExec:\n    def __call__(self, x, callback=None):\n        if callback is not None:\n            return callback(x)\n        return x + 1\n")
    namespace = {"__name__": "easyasc.torchplugin"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    runtime = namespace["OpExec"]()
    x = torch.ones(1)
    with HostGuard(runtime_root=tmp_path, require_runtime=True) as audit:
        result = runtime(x)
    assert result.item() == 2
    assert audit.runtime_operations > 0
    with pytest.raises(HostOperationError, match="aten.add"):
        with HostGuard(runtime_root=tmp_path):
            runtime(x, lambda tensor: tensor + 1)
