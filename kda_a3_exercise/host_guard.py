"""Basic in-process host-operation audit, not a hostile-code sandbox.

Torch modes cover participant tensor operations and data extraction. The
framework exception requires a direct EasyASC caller inside a real OpExec
call; an outer runtime frame alone does not exempt participant callbacks.
The official evaluator must keep this module and the runtime checkout fixed.
"""

import ast
from collections import Counter
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import sysconfig
from typing import Dict, Optional

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


_HERE = str(Path(__file__).resolve())
_TORCH_ROOT = Path(torch.__file__).resolve().parent
_STDLIB_ROOT = Path(sysconfig.get_path("stdlib")).resolve()

# These aliases preserve element order. Slices, transpose/permute, gather,
# contiguous copies and dtype casts are deliberately outside this policy.
_VIEWS = frozenset([
    "aten.view.default", "aten.view.dtype", "aten._unsafe_view.default",
    "aten._reshape_alias.default", "aten.reshape.default", "aten.alias.default",
    "aten.detach.default", "aten.squeeze.default", "aten.squeeze.dim",
    "aten.squeeze.dims", "aten.unsqueeze.default", "aten.flatten.using_ints",
])
_ALLOCATIONS = frozenset([
    "aten.empty.memory_format", "aten.empty_strided.default", "aten.empty_like.default",
    "aten.new_empty.default", "aten.new_empty_strided.default",
])
_EXTRACTION = frozenset([
    "numpy", "tolist", "item", "data_ptr", "storage", "untyped_storage",
    "__array__", "__dlpack__", "__dlpack_device__", "__reduce__", "__reduce_ex__",
    "__float__", "__int__", "__index__", "__bool__", "as_subclass",
])
_FORBIDDEN_IMPORTS = (
    "numpy", "cupy", "scipy", "ctypes", "numba", "threading", "multiprocessing",
    "concurrent.futures", "subprocess", "torch.utils._python_dispatch",
    "torch.overrides", "torch._C", "torch.library", "torch.utils.cpp_extension",
)


class HostOperationError(RuntimeError):
    """A participant performed an operation outside the host whitelist."""


def discover_runtime_root() -> Optional[Path]:
    """Resolve the evaluator's runtime before adding the submission directory."""
    try:
        spec = importlib.util.find_spec("easyasc")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    candidates = [Path(spec.origin).resolve().parent] if spec.origin is not None else []
    candidates.extend(Path(item).resolve() for item in (spec.submodule_search_locations or []))
    roots = {root for root in candidates if (root / "torchplugin.py").is_file()}
    return next(iter(roots)) if len(roots) == 1 else None


@lru_cache(maxsize=1024)
def _path(filename: str) -> Path:
    return Path(filename).resolve()


def _within(path: Path, root: Optional[Path]) -> bool:
    return root is not None and (path == root or root in path.parents)


def audit_source(path: Path, runtime_root: Optional[Path] = None) -> Dict:
    """Reject common alternate engines/reference imports, including local helpers."""
    entry = path.resolve()
    base = entry.parent
    pending, seen = [entry], set()
    while pending:
        source = pending.pop()
        if source in seen or _within(source, runtime_root):
            continue
        seen.add(source)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [module] + [module + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__"):
                    raise HostOperationError("dynamic execution/import is not allowed: %s:%d" % (source.name, node.lineno))
            for name in names:
                if "kda_a3_exercise" in name.split(".") or any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_IMPORTS):
                    raise HostOperationError("submission import is not allowed: %s (%s:%d)" % (name, source.name, node.lineno))
                parts = [part for part in name.split(".") if part and part != "*"]
                if not parts:
                    continue
                roots = [base, source.parent]
                for parent in roots:
                    candidate = parent.joinpath(*parts)
                    for local in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if local.is_file() and _within(local.resolve(), base):
                            pending.append(local.resolve())
    return {"files_checked": len(seen), "passed": True}


class _FunctionAudit(TorchFunctionMode):
    def __init__(self, owner):
        self.owner = owner

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name in _EXTRACTION:
            if self.owner.runtime_call():
                self.owner.runtime_operations += 1
            else:
                self.owner.reject("Tensor." + name)
        return func(*args, **(kwargs or {}))


class _DispatchAudit(TorchDispatchMode):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if self.owner.runtime_call():
            self.owner.runtime_operations += 1
        elif name in _VIEWS or name in _ALLOCATIONS:
            self.owner.host_operations[name] += 1
        elif name == "aten._to_copy.default" and self.owner.is_device_transfer(args, kwargs):
            self.owner.host_operations[name] += 1
        else:
            self.owner.reject(name)
        return func(*args, **kwargs)


class HostGuard:
    def __init__(self, runtime_root: Optional[Path] = None, phase: str = "forward",
                 require_runtime: bool = False):
        self.runtime_root = runtime_root.resolve() if runtime_root is not None else None
        self.phase = phase
        self.require_runtime = require_runtime
        self.host_operations = Counter()
        self.runtime_operations = 0
        self.violations = []
        self.function_mode = _FunctionAudit(self)
        self.dispatch_mode = _DispatchAudit(self)

    def runtime_call(self) -> bool:
        if self.runtime_root is None:
            return False
        caller = None
        active_runtime = False
        framework_import = False
        frame = sys._getframe(1)
        while frame is not None:
            filename = frame.f_code.co_filename
            if not filename.startswith("<"):
                path = _path(filename)
                stdlib = _within(path, _STDLIB_ROOT) and "site-packages" not in path.parts
                ignored = str(path) == _HERE or _within(path, _TORCH_ROOT) or stdlib
                if not ignored and caller is None:
                    caller = path
                if _within(path, self.runtime_root):
                    if frame.f_code.co_name == "<module>":
                        framework_import = True
                    if path == self.runtime_root / "torchplugin.py" and frame.f_code.co_name == "__call__":
                        receiver = frame.f_locals.get("self")
                        active_runtime = (type(receiver).__name__ == "OpExec"
                                          and type(receiver).__module__ == "easyasc.torchplugin")
            elif filename not in ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>") and caller is None:
                # Dynamically compiled participant code is not a runtime caller.
                caller = Path(filename)
            frame = frame.f_back
        return _within(caller, self.runtime_root) if caller is not None and (active_runtime or (self.phase == "import" and framework_import)) else False

    @staticmethod
    def is_device_transfer(args, kwargs) -> bool:
        if not args or not isinstance(args[0], torch.Tensor):
            return False
        source = args[0]
        target = kwargs.get("device")
        if target is None:
            return False
        target = torch.device(target)
        return (target != source.device and target.type in ("cpu", "npu", "privateuseone")
                and kwargs.get("dtype", source.dtype) == source.dtype
                and kwargs.get("memory_format", torch.preserve_format) in (None, torch.preserve_format))

    def reject(self, operation: str) -> None:
        self.violations.append(operation)
        raise HostOperationError("host operation is not allowed: " + operation)

    def report(self) -> Dict:
        return {"enabled": True, "kind": "basic_in_process", "phase": self.phase,
                "host_operations": dict(self.host_operations),
                "runtime_operations": self.runtime_operations, "violations": list(self.violations)}

    def __enter__(self):
        self.function_mode.__enter__()
        self.dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dispatch_mode.__exit__(exc_type, exc_value, traceback)
        self.function_mode.__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            if self.violations:
                raise HostOperationError("submission caught a forbidden host operation: " + self.violations[0])
            if self.require_runtime and self.runtime_operations == 0:
                raise HostOperationError("no trusted EasyASC OpExec runtime operations were observed")
        return False
