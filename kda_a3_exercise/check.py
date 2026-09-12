"""Check chunk-prefill forward outputs without measuring Python/launch time."""

import argparse
from contextlib import nullcontext
from functools import wraps
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kda_a3_exercise.cases import CASES_PATH, SUITES, load_cases, make_inputs
    from kda_a3_exercise.contract import CONTRACT, CONTRACT_PATH, OUTPUT_NAMES, compare_output
    from kda_a3_exercise.golden import forward as golden_forward
    from kda_a3_exercise.host_guard import HostGuard, audit_source, discover_runtime_root
    from kda_a3_exercise.auxiliary import make_auxiliary, clone_auxiliary, assert_auxiliary_unchanged
else:
    from .cases import CASES_PATH, SUITES, load_cases, make_inputs
    from .contract import CONTRACT, CONTRACT_PATH, OUTPUT_NAMES, compare_output
    from .golden import forward as golden_forward
    from .host_guard import HostGuard, audit_source, discover_runtime_root
    from .auxiliary import make_auxiliary, clone_auxiliary, assert_auxiliary_unchanged


def clone_inputs(inputs: Dict) -> Dict:
    return {name: value.clone() if value is not None else None for name, value in inputs.items()}


def load_submission(path: Path, runtime_root=None) -> Callable:
    path = path.resolve()
    if not path.is_file():
        raise ValueError("submission file does not exist: " + str(path))
    source_audit = audit_source(path, runtime_root)
    spec = importlib.util.spec_from_file_location("kda_participant_submission", str(path))
    if spec is None or spec.loader is None:
        raise ValueError("cannot import submission: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Keep the directory available to imports made later by the participant.
    sys.path.insert(0, str(path.parent))
    import_guard = HostGuard(runtime_root, phase="import")
    with import_guard:
        spec.loader.exec_module(module)
    fn = getattr(module, "forward", None)
    if not callable(fn):
        raise ValueError("submission must export callable forward(q,k,v,g,beta,initial_state=None,*,aux=None)")
    @wraps(fn)
    def checked_entry(*args, **kwargs):
        return fn(*args, **kwargs)
    checked_entry.import_audit = {"source": source_audit, "operations": import_guard.report()}
    return checked_entry


def check_case(fn: Callable, case: Dict, *, audit_host: bool = True, runtime_root=None, auxiliary=None) -> Dict:
    inputs = make_inputs(case)
    result = {"case_id": case["id"], "seed": case["seed"],
              "shape": {name: case[name] for name in ("B", "T", "H", "HV", "K", "V")},
              "passed": False}
    guard = HostGuard(runtime_root, require_runtime=True) if audit_host else None
    try:
        auxiliary = make_auxiliary() if auxiliary is None else auxiliary
        submitted_auxiliary = clone_auxiliary(auxiliary)
        with torch.no_grad():
            expected = golden_forward(**clone_inputs(inputs))
            submitted_inputs = clone_inputs(inputs)
            with guard if guard is not None else nullcontext():
                actual = fn(**submitted_inputs, aux=submitted_auxiliary)
        assert_auxiliary_unchanged(submitted_auxiliary, auxiliary)
        for name, original in inputs.items():
            current = submitted_inputs[name]
            if original is not None and (current.dtype != original.dtype or current.device != original.device
                                         or current.shape != original.shape or not torch.equal(current, original)):
                raise ValueError("submission modified input " + name)
        if not isinstance(actual, (tuple, list)) or len(actual) != 2:
            raise ValueError("forward must return exactly (o, final_state)")
        metrics = {name: compare_output(name, out, ref)
                   for name, out, ref in zip(OUTPUT_NAMES, actual, expected)}
        result["outputs"] = metrics
        result["passed"] = all(item["passed"] for item in metrics.values())
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["host_audit"] = guard.report() if guard is not None else {"enabled": False, "kind": "reference_self_check"}
    return result


def run_checks(fn: Callable, cases: List[Dict], *, audit_host: bool = True, runtime_root=None) -> List[Dict]:
    if not cases:
        raise ValueError("no cases selected")
    auxiliary = make_auxiliary()
    return [check_case(fn, case, audit_host=audit_host, runtime_root=runtime_root, auxiliary=auxiliary) for case in cases]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, help="Python file exporting forward")
    parser.add_argument("--suite", choices=SUITES, default="core")
    parser.add_argument("--case", action="append", default=[], help="case ID within the suite; repeatable")
    parser.add_argument("--list", action="store_true", help="list selected cases without running")
    parser.add_argument("--check-golden", action="store_true", help="exercise harness plumbing, not a submission result")
    parser.add_argument("--report", type=Path, help="write correctness JSON; contains no timing")
    parser.add_argument("--threads", type=int, default=1, help="CPU golden PyTorch thread count")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cases = load_cases(args.suite)
    if args.case:
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            parser.error("unknown case IDs in suite: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in args.case]
    if args.list:
        for case in cases:
            print(case["id"] + " " + " ".join(name + "=" + str(case[name]) for name in ("B", "T", "H", "HV", "K", "V")))
        return 0
    if (args.submission is not None) == args.check_golden:
        parser.error("choose exactly one of --submission or --check-golden")
    runtime_root = discover_runtime_root()
    if args.check_golden:
        print("GOLDEN HARNESS CHECK ONLY: this does not validate an A3 implementation.")
        fn = golden_forward
    else:
        try:
            fn = load_submission(args.submission, runtime_root)
        except Exception as exc:
            parser.error(str(exc))
    results = run_checks(fn, cases, audit_host=not args.check_golden, runtime_root=runtime_root)
    for result in results:
        print(("PASS " if result["passed"] else "FAIL ") + result["case_id"])
        if not result["passed"]:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    passed = all(result["passed"] for result in results)
    report = {
        "contract_version": CONTRACT["version"], "suite": args.suite,
        "operator": CONTRACT["operator"], "golden_algorithm": CONTRACT["golden_algorithm"],
        "run_kind": "golden_harness_check" if args.check_golden else "submission_correctness",
        "full_core_suite": args.suite == "core" and not args.case,
        "passed": passed, "device_execution_verified": False, "kernel_timing": None,
        "host_audit_enabled": not args.check_golden,
        "torch_version": torch.__version__, "results": results,
        "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
    }
    if args.submission is not None:
        report["submission_sha256"] = hashlib.sha256(args.submission.read_bytes()).hexdigest()
        report["submission_import_audit"] = fn.import_audit
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("%d/%d cases passed" % (sum(item["passed"] for item in results), len(results)))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
