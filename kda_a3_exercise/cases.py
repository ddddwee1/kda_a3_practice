"""Deterministic public inputs, generated before either implementation runs."""

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .contract import validate_inputs


CASES_PATH = Path(__file__).with_name("cases.json")
SUITES = ("smoke", "core", "extended", "performance")


def load_cases(suite: str = "core") -> List[Dict]:
    if suite not in SUITES:
        raise ValueError("unknown suite: " + suite)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    return [case for case in cases if suite in case["suites"]]


def make_inputs(case: Dict) -> Dict:
    b, t, h, hv, k, v = [case[name] for name in ("B", "T", "H", "HV", "K", "V")]
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    def raw_qk():
        # Per-token gains expose implementations that omit or misplace L2 norm.
        gains = torch.exp(randn((b, t, h, 1)) * 0.6)
        return (randn((b, t, h, k)) * gains).to(torch.bfloat16)

    q, key = raw_qk(), raw_qk()
    qk_mode = case.get("qk_mode", "normal")
    if qk_mode == "tiny":
        q, key = (q.float() * 1e-5).bfloat16(), (key.float() * 1e-5).bfloat16()
    elif qk_mode == "zero_rows":
        q[:, ::3].zero_()
        key[:, 1::3].zero_()
    elif qk_mode != "normal":
        raise ValueError("unknown qk_mode: " + qk_mode)
    value = randn((b, t, hv, v)).to(torch.bfloat16)
    gate_mode = case["gate"]
    if gate_mode == "zero":
        gate = torch.zeros((b, t, hv, k), dtype=torch.float32)
    else:
        divisors = {"weak": 128.0, "moderate": 8.0, "strong": 1.0}
        gate = F.logsigmoid(randn((b, t, hv, k))) / divisors[gate_mode]
    beta = torch.sigmoid(randn((b, t, hv)))
    if case["beta"] == "zero":
        beta.zero_()
    elif case["beta"] == "one":
        beta.fill_(1)
    elif case["beta"] == "edges":
        beta[:, ::4] = 0
        beta[:, 1::4] = 1
        beta[:, 2::4] = 1e-4
        beta[:, 3::4] = 1 - 1e-4
    elif case["beta"] != "random":
        raise ValueError("unknown beta mode: " + case["beta"])
    state = None
    if case["state"] == "nonzero":
        state = randn((b, hv, k, v)) * 0.25
    elif case["state"] == "zero":
        state = torch.zeros((b, hv, k, v), dtype=torch.float32)
    elif case["state"] != "none":
        raise ValueError("unknown state mode: " + case["state"])
    inputs = dict(q=q, k=key, v=value, g=gate, beta=beta, initial_state=state)
    validate_inputs(**inputs)
    return inputs
