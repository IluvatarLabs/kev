"""Execution parity, question isolation, branch-batching invariance, the 255-option boundary and the run/artifact round
trip, on the real parent with a fresh head (GPU). Run: scripts/remote.sh run uv run pytest tests/test_model.py -q -s"""
import json, os, subprocess, sys

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")   # the GPU host sets 1 system-wide; fp32 must mean fp32 (read at CUDA init)
import pytest
import torch

from decision_model import DEFAULT_PARENT as P, DEFAULT_PARENT_REVISION as R
import decision_model.model as M
from decision_model.schema import DecisionRequest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA host")
TOL = {torch.float32: 1e-5, torch.bfloat16: 1e-3}
LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
TEXT = "Order {i} arrived two weeks late, the box was crushed, and two charges of $49.99 appear for invoice {i}. "


def make(i=0, reps=16, extra=False):
    qs = {f"team{k}": {"type": "choice", "instructions": f"Which team handles request {k}?", "criteria": {"billing": "Charges", "shipping": "Parcels", "account": None, "other": None}} for k in range(4)}
    qs |= {f"dup{k}": {"type": "noul", "instructions": f"Is a duplicate charge reported ({k})?"} for k in range(2)}
    qs |= {f"urg{k}": {"type": "score", "instructions": "Rate the urgency.", "criteria": ["routine", "urgent", "immediate"]} for k in range(2)}
    if extra:
        qs["weather"] = {"type": "choice", "instructions": "What is the weather?", "criteria": {"sun": None, "rain": None}}
    return {"state": {"message": "".join(TEXT.format(i=i + r) for r in range(reps))}, "questions": qs}


def probs(res, drop=()):
    return {q: torch.tensor(list(a["probabilities"].values())) for q, a in res["answers"].items() if q not in drop}


def delta(a, b):
    return max((a[q] - b[q]).abs().max().item() for q in b)


# bf16 cross-form gate (lead decision 2026-09-24): bf16 rounding depends on kernel shapes. Measured on this request: the
# identical packed computation at batch 1 vs batch 2 moves probabilities 1.63e-2; bf16 rows/cached vs packed 1.47e-2 /
# 1.81e-2; each bf16 form is 1.2-2.3e-2 from fp32 packed. fp32 (the exactness proof) stays at 1e-5.
# Argmax changes between two bf16 sides are accepted when the question's fp32 top-two margin is <= that comparison's
# measured max |dp|. Each bf16 side drifts from fp32 independently (up to ~2.3e-2), so a flip between sides X and Y is
# bounded by the per-option deltas between them relative to their own margins, not by the fp32 margin; the per-question
# |dp| is therefore not the relevant noise scale. Measured flips (all evaluate_requests vs evaluate, team0), as (fp32
# margin, |dp|): (0.0099, 0.0312) (0.0165, 0.0151) (0.0017, 0.0173) (0.0096, 0.0177) (0.0078, 0.0090) (0.0191, 0.0069);
# that comparison's max |dp| is 3.42e-2. The untrained head makes nearly every 4-choice question a near tie (fp32 margins
# 0.002-0.02), so the evaluator re-reads this case on the trained checkpoint.
BF16_CROSS_FORM = 5e-2


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    fp32 = M.DecisionModel.from_parent(P, R, device="cuda", dtype=torch.float32)
    bf16 = M.DecisionModel.from_parent(P, R, device="cuda", dtype=torch.bfloat16)
    bf16.head.load_state_dict(fp32.head.state_dict())
    return {torch.float32: fp32, torch.bfloat16: bf16}


def comparisons(model, monkeypatch):
    """(name, a, b, request) pairs that must agree: forms, question isolation, branch chunking, batched vs per request."""
    r, rx = make(), make(extra=True)
    assert model.evaluate(**r)["usage"]["state_tokens"] > 450
    ref = probs(model.evaluate(**r, form="packed"))
    out = [(f, probs(model.evaluate(**r, form=f)), ref, r) for f in ("rows", "cached")]
    out += [(f"isolation/{f}", probs(model.evaluate(**rx, form=f), drop=("weather",)), probs(model.evaluate(**r, form=f)), r) for f in ("packed", "rows", "cached")]
    enc, qids = model.encode(M.to_record(DecisionRequest.model_validate(r))[0]), list(r["questions"])
    prefix = model.prefix(enc)
    whole = model.probs_with_prefix(enc, prefix)
    monkeypatch.setattr(M, "BRANCH_TOKEN_BUDGET", 1)      # one branch per pass: 8 replicas of the same prefix
    for _ in range(2):                                      # twice: no pass may have written to the caller's prefix
        out.append(("chunking", dict(zip(qids, model.probs_with_prefix(enc, prefix))), dict(zip(qids, whole)), r))
    monkeypatch.undo()
    reqs = [make(i, reps=1 + i % 7) for i in range(32)]
    batched = model.evaluate_requests([DecisionRequest.model_validate(q) for q in reqs])
    out += [("evaluate_requests/evaluate", probs(b), probs(model.evaluate(**q)), q) for b, q in zip(batched, reqs)]
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_forms_agree_isolate_questions_and_ignore_batching(models, dtype, monkeypatch):
    comps = comparisons(models[dtype], monkeypatch)
    d = {}
    for name, a, b, _ in comps:
        d[name] = max(d.get(name, 0.0), delta(a, b))
    print(f"\n{dtype} max |dp|: " + ", ".join(f"{k} {v:.2e}" for k, v in d.items()))
    if dtype == torch.float32:
        assert all(v <= TOL[dtype] for v in d.values()), d
        return
    flips = []   # (comparison, question, fp32 top-two margin, this question's |dp|)
    for name, a, b, req in comps:
        for q in b:
            if a[q].argmax() != b[q].argmax():
                top2 = probs(models[torch.float32].evaluate(**req, form="packed"))[q].topk(2).values
                flips.append((name, q, (top2[0] - top2[1]).item(), (a[q] - b[q]).abs().max().item()))
    print(f"bf16 argmax changes (comparison, question, fp32 margin, |dp|): {flips}")
    assert all(v <= BF16_CROSS_FORM for v in d.values()), d
    assert all(margin <= d[name] for name, _, margin, _ in flips), flips


def test_255_option_choice_and_score_return_complete_distributions(models):
    model = models[torch.bfloat16]
    qs = {"pick": {"type": "choice", "instructions": "Pick one.", "criteria": {f"o{i}": None for i in range(255)}},
          "rate": {"type": "score", "instructions": "Rate it.", "criteria": [f"level {i}" for i in range(255)]}}
    res = model.evaluate({"message": "A short state."}, qs)
    for q, keys in (("pick", [f"o{i}" for i in range(255)]), ("rate", [str(i) for i in range(255)])):
        p = res["answers"][q]["probabilities"]
        assert list(p) == keys and all(torch.isfinite(torch.tensor(v)) for v in p.values()) and abs(sum(p.values()) - 1) <= 1e-6


def test_run_and_artifact_round_trip_offline(tmp_path):
    torch.manual_seed(0)
    m = M.DecisionModel.from_parent(P, R, device="cuda", lora=LORA)
    with torch.no_grad():   # a non-zero adapter, so the merge actually changes weights
        for n, p in m.lm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    r = make()
    before = probs(m.evaluate(**r))
    m.save_run(tmp_path / "run")
    run = M.DecisionModel.from_run(tmp_path / "run", device="cuda")
    run.merge_lora()
    merged = probs(run.evaluate(**r))
    run.save_pretrained(tmp_path / "art", dtype=torch.bfloat16)   # leaves the live model in bf16
    deployed = probs(run.evaluate(**r))
    (tmp_path / "hf").mkdir()
    code = ("import json, sys; from decision_model.model import DecisionModel; m = DecisionModel.from_pretrained(sys.argv[1]); "
            "res = m.evaluate(**json.loads(sys.argv[2])); c = m.lm.config; "
            "print(json.dumps({'p': {q: list(a['probabilities'].values()) for q, a in res['answers'].items()}, 'layers': c.num_hidden_layers, "
            "'lm_head': hasattr(m.lm, 'lm_head'), 'type': type(m.lm).__name__, 'dtype': str(m.lm.dtype)}))")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")}
    out = json.loads(subprocess.run([sys.executable, "-c", code, str(tmp_path / "art"), json.dumps(r)], env=env, check=True,
                                    capture_output=True, text=True).stdout.strip().splitlines()[-1])
    reloaded = {q: torch.tensor(v) for q, v in out["p"].items()}
    d = {"run": delta(merged, before), "reloaded bf16 vs in-memory bf16": delta(reloaded, deployed)}
    print(f"\nround trip max |dp|: " + ", ".join(f"{k} {v:.2e}" for k, v in d.items()))
    assert d["run"] <= 1e-5 and d["reloaded bf16 vs in-memory bf16"] <= 1e-3
    assert (out["layers"], out["lm_head"], out["type"], out["dtype"]) == (28, False, "Qwen3Model", "torch.bfloat16")
    assert {"config.json", "backbone", "decision_head.safetensors", "tokenizer"} <= set(os.listdir(tmp_path / "art"))
