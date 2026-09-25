"""Editable-question demo: one customer-support record -> several typed decisions from the converted model, beside the
parent's one-token scoring (raw and cyclic) when --parent is given.

    python demo/run_demo.py --model runs/showcase/model --request demo/customer_support.json
    python demo/run_demo.py --model runs/showcase/model --parent Qwen/Qwen3-1.7B --request demo/customer_support_reworded.json

Edit the request JSON (question wording, option descriptions, option order, add or remove questions) and rerun; no
retraining is involved. The converted model is loaded from local files only, with network access disabled: it never
calls a teacher or the parent. The parent is loaded only for the comparison columns.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # no network: the converted model loads from its own directory only

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from decision_model import DecisionModel, DecisionRequest  # noqa: E402
from decision_model.baselines import sync  # noqa: E402


def timed(device, fn, reps=3):
    """Run once untimed (warm-up), then report the median of `reps` timed calls and the last result."""
    fn()
    ms = []
    for _ in range(reps):
        sync(device)
        t = time.perf_counter()
        out = fn()
        sync(device)
        ms.append((time.perf_counter() - t) * 1e3)
    return out, sorted(ms)[len(ms) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="exported artifact directory")
    ap.add_argument("--request", default=str(Path(__file__).with_name("customer_support.json")))
    ap.add_argument("--parent", default=None, help="parent id for the parent/raw and parent/cyclic columns (from the local HF cache)")
    ap.add_argument("--revision", default=None, help="parent revision (default: the one recorded in the artifact's config.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    revision = a.revision or json.loads((Path(a.model) / "config.json").read_text())["parent"]["revision"]
    req = DecisionRequest.model_validate_json(Path(a.request).read_text())
    questions = {k: q.model_dump() for k, q in req.questions.items()}
    model = DecisionModel.from_pretrained(a.model, device=a.device)
    conv, conv_ms = timed(model.device, lambda: model.evaluate(req.state, questions))
    cols = {"converted": {qid: ans["probabilities"] for qid, ans in conv["answers"].items()}}
    lat = {"converted": conv_ms}
    if a.parent:
        del model
        from decision_model.targets import ParentScorer
        scorer = ParentScorer(a.parent, revision, device=a.device)
        for mode in ("raw", "cyclic"):
            out, ms = timed(a.device, lambda mode=mode: scorer.score(req, mode))
            cols[f"parent/{mode}"] = {qid: (o["probs"] if o["eligible"] else None) for qid, o in out.items()}
            lat[f"parent/{mode}"] = ms
    order = [c for c in ("parent/raw", "parent/cyclic", "converted") if c in cols]

    print(f"request: {a.request}")
    print("median warm latency per request: " + ", ".join(f"{c} {lat[c]:.1f} ms" for c in order) + f" ({a.device})\n")
    for qid, q in req.questions.items():
        print(f"{qid} ({q.type}): {q.instructions}")
        keys = list(cols["converted"][qid])
        tops = {c: (max(cols[c][qid], key=cols[c][qid].get) if cols[c][qid] else None) for c in order}
        print("  " + f"{'option':<28}" + "".join(f"{c:>16}" for c in order))
        for k in keys:
            label = k if q.type != "score" else f"{k}: {q.criteria[int(k)]}"
            cells = "".join(f"{'ineligible':>16}" if cols[c][qid] is None else
                            f"{(cols[c][qid][k]):>15.3f}{'*' if tops[c] == k else ' '}" for c in order)
            print(f"  {label[:28]:<28}{cells}")
        diff = [c for c in order if c != "converted" and tops[c] is not None and tops[c] != tops["converted"]]
        if diff:
            print(f"  DISAGREEMENT: converted picks {tops['converted']!r}; " + ", ".join(f"{c} picks {tops[c]!r}" for c in diff))
        print()
    print("* = highest probability. Edit the request file and rerun to change questions, descriptions or option order.")
    print(json.dumps({"latency_ms": lat}, indent=None))


if __name__ == "__main__":
    main()
