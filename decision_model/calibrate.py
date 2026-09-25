# Modified implementation for Kev; see LICENSE and NOTICE.
"""Fit the selected checkpoint's scalar temperature on the calibration partition (SPEC section 8).

    python -m decision_model.calibrate --run runs/showcase/train/selected --data data/prepared --out runs/showcase/calibration.json

Only the calibration partition is read. Logits come from the selected run at T = 1 (packed form, the training
numerics); T minimizes the mean NLL over every calibration question. Raw and calibrated NLL and accuracy are reported
overall, by primitive and by source family.
"""
import argparse
import contextlib
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .render import to_record
from .schema import DecisionRequest, question_keys


def nll_at(z, y, T):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    return float(np.log(np.exp(z).sum()) - z[y])


def fit_temperature(rows, low=0.25, high=4.0, points=121):
    """Temperature fitting (micro): the grid value minimizing mean NLL over all rows."""
    grid = np.exp(np.linspace(np.log(low), np.log(high), points))
    losses = [np.mean([nll_at(r["logits"], r["label"], float(T)) for r in rows]) for T in grid]
    return float(grid[int(np.argmin(losses))]), [float(x) for x in grid], [float(x) for x in losses]


def summarize(rows, T):
    def stats(rs):
        return {"questions": len(rs), "accuracy": float(np.mean([int(np.argmax(r["logits"]) == r["label"]) for r in rs])),
                "nll_raw": float(np.mean([nll_at(r["logits"], r["label"], 1.0) for r in rs])),
                "nll_calibrated": float(np.mean([nll_at(r["logits"], r["label"], T) for r in rs]))}
    by_p, by_s = defaultdict(list), defaultdict(list)
    for r in rows:
        by_p[r["type"]].append(r)
        by_s[r["source_family"]].append(r)
    return {"overall": stats(rows), "by_primitive": {k: stats(v) for k, v in sorted(by_p.items())},
            "by_source_family": {k: stats(v) for k, v in sorted(by_s.items())}}


@torch.no_grad()
def calibration_rows(model, records, autocast, batch=8):
    model.eval()
    rows = []
    for s in range(0, len(records), batch):
        part = records[s:s + batch]
        encs, metas = [], []
        for r in part:
            req = DecisionRequest.model_validate(r["request"])
            rec, meta = to_record(req)
            encs.append(model.encode(rec))
            metas.append([(qid, q.type, question_keys(q.type, q.criteria).index(r["supervision"][qid]["gold"])) for qid, q in req.questions.items()])
        with autocast:
            logits = model.forward_batch(encs, form="packed")
        for r, zs, ms in zip(part, logits, metas):
            for z, (qid, qtype, y) in zip(zs, ms):
                rows.append({"record_id": r["record_id"], "qid": qid, "type": qtype, "source_family": r["source_family"],
                             "logits": z.float().cpu().tolist(), "label": y})
    return rows


def inputs_hash(cfg, run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    h = hashlib.sha256()
    for p in (run_dir / "run.json", run_dir / "head.safetensors", run_dir / "adapter" / "adapter_model.safetensors"):
        h.update(hashlib.sha256(p.read_bytes()).digest())
    h.update(manifest["files"]["calibration.jsonl"]["sha256"].encode())
    h.update(json.dumps(cfg["calibration"], sort_keys=True).encode())
    return h.hexdigest()


def calibrate(cfg, run_dir, data_dir, out_path, device="cuda", batch=8):
    from .data import load_split
    from .model import DecisionModel

    records = load_split(data_dir, "calibration")
    model = DecisionModel.from_run(run_dir, device=device, dtype=getattr(torch, cfg["train"]["weights_dtype"]))   # the training numerics
    cuda = str(device).startswith("cuda")
    autocast = torch.autocast("cuda", dtype=getattr(torch, cfg["train"]["autocast"])) if cuda and cfg["train"].get("autocast") else contextlib.nullcontext()
    rows = calibration_rows(model, records, autocast, batch)
    g = cfg["calibration"]["grid"]
    T, grid, losses = fit_temperature(rows, g["low"], g["high"], g["points"])
    run_meta = json.loads((Path(run_dir) / "run.json").read_text())
    out = {"temperature": T, "fitting_scope": "calibration partition (data/prepared/calibration.jsonl)",
           "method": f"min micro mean NLL over a {g['points']}-point log grid {g['low']}..{g['high']}",
           "logits": f"selected run at T=1, packed form, {'bf16 autocast' if cuda else 'fp32'} over {cfg['train']['weights_dtype']} base weights",
           "run": str(run_dir), "selected_step": run_meta.get("step"), "selected_dev_nll": run_meta.get("dev_nll"),
           "records": len(records), "questions": len(rows), "grid_edge": T in (grid[0], grid[-1]),
           **summarize(rows, T), "inputs_sha256": inputs_hash(cfg, run_dir, data_dir)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    from .data import load_recipe

    ap = argparse.ArgumentParser(description="fit the selected checkpoint's temperature on the calibration partition")
    ap.add_argument("--config", default="recipes/qwen3-1.7b.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = calibrate(load_recipe(a.config), a.run, a.data, a.out, a.device)
    print(json.dumps({k: out[k] for k in ("temperature", "records", "questions", "overall", "by_primitive")}, indent=2))


if __name__ == "__main__":
    main()
