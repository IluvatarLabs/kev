"""The conversion command (SPEC section 9): prepare -> collect targets -> train -> select -> calibrate -> export.

    python -m decision_model.convert --config recipes/qwen3-1.7b.yaml --out runs/showcase

Each stage's output records the hash of its inputs and configuration; a rerun skips a stage whose output exists and
matches, and refuses to overwrite one that exists but does not match (move it aside to redo it). Locked evaluation is
not part of conversion: run `python -m decision_model.evaluate` after the model is exported.

    prepare    data/prepared/manifest.json       config_sha256 = sha256(recipe data section), plus parent/revision
    collect    data/targets/<rev>/meta.json      input_sha256 = targets.collection_hash(...), complete = true
    train      <out>/train/training_config.json  recipe/data-manifest/targets hashes; training_metrics.json complete = true
    select     <out>/train/selected/run.json     the lowest-development-NLL checkpoint of that run
    calibrate  <out>/calibration.json            inputs_sha256 = calibrate.inputs_hash(...)
    export     <out>/model/conversion.json       stage_sha256 over the selected run, calibration and export settings
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def _sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_json(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def _read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _log(stage, msg):
    print(f"[convert:{stage}] {msg}", flush=True)


def stage_prepare(cfg):
    from .data import prepare
    out = Path(cfg["data"]["out"])
    m = _read(out / "manifest.json")
    want = _sha_json(cfg["data"])
    if m is not None:
        if m.get("config_sha256") == want and m["parent"] == {"name": cfg["parent"], "revision": cfg["revision"]}:
            _log("prepare", f"skip: {out} matches the recipe")
            return m
        raise FileExistsError(f"{out} was prepared from a different recipe; move it aside to rebuild")
    _log("prepare", f"building {out}")
    return prepare(cfg, out)


def stage_collect(cfg, device):
    from .targets import collect, collection_hash, target_path
    path = target_path(cfg["teacher"]["out"], cfg["revision"])
    meta = _read(path.parent / "meta.json")
    if meta and meta.get("complete") and meta.get("input_sha256") == collection_hash(cfg, cfg["data"]["out"]):
        _log("collect", f"skip: {path} is complete for these inputs")
        return path, meta
    _log("collect", f"collecting into {path} (resumes by record/question)")
    return collect(cfg, device=device)


def _train_expect(cfg, targets_path):
    return {"recipe": cfg["train"], "parent": cfg["parent"], "revision": cfg["revision"],
            "data_manifest_sha256": _sha_bytes(Path(cfg["data"]["out"]) / "manifest.json"), "targets_sha256": _sha_bytes(targets_path)}


def stage_train(cfg, out, targets_path, device):
    from .train import train
    run = out / cfg["train"]["out"]
    tc, tm = _read(run / "training_config.json"), _read(run / "training_metrics.json")
    expect = _train_expect(cfg, targets_path)
    if tc is not None:
        if any(tc.get(k) != v for k, v in expect.items()):
            raise FileExistsError(f"{run} was trained from different inputs; move it aside to retrain")
        if tm and tm.get("complete"):
            _log("train", f"skip: {run} is complete for these inputs")
            return run, tm
        raise RuntimeError(f"{run} exists but training has not completed (running elsewhere, or interrupted: move it aside to retrain)")
    _log("train", f"training into {run}")
    return run, train(cfg, cfg["data"]["out"], targets_path, run, device)


def stage_select(run, metrics):
    sel = run / "selected"
    meta = _read(sel / "run.json")
    if meta is None or meta.get("step") != metrics["selected_step"]:
        raise RuntimeError(f"{sel} does not hold the selected step {metrics['selected_step']}")
    _log("select", f"step {meta['step']} dev NLL {meta['dev_nll']:.4f}")
    return sel


def stage_calibrate(cfg, out, sel, device):
    from .calibrate import calibrate, inputs_hash
    path = out / "calibration.json"
    have = _read(path)
    want = inputs_hash(cfg, sel, cfg["data"]["out"])
    if have is not None:
        if have.get("inputs_sha256") == want:
            _log("calibrate", f"skip: {path} matches the selected run (T = {have['temperature']:.4f})")
            return have
        raise FileExistsError(f"{path} was fitted on different inputs; move it aside to refit")
    cal = calibrate(cfg, sel, cfg["data"]["out"], path, device)
    _log("calibrate", f"T = {cal['temperature']:.4f}; NLL {cal['overall']['nll_raw']:.4f} -> {cal['overall']['nll_calibrated']:.4f}")
    return cal


def stage_export(cfg, out, sel, cal, targets_path, targets_meta, train_metrics, device):
    from .data import load_split
    from .export import export
    dest = out / "model"
    stage_sha = _sha_json({"run": [_sha_bytes(sel / "run.json"), _sha_bytes(sel / "head.safetensors"), _sha_bytes(sel / "adapter" / "adapter_model.safetensors")],
                           "calibration": _sha_bytes(out / "calibration.json"), "export": cfg["export"]})
    conv = _read(dest / "conversion.json")
    if conv is not None:
        if conv.get("stage_sha256") == stage_sha:
            _log("export", f"skip: {dest} matches the selected run and calibration")
            return conv
        raise FileExistsError(f"{dest} was exported from different inputs; move it aside to re-export")
    manifest = _read(Path(cfg["data"]["out"]) / "manifest.json")
    check = [r["request"] for r in load_split(cfg["data"]["out"], "calibration")[: cfg["export"]["check_requests"]]]
    conversion = {
        "stage_sha256": stage_sha, "recipe": cfg,
        "corpus": {"manifest_sha256": _sha_bytes(Path(cfg["data"]["out"]) / "manifest.json"), "files": manifest["files"],
                   "sources": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in manifest["sources"].items()}},
        "teacher": {"targets_sha256": _sha_bytes(targets_path), "collection_wall_seconds": targets_meta["wall_seconds"],
                    "sessions": targets_meta["sessions"], "totals": targets_meta["totals"], "teacher": targets_meta["teacher"]},
        "training": {k: train_metrics[k] for k in ("wall_seconds", "records_seen", "forward_tokens", "optimizer_steps", "peak_device_bytes",
                                                   "selected_step", "selected_dev_nll")},
        "calibration": {k: cal[k] for k in ("temperature", "method", "records", "questions", "overall")},
    }
    _log("export", f"exporting {sel} -> {dest}")
    report = export(sel, dest, calibration=cal, conversion=conversion, dtype=getattr(torch, cfg["export"]["dtype"]), device=device, check_requests=check)
    return report


def convert(cfg, out, device="cuda"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stage_prepare(cfg)
    targets_path, targets_meta = stage_collect(cfg, device)
    run, metrics = stage_train(cfg, out, targets_path, device)
    sel = stage_select(run, metrics)
    cal = stage_calibrate(cfg, out, sel, device)
    report = stage_export(cfg, out, sel, cal, targets_path, targets_meta, metrics, device)
    _log("done", f"{time.time() - t0:.0f}s this invocation; model at {out / 'model'}")
    return report


def main():
    from .data import load_recipe

    ap = argparse.ArgumentParser(description="convert a parent LLM into a decision model with one recipe")
    ap.add_argument("--config", default="recipes/qwen3-1.7b.yaml")
    ap.add_argument("--out", default="runs/showcase")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    convert(load_recipe(a.config), a.out, a.device)


if __name__ == "__main__":
    main()
