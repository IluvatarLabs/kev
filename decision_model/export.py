# Modified implementation for Kev; see LICENSE and NOTICE.
"""Export a training run as the complete, standalone artifact (SPEC section 9) and prove it reproduces the run.

    export(run_dir, out_dir, calibration=..., conversion=..., device="cuda", check_requests=[...]) -> report

Steps: DecisionModel.from_run (FP32) -> probabilities on check_requests -> merge_lora() in FP32 -> probabilities ->
save_pretrained(out_dir, dtype) -> reload with DecisionModel.from_pretrained in a fresh process with HF_HUB_OFFLINE=1 and
HF_HOME pointed at an empty directory (the parent is unreachable) -> probabilities and answers. Raises if the merged FP32
probabilities differ from the pre-merge ones by more than 3e-5 (TOL_MERGE, justified beside it), or the reloaded artifact's from the merged FP32 ones by
more than 1e-3 (both in the deployed dtype: the same deployed configuration, SPEC section 10). The merged FP32 vs reloaded
bf16 difference is bf16 arithmetic, not an export defect; it is measured and reported, not gated. All comparisons use
T = 1 (raw) and the serving (cached) execution form.

Also writes calibration.json, conversion.json, README.md, NOTICE and the parent's LICENSE into the artifact, and sets
`temperature` in its config.json to the fitted value.

CLI (the reproduction path when a run is exported on its own; decision_model.convert calls export() directly):
    python -m decision_model.export --run RUN --out OUT --calibration calibration.json [--conversion conversion.json]
                                    [--check-data data/prepared/calibration.jsonl] [--n-check 16] [--device cuda]
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import torch

from .render import to_record
from .schema import DecisionRequest

# pre-merge vs merged, both FP32. Justification: on inputs of ~1,000+ tokens each fp32 execution path sits up to 1.2e-5
# from an fp64 reference (fp64 packed vs rows agree to 2.7e-7), so two fp32 paths differ by up to 2e-5 from rounding alone.
TOL_MERGE = 3e-5
TOL_MERGE_NOTE = ("fp32 gate 3e-5: on inputs of ~1,000+ tokens each fp32 path sits up to 1.2e-5 from an fp64 reference "
                  "(fp64 packed vs rows agree to 2.7e-7), so fp32 paths differ by up to 2e-5 from rounding, not logic")
TOL_RELOAD = 1e-3     # merged weights cast to the deployed dtype in memory vs the artifact reloaded in that dtype
REPO = Path(__file__).resolve().parent.parent
LIBRARIES = ("torch", "transformers", "peft", "safetensors", "accelerate", "huggingface-hub", "pydantic", "fastapi")
MEASURED_START, MEASURED_END = "<!-- measured:start -->", "<!-- measured:end -->"

_LONG_STATE = " ".join([
    "I ordered a pair of running shoes on the first of the month and paid with my credit card.",
    "The confirmation email promised delivery in three to five business days, but tracking did not update for a week.",
    "When the package arrived the box was crushed and the shoes were a size ten instead of the nine I selected.",
    "My bank statement also shows two identical charges from your store on the same afternoon.",
    "I called support twice and was disconnected both times after waiting on hold for over twenty minutes.",
] * 11)   # about 1,000 state tokens: exercises the fp32 gate at length

# Check requests always included: new questions and options the model was not trained on, and one ~1,000-token state.
DEFAULT_CHECK = [
    {"state": {"message": "I was charged twice for my March invoice and need the duplicate refunded today."},
     "questions": {"intent": {"type": "choice", "instructions": "Select the customer's main request.",
                              "criteria": {"refund": "Return a payment", "cancel": "End a subscription",
                                           "upgrade": "Move to a larger plan", "info": "Ask a question"}},
                   "duplicate_charge": {"type": "noul", "instructions": "Does the message report a duplicate charge?"},
                   "urgency": {"type": "score", "instructions": "Rate the urgency expressed in the message.",
                               "criteria": ["routine", "soon", "urgent", "immediate"]}}},
    {"state": "The package arrived with a cracked screen. I'd like a replacement rather than my money back.",
     "questions": {"resolution": {"type": "choice", "instructions": "What outcome does the customer want?",
                                  "criteria": {"replacement": "Send the same item again", "refund": None,
                                               "repair": "Fix the damaged item"}},
                   "damaged": {"type": "noul", "instructions": "Did the item arrive damaged?",
                               "criteria": {"true": "Broken or faulty on arrival", "false": "Arrived intact"}}}},
    {"state": {"ticket": _LONG_STATE},
     "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket first?",
                            "criteria": {"returns": "Exchanges and wrong items", "shipping": "Delivery problems",
                                         "billing": "Charges and refunds", "account": "Login and profile"}},
                   "escalate": {"type": "noul", "instructions": "Should a supervisor review this ticket?"},
                   "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                   "criteria": ["calm", "annoyed", "frustrated", "furious"]}}},
]


def _probs(model, requests, temperature=1.0):
    return [[p.tolist() for p in model.probs(model.encode(to_record(r)[0]), form="cached", temperature=temperature)] for r in requests]


def _compare(a, b, requests):
    """(max |dp| over every option of every question, argmax changes with their margins in `a`)."""
    worst, changes = 0.0, []
    for i, (ra, rb, req) in enumerate(zip(a, b, requests)):
        for qid, pa, pb in zip(req.questions, ra, rb):
            worst = max(worst, max(abs(x - y) for x, y in zip(pa, pb)))
            ia, ib = max(range(len(pa)), key=pa.__getitem__), max(range(len(pb)), key=pb.__getitem__)
            if ia != ib:
                top = sorted(pa, reverse=True)
                changes.append({"request": i, "question": qid, "argmax_a": ia, "argmax_b": ib, "margin_a": top[0] - top[1]})
    return worst, changes


def _reload(out_dir, requests, device):
    """Reload the artifact in a fresh process with the network and the parent unavailable; returns its probabilities
    and public answers for `requests`."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "hf_home").mkdir()
        (Path(tmp) / "requests.json").write_text(json.dumps([r.model_dump() for r in requests]))
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(Path(tmp) / "hf_home")}
        env.pop("HF_HUB_CACHE", None)
        subprocess.run([sys.executable, "-m", "decision_model.export", "--reload-check", str(out_dir), "--requests",
                        str(Path(tmp) / "requests.json"), "--device", str(device), "--result", str(Path(tmp) / "out.json")],
                       env=env, check=True)
        return json.loads((Path(tmp) / "out.json").read_text())


def _reload_check(model_dir, requests_path, device, result_path):
    from .model import DecisionModel
    reqs = [DecisionRequest.model_validate(r) for r in json.loads(Path(requests_path).read_text())]
    m = DecisionModel.from_pretrained(model_dir, device=device)
    out = {"probs": _probs(m, reqs), "answers": [m.evaluate(r.state, r.model_dump()["questions"])["answers"] for r in reqs],
           "dtype": str(next(m.lm.parameters()).dtype).removeprefix("torch."), "layers": m.lm.config.num_hidden_layers,
           "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME")}}
    Path(result_path).write_text(json.dumps(out))


def _versions():
    out = {"python": sys.version.split()[0]}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def _parent_license(parent, revision):
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(parent, "LICENSE", revision=revision, local_files_only=True))
    except Exception:   # noqa: BLE001 - reported in the export report rather than failing the export
        return None


def dir_bytes(path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def readme(cfg: dict, calibration: dict, conversion: dict, report: dict) -> str:
    p, fp = cfg["parent"], report["merged_fp32_vs_reloaded"]
    d = report["merged_in_memory_vs_reloaded_same_dtype"]["max_abs_dp"]
    dt = cfg["backbone"]["dtype"]
    same = (f"the saved and reloaded {dt} weights reproduced the in-memory {dt} model exactly (max difference 0)" if d == 0 else
            f"the saved and reloaded {dt} weights matched the in-memory {dt} model within {d:.3g} in any probability")
    return f"""# Decision model converted from {p['name']}

A complete, standalone typed decision model: state + runtime questions + runtime options in, Boolean (`noul`),
categorical (`choice`) and ordinal (`score`) decisions with full probability distributions out. It was converted from
`{p['name']}@{p['revision']}` by merging a trained LoRA adapter into all {cfg['backbone']['num_hidden_layers']} backbone
layers and adding a shared pointer head over runtime options. The parent's weights were modified (LoRA merged) and its
vocabulary projection is not used; no text is generated. The backbone is the parent's size: this conversion does not
make the weights smaller.

## Contents

| Path | What |
|---|---|
| `config.json` | format version, backbone depth, head, renderer, admission limits, temperature ({cfg['temperature']:.4g}) |
| `backbone/` | Hugging Face `Qwen3Model` config and merged `{cfg['backbone']['dtype']}` safetensors (no `lm_head`) |
| `decision_head.safetensors` | the shared pointer head (query/key projections) |
| `tokenizer/` | the exact tokenizer; delimiter ids are in `config.json` |
| `calibration.json` | fitted temperature, fitting scope and counts |
| `conversion.json` | parent revision, recipe, corpus/target hashes, library versions, export equivalence check |
| `evaluation.json` | measured quality and performance (written by `decision_model.evaluate`) |
| `NOTICE`, `LICENSE` | notices for reused code and the parent model; the parent's Apache-2.0 license |

## Load

Nothing is downloaded: no parent, teacher or network access is needed.

```python
from decision_model import DecisionModel
m = DecisionModel.from_pretrained("path/to/this/directory")        # device defaults to cuda when available
out = m.evaluate(state={{"message": "I was charged twice and need the duplicate refunded."}},
                 questions={{"intent": {{"type": "choice", "instructions": "Select the customer's main request.",
                                         "criteria": {{"refund": "Return a payment", "cancel": "End a subscription"}}}},
                            "duplicate_charge": {{"type": "noul", "instructions": "Does the message report a duplicate charge?"}},
                            "urgency": {{"type": "score", "instructions": "Rate the urgency.", "criteria": ["routine", "urgent", "immediate"]}}}})
print(out["answers"])
```

CLI: `python -m decision_model.decide --model <this directory> --request req.json`.
HTTP: `python -m decision_model.serve --model <this directory> --port 8008`, then `POST /v1/systemone` with the same JSON.

Probabilities are calibrated by the stored temperature; pass `temperature=1.0` to `evaluate` for raw probabilities.
`top_probability` is the probability of the modal option, not the probability of being correct. Calibration was fitted
on {calibration.get('fitting_scope', 'the calibration partition')} and does not establish calibration on arbitrary new schemas.

Precision: the weights ship in {cfg['backbone']['dtype']}. On the export check ({report['questions']} questions) the merged FP32
model and this artifact differed by at most {fp['max_abs_dp']:.3g} in any probability, with {len(fp['argmax_changes'])} argmax
change(s); {same}. Details are in `conversion.json` under `export_equivalence`.

Admission limits: {cfg['limits']['max_state_tokens']}-token state, {cfg['limits']['max_row_tokens']} tokens for state plus one
question branch, {cfg['limits']['max_questions']} questions per request, {cfg['limits']['max_options']} options or levels.
Over-limit requests fail explicitly; nothing is truncated.

## Measured capabilities

{MEASURED_START}
Pending: `decision_model.evaluate` fills this section with measured results for this artifact.
{MEASURED_END}
"""


def _check_out_dir(out: Path) -> None:
    """--out may be absent, empty, or a previous export of ours (replaced); anything else is refused, never deleted."""
    if not out.exists() or (out.is_dir() and not any(out.iterdir())):
        return
    try:
        ours = json.loads((out / "config.json").read_text()).get("model_type") == "decision_model"
    except (OSError, ValueError):
        ours = False
    if not (out.is_dir() and (ours or (out / "export_failed.json").exists())):
        raise FileExistsError(f"{out} exists and is not a previous decision-model export; refusing to replace it")


def export(run_dir, out_dir, *, calibration: dict, conversion: dict, dtype=torch.bfloat16, device, check_requests: list) -> dict:
    """Merge, save and verify. Returns the equivalence report (also stored in conversion.json); raises if a tolerance is
    exceeded (the artifact files are left in place for inspection)."""
    from .model import DecisionModel
    run_dir, out = Path(run_dir), Path(out_dir)
    if "temperature" not in calibration:
        raise ValueError("calibration must contain the fitted 'temperature'")
    reqs = [r if isinstance(r, DecisionRequest) else DecisionRequest.model_validate(r) for r in list(check_requests or []) + DEFAULT_CHECK]
    _check_out_dir(out)
    m = DecisionModel.from_run(run_dir, device=device, dtype=torch.float32)
    pre = _probs(m, reqs)
    m.merge_lora()
    merged = _probs(m, reqs)
    if out.exists():
        shutil.rmtree(out)
    m.save_pretrained(out, dtype=dtype)             # casts parameters only and leaves this instance in `dtype`
    merged_cast = _probs(m, reqs)                   # the deployed weights in memory, same form and request shapes as the reload
    parent = dict(m.parent)
    del m
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Artifact metadata before the reload, so the reload sees the final config.json (temperature does not enter T=1 probs).
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature"] = float(calibration["temperature"])
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    reloaded = _reload(out, reqs, device)
    d_merge, flips_merge = _compare(pre, merged, reqs)
    d_reload, flips_reload = _compare(merged, reloaded["probs"], reqs)
    d_serial, flips_serial = _compare(merged_cast, reloaded["probs"], reqs)
    report = {"requests": len(reqs), "questions": sum(len(r.questions) for r in reqs),
              "conditions": {"fp32_tolerance_justification": TOL_MERGE_NOTE, "form": "cached", "temperature": 1.0, "device": str(device), "deployed_dtype": reloaded["dtype"],
                             "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                             "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
              "pre_merge_vs_merged_fp32": {"max_abs_dp": d_merge, "tolerance": TOL_MERGE, "argmax_changes": flips_merge},
              "merged_in_memory_vs_reloaded_same_dtype": {"max_abs_dp": d_serial, "tolerance": TOL_RELOAD, "argmax_changes": flips_serial},
              "merged_fp32_vs_reloaded": {"max_abs_dp": d_reload, "gated": False, "argmax_changes": flips_reload,
                                          "note": "fp32 -> deployed-dtype arithmetic difference; measured, not a tolerance"},
              "reload_env": reloaded["env"], "reloaded_layers": reloaded["layers"], "reloaded_answers": reloaded["answers"],
              "artifact_bytes": None, "artifact_bytes_note": "all artifact files except conversion.json"}

    run_meta = json.loads((run_dir / "run.json").read_text())
    (out / "calibration.json").write_text(json.dumps({"fitting_scope": "calibration partition", **calibration}, indent=2) + "\n")
    notice = REPO / "NOTICE"
    if notice.exists():
        shutil.copy(notice, out / "NOTICE")
    lic = _parent_license(parent["name"], parent["revision"])
    if lic:
        shutil.copy(lic, out / "LICENSE")
    (out / "README.md").write_text(readme(cfg, calibration, conversion, report))
    report["artifact_bytes"] = dir_bytes(out)     # every artifact file except conversion.json, which records this number
    conv = {**conversion, "parent": parent, "selected_checkpoint": {"path": str(run_dir), "run": run_meta},
            "library_versions": _versions(), "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "export_dtype": str(dtype).removeprefix("torch."), "export_equivalence": {k: v for k, v in report.items() if k != "reloaded_answers"},
            "parent_license_copied": bool(lic)}
    problems = []
    if d_merge > TOL_MERGE:
        problems.append(f"pre-merge vs merged FP32 max |dp| {d_merge:.3g} > {TOL_MERGE}")
    if d_serial > TOL_RELOAD:
        problems.append(f"in-memory {reloaded['dtype']} vs reloaded {reloaded['dtype']} max |dp| {d_serial:.3g} > {TOL_RELOAD}")
    if problems:
        # conversion.json marks a complete export (convert's resume check reads it); a failed one never gets it
        (out / "export_failed.json").write_text(json.dumps({"problems": problems, **conv}, indent=2) + "\n")
        raise AssertionError("export equivalence failed: " + "; ".join(problems) + f" (report: {out / 'export_failed.json'})")
    (out / "conversion.json").write_text(json.dumps(conv, indent=2) + "\n")
    report["artifact_bytes_with_conversion_json"] = dir_bytes(out)
    return report


def _check_requests(path, n):
    if not path:
        return None
    reqs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                reqs.append(DecisionRequest.model_validate(json.loads(line)["request"]))
            if len(reqs) >= n:
                break
    return reqs


def main():
    ap = argparse.ArgumentParser(description="Export a training run as a complete standalone artifact")
    ap.add_argument("--run", help="training-run directory (adapter/, head.safetensors, run.json)")
    ap.add_argument("--out", help="artifact directory to write (replaced if present)")
    ap.add_argument("--calibration", help="calibration JSON (decision_model.calibrate output; must contain 'temperature')")
    ap.add_argument("--conversion", help="conversion metadata JSON to include (recipe, manifest hashes, training metrics)")
    ap.add_argument("--check-data", help="prepared JSONL whose first --n-check requests join the equivalence check")
    ap.add_argument("--n-check", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--reload-check", help=argparse.SUPPRESS)
    ap.add_argument("--requests", help=argparse.SUPPRESS)
    ap.add_argument("--result", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.reload_check:
        return _reload_check(a.reload_check, a.requests, a.device, a.result)
    if not (a.run and a.out and a.calibration):
        ap.error("--run, --out and --calibration are required")
    report = export(a.run, a.out, calibration=json.loads(Path(a.calibration).read_text()),
                    conversion=json.loads(Path(a.conversion).read_text()) if a.conversion else {},
                    dtype=getattr(torch, a.dtype), device=a.device, check_requests=_check_requests(a.check_data, a.n_check))
    print(json.dumps({k: v for k, v in report.items() if k != "reloaded_answers"}, indent=2))


if __name__ == "__main__":
    main()
