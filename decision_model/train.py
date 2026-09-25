# Modified implementation for Kev; see LICENSE and NOTICE.
"""Train LoRA plus the pointer head once with the fixed recipe (SPEC section 7).

    python -m decision_model.train --config recipes/qwen3-1.7b.yaml --data data/prepared \
        --targets data/targets/<rev>/targets.jsonl --out runs/showcase/train

Loss per question: -sum(t * log_softmax(z)) with t = 0.5*one_hot(gold) + 0.5*q_cyclic when an eligible cyclic teacher
target exists, else one_hot(gold); questions are averaged within a record and records within an update. Each epoch
presents every Choice question in a fresh random option order (targets move with their stable option ids); noul and
Score are never reordered; no option is ever inserted or removed. Development NLL (gold, canonical order) is evaluated
every `eval_every` optimizer steps and at each epoch end; an improvement saves checkpoints/step-N/ and refreshes selected/.
"""
import argparse
import contextlib
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .render import to_record
from .schema import DecisionRequest, question_keys


# --- targets and augmentation (pure; exercised by tests/test_targets.py) -----------------------------------------------

def load_teacher(path):
    """{(record_id, qid): cyclic {key: p}} for eligible teacher targets."""
    out = {}
    if path is None:
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            if t["eligible"] and t.get("cyclic") is not None:
                out[(t["record_id"], t["qid"])] = t["cyclic"]
    return out


def augment_request(request, rng):
    """Random presentation order for every Choice question (the criteria dict is reordered, ids keep their
    descriptions); noul and Score untouched; nothing inserted or removed."""
    qs = {}
    for qid, q in request["questions"].items():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            rng.shuffle(keys)
            q = {**q, "criteria": {k: q["criteria"][k] for k in keys}}
        qs[qid] = q
    return {**request, "questions": qs}


def example(record, teacher, rng=None, gold_weight=0.5):
    """(DecisionRequest in this epoch's presentation, per-question target vectors in that request's canonical key
    order). Targets are looked up by stable key, so a reorder can never misalign them."""
    request = augment_request(record["request"], rng) if rng is not None else record["request"]
    req = DecisionRequest.model_validate(request)
    vecs = []
    for qid, q in req.questions.items():
        keys = question_keys(q.type, q.criteria)
        gold = record["supervision"][qid]["gold"]
        cyc = teacher.get((record["record_id"], qid))
        if cyc is not None:
            if set(cyc) != set(keys):
                raise ValueError(f"teacher keys for {record['record_id']}/{qid} differ from the request's options")
            t = [gold_weight * (k == gold) + (1 - gold_weight) * cyc[k] for k in keys]
        else:
            t = [float(k == gold) for k in keys]
        vecs.append(t)
    return req, vecs


def soft_ce(z, t):
    return -(t * F.log_softmax(z, -1)).sum()


# --- training -----------------------------------------------------------------------------------------------------------

def _encode(model, req):
    rec, _ = to_record(req)
    return model.encode(rec)


@torch.no_grad()
def dev_nll(model, records, autocast, batch=8):
    """Mean gold NLL over every development question, canonical option order, T = 1."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(records), batch):
        part = records[s:s + batch]
        encs, golds = [], []
        for r in part:
            req = DecisionRequest.model_validate(r["request"])
            encs.append(_encode(model, req))
            golds.append([question_keys(q.type, q.criteria).index(r["supervision"][qid]["gold"]) for qid, q in req.questions.items()])
        with autocast:
            logits = model.forward_batch(encs, form="packed")
        for zs, gs in zip(logits, golds):
            for z, g in zip(zs, gs):
                total += -F.log_softmax(z.float(), -1)[g].item()
                n += 1
    model.train(was_training)
    return total / n, n


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(cfg, data_dir, targets_path, out_dir, device="cuda", max_steps=None):
    from .data import load_split
    from .model import DecisionModel

    t = cfg["train"]
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite the existing run {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    max_steps = max_steps if max_steps is not None else t.get("max_steps")
    records = load_split(data_dir, "train")
    dev = load_split(data_dir, "dev")
    teacher = load_teacher(targets_path)
    torch.manual_seed(t["seed"])
    rng = random.Random(t["seed"])
    cuda = str(device).startswith("cuda")
    autocast = torch.autocast("cuda", dtype=getattr(torch, t["autocast"])) if cuda and t.get("autocast") else contextlib.nullcontext()
    model = DecisionModel.from_parent(cfg["parent"], cfg["revision"], device=device, dtype=getattr(torch, t["weights_dtype"]),
                                      lora=t["lora"], head_dim=t["head_dim"])
    if t.get("activation_checkpointing"):
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    accum = t["accumulation"]
    if t["microbatch"] != 1:
        raise ValueError("the recipe trains with microbatch 1 (effective batch = accumulation)")
    steps_per_epoch = math.ceil(len(records) / accum)
    total_steps = t["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=total_steps, pct_start=t["pct_start"])
    n_teacher = sum((r["record_id"], q) in teacher for r in records for q in r["request"]["questions"])
    config = {"recipe": t, "parent": cfg["parent"], "revision": cfg["revision"], "device": device,
              "device_name": torch.cuda.get_device_name(0) if cuda else str(device),
              "data_manifest_sha256": _sha(Path(data_dir) / "manifest.json"), "targets_path": str(targets_path) if targets_path else None,
              "targets_sha256": _sha(targets_path) if targets_path else None, "train_records": len(records), "dev_records": len(dev),
              "train_questions": sum(len(r["request"]["questions"]) for r in records), "questions_with_teacher_target": n_teacher,
              "optimizer_steps_planned": total_steps, "max_steps": max_steps, "trainable_parameters": sum(p.numel() for p in params),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "versions": _versions()}
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"train: {len(records)} records, {n_teacher} questions with teacher targets, {total_steps} optimizer steps planned, "
          f"{config['trainable_parameters'] / 1e6:.1f}M trainable parameters", flush=True)

    metrics = {"complete": False, "wall_seconds": 0.0, "records_seen": 0, "forward_tokens": 0, "optimizer_steps": 0,
               "peak_device_bytes": 0, "dev_nll_history": [], "selected_step": None, "selected_dev_nll": None}
    best = math.inf
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def write_metrics():
        metrics["wall_seconds"] = time.time() - t0
        if cuda:
            metrics["peak_device_bytes"] = torch.cuda.max_memory_allocated()
        (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    def evaluate(step, epoch):
        nonlocal best
        te = time.time()
        nll, n = dev_nll(model, dev, autocast, t["eval_batch"])
        entry = {"step": step, "epoch": epoch, "dev_nll": nll, "questions": n, "seconds": time.time() - te}
        metrics["dev_nll_history"].append(entry)
        if nll < best:
            best = nll
            ck = out_dir / "checkpoints" / f"step-{step}"
            model.save_run(ck, extra={"step": step, "epoch": epoch, "dev_nll": nll, "recipe": t})
            sel = out_dir / "selected"
            if sel.exists():
                shutil.rmtree(sel)
            shutil.copytree(ck, sel)
            metrics.update(selected_step=step, selected_dev_nll=nll)
            entry["saved"] = str(ck)
        print(f"dev: step {step} epoch {epoch} nll {nll:.4f} over {n} questions ({entry['seconds']:.0f}s){' *' if entry.get('saved') else ''}", flush=True)
        write_metrics()

    model.train()
    step, run_loss, run_n, stop = 0, 0.0, 0, False
    for epoch in range(t["epochs"]):
        order = list(records)
        rng.shuffle(order)
        for mb, r in enumerate(order):
            req, vecs = example(r, teacher, random.Random(f"{t['seed']}:{epoch}:{r['record_id']}"), t["gold_weight"])
            enc = _encode(model, req)
            with autocast:
                logits = model.forward_batch([enc], form="packed")[0]
            loss = sum(soft_ce(z.float(), torch.tensor(v, device=z.device, dtype=torch.float32)) for z, v in zip(logits, vecs)) / len(vecs)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss on {r['record_id']}")
            group = min(accum, len(order) - (mb // accum) * accum)
            (loss / group).backward()
            run_loss += loss.item(); run_n += 1
            metrics["records_seen"] += 1
            metrics["forward_tokens"] += len(enc["ids"])
            if (mb + 1) % accum == 0 or mb + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, t["clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                metrics["optimizer_steps"] = step
                if step % 10 == 0:
                    el = time.time() - t0
                    spr = el / metrics["records_seen"]
                    print(f"ep{epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{spr:.3f}s/record elapsed {el / 60:.1f}m projected {spr * len(records) * t['epochs'] / 3600:.2f}h "
                          f"peak {torch.cuda.max_memory_allocated() / 2**30 if cuda else 0:.1f}GiB", flush=True)
                    run_loss, run_n = 0.0, 0
                    write_metrics()
                if max_steps is not None and step >= max_steps:
                    stop = True
                    break
                if step % t["eval_every"] == 0:
                    evaluate(step, epoch)
        if stop:
            break
        if not metrics["dev_nll_history"] or metrics["dev_nll_history"][-1]["step"] != step:
            evaluate(step, epoch)
    metrics["complete"] = not stop
    metrics["stopped_at_max_steps"] = stop
    write_metrics()
    print(f"train: done, {step} steps, selected step {metrics['selected_step']} dev nll {metrics['selected_dev_nll']}", flush=True)
    return metrics


def _versions():
    import peft
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "cuda": torch.version.cuda}


def main():
    from .data import load_recipe

    ap = argparse.ArgumentParser(description="train LoRA + pointer head with the fixed recipe")
    ap.add_argument("--config", "--recipe", dest="config", default="recipes/qwen3-1.7b.yaml")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--targets", required=True, help="teacher targets.jsonl ('none' = gold only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (throughput measurement)")
    a = ap.parse_args()
    train(load_recipe(a.config), a.data, None if a.targets == "none" else a.targets, a.out, a.device, a.max_steps)


if __name__ == "__main__":
    main()
