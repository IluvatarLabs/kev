# Modified implementation for Kev; see LICENSE and NOTICE.
"""Quality panels, performance profiles and functional acceptance for one exported artifact versus its parent.

    python -m decision_model.evaluate --model runs/showcase/model --data data/prepared --targets data/targets/<rev>/targets.jsonl \
        --out runs/showcase/eval --panels showcase,test,unseen --profiles

Systems (all bf16, same GPU, same attention backend, same inputs):
  converted      the exported artifact as shipped: raw (T=1) and calibrated (stored T) quality rows; performance in the
                 packed form (headline) and the cached form (state prefill + batched branch rows)
  parent_json    the parent generating a JSON object of answers (baselines.ParentGenerator), showcase panel only
  parent_raw     the parent's one-token restricted-answer scoring, one layout (targets.ParentScorer)
  parent_cyclic  the same over every cyclic option layout, log-mean combined (every layout is executed and counted)

Outputs in --out: rows.jsonl (one row per example x question x system x order variant), summary.json, benchmark.json,
summary.md (the short table with conditions and this command). A full run (no --limit) also copies the summary into
<model>/evaluation.json and fills the measured section of <model>/README.md. Nothing is extrapolated: every figure comes
from this process; systems or sections not run are marked "not measured".
"""
import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import random
import shlex
import statistics
import sys
import time
from pathlib import Path

import torch
from pydantic import ValidationError

from .baselines import ParentGenerator, sync
from .export import MEASURED_END, MEASURED_START, dir_bytes
from .render import encode, rows_of, to_record, user_tokens
from .schema import AdmissionError, DecisionRequest, question_keys

SYSTEMS = ("converted", "parent_json", "parent_raw", "parent_cyclic")
EPS = 1e-12

# ---------------------------------------------------------------- performance workload (fixed, deterministic)

SENTENCES = [
    "I ordered a pair of running shoes on the first of the month and paid with my credit card.",
    "The confirmation email promised delivery in three to five business days.",
    "The tracking page did not update for more than a week, and nobody answered the chat.",
    "When the package finally arrived the box was crushed on one side and the lid was taped shut.",
    "The shoes inside were a size ten instead of the size nine I selected at checkout.",
    "My bank statement also shows two identical charges from your store on the same afternoon.",
    "I have been a customer for six years and this has never happened before.",
    "I need the right size before a race next weekend, otherwise the order is useless to me.",
    "Please tell me whether I should send the wrong pair back first or wait for a label.",
    "I would also like the duplicate charge reversed as soon as possible.",
    "Your returns page says exchanges take up to fourteen days, which is too long for me.",
    "I called the support line twice and was disconnected both times after waiting on hold.",
]
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"returns": "Exchanges and wrong items", "shipping": "Delivery delays and damage", "billing": "Charges and refunds", "account": "Login and profile"}},
    "resolution": {"type": "choice", "instructions": "What outcome does the customer want most?",
                   "criteria": {"exchange": "A different size or item", "refund": "Money back", "replacement": "The same item again", "information": "An answer only"}},
    "tone": {"type": "choice", "instructions": "What is the customer's tone?",
             "criteria": {"calm": None, "concerned": None, "frustrated": None, "angry": None}},
    "priority": {"type": "choice", "instructions": "How should this ticket be prioritised?",
                 "criteria": {"low": "Can wait a week", "normal": "Within two days", "high": "Within a day", "urgent": "Within hours"}},
    "channel": {"type": "choice", "instructions": "How should we reply?",
                "criteria": {"email": None, "phone": "Call the customer back", "chat": None, "letter": "Postal mail"}},
    "issue": {"type": "choice", "instructions": "What is the main problem?",
              "criteria": {"wrong_item": "Wrong size or product", "late": "Late delivery", "damaged": "Damaged packaging or item", "charge": "Billing error"}},
    "loyalty": {"type": "choice", "instructions": "How long has the customer been with us?",
                "criteria": {"new": "Under a year", "regular": "One to five years", "longtime": "More than five years", "unknown": "Not stated"}},
    "next_step": {"type": "choice", "instructions": "What should the agent do first?",
                  "criteria": {"label": "Email a return label", "refund": "Reverse the duplicate charge", "callback": "Schedule a call", "escalate": "Escalate to a supervisor"}},
}
PROFILES = {"state128_q1": (128, 1), "state512_q8": (512, 8), "state2048_q32": (2048, 32)}
# 32 four-choice questions for the largest profile: the eight above, each asked about four aspects of the ticket
ASPECTS = ("", " Consider the first issue the customer raises.", " Consider the customer's stated deadline.",
           " Consider the most recent message only.")
QUESTIONS32 = {(qid if k == 0 else f"{qid}_{k}"): {**q, "instructions": q["instructions"] + ASPECTS[k]}
               for k in range(4) for qid, q in QUESTIONS.items()}


def workload(tok, n_state: int, n_q: int, count: int) -> list[DecisionRequest]:
    """`count` distinct requests: a ticket number plus rotated sentences filling about n_state state tokens."""
    qs = dict(list((QUESTIONS if n_q <= len(QUESTIONS) else QUESTIONS32).items())[:n_q])
    out = []
    for i in range(count):
        text, j = f"Ticket {1000 + i}.", i
        while len(user_tokens(tok, text)) < n_state:
            text += " " + SENTENCES[j % len(SENTENCES)]
            j += 1
        ids = user_tokens(tok, text)[:n_state - 1]        # the state delimiter makes n_state
        out.append(DecisionRequest.model_validate({"state": tok.decode(ids), "questions": qs}))
    return out


# ---------------------------------------------------------------- helpers

def device_mem():
    if torch.cuda.is_available():
        return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
    return None


def gpu_activity():
    """Other compute processes and utilisation on the visible GPU (nvidia-smi), so an unidle benchmark GPU is visible."""
    if not torch.cuda.is_available():
        return None
    import subprocess
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    try:
        q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        gpu = [l.split(", ") for l in q("--query-gpu=uuid,utilization.gpu,memory.used").splitlines()]
        apps = [l.split(", ") for l in q("--query-compute-apps=gpu_uuid,pid,used_memory").splitlines() if l.strip()]
    except (OSError, subprocess.CalledProcessError) as e:
        return {"error": str(e)}
    mine = os.getpid()
    util = next(({"utilization_pct": int(g[1]), "memory_used_mib": int(g[2])} for g in gpu if g[0] == uuid), {})
    others = [{"pid": int(x[1]), "used_mib": int(x[2])} for x in apps if x[0] == uuid and int(x[1]) != mine]
    return {**util, "other_processes": others}


def reset_peak(device):
    sync(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak(device):
    sync(device)
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None


def timed(device, fn, *a, **k):
    sync(device)
    t = time.perf_counter()
    out = fn(*a, **k)
    sync(device)
    return out, (time.perf_counter() - t) * 1e3


def free(device):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def qdump(req: DecisionRequest):
    return {k: q.model_dump() for k, q in req.questions.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- quality rows

def load_panel(data, panel, limit):
    recs = [json.loads(l) for l in open(Path(data) / f"{panel}.jsonl") if l.strip()]
    return recs[:limit] if limit else recs


def permuted(req: DecisionRequest, rng) -> DecisionRequest:
    """Every Choice question's options in a random order (ids and descriptions kept together)."""
    qs = {}
    for qid, q in qdump(req).items():
        if q["type"] == "choice":
            items = list(q["criteria"].items())
            rng.shuffle(items)
            q = {**q, "criteria": dict(items)}
        qs[qid] = q
    return DecisionRequest.model_validate({"state": req.state, "questions": qs})


def has_choice(req):
    return any(q.type == "choice" and len(q.criteria) > 1 for q in req.questions.values())


def make_row(panel, rec, qid, q, system, variant, probs=None, pred=None, *, eligible=True, reason=None, **extra):
    keys = question_keys(q.type, q.criteria)
    sup = rec.get("supervision", {}).get(qid, {})
    gold = sup.get("gold")
    gold = None if gold is None else str(gold).lower() if isinstance(gold, bool) else str(gold)
    p = [float(probs[k]) for k in keys] if probs is not None else None
    if pred is None and p is not None:
        pred = keys[max(range(len(p)), key=p.__getitem__)]
    row = {"panel": panel, "record_id": rec["record_id"], "source_family": rec.get("source_family"),
           "task": sup.get("task") or rec.get("schema_family"), "schema_family": rec.get("schema_family"), "question": qid,
           "type": q.type, "system": system, "variant": variant, "keys": keys, "probs": p, "pred": pred, "gold": gold,
           "correct": (pred == gold) if gold is not None else None, "eligible": eligible, "reason": reason, **extra}
    if q.type == "score":
        row["expected_level"] = sum(i * x for i, x in enumerate(p)) if p is not None else (int(pred) if pred is not None else None)
    return row


def converted_rows(model, panel, recs, form, order_variants, seed):
    rows = []
    for rec in recs:
        req = DecisionRequest.model_validate(rec["request"])
        variants = [("canonical", req)] + ([(f"order{v}", permuted(req, random.Random(f"{seed}/{rec['record_id']}/{v}")))
                                             for v in range(order_variants)] if has_choice(req) else [])
        for variant, r in variants:
            for system, T in (("converted_raw", 1.0), ("converted_cal", None)):
                if variant != "canonical" and system == "converted_cal":
                    continue
                try:
                    out, ms = timed(model.device, model.evaluate, r.state, qdump(r), form=form, temperature=T)
                except AdmissionError as e:
                    rows += [make_row(panel, rec, qid, q, system, variant, eligible=False, reason=f"admission: {e}")
                             for qid, q in r.questions.items()]
                    continue
                rows += [make_row(panel, rec, qid, q, system, variant, out["answers"][qid]["probabilities"], latency_ms=ms)
                         for qid, q in r.questions.items()]
    return rows


def scorer_rows(scorer, panel, recs, modes, order_variants, seed, chunk=32):
    rows = []
    jobs = []   # (rec, variant, request)
    for rec in recs:
        req = DecisionRequest.model_validate(rec["request"])
        jobs.append((rec, "canonical", req))
        if order_variants and has_choice(req):
            jobs += [(rec, f"order{v}", permuted(req, random.Random(f"{seed}/{rec['record_id']}/{v}"))) for v in range(order_variants)]
    for mode in modes:
        system = f"parent_{mode}"
        for i in range(0, len(jobs), chunk):
            part = jobs[i:i + chunk]
            res = scorer.score_many([r for _, _, r in part], mode)
            for (rec, variant, r), out in zip(part, res):
                for qid, q in r.questions.items():
                    o = out[qid]
                    rows.append(make_row(panel, rec, qid, q, system, variant, o["probs"] if o["eligible"] else None,
                                         eligible=o["eligible"], reason=o.get("reason"), layouts=o.get("layouts"),
                                         prompt_tokens=o.get("prompt_tokens")))
    return rows


def json_rows(gen, panel, recs, system="parent_json", batch=1):
    """Quality rows for a generating parent; batch > 1 uses left-padded batched generation (latency is per batch)."""
    rows = []
    for i in range(0, len(recs), batch):
        part = recs[i:i + batch]
        reqs = [DecisionRequest.model_validate(rec["request"]) for rec in part]
        outs = gen.answer_batch(reqs) if batch > 1 else [gen.answer(reqs[0])]
        for rec, req, out in zip(part, reqs, outs):
            for qid, q in req.questions.items():
                rows.append(make_row(panel, rec, qid, q, system, "canonical", pred=out["answers"][qid],
                                     parse_ok=out["parse_ok"], answered=out["answers"][qid] is not None, truncated=out.get("truncated"),
                                     latency_ms=out["latency_ms"], generated_tokens=out["generated_tokens"], batch=len(part),
                                     text=out["text"] if out["answers"][qid] is None else None))
    return rows


def generation_stats(rows):
    """Per generating system: generated tokens per request, truncated and unparsed responses (showcase quality rows)."""
    out = {}
    for sysname in sorted({r["system"] for r in rows if "generated_tokens" in r}):
        reqs = {}
        for r in rows:
            if r["system"] == sysname and r["variant"] == "canonical":
                reqs.setdefault((r["panel"], r["record_id"]), r)
        g = [r["generated_tokens"] for r in reqs.values()]
        out[sysname] = {"requests": len(reqs), "generated_tokens_mean": statistics.fmean(g), "generated_tokens_median": statistics.median(g),
                        "generated_tokens_max": max(g), "truncated": sum(bool(r.get("truncated")) for r in reqs.values()),
                        "no_json": sum(not r["parse_ok"] for r in reqs.values()),
                        "questions_unanswered": sum(r["pred"] is None for r in rows if r["system"] == sysname and r["variant"] == "canonical")}
    return out


# ---------------------------------------------------------------- metrics

def metrics(rows):
    """Accuracy over the questions the system is eligible for (`n`); ineligible questions (e.g. more options than the
    parent scorer's labels) are counted beside it, never scored. An eligible question left unanswered (unparsed JSON)
    counts as wrong. accuracy_answered covers only answered questions."""
    all_rows = [r for r in rows if r["gold"] is not None]
    rows = [r for r in all_rows if r["eligible"]]
    if not all_rows:
        return None
    answered = [r for r in rows if r["pred"] is not None]
    out = {"n": len(rows), "n_total": len(all_rows), "answered": len(answered), "ineligible": len(all_rows) - len(rows),
           "unanswered_or_unparsed": sum(r["pred"] is None for r in rows),
           "accuracy": sum(bool(r["correct"]) for r in rows) / len(rows) if rows else None,
           "accuracy_answered": (sum(bool(r["correct"]) for r in answered) / len(answered)) if answered else None}
    prob = [r for r in rows if r["probs"] is not None]
    if prob:
        nll, brier = [], []
        for r in prob:
            g = r["keys"].index(r["gold"])
            nll.append(-math.log(max(r["probs"][g], EPS)))
            brier.append(sum((p - (i == g)) ** 2 for i, p in enumerate(r["probs"])))
        out.update(nll=statistics.fmean(nll), brier=statistics.fmean(brier))
    sc = [r for r in rows if r["type"] == "score" and r["pred"] is not None]
    if sc:
        out["score_mae_expected"] = statistics.fmean(abs(r["expected_level"] - int(r["gold"])) for r in sc)
        out["score_mae_modal"] = statistics.fmean(abs(int(r["pred"]) - int(r["gold"])) for r in sc)
        out["score_n"] = len(sc)
    return out


def grouped(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def panel_summary(rows):
    canon = [r for r in rows if r["variant"] == "canonical"]
    systems = sorted({r["system"] for r in canon})
    by = {s: [r for r in canon if r["system"] == s] for s in systems}
    out = {"records": len({r["record_id"] for r in canon}), "questions": len({(r["record_id"], r["question"]) for r in canon}),
           "systems": {s: {"overall": metrics(v), "by_task": grouped(v, "task"), "by_source": grouped(v, "source_family"),
                           "by_type": grouped(v, "type")} for s, v in by.items()}}
    # agreement: converted argmax vs parent argmax (not accuracy), on questions both answered
    idx = {s: {(r["record_id"], r["question"]): r for r in v} for s, v in by.items()}
    agree = {}
    for c in (s for s in systems if s.startswith("converted")):
        for p in (s for s in systems if s.startswith("parent")):
            both = [(a, idx[p][k]) for k, a in idx[c].items() if k in idx[p] and a["pred"] is not None and idx[p][k]["pred"] is not None]
            agree[f"{c}_vs_{p}"] = {"n": len(both), "agreement": sum(a["pred"] == b["pred"] for a, b in both) / len(both) if both else None}
    out["parent_agreement"] = agree
    # converted accuracy on the questions each parent scorer could answer, for a like-for-like comparison
    for p in ("parent_raw", "parent_cyclic"):
        if p in idx and "converted_raw" in idx:
            keys = [k for k, r in idx[p].items() if r["eligible"]]
            out.setdefault("on_parent_eligible", {})[p] = {s: metrics([idx[s][k] for k in keys if k in idx[s]])
                                                          for s in ("converted_raw", "converted_cal", p) if s in idx}
    # option-order sensitivity (diagnostic): argmax flips and max probability change across random Choice orders
    order = {}
    for s in systems:
        diffs, flips = [], []
        base = idx[s]
        for r in rows:
            if r["system"] != s or r["variant"] == "canonical" or r["type"] != "choice" or r["probs"] is None:
                continue
            b = base.get((r["record_id"], r["question"]))
            if b is None or b["probs"] is None:
                continue
            pv = dict(zip(r["keys"], r["probs"]))
            diffs.append(max(abs(pv[k] - x) for k, x in zip(b["keys"], b["probs"])))
            flips.append(r["pred"] != b["pred"])
        if diffs:
            order[s] = {"n": len(diffs), "flip_rate": sum(flips) / len(flips), "mean_max_dp": statistics.fmean(diffs), "max_dp": max(diffs)}
    out["order_sensitivity"] = order
    return out


# ---------------------------------------------------------------- performance

def profile(device, one, batch_fn, reqs, n_q, a, cost):
    """Batch-1 latency (a.warmup untimed, a.repeats timed, distinct requests) and batch throughput (one untimed batch,
    a.batch_repeats timed). `cost(result, req)` -> (executed tokens, forward passes) per request."""
    it = iter(reqs)
    budget = getattr(a, "profile_budget_s", 1800)
    warm = [timed(device, one, next(it))[1] for _ in range(a.warmup)]
    if warm and statistics.fmean(warm) / 1e3 * a.repeats > budget:
        why = (f"not measured: {a.repeats} timed requests would take about {statistics.fmean(warm) / 1e3 * a.repeats / 60:.0f} min "
               f"(warm-up mean {statistics.fmean(warm):.0f} ms), over the {budget / 60:.0f}-minute budget")
        return {"state_tokens_target": None, "questions": n_q, "batch1": {"not_measured": why, "warmup_ms": warm}, f"batch{a.batch}": {"not_measured": why}}
    reset_peak(device)
    lat, toks, passes = [], [], []
    for _ in range(a.repeats):
        r = next(it)
        out, ms = timed(device, one, r)
        lat.append(ms)
        t, p = cost(out, r)
        toks.append(t)
        passes.append(p)
    peak1 = peak(device)
    b1 = {"n": len(lat), "warmup": a.warmup, "p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95), "mean_ms": statistics.fmean(lat),
          "executed_tokens_per_request": statistics.fmean(toks), "forward_passes_per_request": statistics.fmean(passes),
          "peak_gpu_bytes": peak1}
    if not getattr(a, "batch_throughput", True):
        return {"state_tokens_target": None, "questions": n_q, "batch1": b1, f"batch{a.batch}": None}
    batches = [[next(it) for _ in range(a.batch)] for _ in range(a.batch_repeats + 1)]
    _, warm_b = timed(device, batch_fn, batches[0])
    if warm_b / 1e3 * a.batch_repeats > budget:
        why = (f"not measured: {a.batch_repeats} timed batches would take about {warm_b / 1e3 * a.batch_repeats / 60:.0f} min "
               f"(untimed batch {warm_b / 1e3:.0f} s), over the {budget / 60:.0f}-minute budget")
        return {"state_tokens_target": None, "questions": n_q, "batch1": b1, f"batch{a.batch}": {"not_measured": why, "untimed_batch_s": warm_b / 1e3}}
    reset_peak(device)
    secs = [timed(device, batch_fn, b)[1] / 1e3 for b in batches[1:]]
    peak_b = peak(device)
    med = statistics.median(secs)
    return {"state_tokens_target": None, "questions": n_q, "batch1": b1,
            f"batch{a.batch}": {"batches_timed": len(secs), "batch_seconds": secs, "median_batch_seconds": med,
                                "decisions_per_second": a.batch * n_q / med, "requests_per_second": a.batch / med,
                                "peak_gpu_bytes": peak_b}}


def cold_request(device, fn, tok):
    """The first request after load, before any other work: one headline-profile request (512-token state, 8 questions)."""
    n_state, n_q = PROFILES["state512_q8"]
    req = workload(tok, n_state, n_q, 1)[0]
    _, ms = timed(device, fn, req)
    return {"cold_request_ms": ms, "cold_request_profile": "state512_q8"}


def run_profiles(device, name, one, batch_fn, cost, tok, a):
    """All profiles for one loaded (already warm) system."""
    out = {}
    for pname, (n_state, n_q) in PROFILES.items():
        reqs = workload(tok, n_state, n_q, 1 + a.warmup + a.repeats + a.batch * (a.batch_repeats + 1))
        res = profile(device, one, batch_fn, reqs[1:], n_q, a, cost)
        res["state_tokens"] = n_state
        res.pop("state_tokens_target")
        out[pname] = res
        b1, bb = res["batch1"], res[f"batch{a.batch}"]
        print(f"[{name}] {pname}: " + (f"p50 {b1['p50_ms']:.1f} ms, p95 {b1['p95_ms']:.1f} ms, " if "p50_ms" in b1 else b1["not_measured"] + "; ")
              + (f"{bb['decisions_per_second']:.1f} decisions/s" if bb and "decisions_per_second" in bb else "batch throughput not measured"), flush=True)
    return out


# ---------------------------------------------------------------- functional acceptance (converted model, not timed)

def filler(tok, n):
    """Text of exactly n tokens under the renderer's tokenizer."""
    words = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
    ids = user_tokens(tok, " ".join(words[i % len(words)] for i in range(n)))[:n]
    text = tok.decode(ids)
    for _ in range(8):
        got = user_tokens(tok, text)
        if len(got) == n:
            return text
        ids = got[:n] if len(got) > n else got + user_tokens(tok, " alpha")[: n - len(got)]
        text = tok.decode(ids)
    raise RuntimeError(f"could not build a {n}-token filler")


def acceptance(model):
    tok = model.tok
    noul = {"flag": {"type": "noul", "instructions": "Is anything mentioned?"}}
    s2047 = filler(tok, model.limits["max_state"] - 1)
    cases = {}

    def row_request(target):
        k = 1000
        for _ in range(6):
            q = {"long": {"type": "choice", "instructions": filler(tok, k), "criteria": {"a": None, "b": None, "c": None, "d": None}}}
            rec, _ = to_record(DecisionRequest.model_validate({"state": s2047, "questions": q}))
            enc = encode(tok, rec, max_state=10 ** 9, max_row=10 ** 9)   # measure only; admission is tested below
            n = len(enc["ids"])
            if n == target:
                return {"state": s2047, "questions": q}
            k += target - n
        raise RuntimeError(f"could not build a {target}-token row")

    specs = {
        "state_2048_tokens": ({"state": s2047, "questions": noul}, True),
        "row_4096_tokens": (row_request(model.limits["max_row"]), True),
        "questions_32": ({"state": "A customer asks about a refund for a late order.",
                          "questions": {f"q{i}": ({"type": "choice", "instructions": f"Question {i}", "criteria": {"x": None, "y": None, "z": None}}
                                                  if i % 3 == 0 else {"type": "noul", "instructions": f"Question {i}"} if i % 3 == 1 else
                                                  {"type": "score", "instructions": f"Question {i}", "criteria": ["low", "mid", "high"]})
                                        for i in range(32)}}, True),
        "choice_255_options": ({"state": "Pick the matching code.", "questions": {"code": {"type": "choice", "instructions": "Which code applies?",
                                                                                         "criteria": {f"c{i:03d}": None for i in range(255)}}}}, True),
        "score_255_levels": ({"state": "Rate it.", "questions": {"level": {"type": "score", "instructions": "Which level applies?",
                                                                         "criteria": [f"level {i}" for i in range(255)]}}}, True),
        "state_2049_tokens_refused": ({"state": filler(tok, model.limits["max_state"]), "questions": noul}, False),
        "row_4097_tokens_refused": (row_request(model.limits["max_row"] + 1), False),
        "questions_33_refused": ({"state": "x", "questions": {f"q{i}": {"type": "noul"} for i in range(33)}}, False),
        "choice_256_options_refused": ({"state": "x", "questions": {"c": {"type": "choice", "criteria": {f"c{i}": None for i in range(256)}}}}, False),
    }
    for name, (body, should_pass) in specs.items():
        try:
            req = DecisionRequest.model_validate(body)
            t = time.perf_counter()
            out = model.evaluate(req.state, qdump(req))
            ms = (time.perf_counter() - t) * 1e3
            checks = {}
            for qid, q in req.questions.items():
                p = out["answers"][qid]["probabilities"]
                checks[qid] = {"keys_complete": list(p) == question_keys(q.type, q.criteria), "all_finite": all(math.isfinite(x) for x in p.values()),
                               "sum": sum(p.values())}
            ok = all(c["keys_complete"] and c["all_finite"] and abs(c["sum"] - 1) < 1e-5 for c in checks.values())
            cases[name] = {"expected": "answer" if should_pass else "refusal", "outcome": "answered", "passed": ok and should_pass,
                           "usage": out["usage"], "latency_ms_untimed": ms,
                           "min_sum": min(c["sum"] for c in checks.values()), "max_sum": max(c["sum"] for c in checks.values()),
                           "questions": len(checks)}
        except (AdmissionError, ValidationError) as e:
            cases[name] = {"expected": "answer" if should_pass else "refusal", "outcome": f"refused ({type(e).__name__})",
                           "passed": not should_pass, "message": str(e).splitlines()[0][:300]}
        print(f"[acceptance] {name}: {cases[name]['outcome']} -> {'PASS' if cases[name]['passed'] else 'FAIL'}", flush=True)
    return cases


# ---------------------------------------------------------------- sizes

def parent_snapshot(parent, revision):
    from huggingface_hub import try_to_load_from_cache
    from safetensors import safe_open
    cfg = try_to_load_from_cache(parent, "config.json", revision=revision)   # the snapshot dir, even if a non-weight file is missing
    if not isinstance(cfg, str):
        return {"error": f"parent snapshot {parent}@{revision} not in the local cache"}
    path = Path(cfg).parent
    files = sorted(path.glob("*.safetensors"))
    lm_head = 0
    for f in files:
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith("lm_head"):
                    sl = s.get_slice(k)
                    lm_head += math.prod(sl.get_shape()) * 2 if sl.get_dtype() in ("BF16", "F16") else math.prod(sl.get_shape()) * 4
    return {"path": str(path), "weights_bytes": sum(f.stat().st_size for f in files), "lm_head_bytes": lm_head}


def artifact_sizes(model_dir):
    d = Path(model_dir)
    weights = sum(f.stat().st_size for f in (d / "backbone").glob("*.safetensors")) + (d / "decision_head.safetensors").stat().st_size
    return {"total_bytes": dir_bytes(d), "weights_bytes": weights}


# ---------------------------------------------------------------- report

def peak_of(p, B):
    return max(p["batch1"].get("peak_gpu_bytes") or 0, (p[B] or {}).get("peak_gpu_bytes") or 0)


def fmt_b(x):
    return "not measured" if x is None else f"{x / 1e9:.2f} GB"


def fmt_pct(m, key="accuracy"):
    if m and m.get(key) is None and m.get("ineligible"):
        return f"ineligible (n={m['ineligible']})"
    if not m or m.get(key) is None:
        return "not measured"
    s = f"{100 * m[key]:.1f}%"
    extra = []
    if m.get("ineligible"):
        extra.append(f"{m['ineligible']} ineligible")
    if m.get("unanswered_or_unparsed"):
        extra.append(f"{m['unanswered_or_unparsed']} unparsed")
    return s + f" (n={m['n']}" + ("; " + ", ".join(extra) if extra else "") + ")"


def summary_md(s):
    L = [f"# Benchmark: {s['model']}", ""]
    if s.get("smoke"):
        L += [f"**SMOKE RUN (--limit {s['smoke']}, {s['conditions']['repeats']} timed requests): plumbing check only; these are not benchmark results.**", ""]
    c = s["conditions"]
    for n in s.get("notes", []):
        L += [f"**Note:** {n}", ""]
    L += ["Command (every figure below comes from this run):", "", f"```\n{s['command']}\n```", "",
          f"Hardware and conditions: {c.get('gpu')} ({c.get('device')}), torch {c['torch']}, transformers {c['transformers']}, "
          f"dtype {c['dtype']}, attention {c['attn']}, NVIDIA_TF32_OVERRIDE={c.get('NVIDIA_TF32_OVERRIDE')}; batch 1 = "
          f"{c['warmup']} untimed warm-ups then {c['repeats']} timed distinct requests; batch {c['batch']} = one untimed batch then "
          f"{c['batch_repeats']} timed batches (median); every call timed end to end with device synchronisation. GPU at start: "
          f"{c.get('gpu_activity_at_start')}; at end: {c.get('gpu_activity_at_end')}.", ""]
    busy = [x for x in (c.get("gpu_activity_at_start"), c.get("gpu_activity_at_end"))
            if x and (x.get("utilization_pct", 0) > 5 or any(p["used_mib"] > 1024 for p in x.get("other_processes", [])))]
    if busy:
        L += ["**Warning: the GPU was busy (utilisation above 5% or another process holding more than 1 GiB) at the start or end "
              "of this run; latency and throughput are not idle-GPU measurements.**", ""]
    perf = s.get("performance", {})
    show = s["panels"].get("showcase")
    rows_sys = [("parent_json", "Parent, structured JSON generation", "parent_json"),
                ("parent_json_thinking", "Parent with thinking (reasoning on), structured JSON", "parent_json_thinking"),
                ("parent_raw", "Parent, one-token scoring (raw)", "parent_raw"),
                ("parent_cyclic", "Parent, one-token scoring (cyclic, all layouts)", "parent_cyclic"),
                ("converted_packed", "Converted (packed form), raw probabilities", "converted_raw"),
                ("converted_cached", "Converted (cached form, same model and probabilities up to rounding)", "converted_raw"),
                (None, "Converted, calibrated probabilities (same passes)", "converted_cal")]
    B = f"batch{c['batch']}"
    if show:
        sysm = show["systems"]
        base = (sysm.get("parent_json") or {}).get("overall")
        L += [f"## Showcase panel ({show['records']} records, {show['questions']} questions)", "",
              "Latency and throughput: 512-token state with eight 4-choice questions (the headline profile).", "",
              f"| System | Accuracy | vs parent JSON (pp) | Agreement with parent raw | p50 / p95 latency | decisions/s (batch {c['batch']}) | Peak GPU memory |",
              "|---|---|---|---|---|---|---|"]
        for psys, label, qsys in rows_sys:
            if qsys not in sysm and psys not in perf:
                continue
            m = (sysm.get(qsys) or {}).get("overall")
            same = psys == "converted_cached"   # the cached form computes the same function; quality rows use the packed form
            d = f"{100 * (m['accuracy'] - base['accuracy']):+.1f}" if m and base and m["accuracy"] is not None and not same and qsys != "parent_json" else "–"
            ag = show["parent_agreement"].get(f"{qsys}_vs_parent_raw", {}).get("agreement") if qsys.startswith("converted") and not same else None
            p = (perf.get(psys) or {}).get("profiles", {}).get("state512_q8") if psys else None
            lat = f"{p['batch1']['p50_ms']:.1f} / {p['batch1']['p95_ms']:.1f} ms" if p else "–"
            dps = (f"{p[B]['decisions_per_second']:.1f}" if p[B] else "not measured") if p else "–"
            pk = fmt_b(peak_of(p, B)) if p else "–"
            L.append(f"| {label} | {'see packed row' if same else fmt_pct(m)} | {d} | {'–' if ag is None else f'{100 * ag:.1f}%'} | {lat} | {dps} | {pk} |")
        L += ["", "Accuracy is over the questions each system is eligible for; ineligible questions (more options than the parent "
              "scorer's single-token labels support) are counted beside it. Unparsed JSON answers count as wrong.", ""]
        elig = show.get("on_parent_eligible", {})
        if elig:
            L += ["Like-for-like: accuracy on exactly the questions each parent scorer could answer.", "",
                  "| Questions the parent scorer could answer | Converted raw | Converted calibrated | That parent scorer |", "|---|---|---|---|"]
            for p_, ms in elig.items():
                L.append(f"| {p_} | " + " | ".join(fmt_pct(ms.get(k)) for k in ("converted_raw", "converted_cal", p_)) + " |")
            L.append("")
        pj = (perf.get("parent_json") or {}).get("profiles", {}).get("state512_q8")
        for cs in ("converted_packed", "converted_cached"):
            cp = (perf.get(cs) or {}).get("profiles", {}).get("state512_q8")
            if pj and cp and pj[B] and cp[B]:
                L.append(f"- {cs}: {pj['batch1']['p50_ms'] / cp['batch1']['p50_ms']:.1f}x lower p50 latency and "
                         f"{cp[B]['decisions_per_second'] / pj[B]['decisions_per_second']:.1f}x more decisions/s than structured JSON generation (measured ratio).")
        L.append("")
    if perf:
        L += ["## Performance profiles", "", f"| System | Profile | p50 / p95 ms (batch 1) | tokens / passes per request | decisions/s (batch {c['batch']}) | peak GPU memory | resident weights | cold load | cold request |",
              "|---|---|---|---|---|---|---|---|---|"]
        for sname, sp in perf.items():
            for pname, p in sp["profiles"].items():
                b1, bb = p["batch1"], p[B] or {}
                if "p50_ms" not in b1:
                    L.append(f"| {sname} | {pname} | {b1['not_measured']} | – | – | – | {fmt_b(sp.get('resident_bytes'))} | {sp['cold_load_s']:.1f} s | – |")
                    continue
                L.append(f"| {sname} | {pname} | {b1['p50_ms']:.1f} / {b1['p95_ms']:.1f} | "
                         f"{b1['executed_tokens_per_request']:.0f} / {b1['forward_passes_per_request']:.1f} | "
                         f"{(format(bb['decisions_per_second'], '.1f') if 'decisions_per_second' in bb else bb.get('not_measured', 'not measured'))} | {fmt_b(peak_of(p, B))} | "
                         f"{fmt_b(sp.get('resident_bytes'))} | {sp['cold_load_s']:.1f} s | {sp.get('cold_request_ms', float('nan')):.0f} ms |")
        L += ["", "Peak GPU memory is `torch.cuda.max_memory_allocated` during the measurement (weights included); resident weights = "
              "allocated bytes after load. Cyclic scoring executes and counts every option layout. Throughput counts decisions "
              "(questions answered); a request's state is one piece of work however many questions share it.", ""]
    for gname, g in s.get("generation", {}).items():
        L.append(f"- {gname} (showcase quality run): {g['requests']} requests, generated tokens per request mean {g['generated_tokens_mean']:.0f}, "
                 f"median {g['generated_tokens_median']:.0f}, max {g['generated_tokens_max']}; {g['truncated']} truncated at the token limit, "
                 f"{g['no_json']} with no parseable JSON, {g['questions_unanswered']} questions unanswered (counted wrong).")
    for sname, sp in perf.items():
        if sp.get("timed_generated_tokens_mean") is not None:
            L.append(f"- {sname} (timed profile requests): generated tokens per request mean {sp['timed_generated_tokens_mean']:.0f}, "
                     f"{sp['timed_truncated']} truncated, fully parsed {100 * sp['timed_requests_fully_parsed']:.0f}%, max_new_tokens {sp['max_new_tokens']}.")
    if s.get("generation") or any(sp.get("timed_generated_tokens_mean") is not None for sp in perf.values()):
        L.append("")
    a = s.get("artifact", {})
    ps = s.get("parent_snapshot", {})
    if a:
        L += ["## Checkpoint size", "", f"- Artifact directory: {a['total_bytes']:,} bytes; weights (backbone + head): {a['weights_bytes']:,} bytes.",
              f"- Parent bf16 snapshot weights: {ps.get('weights_bytes', 'not measured'):,} bytes, of which a stored `lm_head` is {ps.get('lm_head_bytes', 0):,} bytes."
              if "weights_bytes" in ps else f"- Parent snapshot: {ps.get('error')}",
              f"- The backbone keeps all {s.get('backbone_layers', 28)} parent layers and is unchanged in size; any difference comes from the vocabulary projection the "
              "decision model does not use. Peak runtime memory is reported separately above.", ""]
    for panel, ps_ in s["panels"].items():
        L += [f"## {panel} panel: accuracy per task", "", "| Task | " + " | ".join(ps_["systems"]) + " |", "|---|" + "---|" * len(ps_["systems"])]
        tasks = sorted({t for v in ps_["systems"].values() for t in v["by_task"]})
        for t in tasks:
            L.append(f"| {t} | " + " | ".join(fmt_pct(v["by_task"].get(t)) for v in ps_["systems"].values()) + " |")
        L.append("| **all** | " + " | ".join(fmt_pct(v["overall"]) for v in ps_["systems"].values()) + " |")
        L += ["", f"{panel} panel: accuracy per source (all of a source's questions together)", "",
              "| Source | " + " | ".join(ps_["systems"]) + " |", "|---|" + "---|" * len(ps_["systems"])]
        for src in sorted({t for v in ps_["systems"].values() for t in v["by_source"]}):
            L.append(f"| {src} | " + " | ".join(fmt_pct(v["by_source"].get(src)) for v in ps_["systems"].values()) + " |")
        elig = ps_.get("on_parent_eligible", {})
        if elig and panel != "showcase":
            L += ["", f"{panel} panel, like-for-like: accuracy on exactly the questions each parent scorer could answer.", "",
                  "| Questions the parent scorer could answer | Converted raw | Converted calibrated | That parent scorer |", "|---|---|---|---|"]
            L += [f"| {p_} | " + " | ".join(fmt_pct(ms.get(k)) for k in ("converted_raw", "converted_cal", p_)) + " |" for p_, ms in elig.items()]
        ag = ", ".join(f"{k} {100 * v['agreement']:.1f}% (n={v['n']})" for k, v in ps_["parent_agreement"].items() if v["agreement"] is not None)
        if ag:
            L += ["", f"Parent agreement (argmax match, not accuracy): {ag}."]
        for sname, v in ps_["systems"].items():
            o = v["overall"] or {}
            if "score_mae_expected" in o:
                L.append(f"- {sname} Score error: MAE of expected level {o['score_mae_expected']:.3f}, of modal level {o['score_mae_modal']:.3f} (n={o['score_n']}).")
        for sname, o in ps_.get("order_sensitivity", {}).items():
            L.append(f"- {sname} option-order sensitivity (diagnostic): flip rate {100 * o['flip_rate']:.1f}%, mean max |dp| {o['mean_max_dp']:.3f} over {o['n']} reordered Choice questions.")
        L.append("")
    acc = s.get("acceptance")
    if acc:
        L += ["## Functional acceptance (converted model, not timed)", "", "| Case | Expected | Outcome | Result |", "|---|---|---|---|"]
        L += [f"| {k} | {v['expected']} | {v['outcome']} | {'pass' if v['passed'] else 'FAIL'} |" for k, v in acc.items()]
        L.append("")
    for k in s.get("not_measured", []):
        L.append(f"- Not measured in this run: {k}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Evaluate and benchmark an exported decision model against its parent")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--targets", help="teacher targets JSONL the conversion used; its hash and collection cost are recorded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default="showcase", help="comma list of showcase,test,unseen (test is the locked panel: read it once, "
                    "last), or 'none' for profiles/acceptance only")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--profiles", action="store_true", help="run the timed performance profiles")
    ap.add_argument("--limit", type=int, default=None, help="records per panel (smoke runs; a limited run never writes into the artifact)")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--order-variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batch-repeats", type=int, default=5)
    ap.add_argument("--no-batch-throughput", dest="batch_throughput", action="store_false", help="batch-1 latency only")
    ap.add_argument("--gen-batch", type=int, default=8, help="batch size for parent_json_thinking quality generation")
    ap.add_argument("--thinking-max-new-tokens", type=int, default=2048)
    ap.add_argument("--note", action="append", default=[], help="a note printed at the top of summary.md (repeatable)")
    ap.add_argument("--panel-file", action="append", default=[], metavar="NAME=PATH",
                    help="a standalone prepared-format JSONL evaluated as panel NAME (repeatable); its sha256 is recorded")
    ap.add_argument("--profile-budget-s", type=float, default=1800,
                    help="skip (and record as not measured) any batch-1 or batch configuration estimated to take longer")
    a = ap.parse_args()
    import transformers
    from .model import BRANCH_TOKEN_BUDGET, DecisionModel, rows_per_pass

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    parent, revision = a.parent or cfg["parent"]["name"], a.revision or cfg["parent"]["revision"]
    systems = [s for s in a.systems.split(",") if s]
    panels = [] if a.panels == "none" else [p for p in a.panels.split(",") if p]
    data = {p: load_panel(a.data, p, a.limit) for p in panels}
    panel_files = {}
    for spec in a.panel_file:
        name, _, path = spec.partition("=")
        if not name or not path or name in data:
            ap.error(f"--panel-file needs a new name=path, got {spec!r}")
        recs = [json.loads(l) for l in open(path) if l.strip()]
        for r in recs:
            DecisionRequest.model_validate(r["request"])
        data[name] = recs[:a.limit] if a.limit else recs
        panel_files[name] = {"path": path, "sha256": sha256(path), "records": len(data[name])}
        panels.append(name)
    dev, dtype = a.device, torch.bfloat16
    s = {"command": "python -m decision_model.evaluate " + shlex.join(sys.argv[1:]), "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "model": a.model, "parent": {"name": parent, "revision": revision}, "smoke": a.limit,
         "backbone_layers": cfg["backbone"]["num_hidden_layers"],
         "conditions": {"device": dev, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
                        "dtype": "bfloat16", "attn": a.attn, "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                        "torch_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu_activity_at_start": gpu_activity(),
                        "warmup": a.warmup, "repeats": a.repeats, "batch": a.batch, "batch_repeats": a.batch_repeats,
                        "branch_token_budget": BRANCH_TOKEN_BUDGET, "order_variants": a.order_variants, "seed": a.seed},
         "panels": {}, "performance": {}, "not_measured": [], "notes": a.note}
    s["conditions"]["panel_files"] = panel_files
    s["conditions"]["profile_budget_s"] = a.profile_budget_s
    if a.targets:
        t = Path(a.targets)
        cost = {"lines": 0, "layouts": 0, "prompt_tokens": 0, "eligible": 0}
        with open(t) as f:
            for line in f:
                x = json.loads(line)
                cost["lines"] += 1
                cost["eligible"] += bool(x.get("eligible"))
                cost["layouts"] += (x.get("cost") or {}).get("layouts", 0) or 0
                cost["prompt_tokens"] += (x.get("cost") or {}).get("prompt_tokens", 0) or 0
        s["teacher_targets"] = {"path": str(t), "sha256": sha256(t), "collection_cost": cost}
    all_rows = []
    base_mem = device_mem()

    def add(rows):
        all_rows.extend(rows)

    # converted: the shipped bf16 artifact
    if "converted" in systems:
        t0 = time.perf_counter()
        model = DecisionModel.from_pretrained(a.model, device=dev, dtype=dtype, attn=a.attn)
        sync(dev)
        load_s = time.perf_counter() - t0
        resident = (device_mem() or {}).get("allocated")
        cold = cold_request(dev, lambda r: model.evaluate(r.state, qdump(r), form="packed"), model.tok)
        cold["cold_request_form"] = "packed (one cold request, shared by both converted rows)"
        s["temperature"] = model.head.temperature
        for p, recs in data.items():
            add(converted_rows(model, p, recs, "packed", a.order_variants if p == "showcase" else 0, a.seed))
            print(f"[converted] {p}: {len(recs)} records", flush=True)
        s["acceptance"] = acceptance(model)
        if a.profiles:
            for form in ("packed", "cached"):
                def cost(out, r, form=form):
                    u = out["usage"]
                    if form == "packed":
                        return u["state_tokens"] + u["branch_tokens"], 1
                    enc = model.encode(to_record(r)[0])
                    _, _, rows = rows_of(enc)
                    per = rows_per_pass([x["ids"] for x in rows], enc["n_state"])
                    return u["state_tokens"] + u["branch_tokens"], 1 + math.ceil(len(rows) / per)
                prof = run_profiles(dev, f"converted_{form}", lambda r, form=form: model.evaluate(r.state, qdump(r), form=form),
                                    lambda rs, form=form: model.evaluate_requests(rs, form=form), cost, model.tok, a)
                s["performance"][f"converted_{form}"] = {"cold_load_s": load_s, "resident_bytes": resident - (base_mem or {}).get("allocated", 0) if resident is not None else None,
                                                        **cold, "profiles": prof}
        del model
        free(dev)

    # parent one-token scoring, raw and cyclic (one loaded scorer)
    modes = [m for m in ("raw", "cyclic") if f"parent_{m}" in systems]
    if modes:
        from .targets import ParentScorer
        b0 = device_mem()
        t0 = time.perf_counter()
        scorer = ParentScorer(parent, revision, device=dev, dtype=dtype, attn=a.attn)
        sync(dev)
        load_s = time.perf_counter() - t0
        resident = (device_mem() or {}).get("allocated")
        cold = cold_request(dev, lambda r: scorer.score(r, modes[0]), scorer_tok(scorer, parent, revision))
        cold["cold_request_mode"] = f"{modes[0]} (one cold request, shared by the parent scoring rows)"
        for p, recs in data.items():
            add(scorer_rows(scorer, p, recs, modes, a.order_variants if p == "showcase" else 0, a.seed))
            print(f"[parent {'/'.join(modes)}] {p}: {len(recs)} records", flush=True)
        if a.profiles:
            for mode in modes:
                def cost(out, r):
                    return sum(v.get("prompt_tokens") or 0 for v in out.values()), sum(v.get("layouts") or 0 for v in out.values())
                prof = run_profiles(dev, f"parent_{mode}", lambda r, mode=mode: scorer.score(r, mode),
                                    lambda rs, mode=mode: scorer.score_many(rs, mode), cost, scorer_tok(scorer, parent, revision), a)
                s["performance"][f"parent_{mode}"] = {"cold_load_s": load_s, "resident_bytes": resident - (b0 or {}).get("allocated", 0) if resident is not None else None,
                                                      **cold, "profiles": prof}
        del scorer
        free(dev)

    # parent structured JSON generation (showcase quality + profiles), reasoning off and (opt-in) on
    for jsys, thinking in (("parent_json", False), ("parent_json_thinking", True)):
        if jsys not in systems:
            continue
        b0 = device_mem()
        t0 = time.perf_counter()
        gen = ParentGenerator(parent, revision, device=dev, dtype=dtype, attn=a.attn, thinking=thinking,
                              max_new_tokens=a.thinking_max_new_tokens if thinking else None)
        sync(dev)
        load_s = time.perf_counter() - t0
        resident = (device_mem() or {}).get("allocated")
        cold = cold_request(dev, gen.answer, gen.tok)
        for p in [x for x in data if x == "showcase" or x in panel_files]:
            add(json_rows(gen, p, data[p], jsys, a.gen_batch if thinking else 1))
            print(f"[{jsys}] {p}: {len(data[p])} records", flush=True)
        if a.profiles:
            parse, gen_toks, trunc = [], [], []

            def cost(out, r):
                parse.append(out["parse_ok"] and all(v is not None for v in out["answers"].values()))
                gen_toks.append(out["generated_tokens"])
                trunc.append(bool(out.get("truncated")))
                return out["prompt_tokens"] + out["generated_tokens"], out["generated_tokens"]
            prof = run_profiles(dev, jsys, gen.answer, gen.answer_batch, cost, gen.tok, a)
            s["performance"][jsys] = {"cold_load_s": load_s, "resident_bytes": resident - (b0 or {}).get("allocated", 0) if resident is not None else None,
                                      **cold, "profiles": prof, "timed_requests_fully_parsed": sum(parse) / len(parse) if parse else None,
                                      "timed_generated_tokens_mean": statistics.fmean(gen_toks) if gen_toks else None,
                                      "timed_truncated": sum(trunc), "max_new_tokens": gen.max_new_tokens,
                                      "thinking": thinking}
        del gen
        free(dev)

    s["conditions"]["gpu_activity_at_end"] = gpu_activity()
    s["generation"] = generation_stats(all_rows)
    for p in panels:
        s["panels"][p] = panel_summary([r for r in all_rows if r["panel"] == p])
    s["artifact"] = artifact_sizes(a.model)
    s["parent_snapshot"] = parent_snapshot(parent, revision)
    if not a.profiles:
        s["not_measured"].append("performance profiles (run with --profiles)")
    s["not_measured"] += [f"system {x}" for x in SYSTEMS if x not in systems]

    with open(out_dir / "rows.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    bench = {"command": s["command"], "conditions": s["conditions"], "performance": s["performance"], "acceptance": s.get("acceptance"),
             "artifact": s["artifact"], "parent_snapshot": s["parent_snapshot"], "profiles": {k: {"state_tokens": v[0], "questions": v[1]} for k, v in PROFILES.items()},
             "workload": {"questions": QUESTIONS, "sentences": SENTENCES}}
    (out_dir / "benchmark.json").write_text(json.dumps(bench, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    md = summary_md(s)
    (out_dir / "summary.md").write_text(md)
    print(md)
    if a.limit is None:
        (Path(a.model) / "evaluation.json").write_text(json.dumps(s, indent=2) + "\n")
        readme = Path(a.model) / "README.md"
        if readme.exists():
            text = readme.read_text()
            i, j = text.find(MEASURED_START), text.find(MEASURED_END)
            if i != -1 and j != -1:
                body = md.split("\n", 1)[1]
                readme.write_text(text[:i + len(MEASURED_START)] + "\n" + body + text[j:])
    else:
        print(f"--limit {a.limit}: smoke run; {a.model}/evaluation.json and README.md left unchanged", flush=True)


def scorer_tok(scorer, parent, revision):
    tok = getattr(scorer, "tok", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(parent, revision=revision)
    return tok


if __name__ == "__main__":
    main()
