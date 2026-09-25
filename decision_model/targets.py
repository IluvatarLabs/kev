# Modified teacher-target implementation; licensed under Apache-2.0 (see LICENSE).
"""Teacher targets from the parent LLM (SPEC section 5).

ParentScorer asks the parent each question as an label-based multiple-choice prompt (thinking disabled through the chat
template) and reads the logits of the answer-label tokens at the first answer position. Probabilities are the softmax
over those label logits only; `mass_on_labels` records their share of the full-vocabulary softmax.

    raw     one canonical layout: the canonical option order (choice, score), "Yes or No" (noul)
    cyclic  every cyclic rotation of the options (choice), both answer orders (noul), the one ordered layout (score),
            each mapped back to stable option keys and combined by q(i) = normalize(exp(mean_s log max(p_s(i), eps)))

Declared caps: choice <= 26 letter labels A..Z, score <= 10 levels labelled "0".."K-1", noul labels Yes/No; each label
must be one token (bare, else leading space) with no collisions, and must survive the answer boundary of the rendered
prompt (encode(prompt + label) == encode(prompt) + [id]). Anything else is recorded ineligible with a reason, never
shortlisted. No prior correction.

    python -m decision_model.targets --config recipes/qwen3-1.7b.yaml      # collect for every train-split question
"""
import argparse
import hashlib
import json
import string
import time
from pathlib import Path

import numpy as np
import torch

from .render import RENDERER_VERSION, _SPECIAL_RE, render_content, to_record
from .schema import DecisionRequest, question_keys

PROMPT_FORMAT = "label-chat-v1"
SYSTEM = ("You are a decision function. You will be given a state and one question. "
          "Reply with the answer label only: no words, no punctuation, no explanation.")
LETTERS = string.ascii_uppercase
MAX_CHOICE, MAX_SCORE = 26, 10
EPS = 1e-9


class LabelTokenError(ValueError):
    pass


def map_label_tokens(tok, labels):
    """Map each label -> exactly one token id (bare string, then a leading-space variant); refuses
    multi-token labels and collisions. Returns (ids, surface strings used)."""
    ids, used = [], []
    for lab in labels:
        found = None
        for cand in (lab, " " + lab):
            toks = tok.encode(cand, add_special_tokens=False)
            if len(toks) == 1:
                found = (toks[0], cand)
                break
        if found is None:
            raise LabelTokenError(f"label {lab!r} is not a single token for this tokenizer: {tok.encode(lab, add_special_tokens=False)}")
        ids.append(found[0])
        used.append(found[1])
    if len(set(ids)) != len(ids):
        raise LabelTokenError(f"label tokens collide: {dict(zip(labels, ids))}")
    return ids, used


def answer_labels(qtype, k):
    """Label strings in teacher-option order. choice/score labels name a position; noul labels name the option
    (teacher option 0 = Yes = key "true", option 1 = No = key "false") and follow it when the phrasing order changes."""
    if qtype == "choice":
        return list(LETTERS[:k])
    if qtype == "score":
        return [str(i) for i in range(k)]
    return ["Yes", "No"]


def ineligible_reason(qtype, k):
    if qtype == "choice" and k > MAX_CHOICE:
        return f"choice has {k} options; the letter-label teacher supports at most {MAX_CHOICE}"
    if qtype == "score" and k > MAX_SCORE:
        return f"score has {k} levels; the digit-label teacher supports at most {MAX_SCORE}"
    return None


def layouts(qtype, k, mode):
    """Teacher layouts as perms over teacher options: perm[j] = teacher option shown at position j under cyclic
    shifts. The first layout is always the canonical one."""
    if mode not in ("raw", "cyclic"):
        raise ValueError("mode must be 'raw' or 'cyclic'")
    if qtype == "noul":
        return [[0, 1]] if mode == "raw" else [[0, 1], [1, 0]]
    if qtype == "score" or mode == "raw":
        return [list(range(k))]
    return [[(j + s) % k for j in range(k)] for s in range(k)]


def teacher_keys(qtype, criteria):
    """Stable key of each teacher option (same order as answer_labels)."""
    return ["true", "false"] if qtype == "noul" else question_keys(qtype, criteria)


def _escape(text):
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


def build_user(state_text, question, rec_q, perm, labels):
    """Build the prompt over the renderer's text: the state is render.render_content(state), choice options are the
    renderer's option strings ("id" or "id: description"), score levels its level descriptions."""
    lines = ["State:", state_text if state_text else "(empty)", "", f"Question: {rec_q['instr']}"]
    if question.type == "noul":
        c = question.criteria or {}
        if c.get("true") not in (None, ""):
            lines.append(f"Yes means: {render_content(c['true'])}")
        if c.get("false") not in (None, ""):
            lines.append(f"No means: {render_content(c['false'])}")
        order = [labels[i] for i in perm]
        lines.append(f"Answer {order[0]} or {order[1]}.")
    elif question.type == "score":
        lines.append("Pick the level that applies (the levels are ordered):")
        lines += [f"{labels[j]}. {rec_q['options'][i]}" for j, i in enumerate(perm)]
        lines.append("Answer with the number only.")
    else:
        lines.append("Options:")
        lines += [f"{labels[j]}. {rec_q['options'][i]}" for j, i in enumerate(perm)]
        lines.append("Answer with the letter only.")
    return _escape("\n".join(lines))


def render_chat(tok, user):
    """Render system + user through the chat template with the generation prompt, thinking disabled, so
    the next token is the model's first answer token."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def cyclic_combine(p_by_layout, eps=EPS):
    """[S, K] per-layout distributions already mapped to key order -> normalize(exp(mean_s log max(p, eps)))."""
    z = np.log(np.maximum(np.asarray(p_by_layout, dtype=np.float64), eps)).mean(axis=0)
    q = np.exp(z - z.max())
    return q / q.sum()


def _restricted(lp):
    """Full-vocabulary label log-probs -> (softmax over the label logits, mass_on_labels)."""
    lp = np.asarray(lp, dtype=np.float64)
    p = np.exp(lp - lp.max())
    return p / p.sum(), float(np.exp(lp).sum())


class ParentScorer:
    """Restricted-token parent scoring. `forward(prompts, label_ids)` returns, per prompt, the full-vocabulary log-probs
    of its label tokens at the answer position; it is the one model-dependent piece (tests replace it)."""

    def __init__(self, parent, revision, *, device, dtype=torch.bfloat16, attn=None, batch_tokens=65536, load_model=True):
        from transformers import AutoTokenizer

        self.parent, self.revision, self.device, self.dtype = parent, revision, device, dtype
        self.batch_tokens = batch_tokens
        self.tok = AutoTokenizer.from_pretrained(parent, revision=revision)
        self.tok.padding_side = "left"
        self.tokenizer_sha256 = hashlib.sha256(self.tok.backend_tokenizer.to_str().encode()).hexdigest()
        self._labels = {}
        self.model = None
        self.forward = self._hf_forward
        if load_model:
            from transformers import AutoModelForCausalLM
            kw = {"attn_implementation": attn} if attn else {}
            self.model = AutoModelForCausalLM.from_pretrained(parent, revision=revision, dtype=dtype, **kw).to(device).eval()

    # --- labels ----------------------------------------------------------------------------------------------------
    def resolve(self, qtype, k, sample_prompt):
        """(labels, ids) for this question kind/size, verified once: single tokens, no collisions, and each label is
        exactly one token appended at the rendered prompt's answer boundary."""
        key = (qtype, k)
        if key not in self._labels:
            labels = answer_labels(qtype, k)
            ids, used = map_label_tokens(self.tok, labels)
            base = self.tok.encode(sample_prompt, add_special_tokens=False)
            for lab, i in zip(used, ids):
                if self.tok.encode(sample_prompt + lab, add_special_tokens=False) != base + [i]:
                    raise LabelTokenError(f"label {lab!r} does not tokenize as token {i} at the answer boundary")
            self._labels[key] = (labels, ids)
        return self._labels[key]

    # --- model -----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _hf_forward(self, prompts, label_ids):
        """Next-token log probabilities: length-sorted left-padded batches under a padded-token budget, explicit
        position ids, logits at the last position only."""
        enc = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(enc[i]))
        out = [None] * len(prompts)
        pad = self.tok.pad_token_id
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end + 1 - start) * len(enc[order[end]]) <= self.batch_tokens:
                end += 1
            idx = order[start:end]
            L = len(enc[idx[-1]])
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, L - len(enc[i]):] = torch.tensor(enc[i])
                mask[r, L - len(enc[i]):] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            for r, i in enumerate(idx):
                out[i] = lp[r, torch.tensor(label_ids[i], device=lp.device)].double().cpu().numpy()
            start = end
        return out

    # --- scoring ---------------------------------------------------------------------------------------------------
    def plan(self, request: DecisionRequest, mode):
        """Per question: (qid, meta dict, list of (perm, prompt, label ids in position order)) or an ineligible entry."""
        rec, meta = to_record(request)
        items = []
        for (qid, q), rq, m in zip(request.questions.items(), rec["questions"], meta):
            k = len(m["keys"])
            reason = ineligible_reason(q.type, k)
            entry = {"qid": qid, "type": q.type, "keys": m["keys"], "tkeys": teacher_keys(q.type, q.criteria), "reason": reason, "layouts": []}
            if reason is None:
                labels0 = answer_labels(q.type, k)
                perms = layouts(q.type, k, mode)
                prompts = [render_chat(self.tok, build_user(rec["state"], q, rq, perm, labels0)) for perm in perms]
                try:
                    labels, ids = self.resolve(q.type, k, prompts[0])
                except LabelTokenError as e:
                    entry["reason"] = str(e)
                else:
                    for perm, prompt in zip(perms, prompts):
                        # choice/score labels name positions; noul labels follow their option
                        pos_labels = [labels[i] for i in perm] if q.type == "noul" else labels
                        pos_ids = [ids[i] for i in perm] if q.type == "noul" else ids
                        entry["layouts"].append({"perm_t": perm, "prompt": prompt, "labels": pos_labels, "ids": pos_ids})
            items.append(entry)
        return items

    def score_detail(self, requests, mode):
        """Full provenance per request and question: layouts (perm over canonical keys, labels, label token ids, label
        log-probs by position, mass_on_labels), raw (layout 0) and combined probabilities keyed by stable id."""
        t0 = time.perf_counter()
        plans = [self.plan(r, mode) for r in requests]
        flat = [(ri, qi, li) for ri, items in enumerate(plans) for qi, e in enumerate(items) for li in range(len(e["layouts"]))]
        prompts = [plans[ri][qi]["layouts"][li]["prompt"] for ri, qi, li in flat]
        lps = self.forward(prompts, [plans[ri][qi]["layouts"][li]["ids"] for ri, qi, li in flat]) if flat else []
        ntok = {}
        for (ri, qi, li), p, lp in zip(flat, prompts, lps):
            plans[ri][qi]["layouts"][li]["lp"] = lp
            ntok[(ri, qi)] = ntok.get((ri, qi), 0) + len(self.tok.encode(p, add_special_tokens=False))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        results = []
        for ri, items in enumerate(plans):
            out = {}
            for qi, e in enumerate(items):
                keys, tkeys = e["keys"], e["tkeys"]
                res = {"eligible": e["reason"] is None, "reason": e["reason"], "keys": keys, "layouts": [], "raw": None, "cyclic": None,
                       "probs": None, "prompt_tokens": ntok.get((ri, qi), 0), "latency_ms": latency_ms}
                if e["reason"] is None:
                    mapped = []
                    for lay in e["layouts"]:
                        p_pos, mass = _restricted(lay["lp"])
                        by_key = {tkeys[i]: float(p_pos[j]) for j, i in enumerate(lay["perm_t"])}
                        mapped.append([by_key[k] for k in keys])
                        res["layouts"].append({"perm": [keys.index(tkeys[i]) for i in lay["perm_t"]], "labels": lay["labels"],
                                               "label_token_ids": lay["ids"], "logprobs_by_position": [float(x) for x in lay["lp"]],
                                               "mass_on_labels": mass})
                    res["raw"] = dict(zip(keys, mapped[0]))
                    if mode == "cyclic":
                        res["cyclic"] = dict(zip(keys, (float(x) for x in cyclic_combine(mapped))))
                    res["probs"] = res["cyclic"] if mode == "cyclic" else res["raw"]
                res["layouts_count"] = len(res["layouts"])
                out[e["qid"]] = res
            results.append(out)
        return results

    def score_many(self, requests, mode):
        """Batched scoring. Per qid: {"probs": {key: p} | None, "eligible", "reason", "layouts" (count), "prompt_tokens",
        "latency_ms" (wall time of this whole call), "mass_on_labels" (mean over layouts), "raw"}."""
        out = []
        for detail in self.score_detail(requests, mode):
            out.append({qid: {"probs": d["probs"], "eligible": d["eligible"], "reason": d["reason"], "layouts": d["layouts_count"],
                              "prompt_tokens": d["prompt_tokens"], "latency_ms": d["latency_ms"], "raw": d["raw"],
                              "mass_on_labels": float(np.mean([l["mass_on_labels"] for l in d["layouts"]])) if d["layouts"] else None}
                        for qid, d in detail.items()})
        return out

    def score(self, request, mode):
        return self.score_many([request], mode)[0]

    def teacher_info(self):
        return {"parent": self.parent, "revision": self.revision, "prompt_format": PROMPT_FORMAT, "tokenizer_sha256": self.tokenizer_sha256,
                "renderer_version": RENDERER_VERSION, "thinking": False, "dtype": str(self.dtype).replace("torch.", ""), "prior_correction": None}


# --- collection ----------------------------------------------------------------------------------------------------------

def target_path(out_root, revision):
    return Path(out_root) / revision[:12] / "targets.jsonl"


def load_targets(path):
    """{(record_id, qid): target record}"""
    out = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                out[(t["record_id"], t["qid"])] = t
    return out


def collection_hash(cfg, data_dir):
    """Everything that determines the collected targets: the train split, the parent, and the teacher settings."""
    t = cfg["teacher"]
    train_sha = json.loads((Path(data_dir) / "manifest.json").read_text())["files"]["train.jsonl"]["sha256"]
    key = {"train_sha256": train_sha, "parent": cfg["parent"], "revision": cfg["revision"], "dtype": t["dtype"],
           "batch_tokens": t["batch_tokens"], "prompt_format": PROMPT_FORMAT, "epsilon": EPS}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def collect(cfg, data_dir=None, out_root=None, device="cuda", limit=None, chunk=256):
    """Teacher targets for every train-split question, appended to targets.jsonl and resumable by (record_id, qid)."""
    from .data import load_split

    t_cfg = cfg["teacher"]
    data_dir = Path(data_dir or cfg["data"]["out"])
    path = target_path(out_root or t_cfg["out"], cfg["revision"])
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.parent / "meta.json"
    records = load_split(data_dir, "train")
    if limit:
        records = records[:limit]
    done = {k for k in load_targets(path)}
    todo = [r for r in records if any((r["record_id"], q) not in done for q in r["request"]["questions"])]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"wall_seconds": 0.0, "sessions": []}
    input_sha = collection_hash(cfg, data_dir)
    if meta.get("input_sha256") not in (None, input_sha):
        raise ValueError(f"{path} was collected for different inputs; move it aside to recollect")
    print(f"targets: {len(records)} train records, {len(records) - len(todo)} already complete, {len(todo)} to score", flush=True)
    t0 = time.time()
    scorer = ParentScorer(cfg["parent"], cfg["revision"], device=device, dtype=getattr(torch, t_cfg["dtype"]), batch_tokens=t_cfg["batch_tokens"])
    load_s = time.time() - t0
    t1 = time.time()
    session = {"records": 0, "questions": 0, "layouts": 0, "prompt_tokens": 0}
    with path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), chunk):
            part = todo[start:start + chunk]
            details = scorer.score_detail([DecisionRequest.model_validate(r["request"]) for r in part], "cyclic")
            for r, det in zip(part, details):
                for qid, d in det.items():
                    if (r["record_id"], qid) in done:
                        continue
                    f.write(json.dumps({"record_id": r["record_id"], "qid": qid, "teacher": scorer.teacher_info(), "keys": d["keys"],
                                        "eligible": d["eligible"], "reason": d["reason"], "layouts": d["layouts"], "raw": d["raw"], "cyclic": d["cyclic"],
                                        "cost": {"layouts": d["layouts_count"], "prompt_tokens": d["prompt_tokens"]}}, ensure_ascii=False) + "\n")
                    session["questions"] += 1
                    session["layouts"] += d["layouts_count"]
                    session["prompt_tokens"] += d["prompt_tokens"]
                session["records"] += 1
            f.flush()
            el = time.time() - t1
            print(f"targets: {start + len(part)}/{len(todo)} records, {session['layouts']} layouts, {session['prompt_tokens']} tokens, "
                  f"{el:.0f}s ({session['prompt_tokens'] / max(el, 1e-9):.0f} tok/s)", flush=True)
    session.update(wall_seconds=time.time() - t1, model_load_seconds=load_s,
                   device=torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else device,
                   peak_device_bytes=torch.cuda.max_memory_allocated() if device == "cuda" and torch.cuda.is_available() else None)
    meta["sessions"].append(session)
    meta["wall_seconds"] += session["wall_seconds"]
    meta.update(input_sha256=input_sha, teacher=scorer.teacher_info(), batch_tokens=t_cfg["batch_tokens"], epsilon=EPS)
    all_t = load_targets(path)
    want = {(r["record_id"], q) for r in records for q in r["request"]["questions"]}
    meta["complete"] = want <= set(all_t) and limit is None
    meta["totals"] = summarize(all_t.values())
    meta["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta


def summarize(targets):
    by_source, layouts, tokens, reasons = {}, 0, 0, {}
    for t in targets:
        src = t["record_id"].split("/")[0]
        s = by_source.setdefault(src, {"eligible": 0, "ineligible": 0})
        s["eligible" if t["eligible"] else "ineligible"] += 1
        layouts += t["cost"]["layouts"]
        tokens += t["cost"]["prompt_tokens"]
        if not t["eligible"]:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return {"questions": sum(s["eligible"] + s["ineligible"] for s in by_source.values()), "by_source": by_source,
            "layouts": layouts, "prompt_tokens": tokens, "ineligible_reasons": reasons}


def main():
    from .data import load_recipe

    ap = argparse.ArgumentParser(description="collect parent teacher targets for the train split")
    ap.add_argument("--config", default="recipes/qwen3-1.7b.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N train records only (throughput check)")
    a = ap.parse_args()
    path, meta = collect(load_recipe(a.config), a.data, a.out, a.device, a.limit)
    print(json.dumps({"path": str(path), "wall_seconds": meta["wall_seconds"], "complete": meta["complete"], **meta["totals"]}, indent=2))


if __name__ == "__main__":
    main()
