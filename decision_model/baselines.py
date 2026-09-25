"""Parent baselines for the benchmark.

ParentGenerator is the structured-answer baseline: the parent (with its causal-LM head) reads one chat prompt holding the
same rendered state and every question with its type, instructions and options, and generates a JSON object mapping
question id to answer (thinking disabled, greedy). The whole response is timed. Unparseable or missing answers are
returned as None; the evaluator counts them as wrong and reports them.

The optimized one-token parent rows (raw and cyclic restricted-answer scoring) are decision_model.targets.ParentScorer,
used as-is.
"""
import copy
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .render import render_content
from .schema import DecisionRequest, question_keys

TOKENS_PER_QUESTION, TOKENS_BASE = 24, 16   # max_new_tokens = 24 per question + 16


def sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def _describe(qid: str, q) -> str:
    lines = [f"Question {json.dumps(qid)} ({q.type})"]
    instr = render_content(q.instructions)
    if instr:
        lines.append("  instructions: " + instr.replace("\n", "\n    "))
    if q.type == "choice":
        lines.append("  answer: exactly one option id from this list, as a JSON string")
        for k, v in q.criteria.items():
            d = render_content(v)
            lines.append(f"    {json.dumps(k)}" + (": " + d.replace(chr(10), " ") if d else ""))
    elif q.type == "noul":
        lines.append("  answer: true or false, as a JSON boolean")
        for k in ("true", "false"):
            d = render_content((q.criteria or {}).get(k))
            if d:
                lines.append(f"    {k}: " + d.replace("\n", " "))
    else:
        lines.append(f"  answer: one level index from 0 to {len(q.criteria) - 1}, as a JSON integer")
        for i, v in enumerate(q.criteria):
            lines.append(f"    {i}: " + render_content(v).replace("\n", " "))
    return "\n".join(lines)


def prompt_text(request: DecisionRequest) -> str:
    """The user message: rendered state, every question, and the JSON-only answer instruction."""
    qs = "\n".join(_describe(qid, q) for qid, q in request.questions.items())
    return ("Read the state and answer every question.\n\n"
            f"State:\n{render_content(request.state)}\n\n"
            f"Questions:\n{qs}\n\n"
            "Respond with only a JSON object whose keys are the question ids ("
            + ", ".join(json.dumps(k) for k in request.questions) + ") and whose values are the answers. No other text.")


def _first_json_object(text: str):
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_answers(text: str, request: DecisionRequest) -> tuple[dict, bool]:
    """({qid: stable key or None}, parse_ok). A key is the option id (choice), "true"/"false" (noul) or the level index
    as a string (score), i.e. the same keys the decision model reports. parse_ok = a JSON object was found."""
    obj = _first_json_object(text)
    out = {}
    for qid, q in request.questions.items():
        v = None if obj is None else obj.get(qid)
        keys = question_keys(q.type, q.criteria)
        if q.type == "noul":
            if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                v = v.strip().lower() == "true"
            v = ("true" if v else "false") if isinstance(v, bool) else None
        elif q.type == "score":
            if isinstance(v, str) and v.strip().isdigit():
                v = int(v.strip())
            v = str(v) if isinstance(v, int) and not isinstance(v, bool) else None
        else:
            # larger parents echo the prompt's JSON-quoted option ids ("\"world\""): strip surrounding quotes
            v = v.strip().strip("\"'") if isinstance(v, str) else None
        out[qid] = v if v in keys else None
    return out, obj is not None


class ParentGenerator:
    """The parent generating structured JSON answers. `attn` should match the other benchmarked systems.
    thinking=True turns Qwen3's reasoning on (enable_thinking=True): the answer is parsed only from the text after
    `</think>`, and max_new_tokens (default 2048) covers the reasoning plus the JSON; a response that reaches the limit
    without finishing is reported as truncated."""

    def __init__(self, parent, revision, *, device, dtype=torch.bfloat16, attn=None, thinking=False, max_new_tokens=None):
        self.device = torch.device(device)
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens or (2048 if thinking else None)
        self.tok = AutoTokenizer.from_pretrained(parent, revision=revision)
        self.tok.padding_side = "left"
        self.lm = AutoModelForCausalLM.from_pretrained(parent, revision=revision, dtype=dtype, attn_implementation=attn).to(self.device).eval()
        self.gen = GenerationConfig(do_sample=False, eos_token_id=self.lm.generation_config.eos_token_id,
                                    pad_token_id=self.tok.pad_token_id, bos_token_id=self.lm.generation_config.bos_token_id)
        eos = self.gen.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def chat(self, request: DecisionRequest) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": prompt_text(request)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=self.thinking)

    @torch.no_grad()
    def answer_batch(self, requests: list[DecisionRequest]) -> list[dict]:
        """Greedy generation for a left-padded batch of requests. latency_ms is the wall time of the whole call."""
        sync(self.device)
        t0 = time.perf_counter()
        enc = self.tok([self.chat(r) for r in requests], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        n_new = self.max_new_tokens or TOKENS_PER_QUESTION * max(len(r.questions) for r in requests) + TOKENS_BASE
        gen = copy.copy(self.gen)
        gen.max_new_tokens = n_new
        out = self.lm.generate(**enc, generation_config=gen)
        new = out[:, enc.input_ids.shape[1]:].tolist()
        results = []
        for r, ids, mask in zip(requests, new, enc.attention_mask.sum(1).tolist()):
            n = next((i + 1 for i, t in enumerate(ids) if t in self.eos), None)
            truncated = n is None
            n = len(ids) if truncated else n
            text = self.tok.decode(ids[:n], skip_special_tokens=True)
            answer_text = text.split("</think>")[-1] if not self.thinking else (text.split("</think>", 1)[1] if "</think>" in text else "")
            answers, ok = parse_answers(answer_text, r)
            results.append({"answers": answers, "text": text, "generated_tokens": n, "prompt_tokens": int(mask), "parse_ok": ok,
                            "truncated": truncated})
        sync(self.device)
        ms = (time.perf_counter() - t0) * 1e3
        for x in results:
            x["latency_ms"] = ms
        return results

    def answer(self, request: DecisionRequest) -> dict:
        """{"answers": {qid: key or None}, "text", "generated_tokens", "prompt_tokens", "latency_ms", "parse_ok"}; the
        whole response (templating, tokenizing, generation, decoding, parsing) is timed with device sync."""
        return self.answer_batch([request])[0]
