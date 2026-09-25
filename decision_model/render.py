# Modified implementation for Kev; see LICENSE and NOTICE.
"""The one renderer and token encoder (lead-owned). Training, evaluation, serving and the teacher collector all pass
requests through to_record() and encode(); nothing else decides how a state, question or option becomes tokens.

Layout of one encoded request:

    [<state> state tokens...]  [<q> instr <opt> o1 </opt> <opt> o2 </opt> ... <decide>]  [<q> ... <decide>] ...
     seg 0                       seg 1                                                    seg 2
     pos 0..S-1                  pos S..                                                  pos S..   (positions restart after the state)

The five delimiters reuse rarely-used Qwen special tokens, so no embedding rows are added or trained; LoRA adapts their
meaning. User text is escaped so it can never produce a delimiter/control token (option boundaries are unforgeable).
The renderer is a deterministic serialization of the supplied state and schema: no date facts or other application
feature engineering.

encode() never truncates: a request over the limits raises AdmissionError.
"""
import re

from .schema import (MAX_ROW_TOKENS, MAX_STATE_TOKENS, AdmissionError, DecisionRequest, JSONContent, question_keys)

# state, question, option, end-of-option, decide
SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
RENDERER_VERSION = 1

OPT_NONE, OPT_DECIDE = -1, -2   # enc["opt"] values for state/instruction tokens and for the <decide> token


def render_content(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees. Field names are kept as labels. Deterministic."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render_content(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render_content(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render_content(x)}"
                     for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render_content(desc)}"


def to_record(req: DecisionRequest):
    """DecisionRequest -> (internal record, per-question meta).

    record: {"state": str, "questions": [{"instr": str, "options": [str, ...]}, ...]}   (canonical option order)
    meta:   [{"id": qid, "type": "noul"|"choice"|"score", "keys": [...], "legend": {...} (score only)}, ...]
    noul renders as two options [no, yes] whose keys are ["false", "true"]; choice options render as "id" or "id: desc";
    score options are the ordered level descriptions and keys are "0".."K-1"."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            opts = [render_content(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render_content(q.instructions), "options": opts})
        meta.append(m)
    return {"state": render_content(req.state), "questions": qs}, meta


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text: str) -> list[int]:
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens: `<|name|>` becomes `<¦name¦>`
    before tokenizing (fast tokenizers ignore split_special_tokens)."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def delimiter_ids(tok) -> dict[str, int]:
    ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL]
    if any(i is None or i == tok.unk_token_id for i in ids):
        raise ValueError(f"tokenizer lacks the delimiter tokens {SPECIAL}")
    return dict(zip(("state", "question", "option", "end_option", "decide"), ids))


def encode(tok, rec: dict, max_state: int = MAX_STATE_TOKENS, max_row: int = MAX_ROW_TOKENS) -> dict:
    """Pack one internal record into the layout above.

    Returns {"ids", "seg", "pos", "opt", "n_state", "decide_idx" [Q], "opt_idx" [Q][K]}:
      seg[i]       0 for state tokens, k for question k (1-based)
      pos[i]       position ids; every branch restarts at n_state
      opt[i]       OPT_NONE for state/instruction tokens, option index within its question for option-span tokens
                   (including <opt> and </opt>), OPT_DECIDE for <decide>
      decide_idx   index of each question's <decide> token in ids
      opt_idx      index of each option's </opt> token (the option representation the head reads)
    Raises AdmissionError when the state exceeds max_state tokens or any state+branch row exceeds max_row tokens."""
    d = delimiter_ids(tok)
    state_tokens = user_tokens(tok, rec["state"])
    if len(state_tokens) + 1 > max_state:
        raise AdmissionError(f"state is {len(state_tokens) + 1} tokens; the limit is {max_state}")
    S = [d["state"]] + state_tokens
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [d["question"]] + user_tokens(tok, q["instr"])
        spans = [[d["option"]] + user_tokens(tok, o) + [d["end_option"]] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d["decide"]]
        if len(S) + len(br) > max_row:
            raise AdmissionError(f"question {k} row is {len(S) + len(br)} tokens (state {len(S)} + branch {len(br)}); the limit is {max_row}")
        base, p0 = len(ids), len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += br
        seg += [k] * len(br)
        pos += list(range(p0, p0 + len(br)))
        opt += br_opt
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "n_state": len(S), "decide_idx": decide_idx, "opt_idx": opt_idx}


def rows_of(enc: dict):
    """Split a packed encoding into (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide", "opts"}: the
    branch tokens of question k with their state-continuing positions and readout offsets *within the branch*. Feeding
    state + rows[k] as one causal row computes exactly what question k sees under the packed block-causal mask."""
    seg, Ls = enc["seg"], enc["n_state"]
    rows, start = [], Ls
    for k, (dix, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = dix + 1
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": dix - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


def encode_request(tok, req: DecisionRequest, **limits):
    """Validated request -> (enc, meta). The single entry point serving/evaluation use."""
    rec, meta = to_record(req)
    return encode(tok, rec, **limits), meta
