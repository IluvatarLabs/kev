# Modified implementation for Kev; see LICENSE and NOTICE.
"""Decision model: Qwen3 backbone + one shared pointer head over runtime options.

Execution forms, all computing the same function of one encoded request (render.encode):
  packed  one pass over state + every branch under the block-causal mask (training reference; batched throughput path)
  rows    one causal row per question = state + its branch (the state is recomputed per question)
  cached  one state pass (prefix), then the branches as batched causal rows continuing a per-chunk replica of its KV
Option i of a question scores z_i = key(h_end_i) . query(h_decide) / sqrt(dim) on the backbone's final-normed hidden
states. No generation, no vocabulary projection. At inference (probs, evaluate, evaluate_requests) a record whose packed
length exceeds PACKED_TOKEN_LIMIT runs in the cached form instead of packed (its L x L mask would not fit), reported as "cached".

Dtype contract: never call nn.Module.to(dtype) on the backbone anywhere in decision_model (it would also cast the rotary
inv_freq buffer that from_pretrained keeps in fp32). decision_model.backbones.qwen3.cast_parameters(module, dtype) is the
only cast, and merge_lora uses it. save_pretrained(out_dir, dtype) leaves the live instance in that dtype (parameters
cast, buffers untouched), so a caller can compare the in-memory model with the reloaded artifact under identical numerics.
"""
import json
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, DynamicCache

from .backbones import qwen3
from .render import RENDERER_VERSION, delimiter_ids, encode, encode_request, rows_of, to_record
from .schema import MAX_OPTIONS, MAX_QUESTIONS, MAX_ROW_TOKENS, MAX_STATE_TOKENS, DecisionRequest, to_answers

BRANCH_TOKEN_BUDGET = 16384   # tokens per pass, counting each row's cached prefix: (prefix_len + longest row) * rows; read at call time
PACKED_TOKEN_LIMIT = 8192     # inference: longer packed records run in the cached form (Kev's SERVE_MAX_PACKED); read at call time
FORMAT_VERSION = 1


class PointerHead(nn.Module):
    def __init__(self, hidden: int, dim: int = 256):
        super().__init__()
        self.q, self.k = nn.Linear(hidden, dim), nn.Linear(hidden, dim)
        self.scale = dim ** -0.5
        # calibration scalar: DecisionModel.probs (inference) divides the logits by it; forward() and the training path
        # never do, so training always sees T=1 and a fitted value stays meaningful. 1.0 = raw.
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):   # [d], [K, d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class Prefix(NamedTuple):
    n_state: int
    cache: DynamicCache
    hidden: torch.Tensor   # [n_state, d] fp32


def branch_mask_batch(segs, length, device, dtype):
    """attend(i, j) iff j <= i and (seg[j] == 0 or seg[j] == seg[i]); additive [B, 1, L, L], right-padded to `length`.
    Pad keys are masked for every query; pad query rows keep their diagonal so no row is fully masked (finfo.min, not
    -inf, keeps softmax finite). Real tokens never see pads: pads sit after them and belong to no segment (-1)."""
    s = torch.full((len(segs), length), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    allow = causal[None] & ((s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)) & (s != -1)[:, None, :]
    allow = allow | torch.eye(length, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), length, length, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_per_pass(rows, prefix_len=0):
    """Causal rows per forward pass: as many as fit BRANCH_TOKEN_BUDGET counting the prefix each row carries, at least one.
    Rows are independent, so the split never changes an answer; it bounds memory (each chunk replicates the prefix)."""
    return max(1, BRANCH_TOKEN_BUDGET // (prefix_len + max(len(r) for r in rows)))


class DecisionModel(nn.Module):
    def __init__(self, tok, lm, head, device, *, parent: dict, lora: dict | None, limits: dict | None = None):
        super().__init__()
        self.tok, self.lm, self.head, self.device = tok, lm, head, torch.device(device)
        self.parent, self.lora = parent, lora
        self.limits = limits or {"max_state": MAX_STATE_TOKENS, "max_row": MAX_ROW_TOKENS}
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.to(self.device)
        self.eval()

    # ---- construction and persistence

    @classmethod
    def from_parent(cls, parent, revision, *, device, dtype=torch.float32, lora: dict | None = None, head_dim=256, attn=None) -> "DecisionModel":
        """The parent's Qwen3Model (frozen) + optional PEFT LoRA + a fresh pointer head (seed with torch.manual_seed first)."""
        tok = AutoTokenizer.from_pretrained(parent, revision=revision)
        delimiter_ids(tok)
        lm = qwen3.load_backbone(parent, revision=revision, dtype=dtype, attn=attn, device=device)
        lm.requires_grad_(False)
        if lora:
            from peft import LoraConfig, get_peft_model
            lm = get_peft_model(lm, LoraConfig(task_type="FEATURE_EXTRACTION", r=lora["r"], lora_alpha=lora["alpha"],
                                               lora_dropout=lora["dropout"], target_modules=list(lora["targets"])))
        head = PointerHead(lm.config.hidden_size, head_dim)
        return cls(tok, lm, head, device, parent={"name": parent, "revision": revision}, lora=lora)

    @classmethod
    def from_run(cls, run_dir, *, device, dtype=torch.float32, attn=None) -> "DecisionModel":
        """A training run (adapter/, head.safetensors, run.json) over the parent resolved through the HF cache."""
        from peft import PeftModel
        run = Path(run_dir)
        meta = json.loads((run / "run.json").read_text())
        m = cls.from_parent(meta["parent"], meta["revision"], device=device, dtype=dtype, head_dim=meta["head_dim"], attn=attn)
        m.lm = PeftModel.from_pretrained(m.lm, str(run / "adapter")).to(m.device)
        m.lora = meta["lora"]
        m.head.load_state_dict(load_file(run / "head.safetensors"))
        m.head.temperature = float(meta["temperature"])
        return m.eval()

    @classmethod
    def from_pretrained(cls, path, *, device=None, dtype=None, attn=None) -> "DecisionModel":
        """An exported artifact directory. Local files only; the parent is never resolved."""
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        found = (cfg.get("model_type"), cfg.get("format_version"), cfg.get("renderer", {}).get("version"))
        if found != ("decision_model", FORMAT_VERSION, RENDERER_VERSION):
            raise ValueError(f"{path}: (model_type, format_version, renderer version) is {found}; this runtime reads "
                             f"('decision_model', {FORMAT_VERSION}, {RENDERER_VERSION})")
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = dtype or getattr(torch, cfg["backbone"]["dtype"])
        tok = AutoTokenizer.from_pretrained(str(path / cfg["tokenizer"]["path"]), local_files_only=True)
        if delimiter_ids(tok) != cfg["tokenizer"]["delimiters"]:
            raise ValueError(f"{path}: tokenizer delimiter ids differ from config.json")
        lm = qwen3.load_backbone(path / cfg["backbone"]["path"], dtype=dtype, attn=attn, device=device, local_files_only=True)
        if lm.config.num_hidden_layers != cfg["backbone"]["num_hidden_layers"]:
            raise ValueError(f"{path}: backbone has {lm.config.num_hidden_layers} layers, config.json says {cfg['backbone']['num_hidden_layers']}")
        lm.requires_grad_(False)
        head = PointerHead(lm.config.hidden_size, cfg["head"]["dim"])
        head.load_state_dict(load_file(path / cfg["head"]["path"]))
        head.temperature = float(cfg["temperature"])
        limits = {"max_state": cfg["limits"]["max_state_tokens"], "max_row": cfg["limits"]["max_row_tokens"]}
        return cls(tok, lm, head, device, parent=cfg["parent"], lora=None, limits=limits)

    def _head_tensors(self):
        return {k: v.detach().float().cpu().contiguous() for k, v in self.head.state_dict().items()}

    def save_run(self, run_dir, extra: dict | None = None) -> None:
        from peft import PeftModel
        if not isinstance(self.lm, PeftModel):
            raise ValueError("save_run needs a LoRA adapter (from_parent(..., lora=...) or from_run)")
        run = Path(run_dir)
        run.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(str(run / "adapter"))
        save_file(self._head_tensors(), run / "head.safetensors")
        meta = {**(extra or {}), "parent": self.parent["name"], "revision": self.parent["revision"], "head_dim": self.head.q.out_features,
                "lora": self.lora, "temperature": self.head.temperature}
        (run / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    def merge_lora(self) -> None:
        """Fold the adapter into the backbone in FP32 (one rounding back to the load dtype); no-op without an adapter."""
        from peft import PeftModel
        if not isinstance(self.lm, PeftModel):
            return
        dt = next(self.lm.parameters()).dtype
        self.lm = qwen3.cast_parameters(qwen3.cast_parameters(self.lm, torch.float32).merge_and_unload(), dt)
        self.lm.requires_grad_(False)

    def save_pretrained(self, out_dir, dtype=torch.bfloat16) -> None:
        """Write the model part of the artifact: backbone/, decision_head.safetensors, tokenizer/, config.json. The live
        backbone's parameters are left in `dtype` (see the module docstring)."""
        from peft import PeftModel
        if isinstance(self.lm, PeftModel):
            raise ValueError("merge_lora() before save_pretrained(): the artifact holds merged weights only")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        qwen3.save_backbone(self.lm, out / "backbone", dtype)
        save_file(self._head_tensors(), out / "decision_head.safetensors")
        self.tok.save_pretrained(str(out / "tokenizer"))
        c = self.lm.config
        cfg = {"format_version": FORMAT_VERSION, "model_type": "decision_model",
               "backbone": {"type": "qwen3", "path": "backbone", "num_hidden_layers": c.num_hidden_layers, "hidden_size": c.hidden_size,
                            "dtype": str(dtype).removeprefix("torch.")},
               "head": {"type": "pointer", "dim": self.head.q.out_features, "path": "decision_head.safetensors"},
               "tokenizer": {"path": "tokenizer", "delimiters": delimiter_ids(self.tok)},
               "renderer": {"version": RENDERER_VERSION},
               "limits": {"max_state_tokens": self.limits["max_state"], "max_row_tokens": self.limits["max_row"],
                          "max_questions": MAX_QUESTIONS, "max_options": MAX_OPTIONS},
               "temperature": self.head.temperature, "parent": self.parent}
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # ---- execution

    def encode(self, rec) -> dict:
        return encode(self.tok, rec, **self.limits)

    def _pad(self, rows):
        """Right-pad (ids, pos) rows into [N, L] ids / positions and a [N, L] padding mask (1 = real token)."""
        L = max(len(ids) for ids, _ in rows)
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device)
            pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device)
            att[i, : len(rid)] = 1
        return ids, pos, att

    def _logits(self, h, decide, opts):
        return self.head(h[decide], h[torch.tensor(opts, device=self.device)])

    def _packed_hidden(self, encs):
        ids, pos, _ = self._pad([(e["ids"], e["pos"]) for e in encs])
        mask = branch_mask_batch([e["seg"] for e in encs], ids.shape[1], self.device, next(self.lm.parameters()).dtype)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False).last_hidden_state.float()

    def _rows_hidden(self, rows, cache=None, prefix_len=0):
        """[L_i, d] hidden states per causal (ids, pos) row. Eval mode runs rows_per_pass rows per pass; training keeps
        one pass. With `cache` the rows continue the cached prefix: each chunk gets its own replica of it."""
        chunk = len(rows) if self.training else rows_per_pass([ids for ids, _ in rows], prefix_len)
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad(part)
            kw = {"use_cache": False}
            if cache is not None:
                att = torch.cat([torch.ones((len(part), prefix_len), dtype=torch.long, device=self.device), att], 1)
                kw = {"past_key_values": qwen3.replicate_cache(cache, len(part), self.lm.config), "use_cache": True}
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, **kw).last_hidden_state.float()
            out += [h[i, : len(rid)] for i, (rid, _) in enumerate(part)]
        return out

    def forward_batch(self, encs, form="packed") -> list[list[torch.Tensor]]:
        """Logits at T=1, per record, per question. Differentiable (the training path)."""
        if form == "packed":
            hs = self._packed_hidden(encs)
            return [[self._logits(hs[b], d, oi) for d, oi in zip(e["decide_idx"], e["opt_idx"])] for b, e in enumerate(encs)]
        if form != "rows":
            raise ValueError(f"forward_batch form must be 'packed' or 'rows', not {form!r}")
        rows, readouts = [], []
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"]))
                readouts.append((b, len(S) + r["decide"], [len(S) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self._logits(h, d, oi))
        return out

    def _softmax(self, logits, temperature):
        T = self.head.temperature if temperature is None else temperature
        ps = [F.softmax(z / T, -1) for z in logits]
        return list(torch.cat(ps).cpu().split([len(p) for p in ps]))   # one device sync per call

    @staticmethod
    def _inference_form(enc, form):
        return "cached" if form == "packed" and len(enc["ids"]) > PACKED_TOKEN_LIMIT else form

    @torch.no_grad()
    def probs(self, enc, form="cached", temperature=None) -> list[torch.Tensor]:
        """Per-question probabilities (CPU fp32, canonical option order). temperature None = the stored one; 1.0 = raw."""
        form = self._inference_form(enc, form)
        if form == "cached":
            return self.probs_with_prefix(enc, self.prefix(enc), temperature)
        return self._softmax(self.forward_batch([enc], form)[0], temperature)

    @torch.no_grad()
    def prefix(self, enc) -> Prefix:
        L = enc["n_state"]
        ids = torch.tensor([enc["ids"][:L]], device=self.device)
        pos = torch.tensor([enc["pos"][:L]], device=self.device)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        return Prefix(L, out.past_key_values, out.last_hidden_state[0].float())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix, temperature=None) -> list[torch.Tensor]:
        """Only the branches run, as causal rows on replicas of the prefix cache; the caller's prefix is not modified."""
        if enc["n_state"] != prefix.n_state:
            raise ValueError("prefix does not match this record's state")
        _, _, rows = rows_of(enc)
        hs = self._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=prefix.cache, prefix_len=prefix.n_state)
        return self._softmax([self._logits(h, r["decide"], r["opts"]) for h, r in zip(hs, rows)], temperature)

    @staticmethod
    def _result(enc, meta, ps, form, latency_ms):
        return {"answers": to_answers([p.tolist() for p in ps], meta),
                "usage": {"state_tokens": enc["n_state"], "branch_tokens": len(enc["ids"]) - enc["n_state"], "questions": len(meta)},
                "latency_ms": latency_ms, "form": form}

    def evaluate(self, state, questions, *, form="cached", temperature=None) -> dict:
        """Public request -> public result. latency_ms covers validation, encoding and the forward passes."""
        t0 = time.perf_counter()
        rec, meta = to_record(DecisionRequest.model_validate({"state": state, "questions": questions}))
        enc = self.encode(rec)
        ps = self.probs(enc, form=form, temperature=temperature)
        return self._result(enc, meta, ps, self._inference_form(enc, form), (time.perf_counter() - t0) * 1e3)

    @torch.no_grad()
    def evaluate_requests(self, requests: list[DecisionRequest], *, form="packed", temperature=None) -> list[dict]:
        """Throughput path. packed: right-padded batches of packed records, as many per pass as fit BRANCH_TOKEN_BUDGET
        tokens; other forms: per request. Every result's latency_ms is the wall time of the whole call."""
        t0 = time.perf_counter()
        encoded = [encode_request(self.tok, r, **self.limits) for r in requests]
        encs = [e for e, _ in encoded]
        forms = [self._inference_form(e, form) for e in encs]
        ps = [None] * len(encs)
        packed = [i for i, f in enumerate(forms) if f == "packed"]
        if packed:
            n = max(1, BRANCH_TOKEN_BUDGET // max(len(encs[i]["ids"]) for i in packed))
            for c in range(0, len(packed), n):
                idx = packed[c:c + n]
                for i, z in zip(idx, self.forward_batch([encs[i] for i in idx], "packed")):
                    ps[i] = self._softmax(z, temperature)
        for i, f in enumerate(forms):
            if f != "packed":
                ps[i] = self.probs(encs[i], form=f, temperature=temperature)
        ms = (time.perf_counter() - t0) * 1e3
        return [self._result(e, meta, p, f, ms) for (e, meta), p, f in zip(encoded, ps, forms)]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
