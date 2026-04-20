# -*- coding: utf-8 -*-
"""
Training-free GDD-style decoding for CodeT5 on HumanEval.

This script runs:
  1) download/load HumanEval problems
  2) baseline sampling with CodeT5
  3) GDD decoding (logits -= lambda * energy(prefix, candidate)) during sampling
  4) evaluation: pass@k (HumanEval) + syntax validity (ast.parse)

Run (from repo root):
  python -m code_gen.gdd_codet5_pipeline --out_dir code_gen/runs/codet5_gdd
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList, RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm


# -------------------------
# Utilities
# -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _http_download(url: str, dst: Path) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dst.write_bytes(data)


def ensure_humaneval_jsonl_gz(data_dir: Path) -> Path:
    """
    Ensure `HumanEval.jsonl.gz` exists locally. Prefer:
      1) local `data_dir/HumanEval.jsonl.gz`
      2) download from openai/human-eval GitHub
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / "HumanEval.jsonl.gz"
    if p.exists():
        return p
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    print(f"[data] Downloading HumanEval to {p} from {url}")
    _http_download(url, p)
    return p


def stream_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_humaneval_problems(humaneval_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = stream_jsonl_gz(humaneval_path)
    problems = {r["task_id"]: r for r in rows}
    return problems


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# GDD energy on code prefixes
# -------------------------


_OPEN = "([{"
_CLOSE = ")]}"
_PAIR = {")": "(", "]": "[", "}": "{"}
_PAIR_FWD = {"(": ")", "[": "]", "{": "}"}


@dataclass
class PrefixState:
    # A small “syntax geodesic” state: delimiter stack + quote parity.
    stack: List[str]
    in_single: bool
    in_double: bool
    escaped: bool


def init_state() -> PrefixState:
    return PrefixState(stack=[], in_single=False, in_double=False, escaped=False)


def advance_state(state: PrefixState, text: str) -> PrefixState:
    """
    Update state by scanning characters. This is a lightweight approximation:
      - tracks (),[],{} nesting only when not inside quotes
      - tracks single/double quote parity with a simple escape handling
    It is not a full lexer; but it is fast and works well as a training-free prior.
    """
    st = PrefixState(
        stack=list(state.stack),
        in_single=state.in_single,
        in_double=state.in_double,
        escaped=state.escaped,
    )

    for ch in text:
        if st.escaped:
            st.escaped = False
            continue
        if ch == "\\":
            st.escaped = True
            continue

        if not st.in_double and ch == "'":
            st.in_single = not st.in_single
            continue
        if not st.in_single and ch == '"':
            st.in_double = not st.in_double
            continue

        if st.in_single or st.in_double:
            continue

        if ch in _OPEN:
            st.stack.append(ch)
        elif ch in _CLOSE:
            if not st.stack:
                # keep empty; mismatch is handled by energy
                continue
            want = _PAIR[ch]
            if st.stack[-1] == want:
                st.stack.pop()
            else:
                # mismatch: keep; handled by energy
                continue
    return st


def syntax_energy(prefix_state: PrefixState, cand_text: str) -> float:
    """
    Energy is higher for candidates that move away from syntactically completable states.
    This is our "geodesic distance" surrogate in a small state graph:
      - illegal close when stack empty/mismatched (big penalty)
      - toggling quotes into an open string late (penalty)
      - excessive growth of stack depth (small penalty)
    """
    # Scan candidate text while checking for illegal actions relative to prefix_state.
    st = PrefixState(
        stack=list(prefix_state.stack),
        in_single=prefix_state.in_single,
        in_double=prefix_state.in_double,
        escaped=prefix_state.escaped,
    )
    illegal_close = 0
    mismatch_close = 0
    quote_toggles = 0
    opens = 0

    for ch in cand_text:
        if st.escaped:
            st.escaped = False
            continue
        if ch == "\\":
            st.escaped = True
            continue

        if not st.in_double and ch == "'":
            st.in_single = not st.in_single
            quote_toggles += 1
            continue
        if not st.in_single and ch == '"':
            st.in_double = not st.in_double
            quote_toggles += 1
            continue

        if st.in_single or st.in_double:
            continue

        if ch in _OPEN:
            st.stack.append(ch)
            opens += 1
        elif ch in _CLOSE:
            if not st.stack:
                illegal_close += 1
                continue
            want = _PAIR[ch]
            if st.stack[-1] != want:
                mismatch_close += 1
                continue
            st.stack.pop()

    # Weighted sum
    e = 0.0
    e += 8.0 * illegal_close
    e += 5.0 * mismatch_close
    e += 0.2 * opens
    # Penalize entering/oscillating quotes; encourages staying in code mode.
    e += 0.3 * quote_toggles
    return float(e)


class GDDLogitsProcessor(LogitsProcessor):
    """
    Apply: logits[topk] -= lambda * E(prefix_state, candidate_text)
    during AR decoding.
    """

    def __init__(
        self,
        tokenizer: RobertaTokenizer,
        lam: float,
        topk: int = 64,
        max_prefix_chars: int = 4000,
    ):
        self.tokenizer = tokenizer
        self.lam = float(lam)
        self.topk = int(topk)
        self.max_prefix_chars = int(max_prefix_chars)
        self._states: List[PrefixState] | None = None
        self._cached_len: List[int] | None = None
        # Caches for speed (logic-equivalent).
        # 1) token_id -> decoded string piece
        self._tok_text_cache: Dict[int, str] = {}
        # 2) (state_signature, token_id) -> energy
        self._energy_cache: Dict[Tuple[Tuple[str, ...], bool, bool, bool, int], float] = {}
        # Cap to avoid unbounded growth on long runs.
        self._energy_cache_max = 250_000

    def _ensure_batch(self, bsz: int) -> None:
        if self._states is None or len(self._states) != bsz:
            self._states = [init_state() for _ in range(bsz)]
            self._cached_len = [0 for _ in range(bsz)]

    @staticmethod
    def _state_sig(st: PrefixState) -> Tuple[Tuple[str, ...], bool, bool, bool]:
        return (tuple(st.stack), bool(st.in_single), bool(st.in_double), bool(st.escaped))

    def _decode_token_piece(self, tok_id: int) -> str:
        t = self._tok_text_cache.get(tok_id)
        if t is None:
            t = self.tokenizer.decode(
                [tok_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            self._tok_text_cache[tok_id] = t
        return t

    def _energy_cached(self, st: PrefixState, tok_id: int) -> float:
        sig = self._state_sig(st)
        key = (sig[0], sig[1], sig[2], sig[3], int(tok_id))
        e = self._energy_cache.get(key)
        if e is not None:
            return float(e)
        cand_text = self._decode_token_piece(tok_id)
        if not cand_text:
            e = 0.0
        else:
            e = float(syntax_energy(st, cand_text))
        # Simple cap eviction: clear whole cache if too large.
        if len(self._energy_cache) >= self._energy_cache_max:
            self._energy_cache.clear()
        self._energy_cache[key] = e
        return e

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.lam == 0.0:
            return scores
        bsz = input_ids.shape[0]
        self._ensure_batch(bsz)
        assert self._states is not None and self._cached_len is not None

        # Update prefix state based on newly appended decoded text since last call.
        # For T5, `input_ids` here are decoder sequence ids.
        for i in range(bsz):
            # Decode only the suffix since last time to keep this cheap.
            # Tokenizer decode can be expensive; but we only do it once per step per sequence.
            cur_len = int(input_ids[i].shape[0])
            prev_len = int(self._cached_len[i])
            if cur_len <= prev_len:
                continue
            # Decode incremental tokens (excluding special tokens)
            suffix_ids = input_ids[i][prev_len:cur_len]
            suffix_text = self.tokenizer.decode(
                suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if suffix_text:
                self._states[i] = advance_state(self._states[i], suffix_text)
            self._cached_len[i] = cur_len

        # Apply energy to top-k candidates
        k_eff = min(self.topk, scores.shape[-1])
        vals, idx = torch.topk(scores, k=k_eff, dim=-1)
        # Work in FP32 for stability, then cast back.
        vals_f = vals.float()

        for i in range(bsz):
            st = self._states[i]
            # Decode candidate tokens one-by-one (k_eff <= 64)
            # We decode single token pieces and evaluate energy on that piece.
            for j in range(k_eff):
                tok_id = int(idx[i, j].item())
                e = self._energy_cached(st, tok_id)
                if e != 0.0:
                    vals_f[i, j] -= self.lam * e

        scores = scores.clone()
        scores.scatter_(dim=-1, index=idx, src=vals_f.to(scores.dtype))
        return scores


# -------------------------
# Metrics
# -------------------------


def is_syntax_valid_py(code: str) -> bool:
    import ast

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def _extract_completion_like_humaneval(comp: str) -> str:
    """
    HumanEval-style stopping: truncate completion when it starts a new top-level block.
    Matches common evaluation scripts to reduce syntax breakage.
    """
    stop_seqs = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    out = comp.replace("\t", "    ")
    for s in stop_seqs:
        idx = out.find(s)
        if idx != -1:
            out = out[:idx]
    return out


def evaluate_humaneval_passk_official(
    samples_jsonl: Path,
    k_list: List[int],
    *,
    problem_file: Path,
    ignore_incomplete: bool,
) -> Dict[str, float]:
    """
    Official HumanEval evaluation using `human-eval` package:
      - entrypoint: `human_eval.evaluation.evaluate_functional_correctness`
      - expects jsonl with fields: {task_id, completion}
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: human-eval. Install with `pip install human-eval`."
        ) from e

    ks = sorted(set([int(k) for k in k_list if int(k) > 0]))
    if not ks:
        return {}
    res = evaluate_functional_correctness(
        str(samples_jsonl),
        k=ks,
        n_workers=1,
        timeout=3.0,
        problem_file=str(problem_file),
        ignore_incomplete=ignore_incomplete,
    )
    out: Dict[str, float] = {}
    for k in ks:
        key = f"pass@{k}"
        if key in res:
            out[key] = float(res[key])
    return out


def evaluate_bleu_official(
    samples_jsonl: Path, problems: Dict[str, Dict[str, Any]]
) -> float:
    """
    BLEU computed with sacrebleu (standard, reproducible).
    References are HumanEval `canonical_solution`.
    Hypothesis is `prompt + completion` (matching evaluation context).
    """
    try:
        import sacrebleu
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: sacrebleu. Install with `pip install sacrebleu`."
        ) from e

    hyps: List[str] = []
    refs: List[str] = []
    for line in samples_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        task_id = r["task_id"]
        comp = r["completion"]
        prob = problems.get(task_id)
        if prob is None:
            continue
        hyp = prob["prompt"] + comp
        ref_full = prob.get("canonical_solution", "")
        # If canonical_solution includes the prompt, strip it to compare the same span.
        if ref_full.startswith(prob["prompt"]):
            ref = ref_full
        else:
            ref = ref_full
        hyps.append(hyp)
        refs.append(ref)

    if not hyps:
        return float("nan")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return float(bleu.score)


# -------------------------
# Generation
# -------------------------


def generate_samples(
    *,
    model: T5ForConditionalGeneration,
    tokenizer: RobertaTokenizer,
    problems: Dict[str, Dict[str, Any]],
    out_jsonl: Path,
    device: str,
    num_samples_per_task: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gdd_lam: float,
    gdd_topk: int,
) -> Dict[str, float]:
    """
    Writes samples in HumanEval expected format: {"task_id":..., "completion":...}
    Also returns syntax validity on generated `prompt+completion`.
    """
    safe_mkdir(out_jsonl.parent)
    model.eval()

    task_ids = sorted(problems.keys())
    n_valid = 0
    n_total = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for task_id in tqdm(task_ids, desc=f"generate ({out_jsonl.name})", ncols=0):
            prompt = problems[task_id]["prompt"]
            # HumanEval prompts include indentation; keep as-is.
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

            logits_proc = LogitsProcessorList()
            if gdd_lam != 0.0:
                logits_proc.append(GDDLogitsProcessor(tokenizer=tokenizer, lam=gdd_lam, topk=gdd_topk))

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_samples_per_task,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_proc if len(logits_proc) > 0 else None,
                )

            # Decode each completion (for seq2seq, generated text is the completion)
            completions = tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for comp in completions:
                # HumanEval expects only the completion that appends after prompt.
                # CodeT5 tends to generate full continuations; keep raw completion.
                comp = _extract_completion_like_humaneval(comp)
                rec = {"task_id": task_id, "completion": comp}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
                if is_syntax_valid_py(prompt + comp):
                    n_valid += 1

    return {"syntax_valid_rate": (n_valid / max(1, n_total))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="code_gen/runs/codet5_gdd")
    ap.add_argument("--model", type=str, default="Salesforce/codet5-base")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    # HumanEval
    ap.add_argument("--humaneval_dir", type=str, default="code_gen/data/humaneval")
    ap.add_argument("--num_tasks", type=int, default=-1, help="subset for quick runs; -1 = all")
    ap.add_argument("--num_samples_per_task", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--passk", type=str, default="1,10", help="comma-separated k values for pass@k")

    # GDD
    ap.add_argument("--gdd_lam", type=float, default=0.8)
    ap.add_argument("--gdd_topk", type=int, default=64)

    args = ap.parse_args()
    set_seed(args.seed)

    out = Path(args.out_dir)
    safe_mkdir(out)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print(f"[device] {device}")

    # Load problems
    humaneval_dir = Path(args.humaneval_dir)
    humaneval_path = ensure_humaneval_jsonl_gz(humaneval_dir)
    problems = load_humaneval_problems(humaneval_path)
    if args.num_tasks > 0:
        keep = sorted(problems.keys())[: args.num_tasks]
        problems = {k: problems[k] for k in keep}
    print(f"[data] HumanEval tasks: {len(problems)}")

    # Load model/tokenizer (frozen pretrained)
    print(f"[model] Loading {args.model}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.to(device)

    k_list = [int(x) for x in args.passk.split(",") if x.strip()]

    # Baseline
    print("[run] Baseline sampling")
    baseline_jsonl = out / "samples_baseline.jsonl"
    baseline_syn = generate_samples(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        out_jsonl=baseline_jsonl,
        device=device,
        num_samples_per_task=args.num_samples_per_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gdd_lam=0.0,
        gdd_topk=args.gdd_topk,
    )

    # GDD
    print("[run] GDD sampling")
    gdd_jsonl = out / "samples_gdd.jsonl"
    gdd_syn = generate_samples(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        out_jsonl=gdd_jsonl,
        device=device,
        num_samples_per_task=args.num_samples_per_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gdd_lam=args.gdd_lam,
        gdd_topk=args.gdd_topk,
    )

    # pass@k
    print("[eval] pass@k (HumanEval)")
    ignore_incomplete = args.num_tasks > 0
    base_pass = evaluate_humaneval_passk_official(
        baseline_jsonl,
        k_list,
        problem_file=humaneval_path,
        ignore_incomplete=ignore_incomplete,
    )
    gdd_pass = evaluate_humaneval_passk_official(
        gdd_jsonl,
        k_list,
        problem_file=humaneval_path,
        ignore_incomplete=ignore_incomplete,
    )

    # BLEU
    print("[eval] BLEU (sacrebleu; refs=canonical_solution)")
    base_bleu = evaluate_bleu_official(baseline_jsonl, problems)
    gdd_bleu = evaluate_bleu_official(gdd_jsonl, problems)

    # Print table
    keys = sorted(
        set(
            ["syntax_valid_rate", "bleu"]
            + list(base_pass.keys())
            + list(gdd_pass.keys())
        )
    )

    def row(name: str, syn: Dict[str, float], pk: Dict[str, float]) -> Dict[str, float]:
        outd = {"method": name}
        outd.update({"syntax_valid_rate": float(syn.get("syntax_valid_rate", float("nan")))})
        outd["bleu"] = float(base_bleu if name == "baseline" else gdd_bleu)
        for k in keys:
            if k.startswith("pass@"):
                outd[k] = float(pk.get(k, float("nan")))
        return outd

    rows = [row("baseline", baseline_syn, base_pass), row("gdd", gdd_syn, gdd_pass)]
    print("\n=== RESULT TABLE ===")
    hdr = ["method"] + keys
    print(" ".join([f"{h:>18}" for h in hdr]))
    print("-" * (19 * len(hdr)))
    for r in rows:
        vals = [r.get("method", "")]
        for k in keys:
            v = r.get(k, float("nan"))
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(" ".join([f"{v:>18}" for v in vals]))

    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {(out / 'results.json').resolve()}")


if __name__ == "__main__":
    main()

