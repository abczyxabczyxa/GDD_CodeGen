# GDD-style training-free decoding for CodeT5

This folder provides a **single-script** pipeline that evaluates **baseline CodeT5 decoding** vs a **training-free GDD-style constrained sampler** during autoregressive generation.

The core idea is to modify decoding logits at each step:

`logits(token) ← logits(token) - λ · E(prefix_state, token)`

where `E(·)` is a lightweight **syntax-geodesic energy** that penalizes steps moving the partial program away from *syntactically completable* states.

## What is implemented

- **Model**: `Salesforce/codet5-base` (frozen; downloaded automatically via Hugging Face).
- **Benchmark**: **HumanEval** (Python).
- **Metrics**:
  - `pass@k` via the **official `human-eval` evaluator**
  - `BLEU` via **sacrebleu** (reproducible standard BLEU)
  - `syntax_valid_rate`: fraction of generated `prompt + completion` that can be parsed by `ast.parse`
- **Baseline**: standard sampling (`do_sample=True`, `temperature`, `top_p`).
- **GDD decoding (training-free)**: a custom logits processor that, for **top‑k** candidate tokens per step, decodes each candidate into text and applies a penalty based on:
  - illegal closing brackets when no bracket is open
  - mismatched bracket types
  - entering/oscillating quotes
  - excessive growth of bracket stack depth

This is a practical approximation of “geodesic distance in a syntax-state graph” (a small automaton defined by delimiter stack + quote parity). It is **fast**, differentiable-free, and works with standard `model.generate`.

## Install

Recommended (in your active env):

```bash
pip install transformers torch human-eval sacrebleu
```

## Run

From repo root (`GDD/`):

```bash
python -m code_gen.gdd_codet5_pipeline --out_dir code_gen/runs/codet5_gdd
```

The script will:
1. download `HumanEval.jsonl.gz` (cached under `--humaneval_dir`)
2. generate baseline samples → `samples_baseline.jsonl`
3. generate GDD samples → `samples_gdd.jsonl`
4. compute `BLEU`, `pass@k`, and `syntax_valid_rate`
5. write `results.json`

## Useful flags

- `--num_tasks`: evaluate only the first N tasks (quick debug)
- `--num_samples_per_task`: number of samples per task (controls pass@k ceiling)
- `--passk`: e.g. `1,10`
- `--temperature`, `--top_p`: baseline sampling parameters
- `--gdd_lam`: strength of GDD constraint
- `--gdd_topk`: apply energy only to top‑k logits each step (speed/quality trade-off)

Example quick run:

```bash
python -m code_gen.gdd_codet5_pipeline \
  --out_dir code_gen/runs/codet5_gdd_quick \
  --num_tasks 10 \
  --num_samples_per_task 5 \
  --passk 1 \
  --gdd_lam 0.6 \
  --gdd_topk 32
```

## Notes / limitations

- The energy is a **syntax prior**, not a full CFG/AST checker at each step. It is designed to be cheap and robust.
- `pass@k` runs the official HumanEval evaluator, which executes generated code during unit testing.
- CodeT5 is seq2seq; we treat `model.generate` output as the completion appended to the HumanEval prompt.

