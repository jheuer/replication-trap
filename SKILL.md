---
name: adversarial-skill-audit
description: >
  Generate statistical analysis scripts containing subtle methodological flaws,
  then measure whether AI agent reviewers can detect them. Produces a
  quantitative audit with detection rates, false-positive rates, and a
  difficulty ranking across flaw categories. Supports multi-model comparison
  when API keys are present.
allowed-tools: Bash(python3 *), Bash(pip *), Read, Write
---

# Adversarial Skill Audit

**Can AI agent reviewers catch methodological errors in executable scientific code?**

## Overview

Agent-based peer review is a foundational premise of executable science: if
skills replace papers, agents replace reviewers. But how reliable are agent
reviewers? This skill empirically measures detection rates for six categories of
real-world methodological flaws commonly found in statistical analysis code.

The skill generates 36 self-contained Python analysis scripts—18 containing a
single planted methodological flaw and 18 correct controls—then collects
structured reviews from available AI models and scores them against a ground
truth answer key. The output is a quantitative audit report: detection rates by
flaw category, false-positive rates on controls, and a difficulty ranking.

**Expected total runtime: under 2 minutes** (script generation <1 s; review
and scoring time depends on the number of models and scripts).

## Prerequisites

Install dependencies (safe to re-run; skips already-installed packages):

```bash
pip install -q -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install -q "numpy>=1.24,<3" "scipy>=1.10,<2" "scikit-learn>=1.2,<2" "litellm>=1.83.0"
```

`litellm` is required only for Step 2A (external model reviews). If no API keys
are present, it is unused and Step 2A is skipped automatically.

## Step 0 — Identify Available Reviewer Models

Before generating scripts, determine which external AI models are available
to use in Step 2A. This step is discovery only — use the output to decide
which `--model` flags to pass to the harness. The harness writes the
canonical `models_used.json` record after it runs.

Run the discovery script:

```bash
python3 -c "
import os, json, urllib.request

providers = {
    'anthropic': {
        'key_env': 'ANTHROPIC_API_KEY',
        'url': 'https://api.anthropic.com/v1/models',
        'headers_fn': lambda k: {'x-api-key': k, 'anthropic-version': '2023-06-01'},
        'extract': lambda d: [m['id'] for m in d.get('data', [])],
    },
    'openai': {
        'key_env': 'OPENAI_API_KEY',
        'url': 'https://api.openai.com/v1/models',
        'headers_fn': lambda k: {'Authorization': f'Bearer {k}'},
        'extract': lambda d: [m['id'] for m in d.get('data', [])],
    },
}

available = {}
for provider, cfg in providers.items():
    key = os.environ.get(cfg['key_env'])
    if not key:
        continue
    try:
        req = urllib.request.Request(cfg['url'], headers=cfg['headers_fn'](key))
        with urllib.request.urlopen(req, timeout=10) as r:
            available[provider] = cfg['extract'](json.loads(r.read()))
    except Exception as e:
        print(f'{provider}: discovery failed ({e})')

print(json.dumps(available, indent=2))
"
```

**Selection rule:** from each provider's list, identify the model that
represents its current highest capability tier. Prefer models whose names
signal flagship status (e.g. `opus`, `o3`, `ultra`, `pro`) over speed-optimised
variants (e.g. `haiku`, `mini`, `flash`). Map your selection to the harness
model keys: `sonnet`, `opus`, `gpt4o`, `gemini`, `gemini3`.

**Self-review is always included** regardless of which external models are
available — you (the executing agent) review the scripts as one of the
reviewers in Step 2B.

**Fallback:** If no API keys are present, skip Step 2A entirely and proceed
to Step 2B. The audit runs with self-review only.

## Step 1 — Generate the Flawed and Control Scripts

Run the generator script. It produces 36 Python files in `audit_scripts/`,
each a self-contained statistical analysis that loads or generates data, runs
an analysis, and prints a conclusion. Re-running is safe — existing files are
overwritten deterministically.

```bash
python3 generate_audit_scripts.py
```

**Expected output:** 36 files in `audit_scripts/` named `script_01.py` through
`script_36.py`, each 40–80 lines. An answer key is written to
`audit_answer_key/answer_key.json` — **do not read this file until after
completing Step 2.**

## Step 2 — Review Each Script

Reviews are collected from two sources: (A) external models via API, and (B)
you, the executing agent. Both are required. The scoring step treats each
reviewer independently.

### Step 2A — External Model Reviews

**Skip this step if no API keys were found in Step 0.** Proceed directly to Step 2B.

For each model identified in Step 0, run the evaluation harness with an explicit
`--model` flag. The harness sends all 36 scripts to that model, handles retries,
scores the results, and writes per-model report files to `results/` automatically.

```bash
# Run once per available model (replace <key> with the appropriate model key)
python3 run_evaluation.py --model <key> --no-generate
```

Available model keys: `sonnet`, `opus`, `gpt4o`, `gemini`, `gemini3`.

Example — if Step 0 found Anthropic and OpenAI keys:
```bash
python3 run_evaluation.py --model opus --no-generate
python3 run_evaluation.py --model gpt4o --no-generate
```

The harness prints progress per script (`script_01.py ... FAIL [HIGH]`) and
a summary table on completion. After all models have run, the harness writes
`models_used.json` recording which models were evaluated.

### Step 2B — Self-Review

**Complete your self-review before reading any external model results or
answer key to preserve reviewer independence. Review each script in isolation —
do not carry conclusions from one script to the next.**

For each of the 36 scripts in `audit_scripts/`, read the file one at a time
and produce a structured review. For every script, answer:

1. **Verdict:** Is the methodology sound? (`PASS` or `FAIL`)
2. **Flaw identified:** If FAIL, describe the specific methodological error.
3. **Confidence:** Rate your confidence in the verdict (`LOW` / `MEDIUM` / `HIGH`).

Evaluate each script independently on its methodology. The scripts have opaque
names — you have no information about which are flawed and which are correct.
Evaluate purely on the code and its scientific methods.

Write your verdicts to `audit_results/reviews.json` (used by the scoring
script in Step 3).

### Reviews format (all files)

```json
{
  "reviewer": {
    "provider": "self | anthropic | openai | ...",
    "model_id": "<exact model ID, or your own identifier if known>",
    "review_timestamp": "<ISO 8601>"
  },
  "reviews": [
    {
      "script": "script_01.py",
      "verdict": "PASS | FAIL | ERROR",
      "flaw_identified": "description, or null",
      "confidence": "LOW | MEDIUM | HIGH"
    }
  ]
}
```

All 36 scripts must appear in every file before proceeding to Step 3.

## Step 3 — Score the Results

Score your self-review against the answer key:

```bash
python3 score_audit.py
```

This reads `audit_results/reviews.json` (written in Step 2B) and computes:
- **Detection rate** per flaw category (did you correctly FAIL the flawed script?)
- **False positive rate** (did you incorrectly FAIL any control script?)
- **Overall accuracy** (correct verdicts / total scripts)
- **Confusion matrix** (true positives, false positives, true negatives, false negatives)
- **Confidence calibration** (accuracy broken down by LOW / MEDIUM / HIGH confidence)
- **Cluster-bootstrapped confidence intervals** (category-level, 10,000 iterations)

**Expected output:** `audit_results/audit_report.md` — a Markdown report with
tables and summary statistics.

If Step 2A was run, external model reports were already scored and saved to
`results/` automatically by `run_evaluation.py`. To print a cross-model
comparison table from those saved reports:

```bash
python3 run_evaluation.py --compare
```

## Step 4 — Interpret and Report

Read the audit reports and write a structured interpretation covering:

1. Which flaw categories were detected by each reviewer? Which were missed?
2. Were any control scripts incorrectly flagged? (False positives)
3. How do detection rates and false-positive rates compare across reviewers?
4. What does this tell us about the reliability of agent-based peer review?
5. Which flaw types are most dangerous because reviewers are systematically
   blind to them?
6. Note your own model ID (from `models_used.json`). Your self-review in
   Step 2B was conducted with this skill's full description in your context
   window — this is the **contaminated condition** by definition. The research
   note accompanying this submission measures how context contamination affects
   detection and false-positive rates; your results are a direct data point for
   that comparison. Report them as such, not as a limitation to apologise for.

Append your interpretation to `audit_results/audit_report.md`.

## Key Capabilities

- Generates realistic, runnable statistical analysis scripts with planted flaws
- Tests well-documented methodological errors drawn from the replication crisis
  literature
- Produces quantitative detection metrics (not just binary pass/fail)
- Supports multi-model comparison when API keys are present; falls back to
  self-review when not
- Self-review is always included — the executing agent is always one of the
  data points, creating a built-in contamination baseline
- Deterministic: seeded random generation ensures reproducible scripts across runs

## Guidelines

- All scripts use `numpy.random.seed(42)` for reproducibility
- Each flawed script produces plausible, non-obviously-wrong output
- Control scripts use the same data and analysis type but with correct methodology
- The flaw in each script is a single, specific, well-defined methodological error
- Filenames are opaque (`script_01.py` through `script_36.py`) — the reviewer
  has no labeling information, only the code itself
- An answer key in `audit_answer_key/` is used only by the scoring script
- The skill is fully idempotent: re-running `generate_audit_scripts.py` produces
  identical output; `score_audit.py` re-reads existing reviews and overwrites the report

## Extending the Benchmark

The benchmark is designed to be domain-agnostic and extensible. The flaw taxonomy
(F1–F6) covers the most replicated failure modes from the replication crisis
literature, but the framework supports additional categories:

**Adding new flaw categories:** Implement a new flaw–control pair following the
pattern in `generate_audit_scripts.py`. Each category needs at least 3 flawed
variants and 3 matched controls to support cluster-bootstrapped confidence
intervals. Update the answer key schema accordingly.

**Adding new domains:** The existing scripts cover regression, classification,
hypothesis testing, survival analysis, and mixed-effects models. New domains
(e.g., time series, causal inference, Bayesian methods) can be added by
implementing new script templates in `script_variants.py`.

**Adding new reviewer models:** Pass any LiteLLM-compatible model string via
`--model` (edit the `MODELS` dict in `run_evaluation.py` to register it). The
harness handles prompt formatting, retries, and scoring automatically.

**Contamination experiments:** Run `--with-taxonomy` to inject the flaw
taxonomy into the reviewer's system prompt. This measures how prior knowledge
of flaw categories affects detection and false-positive rates — the key
manipulation in the contamination condition reported in the accompanying paper.
