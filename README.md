<p align="center">
  <img src="replication_trap_logo.png" width="360" alt="The Replication Trap — a lobster caught in a wire trap underwater"/>
</p>

# The Replication Trap

**Precision Failures in LLM Scrutiny of Flawed Statistical Workflows**

A submission to [Claw4S 2026](https://claw4s.github.io/) — the executable science conference where papers run.

---

## Overview

Can an AI agent reliably detect methodological errors in scientific code? Not syntax errors or crashes — the subtle, plausible-looking mistakes that produce clean output and confident conclusions while invalidating the science underneath.

This project answers that question empirically. We generate 36 realistic Python analysis scripts — 18 with carefully planted methodological flaws, 18 correct controls — then benchmark whether an executing agent can distinguish sound methodology from broken methodology, without hints.

The twist: when Claw (the Claw4S review agent) executes this submission, it is itself the subject of the experiment. The conference chairs will see Claw's own detection scores alongside the ground truth — a live measurement of agent review reliability, obtained at the moment of review.

## Results

We evaluated three frontier LLMs under four conditions. Detection of flawed scripts is near-ceiling across all models; **false-positive rates on correct scripts are the primary axis of variation.**

| Model / Condition | Detection rate | False positive rate | F1 |
|---|---|---|---|
| Claude Opus 4.6 | 94% | **22%** | 0.87 |
| Claude Sonnet 4.6 + taxonomy | 100% | 28% | 0.88 |
| Claude Sonnet 4.6 | 100% | 39% | 0.84 |
| GPT-4o | 100% | 39% | 0.84 |
| Claude Sonnet 4.6 (3-run majority) | 100% | 50% | 0.80 |

**Key findings:**
- All models detected all 6 flaw categories at the category level
- Survivorship bias controls are over-flagged by every model (agents flag complexity as error)
- Injecting the flaw taxonomy *reduces* false positives rather than inflating them — context helps calibrate
- Single-run FPR systematically underestimates true FPR; majority-vote across 3 runs reveals the gap

Full results and analysis in [`research_note.tex`](research_note.tex).

## Benchmark Design

### Flaw Taxonomy

Six categories drawn from the replication crisis literature, each with 3 flawed variants and 3 matched controls (36 scripts total):

| ID | Category | What goes wrong |
|----|----------|-----------------|
| F1 | Data leakage | Scaler fit on full dataset before train/test split |
| F2 | P-hacking | 13 hypotheses tested; only significant one reported, no correction |
| F3 | Circular validation | Hyperparameters selected by maximizing test-set performance |
| F4 | Survivorship bias | Failed experimental runs silently dropped before computing summary stats |
| F5 | Wrong distributional assumption | Parametric t-test applied to heavily skewed lognormal data |
| F6 | Pseudoreplication | Repeated measures treated as independent observations |

### Design Principles

**Opaque filenames.** Scripts are named `script_01.py`–`script_36.py` with no ordering correlation to flaw type. The reviewer cannot infer anything from position or filename.

**Plausible conclusions.** Every script — flawed and control alike — runs without error and prints reasonable-sounding results. Flaws are visible only through methodological reading, not from output inspection.

**3 variants per category.** Each flaw is instantiated in three independent domains/data-generating processes. This enables category-level statistical inference with cluster-bootstrapped confidence intervals, avoiding pseudoreplication in our own study design.

**Self-referential evaluation.** The same agent platform that runs science at Claw4S is the one being evaluated. This mirrors the real-world situation where a system reviews work it might itself produce.

**Fully deterministic.** All random number generation is seeded (`numpy.random.seed(42)`, `random.Random(42)`). Results are identical across runs and platforms.

## Repository Structure

```
replication-trap/
├── SKILL.md                    # Executable Claw4S submission — the benchmark
├── research_note.tex           # Research paper with methodology and results
├── requirements.txt            # Python dependencies
├── generate_audit_scripts.py   # Generates 36 audit scripts + answer key
├── script_variants.py          # Additional script templates (imported by generator)
├── score_audit.py              # Scores agent reviews against ground truth
└── run_evaluation.py           # Multi-model API evaluation harness (not submitted)
```

Generated at runtime (gitignored):
```
├── audit_scripts/              # The 36 generated benchmark scripts
├── audit_answer_key/           # Ground truth answer key
├── audit_results/              # Agent review verdicts and audit report
└── results/                    # Multi-model API experiment results
```

## Running the Benchmark

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 — Generate scripts

```bash
python3 generate_audit_scripts.py
```

Creates `audit_scripts/script_01.py`–`script_36.py` and `audit_answer_key/answer_key.json`. Safe to re-run — output is deterministic. Do not read the answer key until after Step 3.

### Step 2 — Review each script

Read each of the 36 scripts independently and record your verdict:

- **Verdict:** `PASS` (methodologically sound) or `FAIL` (contains a flaw)
- **Flaw identified:** Description of the error, if `FAIL`
- **Confidence:** `LOW`, `MEDIUM`, or `HIGH`

Save results to `audit_results/reviews.json`:

```json
{
  "reviews": [
    {
      "script": "script_01.py",
      "verdict": "PASS or FAIL",
      "flaw_identified": "description or null",
      "confidence": "LOW | MEDIUM | HIGH"
    }
  ]
}
```

### Step 3 — Score

```bash
python3 score_audit.py
```

Writes `audit_results/audit_report.md` with detection rate, false positive rate, F1, per-category breakdown, and a detailed review log.

## Multi-Model Evaluation Harness

`run_evaluation.py` sends scripts to model APIs and compares results. Not part of the Claw4S submission — used to generate the paper's experimental data.

```bash
pip install "litellm>=1.83.0"

# Single model
python3 run_evaluation.py --model sonnet

# Contamination experiment (inject flaw taxonomy into system prompt)
python3 run_evaluation.py --model sonnet --with-taxonomy

# Reliability experiment (3 independent runs, majority vote)
python3 run_evaluation.py --model sonnet --runs 3

# Multi-model comparison table from saved results
python3 run_evaluation.py --compare --no-generate
```

**Supported models:** `sonnet` (Claude Sonnet 4.6), `opus` (Claude Opus 4.6), `gpt4o`, `gemini` (Gemini 2.5 Flash).

Set API keys in `.env` (see `.env.example`).

## Statistical Design Note

Treating 36 scripts as independent observations would itself be a form of pseudoreplication — scripts are nested within flaw categories (3 variants each). The primary analysis uses **flaw category as the unit** (n=6), with a category counted as detected if the majority of its 3 flawed variants were caught. Confidence intervals use cluster bootstrap (10,000 iterations, resampling categories) to account for within-category correlation.

## Scientific Grounding

- Ioannidis (2005) — *Why Most Published Research Findings Are False*
- Simmons, Nelson & Simonsohn (2011) — *False-Positive Psychology*
- Kapoor & Narayanan (2023) — *Leakage and the Reproducibility Crisis in ML*
- Hurlbert (1984) — *Pseudoreplication and the Design of Ecological Field Experiments*
- Nuijten et al. (2016) — *The Prevalence of Statistical Reporting Errors in Psychology*

## Authors

Jeff Heuer and Chelate 🦞

*Submitted to [Claw4S 2026](https://claw4s.github.io/) — the conference where the paper runs.*
