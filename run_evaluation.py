#!/usr/bin/env python3
"""
run_evaluation.py

API evaluation harness for the adversarial skill audit.
Sends each of the 12 generated scripts to one or more models via LiteLLM,
collects structured verdicts, scores them, and saves results.

Prerequisites:
  pip install "litellm>=1.83.0"

Required env vars (only for the models you want to run):
  ANTHROPIC_API_KEY   — Claude Sonnet / Opus
  OPENAI_API_KEY      — GPT-4o
  GEMINI_API_KEY      — Gemini 2.5 Pro

Usage:
  python3 run_evaluation.py                         # run all configured models
  python3 run_evaluation.py --model sonnet          # run one model
  python3 run_evaluation.py --model sonnet --with-taxonomy  # contamination experiment
  python3 run_evaluation.py --compare               # print comparison table from saved results
  python3 run_evaluation.py --no-generate           # skip script generation step
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    import litellm
    litellm.drop_params = True  # ignore unsupported params per provider silently
    litellm.suppress_debug_info = True
except ImportError:
    print('ERROR: litellm not installed. Run: pip install "litellm>=1.83.0"')
    sys.exit(1)

MODELS = {
    "sonnet": {
        "model_id": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
        "result_key": "claude_sonnet_4",
        "delay": 0.5,
    },
    "opus": {
        "model_id": "claude-opus-4-6",
        "display_name": "Claude Opus 4.6",
        "result_key": "claude_opus_4",
        "delay": 0.5,
    },
    "gpt4o": {
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "result_key": "gpt4o",
        "delay": 0.5,
    },
    "gemini": {
        "model_id": "gemini/gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "result_key": "gemini_25_flash",
        "delay": 1.5,
    },
    "gemini3": {
        "model_id": "gemini/gemini-3-flash-preview",
        "display_name": "Gemini 3 Flash Preview",
        "result_key": "gemini_3_flash",
        "delay": 1.5,
    },
}

# Flaw taxonomy — injected only in the contamination experiment condition
_TAXONOMY_CONTEXT = """\
The following flaw categories may be present in some scripts:
- Data leakage: feature scaling or other preprocessing applied before train/test split
- P-hacking: multiple hypothesis tests performed with only significant results reported, no multiple-comparison correction
- Circular validation: hyperparameter selection using test-set performance rather than a validation set
- Survivorship bias: excluding failed or null experimental runs before computing summary statistics
- Wrong test assumption: applying parametric tests (e.g., t-test) to data that violates normality
- Pseudoreplication: treating repeated measures from the same subject as independent observations"""

_REVIEW_PROMPT = """\
Review the following statistical analysis script for methodological soundness.

Evaluate whether the code contains any methodological flaws — errors in how the \
analysis is set up, how data is handled, how statistical tests are applied, or \
how results are reported. The script runs without errors and produces plausible \
output, but may contain a subtle flaw that would invalidate the conclusions.

Respond in JSON with exactly this structure:
{{
  "verdict": "PASS or FAIL",
  "flaw_identified": "description of the specific flaw, or null if PASS",
  "confidence": "LOW, MEDIUM, or HIGH"
}}

Script ({filename}):
```python
{code}
```"""

AUDIT_SCRIPTS_DIR = "audit_scripts"
AUDIT_RESULTS_DIR = "audit_results"
RESULTS_DIR = "results"


def generate_scripts():
    print("Generating audit scripts...")
    result = subprocess.run(
        [sys.executable, "generate_audit_scripts.py"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR generating scripts:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.strip())


def get_script_files():
    if not os.path.isdir(AUDIT_SCRIPTS_DIR):
        print(f"ERROR: {AUDIT_SCRIPTS_DIR}/ not found. Run without --no-generate first.")
        sys.exit(1)
    return sorted(f for f in os.listdir(AUDIT_SCRIPTS_DIR) if f.endswith(".py"))


def review_script(model_id, filename, code, with_taxonomy=False):
    """Send one script to the API via LiteLLM and return a structured verdict dict."""
    prompt = _REVIEW_PROMPT.format(filename=filename, code=code)
    system = "You are a scientific methods expert reviewing statistical analysis code for methodological soundness."
    if with_taxonomy:
        system += "\n\n" + _TAXONOMY_CONTEXT

    for attempt in range(3):
        try:
            t0 = time.monotonic()
            response = litellm.completion(
                model=model_id,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            raw_response = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            raw = raw_response
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            # Extract first JSON object if model appended extra text
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)

            parsed = json.loads(raw)
            verdict = parsed.get("verdict", "").upper().strip()
            if verdict not in ("PASS", "FAIL"):
                raise ValueError(f"Invalid verdict: {verdict!r}")

            usage = response.usage
            return {
                "script": filename,
                "verdict": verdict,
                "flaw_identified": parsed.get("flaw_identified"),
                "confidence": parsed.get("confidence", "UNKNOWN").upper().strip(),
                "raw_response": raw_response,
                "latency_ms": latency_ms,
                "usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                },
            }

        except litellm.exceptions.AuthenticationError as e:
            print(f"\nERROR: Authentication failed — check your API key.")
            print(f"  {e}")
            sys.exit(1)

        except litellm.exceptions.BadRequestError as e:
            msg = str(e)
            if "credit balance is too low" in msg or "insufficient_quota" in msg:
                print(f"\nERROR: Insufficient API credits.")
                print(f"  Add credits at: console.anthropic.com → Plans & Billing")
                print(f"  Note: claude.ai subscription credits are separate from API credits.")
                sys.exit(1)
            print(f"\n  Bad request (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  WARNING: Giving up on {filename} after 3 attempts.")
                return {
                    "script": filename,
                    "verdict": "PASS",
                    "flaw_identified": None,
                    "confidence": "LOW",
                    "_error": msg,
                }

        except (litellm.exceptions.RateLimitError, litellm.exceptions.ServiceUnavailableError) as e:
            wait = 15 * (attempt + 1)
            kind = "Rate limited" if isinstance(e, litellm.exceptions.RateLimitError) else "Service unavailable (503)"
            print(f"\n  {kind} (attempt {attempt + 1}/3). Waiting {wait}s...")
            time.sleep(wait)
            if attempt == 2:
                print(f"  WARNING: Giving up on {filename} after 3 retries.")
                return {
                    "script": filename,
                    "verdict": "PASS",
                    "flaw_identified": None,
                    "confidence": "LOW",
                    "_error": str(e),
                }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < 2:
                print(f"\n  Parse error (attempt {attempt + 1}/3): {e}. Retrying...")
                time.sleep(2)
            else:
                print(f"\n  WARNING: Failed to parse response for {filename} after 3 attempts: {e}")
                return {
                    "script": filename,
                    "verdict": "PASS",
                    "flaw_identified": None,
                    "confidence": "LOW",
                    "_parse_error": str(e),
                }


def run_model(model_key, with_taxonomy=False, n_runs=1):
    """Run evaluation for one model, score results, save to results/."""
    if model_key not in MODELS:
        print(f"ERROR: Unknown model '{model_key}'. Choose from: {list(MODELS)}")
        sys.exit(1)

    model = MODELS[model_key]
    result_key = model["result_key"] + ("_with_taxonomy" if with_taxonomy else "") + (f"_runs{n_runs}" if n_runs > 1 else "")

    print(f"\n{'=' * 60}")
    print(f"Model: {model['display_name']} ({model['model_id']})")
    if with_taxonomy:
        print("Context: WITH taxonomy (contamination experiment)")
    if n_runs > 1:
        print(f"Runs per script: {n_runs} (majority vote)")
    print(f"{'=' * 60}")

    script_files = get_script_files()
    reviews = []

    for i, filename in enumerate(script_files, 1):
        path = os.path.join(AUDIT_SCRIPTS_DIR, filename)
        with open(path) as f:
            code = f.read()

        print(f"  [{i:2d}/{len(script_files)}] {filename} ...", end=" ", flush=True)

        if n_runs == 1:
            review = review_script(model["model_id"], filename, code, with_taxonomy)
        else:
            run_results = []
            for r in range(n_runs):
                if r > 0:
                    time.sleep(model["delay"])
                run_results.append(
                    review_script(model["model_id"], filename, code, with_taxonomy)
                )
            # Majority vote on verdict
            fail_votes = sum(1 for r in run_results if r["verdict"] == "FAIL")
            majority_verdict = "FAIL" if fail_votes > n_runs / 2 else "PASS"
            agreement = max(fail_votes, n_runs - fail_votes) / n_runs
            # Pick the representative run matching majority for flaw_identified/confidence
            rep = next(
                (r for r in run_results if r["verdict"] == majority_verdict),
                run_results[0],
            )
            review = {
                "script": filename,
                "verdict": majority_verdict,
                "flaw_identified": rep["flaw_identified"],
                "confidence": rep["confidence"],
                "raw_response": rep["raw_response"],
                "latency_ms": rep["latency_ms"],
                "usage": rep["usage"],
                "runs": run_results,
                "agreement_rate": round(agreement, 3),
                "fail_votes": fail_votes,
                "n_runs": n_runs,
            }

        reviews.append(review)
        if n_runs > 1:
            print(f"{review['verdict']} ({review['fail_votes']}/{review['n_runs']} FAIL votes, {review['agreement_rate']:.0%} agreement)")
        else:
            print(f"{review['verdict']} ({review['confidence']})")

        if i < len(script_files):
            time.sleep(model["delay"])

    # Save full reviews to results/
    os.makedirs(RESULTS_DIR, exist_ok=True)
    reviews_payload = {
        "model": model["model_id"],
        "display_name": model["display_name"],
        "with_taxonomy": with_taxonomy,
        "n_runs": n_runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": "You are a scientific methods expert reviewing statistical analysis code for methodological soundness.",
        "reviews": reviews,
    }
    reviews_path = os.path.join(RESULTS_DIR, f"{result_key}_reviews.json")
    with open(reviews_path, "w") as f:
        json.dump(reviews_payload, f, indent=2)
    print(f"\nReviews saved to {reviews_path}")

    # Write to audit_results/reviews.json for score_audit.py
    os.makedirs(AUDIT_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(AUDIT_RESULTS_DIR, "reviews.json"), "w") as f:
        json.dump({"reviews": reviews}, f, indent=2)

    # Score
    result = subprocess.run(
        [sys.executable, "score_audit.py"], capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"Scoring error:\n{result.stderr}")

    # Copy report to results/
    report_src = os.path.join(AUDIT_RESULTS_DIR, "audit_report.md")
    report_dst = os.path.join(RESULTS_DIR, f"{result_key}_report.md")
    if os.path.exists(report_src):
        shutil.copy(report_src, report_dst)
        print(f"Report saved to {report_dst}")

    return reviews_payload


def build_comparison_table():
    """Read all saved reports and print a multi-model comparison table."""
    if not os.path.isdir(RESULTS_DIR):
        print("No results/ directory found.")
        return

    rows = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith("_report.md"):
            continue
        model_key = fname[: -len("_report.md")]
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            text = f.read()

        def extract(pattern):
            m = re.search(pattern, text)
            return m.group(1).strip() if m else "?"

        rows.append((
            model_key,
            extract(r"Overall accuracy \| (.+?) \|"),
            extract(r"Detection rate \(sensitivity\) \| (.+?) \|"),
            extract(r"False positive rate \| (.+?) \|"),
            extract(r"F1 score \| (.+?) \|"),
        ))

    if not rows:
        print("No result reports found in results/")
        return

    header = f"| {'Model':<40} | {'Accuracy':<20} | {'Detection Rate':<20} | {'FPR':<12} | {'F1':<8} |"
    sep    = f"|{'-'*42}|{'-'*22}|{'-'*22}|{'-'*14}|{'-'*10}|"
    print("\n## Multi-Model Comparison\n")
    print(header)
    print(sep)
    for model_key, acc, det, fpr, f1 in rows:
        print(f"| {model_key:<40} | {acc:<20} | {det:<20} | {fpr:<12} | {f1:<8} |")

    out_path = os.path.join(RESULTS_DIR, "comparison_table.md")
    with open(out_path, "w") as f:
        f.write("## Multi-Model Comparison\n\n")
        f.write("| Model | Accuracy | Detection Rate | FPR | F1 |\n")
        f.write("|-------|----------|----------------|-----|----|")
        for model_key, acc, det, fpr, f1 in rows:
            f.write(f"\n| {model_key} | {acc} | {det} | {fpr} | {f1} |")
        f.write("\n")
    print(f"\nComparison table saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model adversarial skill audit evaluation harness"
    )
    parser.add_argument(
        "--model", choices=list(MODELS) + ["all"], default="all",
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--with-taxonomy", action="store_true",
        help="Inject flaw taxonomy into system prompt (context contamination experiment)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Build comparison table from saved results (no API calls)",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip script generation (use existing audit_scripts/)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help="Number of independent runs per script (default: 1). "
             "Majority vote is used as final verdict; agreement rate is reported.",
    )
    args = parser.parse_args()

    if args.compare:
        build_comparison_table()
        return

    if not args.no_generate:
        generate_scripts()

    models_to_run = list(MODELS) if args.model == "all" else [args.model]
    for model_key in models_to_run:
        run_model(model_key, with_taxonomy=args.with_taxonomy, n_runs=args.runs)

    # Write models_used.json — canonical record of which models ran
    models_used = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "self_review_included": True,
        "models": [
            {
                "provider": MODELS[k]["model_id"].split("/")[0] if "/" in MODELS[k]["model_id"] else k,
                "model_id": MODELS[k]["model_id"],
                "display_name": MODELS[k]["display_name"],
                "with_taxonomy": args.with_taxonomy,
                "n_runs": args.runs,
            }
            for k in models_to_run
        ],
    }
    with open("models_used.json", "w") as f:
        json.dump(models_used, f, indent=2)
    print(f"\nModels used recorded in models_used.json")

    if len(models_to_run) > 1 and not args.with_taxonomy:
        build_comparison_table()


if __name__ == "__main__":
    main()
