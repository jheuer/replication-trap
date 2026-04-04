#!/usr/bin/env python3
"""
score_audit.py

Reads the agent's reviews from audit_results/reviews.json,
scores them against ground truth, and produces audit_results/audit_report.md.
"""

import json
import os
import random
import sys

RESULTS_DIR = "audit_results"
REVIEWS_PATH = os.path.join(RESULTS_DIR, "reviews.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "audit_report.md")
ANSWER_KEY_PATH = os.path.join("audit_answer_key", "answer_key.json")


def load_ground_truth():
    if not os.path.exists(ANSWER_KEY_PATH):
        print(f"ERROR: {ANSWER_KEY_PATH} not found. Run generate_audit_scripts.py first.")
        sys.exit(1)
    with open(ANSWER_KEY_PATH) as f:
        raw = json.load(f)
    # Convert to the format the scorer expects
    ground_truth = {}
    for script_name, info in raw.items():
        ground_truth[script_name] = {
            "is_flawed": info["is_flawed"],
            "category": info["category"],
            "description": info["description"],
        }
    return ground_truth

FLAW_CATEGORIES = [
    "Data leakage",
    "P-hacking",
    "Circular validation",
    "Survivorship bias",
    "Wrong test assumption",
    "Pseudoreplication",
]


def load_reviews():
    if not os.path.exists(REVIEWS_PATH):
        print(f"ERROR: {REVIEWS_PATH} not found.")
        print("Please complete Step 2 of the SKILL.md first.")
        sys.exit(1)

    with open(REVIEWS_PATH) as f:
        data = json.load(f)

    return data.get("reviews", [])


def score_reviews(reviews, ground_truth):
    """Score reviews against ground truth. Returns detailed results."""
    results = []
    for review in reviews:
        script = review["script"]
        verdict = review.get("verdict", "").upper().strip()
        confidence = review.get("confidence", "UNKNOWN").upper().strip()
        flaw_desc = review.get("flaw_identified", None)

        if script not in ground_truth:
            print(f"WARNING: Unknown script '{script}', skipping.")
            continue

        truth = ground_truth[script]
        agent_says_flawed = verdict == "FAIL"
        actually_flawed = truth["is_flawed"]

        if actually_flawed and agent_says_flawed:
            outcome = "TP"  # True positive: correctly identified flaw
        elif actually_flawed and not agent_says_flawed:
            outcome = "FN"  # False negative: missed the flaw
        elif not actually_flawed and agent_says_flawed:
            outcome = "FP"  # False positive: flagged correct code
        else:
            outcome = "TN"  # True negative: correctly passed clean code

        results.append({
            "script": script,
            "category": truth["category"],
            "is_flawed": actually_flawed,
            "verdict": verdict,
            "outcome": outcome,
            "confidence": confidence,
            "flaw_identified": flaw_desc,
            "ground_truth_desc": truth["description"],
        })

    return results


def category_level_analysis(results):
    """
    Primary analysis: category is the unit of observation, not individual script.
    A flaw category is 'detected' if the majority of its flawed variants were
    correctly flagged. A category has a false positive if the majority of its
    control variants were incorrectly flagged.
    """
    from collections import defaultdict
    cats = defaultdict(lambda: {"flawed": [], "controls": []})
    for r in results:
        if r["is_flawed"]:
            cats[r["category"]]["flawed"].append(r["outcome"])
        else:
            cats[r["category"]]["controls"].append(r["outcome"])

    cat_results = {}
    for cat, data in cats.items():
        n_flawed = len(data["flawed"])
        n_tp = sum(1 for o in data["flawed"] if o == "TP")
        n_controls = len(data["controls"])
        n_fp = sum(1 for o in data["controls"] if o == "FP")

        # Majority rule: detected if >50% of flawed variants caught
        detected = (n_tp / n_flawed) > 0.5 if n_flawed else False
        # FP problem if >50% of controls incorrectly flagged
        fp_problem = (n_fp / n_controls) > 0.5 if n_controls else False

        cat_results[cat] = {
            "detected": detected,
            "fp_problem": fp_problem,
            "tp": n_tp, "fn": n_flawed - n_tp,
            "fp": n_fp, "tn": n_controls - n_fp,
            "n_flawed": n_flawed, "n_controls": n_controls,
            "detection_rate": n_tp / n_flawed if n_flawed else 0,
            "fp_rate": n_fp / n_controls if n_controls else 0,
        }

    n_cats = len(cat_results)
    n_detected = sum(1 for c in cat_results.values() if c["detected"])
    n_fp_cats = sum(1 for c in cat_results.values() if c["fp_problem"])

    return {
        "by_category": cat_results,
        "n_categories": n_cats,
        "categories_detected": n_detected,
        "categories_with_fp": n_fp_cats,
        "category_detection_rate": n_detected / n_cats if n_cats else 0,
        "category_fp_rate": n_fp_cats / n_cats if n_cats else 0,
    }


def calibration_analysis(results):
    """
    Are high-confidence verdicts actually correct more often?
    Returns accuracy broken down by confidence tier.
    """
    tiers = {"LOW": [], "MEDIUM": [], "HIGH": [], "UNKNOWN": []}
    for r in results:
        correct = r["outcome"] in ("TP", "TN")
        tier = r.get("confidence", "UNKNOWN").upper()
        if tier not in tiers:
            tier = "UNKNOWN"
        tiers[tier].append(correct)

    out = {}
    for tier, outcomes in tiers.items():
        if outcomes:
            out[tier] = {
                "n": len(outcomes),
                "correct": sum(outcomes),
                "accuracy": sum(outcomes) / len(outcomes),
            }
    return out


def within_category_consistency(results):
    """
    For each flaw category, how many of the N flawed variants were detected?
    High consistency = model understands the flaw concept.
    Low consistency = model sensitive to surface implementation details.
    """
    from collections import defaultdict
    cat_flawed = defaultdict(list)
    cat_controls = defaultdict(list)
    for r in results:
        if r["is_flawed"]:
            cat_flawed[r["category"]].append(r["outcome"] == "TP")
        else:
            cat_controls[r["category"]].append(r["outcome"] == "FP")

    out = {}
    for cat in sorted(cat_flawed):
        flawed = cat_flawed[cat]
        controls = cat_controls.get(cat, [])
        out[cat] = {
            "variants_detected": sum(flawed),
            "variants_total": len(flawed),
            "detection_consistency": sum(flawed) / len(flawed) if flawed else 0,
            "fp_count": sum(controls),
            "controls_total": len(controls),
        }
    return out


def cluster_bootstrap_ci(results, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Bootstrap CIs using category as the resampling unit (cluster bootstrap).
    Correctly accounts for within-category correlation across script variants.
    """
    from collections import defaultdict
    rng = random.Random(seed)
    alpha = (1 - ci) / 2

    # Group results by category
    cat_groups = defaultdict(list)
    for r in results:
        cat_groups[r["category"]].append(r)
    categories = list(cat_groups.keys())
    n_cats = len(categories)

    det_boot, fp_boot, acc_boot, f1_boot = [], [], [], []

    for _ in range(n_bootstrap):
        # Resample categories with replacement
        sampled_cats = [categories[rng.randint(0, n_cats - 1)] for _ in range(n_cats)]
        sample = [r for cat in sampled_cats for r in cat_groups[cat]]

        tp = sum(1 for r in sample if r["outcome"] == "TP")
        fp = sum(1 for r in sample if r["outcome"] == "FP")
        tn = sum(1 for r in sample if r["outcome"] == "TN")
        fn = sum(1 for r in sample if r["outcome"] == "FN")
        n_f = tp + fn
        n_c = fp + tn

        det_boot.append(tp / n_f if n_f else 0)
        fp_boot.append(fp / n_c if n_c else 0)
        acc_boot.append((tp + tn) / len(sample) if sample else 0)
        f1_boot.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0)

    def pct(vals):
        s = sorted(vals)
        return s[int(alpha * n_bootstrap)], s[int((1 - alpha) * n_bootstrap) - 1]

    return {
        "detection_rate": pct(det_boot),
        "false_positive_rate": pct(fp_boot),
        "accuracy": pct(acc_boot),
        "f1": pct(f1_boot),
    }


def bootstrap_ci(results, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence intervals for detection rate, FP rate, accuracy, F1."""
    rng = random.Random(seed)
    n = len(results)
    alpha = (1 - ci) / 2

    acc_boot, det_boot, fp_boot, f1_boot = [], [], [], []

    for _ in range(n_bootstrap):
        sample = [results[rng.randint(0, n - 1)] for _ in range(n)]
        tp = sum(1 for r in sample if r["outcome"] == "TP")
        fp = sum(1 for r in sample if r["outcome"] == "FP")
        tn = sum(1 for r in sample if r["outcome"] == "TN")
        fn = sum(1 for r in sample if r["outcome"] == "FN")
        n_f = tp + fn
        n_c = fp + tn
        acc_boot.append((tp + tn) / n if n else 0)
        det_boot.append(tp / n_f if n_f else 0)
        fp_boot.append(fp / n_c if n_c else 0)
        f1_boot.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0)

    def pct(vals):
        s = sorted(vals)
        lo = s[int(alpha * n_bootstrap)]
        hi = s[int((1 - alpha) * n_bootstrap) - 1]
        return lo, hi

    return {
        "accuracy":           pct(acc_boot),
        "detection_rate":     pct(det_boot),
        "false_positive_rate": pct(fp_boot),
        "f1":                 pct(f1_boot),
    }


def generate_report(results):
    """Generate the audit report as Markdown."""
    tp = sum(1 for r in results if r["outcome"] == "TP")
    fp = sum(1 for r in results if r["outcome"] == "FP")
    tn = sum(1 for r in results if r["outcome"] == "TN")
    fn = sum(1 for r in results if r["outcome"] == "FN")

    total = len(results)
    correct = tp + tn
    accuracy = correct / total if total > 0 else 0

    n_flawed = tp + fn
    n_controls = tn + fp
    detection_rate = tp / n_flawed if n_flawed > 0 else 0
    false_positive_rate = fp / n_controls if n_controls > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    cat = category_level_analysis(results)
    cci = cluster_bootstrap_ci(results)
    sci = bootstrap_ci(results)

    def fmt_ci(lo, hi, pct=True):
        if pct:
            return f"[{lo:.1%}, {hi:.1%}]"
        return f"[{lo:.3f}, {hi:.3f}]"

    lines = []
    lines.append("# Adversarial Skill Audit — Results Report")
    lines.append("")

    # --- Primary analysis: category level ---
    lines.append("## Primary Analysis — Flaw Category Detection")
    lines.append("")
    lines.append("> Unit of analysis: flaw category (n=6). A category is 'detected' if")
    lines.append("> the majority of its flawed variants were correctly flagged.")
    lines.append("> CIs use cluster bootstrap (resampling categories, not individual scripts)")
    lines.append("> to account for within-category correlation across variants.")
    lines.append("")
    lines.append(f"| Metric | Value | 95% Cluster Bootstrap CI |")
    lines.append(f"|--------|-------|--------------------------|")
    lines.append(f"| Categories evaluated | {cat['n_categories']} | — |")
    lines.append(f"| Categories detected | {cat['categories_detected']}/{cat['n_categories']} ({cat['category_detection_rate']:.1%}) | {fmt_ci(*cci['detection_rate'])} |")
    lines.append(f"| Categories with FP | {cat['categories_with_fp']}/{cat['n_categories']} ({cat['category_fp_rate']:.1%}) | {fmt_ci(*cci['false_positive_rate'])} |")
    lines.append("")

    # Per-category breakdown
    lines.append("### Detection by Category")
    lines.append("")
    lines.append("| Flaw Category | Variants (TP/total) | Detected? | Controls (FP/total) | FP problem? |")
    lines.append("|---------------|---------------------|-----------|---------------------|-------------|")
    for cat_name in sorted(cat["by_category"]):
        c = cat["by_category"][cat_name]
        det_mark = "✓" if c["detected"] else "✗"
        fp_mark = "✓" if c["fp_problem"] else "—"
        lines.append(
            f"| {cat_name} | {c['tp']}/{c['n_flawed']} ({c['detection_rate']:.0%}) "
            f"| {det_mark} | {c['fp']}/{c['n_controls']} ({c['fp_rate']:.0%}) | {fp_mark} |"
        )
    lines.append("")

    # --- Secondary analysis: script level ---
    lines.append("## Secondary Analysis — Script-Level Statistics")
    lines.append("")
    lines.append("> Script-level metrics treat each of the 36 scripts as an observation.")
    lines.append("> Note: scripts are nested within flaw categories (3 variants each),")
    lines.append("> so these figures are correlated. The category-level analysis above")
    lines.append("> is the primary inferential measure.")
    lines.append("")
    lines.append(f"| Metric | Value | 95% Bootstrap CI |")
    lines.append(f"|--------|-------|-----------------|")
    lines.append(f"| Scripts reviewed | {total} | — |")
    lines.append(f"| Overall accuracy | {correct}/{total} ({accuracy:.1%}) | {fmt_ci(*sci['accuracy'])} |")
    lines.append(f"| Detection rate | {tp}/{n_flawed} ({detection_rate:.1%}) | {fmt_ci(*sci['detection_rate'])} |")
    lines.append(f"| False positive rate | {fp}/{n_controls} ({false_positive_rate:.1%}) | {fmt_ci(*sci['false_positive_rate'])} |")
    lines.append(f"| Precision | {precision:.1%} | — |")
    lines.append(f"| F1 score | {f1:.3f} | {fmt_ci(*sci['f1'], pct=False)} |")
    lines.append("")

    # Confusion matrix
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("| | Agent: FAIL | Agent: PASS |")
    lines.append("|---|---|---|")
    lines.append(f"| Actually flawed | TP = {tp} | FN = {fn} |")
    lines.append(f"| Actually correct | FP = {fp} | TN = {tn} |")
    lines.append("")

    # Within-category consistency
    consistency = within_category_consistency(results)
    lines.append("## Within-Category Detection Consistency")
    lines.append("")
    lines.append("> How many of each category's N variants were detected?")
    lines.append("> Low consistency suggests detection is sensitive to surface implementation details.")
    lines.append("")
    lines.append("| Flaw Category | Detected | Consistency | FP count |")
    lines.append("|---------------|----------|-------------|----------|")
    for cat_name, c in consistency.items():
        bar = "█" * c["variants_detected"] + "░" * (c["variants_total"] - c["variants_detected"])
        lines.append(
            f"| {cat_name} | {c['variants_detected']}/{c['variants_total']} {bar}"
            f" | {c['detection_consistency']:.0%} | {c['fp_count']}/{c['controls_total']} |"
        )
    lines.append("")

    # Confidence calibration
    cal = calibration_analysis(results)
    lines.append("## Confidence Calibration")
    lines.append("")
    lines.append("> Is the model's stated confidence predictive of accuracy?")
    lines.append("")
    lines.append("| Confidence | N | Correct | Accuracy |")
    lines.append("|------------|---|---------|----------|")
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        if tier in cal:
            c = cal[tier]
            lines.append(f"| {tier} | {c['n']} | {c['correct']} | {c['accuracy']:.1%} |")
    lines.append("")

    # False positives detail
    fps = [r for r in results if r["outcome"] == "FP"]
    if fps:
        lines.append("## False Positives (Correct Scripts Flagged as Flawed)")
        lines.append("")
        for r in fps:
            lines.append(f"- **{r['script']}** ({r['category']}): "
                        f"Agent said: {r['flaw_identified']}")
        lines.append("")

    # Missed flaws detail
    fns = [r for r in results if r["outcome"] == "FN"]
    if fns:
        lines.append("## Missed Flaws (Flawed Scripts Passed as Correct)")
        lines.append("")
        for r in fns:
            lines.append(f"- **{r['script']}** ({r['category']}): "
                        f"Actual flaw: {r['ground_truth_desc']}")
        lines.append("")

    # Raw data
    lines.append("## Detailed Review Log")
    lines.append("")
    lines.append("| Script | Flawed? | Verdict | Outcome | Confidence |")
    lines.append("|--------|---------|---------|---------|------------|")
    for r in sorted(results, key=lambda x: x["script"]):
        flawed = "Yes" if r["is_flawed"] else "No"
        lines.append(
            f"| {r['script']} | {flawed} | {r['verdict']} | "
            f"{r['outcome']} | {r['confidence']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by score_audit.py — "
                "Adversarial Skill Audit, Claw4S 2026*")

    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ground_truth = load_ground_truth()
    reviews = load_reviews()

    if len(reviews) < len(ground_truth):
        print(f"WARNING: Only {len(reviews)} reviews found (expected {len(ground_truth)}).")

    results = score_reviews(reviews, ground_truth)
    report = generate_report(results)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    # Also print summary to stdout
    tp = sum(1 for r in results if r["outcome"] == "TP")
    fp = sum(1 for r in results if r["outcome"] == "FP")
    tn = sum(1 for r in results if r["outcome"] == "TN")
    fn = sum(1 for r in results if r["outcome"] == "FN")
    total = len(results)
    correct = tp + tn

    cat = category_level_analysis(results)
    cci = cluster_bootstrap_ci(results)
    sci = bootstrap_ci(results)

    print(f"\n{'='*50}")
    print(f"ADVERSARIAL SKILL AUDIT — RESULTS")
    print(f"{'='*50}")
    print(f"PRIMARY (category-level, n={cat['n_categories']} flaw types, cluster bootstrap CI):")
    print(f"  Categories detected:  {cat['categories_detected']}/{cat['n_categories']} ({cat['category_detection_rate']:.1%})"
          f"  95% CI [{cci['detection_rate'][0]:.1%}, {cci['detection_rate'][1]:.1%}]")
    print(f"  Categories with FP:   {cat['categories_with_fp']}/{cat['n_categories']} ({cat['category_fp_rate']:.1%})"
          f"  95% CI [{cci['false_positive_rate'][0]:.1%}, {cci['false_positive_rate'][1]:.1%}]")
    print(f"\nSECONDARY (script-level, n={total} scripts, naive bootstrap CI):")
    print(f"  Overall accuracy:     {correct}/{total} ({correct/total:.1%})"
          f"  95% CI [{sci['accuracy'][0]:.1%}, {sci['accuracy'][1]:.1%}]")
    if (tp+fn) > 0:
        print(f"  Detection rate:       {tp}/{tp+fn} ({tp/(tp+fn):.1%})"
              f"  95% CI [{sci['detection_rate'][0]:.1%}, {sci['detection_rate'][1]:.1%}]")
    if (fp+tn) > 0:
        print(f"  False positive rate:  {fp}/{fp+tn} ({fp/(fp+tn):.1%})"
              f"  95% CI [{sci['false_positive_rate'][0]:.1%}, {sci['false_positive_rate'][1]:.1%}]")
    if (2*tp+fp+fn) > 0:
        print(f"  F1 score:             {2*tp/(2*tp+fp+fn):.3f}"
              f"  95% CI [{sci['f1'][0]:.3f}, {sci['f1'][1]:.3f}]")
    print(f"\nFull report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
