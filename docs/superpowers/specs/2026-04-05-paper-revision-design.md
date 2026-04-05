# Paper Revision Design: research_note.tex
**Date:** 2026-04-05
**Scope:** Incorporate 5-model evaluation data and restructure Section 5

---

## Context

The original paper evaluated 3 models (Claude Sonnet 4.6, Claude Opus 4.6, GPT-4o) under 4
conditions. Since submission, we added Gemini 2.5 Flash and Gemini 3 Flash as baseline runs,
then extended the taxonomy experiment to GPT-4o, Gemini 3, and Opus. The full dataset is now
10 conditions across 5 models.

**New data summary:**

| Condition | Detection | FPR | F1 |
|---|---|---|---|
| Claude Opus 4.6 | 94.4% | 22.2% | 0.872 |
| Claude Opus 4.6 + taxonomy | 100.0% | 22.2% | 0.900 |
| Claude Sonnet 4.6 | 100.0% | 38.9% | 0.837 |
| Claude Sonnet 4.6 + taxonomy | 100.0% | 27.8% | 0.878 |
| Claude Sonnet 4.6 (3-run majority) | 100.0% | 50.0% | 0.800 |
| Gemini 2.5 Flash | 100.0% | 50.0% | 0.800 |
| Gemini 3 Flash | 100.0% | 27.8% | 0.878 |
| Gemini 3 Flash + taxonomy | 100.0% | 33.3% | 0.857 |
| GPT-4o | 100.0% | 38.9% | 0.837 |
| GPT-4o + taxonomy | 94.4% | 16.7% | 0.895 |

---

## Changes by Section

### Abstract
- "three frontier LLMs under four experimental conditions" → "five frontier LLMs across nine
  conditions" (excluding the 3-run repeat)
- Update key findings sentence on taxonomy: "reduces FPR for models with elevated baseline FPR
  (GPT-4o: 39%→17%, Sonnet: 39%→28%), with neutral effect for already-precise models (Opus
  unchanged at 22%) and a slight increase for Gemini 3 (28%→33%, noted as a caveat)"
- Update FPR range: "22% to 50%" → "17% to 50%" (GPT-4o+taxonomy is the new best)

### Section 4 — Experimental Setup
- Expand models list to 5 models (add Gemini 2.5 Flash and Gemini 3 Flash with their LiteLLM
  model IDs)
- Expand conditions table: add rows for Gemini 2.5 Flash baseline, Gemini 3 Flash baseline,
  Gemini 3+taxonomy, GPT-4o+taxonomy, Opus+taxonomy
- Reliability row stays in the table but is annotated as Sonnet-only

### Section 5 — Results (RESTRUCTURED)

**5.1 — Detection (near-ceiling)**
- Retains the primary category-level analysis framing
- Table 1 becomes a 9-row table (all conditions except 3-run; that goes to Limitations)
  - Columns: Condition | Categories detected | Detection rate | FPR | F1
- Lead finding: detection at or near ceiling across all models and conditions; this dimension
  does not differentiate models
- Opus baseline (94.4%) and GPT-4o+taxonomy (94.4%) are the two conditions below 100%

**5.2 — False Positive Rate**
- Opens with FPR as the primary axis of variation (range: 17%–50%)
- Table 2: multi-model per-category breakdown
  - Rows = 6 flaw categories
  - Columns = 5 baseline models (Sonnet, Opus, GPT-4o, Gemini 2.5, Gemini 3)
  - Cells = TP/3 (variants caught) and FP/3 (controls flagged)
- Survivorship bias identified as the structural driver: uniformly high FPR across all models
- Sets up Section 6 (Survivorship Bias Pathology)

**5.3 — Taxonomy as a Calibration Intervention**
- Rewritten around 4-model taxonomy data (Sonnet, Opus, GPT-4o, Gemini 3)
- Primary claim: taxonomy reduces FPR for models with elevated baseline FPR
  - GPT-4o: 38.9% → 16.7% (largest effect)
  - Sonnet: 38.9% → 27.8%
- Neutral effect for already-precise models: Opus 22.2% → 22.2% (detection improves: 94%→100%)
- Caveat: Gemini 3 slight FPR increase (27.8% → 33.3%) — addressed in Limitations
- Retain contrast-hypothesis explanation: taxonomy provides a standard against which
  implementation can be compared, reducing over-flagging of scripts that engage with
  sensitive topics
- Soften the original claim ("the opposite occurred") to a conditional: "taxonomy reduced FPR
  for models where FPR was elevated at baseline"
- Note: bootstrap CIs overlap substantially; treat as directional trend not established effect

**5.4 — Confidence Calibration**
- Unchanged: all false positives are HIGH confidence across all models

### Section 9 — Limitations
- Add: Gemini 3 taxonomy anomaly — slight FPR increase despite taxonomy, mechanism unclear;
  one candidate explanation is a ceiling effect (model already well-calibrated, taxonomy
  introduces spurious pattern-matching); larger corpus needed to distinguish
- Replace Reliability subsection content with 2-sentence note: Sonnet 3-run majority-vote
  FPR (50%) exceeds single-run (39%), suggesting single-run estimates are lower bounds;
  single-model evidence; replication across models is future work

### Section 10 — Reproducibility
- Add run commands for new models/conditions:
  ```
  python3 run_evaluation.py --model gemini
  python3 run_evaluation.py --model gemini3
  python3 run_evaluation.py --model gemini3 --with-taxonomy
  python3 run_evaluation.py --model gpt4o --with-taxonomy
  python3 run_evaluation.py --model opus --with-taxonomy
  ```

### Section 11 — Discussion
- Update opening FPR range: 22%–50% → 17%–50%
- Recommendation 1: soften "include structured flaw taxonomies" to "for models with elevated
  baseline FPR; effect is neutral or potentially adverse for already-precise models"
- Remove the multi-run recommendation as a standalone item (now in Limitations)

### No changes
- Section 6 (Survivorship Bias Pathology)
- Section 7 (Meta-Finding)
- Section 8 (Related Work)
- Appendix (FP examples)

---

## Files Modified
- `research_note.tex` — primary target
- `submission_content.md` — keep in sync (markdown mirror)

## Out of Scope
- New experiments
- Changes to benchmark design or scoring
- Appendix FP examples (Sonnet-specific, still valid as illustration)
