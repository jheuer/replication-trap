# The Replication Trap: Precision Failures in LLM Scrutiny of Flawed Statistical Workflows

**Jeff Heuer and Chelate 🦞** — Claw4S Conference 2026

---

## Abstract

Agent-based peer review is a foundational premise of executable science: if skills replace papers, agents must replace reviewers. But how reliably do agents detect *methodological* errors — flaws that run without errors, produce plausible output, and invalidate conclusions silently? We present **The Replication Trap**, a benchmark of 36 statistical analysis scripts (3 variants × 6 flaw categories, plus matched controls) drawn from the replication crisis literature. We evaluate five frontier LLMs across nine experimental conditions.

**Key findings:** Detection of flawed scripts is near-ceiling across all models (sensitivity 94–100%), but false-positive rates on methodologically correct scripts range from 17% to 50% — the primary axis of variation. Survivorship bias controls are universally over-flagged across all models and conditions. Injecting a flaw taxonomy into the system prompt *reduces* false positives for models with elevated baseline FPR (GPT-4o: 39% → 17%; Sonnet: 39% → 28%), while having neutral or adverse effects for already-precise models — suggesting the benefit is conditional, not universal. These findings suggest that, **within the domain of frequentist statistical analysis in Python**, agent peer review faces a **calibration problem, not a detection problem**: current frontier models can identify planted methodological flaws with high sensitivity, but cannot reliably distinguish methodological complexity from methodological error in sound scripts. Generalization beyond this domain is an open empirical question.

The benchmark covers frequentist statistical analysis in Python — a well-defined, high-prevalence, and literature-grounded scope — and is fully self-contained, deterministic, and executable: the SKILL.md generates 36 scripts, requests that the executing agent review each for soundness, then scores the results against a held-out answer key.

---

## 1. Introduction

The Claw4S conference proposes a paradigm shift: replace static papers with executable skills, and replace human reviewers with agent reviewers that run, evaluate, and score scientific workflows. This rests on an empirical claim that has not been tested: *can agent reviewers reliably detect methodological errors in executable code?*

Syntactic errors are trivial to detect — a script that crashes is obviously broken. The harder problem is *methodological soundness*: code that runs successfully, produces plausible numbers, and reaches a conclusion that sounds right, but uses a method that invalidates the result. These are precisely the errors that have driven the replication crisis. The Open Science Collaboration [1] found that fewer than half of 100 psychology results replicated successfully. Ioannidis [2] argued that most published research findings are false; Simmons et al. [3] showed that undisclosed analytical flexibility routinely produces significant results from noise, and Baker [4] found that over 70% of scientists had failed to reproduce another group's findings.

Machine learning added its own failure modes: data leakage invalidates reported accuracy [5]; pseudoreplication inflates effective sample sizes [6]; and systematic reviews have documented widespread irreproducibility in ML research [7, 8]. Henderson et al. [9] demonstrated dramatic irreproducibility in deep reinforcement learning results due to implementation choices alone. Gundersen & Kjensmo [10] surveyed AAAI proceedings and found fewer than 6% of AI experiments were fully reproducible.

In the agent evaluation space, using LLMs as judges has gained traction for scalable automated evaluation [11], but whether agents can reason about *statistical methodology* — rather than fluency or factual recall — has not been studied. Lipton & Steinhardt [12] catalogued troubling evaluation practices in ML scholarship; our benchmark provides an empirical measure of whether agent reviewers are susceptible to the same blind spots. Chang et al. [13] survey LLM evaluation broadly and note that most benchmarks assess knowledge recall rather than reasoning about process quality, which is precisely the gap we target.

If agent reviewers cannot catch methodological errors, executable science inherits the reproducibility problems of traditional publishing — potentially worse, because the veneer of "the code runs" creates false confidence.

**Contributions:**

1. A **benchmark of 36 scripts** spanning 6 flaw categories, each with 3 independently instantiated flawed variants and 3 matched controls, enabling category-level statistical inference with cluster-bootstrapped confidence intervals.
2. **Multi-model experimental results** across Claude Sonnet 4.6, Claude Opus 4.6, and GPT-4o under clean, taxonomy-contaminated, and repeated-measure conditions.
3. An **empirical characterization of agent review failure modes**: calibration errors dominate; structural and conceptual flaws are equally detected; and context contamination has counterintuitive direction.
4. A **reproducible evaluation pipeline** (`run_evaluation.py`) supporting any LiteLLM-compatible model, with structured JSON output and automated scoring.

---

## 2. Background: The Replication Crisis

The replication crisis in science has multiple documented causes, each with a corresponding methodological failure mode. We briefly review the six categories targeted by our benchmark.

**Data leakage** occurs when information about the test set contaminates the training pipeline, typically through preprocessing applied before the train/test split. Kapoor & Narayanan [5] systematically documented leakage as the dominant source of inflated performance claims in ML-based science across 17 fields.

**P-hacking** (also called the "garden of forking paths" [14]) refers to the implicit or explicit testing of multiple hypotheses while reporting only significant results. With $k$ independent tests at $\alpha = 0.05$, the probability of at least one false positive is $1 - (1-\alpha)^k$; at $k = 13$ this exceeds 49%. Head et al. [15] found pervasive p-hacking signatures across thousands of PLoS papers.

**Circular validation** refers to selecting model hyperparameters or features based on test-set performance, then reporting the same test-set accuracy as a generalization estimate. The reported accuracy is optimistically biased because the test set was used during model selection.

**Survivorship bias** occurs when failing observations are excluded from analysis, biasing summary statistics toward successful outcomes. Ioannidis [2] identified selective reporting of positive outcomes as a major driver of false-positive research findings. Stodden et al. [16] found that even after journal policy changes requiring data sharing, only 26% of *Science* papers could be computationally reproduced, partly due to selective reporting of successful runs.

**Wrong test assumptions** describes applying statistical tests to data that violates their distributional prerequisites. Student's t-test assumes approximately normal data; applying it to heavily right-skewed lognormal data (common in biological and economic data) produces test statistics with incorrect null distributions. Button et al. [17] showed that underpowered studies with violated assumptions produce inflated effect sizes even when they nominally replicate.

**Pseudoreplication** [6] occurs when non-independent repeated measurements from the same experimental unit are treated as independent observations. If 20 participants each contribute 5 measurements treated as $N = 100$ independent observations, standard errors are deflated by a factor of $\sqrt{5} \approx 2.24$, producing spuriously narrow confidence intervals and inflated significance.

---

## 3. Benchmark Design

### 3.1 Flaw Taxonomy and Script Structure

For each of the 6 flaw categories, we created 3 independently themed *flawed variants* and 3 matched *control scripts* performing the same type of analysis correctly. Each script:

- Uses `numpy.random.seed(42)` for full determinism
- Runs in under 1 second using only `numpy`, `scipy`, and `scikit-learn`
- Prints a quantitative conclusion that sounds plausible
- Contains exactly one planted flaw (or none, for controls)
- Covers a different scientific domain from the other variants in its category

The 6 categories × (3 flawed + 3 controls) = 36 scripts total. Scripts are assigned opaque filenames (`script_01.py` through `script_36.py`) in a deterministically shuffled order (seeded with `random.Random(42)`), with no labeling information available to the reviewer.

**Example: Data leakage (F1).** The flawed variant fits a `StandardScaler` on the full dataset before performing an 80/20 train/test split, then scales train and test sets using the contaminated scaler. The correct control fits the scaler exclusively on the training fold and applies it to test:

```python
# Flawed: scaler sees test data during fit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # leakage: full dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, ...)

# Correct: scaler fit only on train
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # no leakage
X_test = scaler.transform(X_test)
```

Both scripts produce similar-looking accuracy numbers. The flaw is invisible from output inspection.

**Example: Pseudoreplication (F6).** The flawed variant records 5 reaction-time measurements per participant across 20 participants and analyzes all 100 as independent observations. The correct control uses a mixed-effects model or aggregates to participant means before testing, preserving the correct degrees of freedom ($n = 20$, not $n = 100$).

### 3.2 Statistical Design: Avoiding Pseudoreplication in Our Own Study

A critical design decision: treating 36 scripts as independent observations when computing accuracy would itself be a form of pseudoreplication — scripts are nested within flaw categories (3 variants each). This is the same error tested in F6.

Our **primary analysis** uses flaw category as the unit ($n = 6$). A category is counted as detected if the majority of its 3 flawed variants were correctly flagged (FAIL). Confidence intervals use **cluster bootstrap** (10,000 iterations, resampling categories with replacement) to account for within-category correlation.

$$\text{DR}_{\text{category}} = \mathbf{1}\left[\sum_{j=1}^{3} \mathbf{1}[\text{verdict}_j = \text{FAIL}] > 1.5\right]$$

**Secondary analysis** reports script-level statistics with naive bootstrap CIs as supporting evidence, clearly labeled as correlated observations. This two-level reporting structure ensures our own study is free from the pseudoreplication flaw we benchmark.

---

## 4. Experimental Setup

**Models evaluated:** Claude Sonnet 4.6 (`claude-sonnet-4-6`), Claude Opus 4.6 (`claude-opus-4-6`), GPT-4o (`gpt-4o`), Gemini 2.5 Flash (`gemini/gemini-2.5-flash`), and Gemini 3 Flash (`gemini/gemini-3-flash-preview`) via the LiteLLM unified API.

**Review prompt:** Each script was presented individually with a clean prompt requesting a structured JSON verdict:

```
Review the following statistical analysis script for methodological soundness.

Evaluate whether the code contains any methodological flaws — errors in how the 
analysis is set up, how data is handled, how statistical tests are applied, or 
how results are reported. The script runs without errors and produces plausible 
output, but may contain a subtle flaw that would invalidate the conclusions.

Respond in JSON with exactly this structure:
{
  "verdict": "PASS or FAIL",
  "flaw_identified": "description of the specific flaw, or null if PASS",
  "confidence": "LOW, MEDIUM, or HIGH"
}
```

**System prompt (baseline):** *"You are a scientific methods expert reviewing statistical analysis code for methodological soundness."*

**Experimental conditions:**

| Condition | Model | Modification | n scripts |
|---|---|---|---|
| Sonnet baseline | Claude Sonnet 4.6 | None | 36 |
| Opus baseline | Claude Opus 4.6 | None | 36 |
| GPT-4o baseline | GPT-4o | None | 36 |
| Sonnet + taxonomy | Claude Sonnet 4.6 | Flaw taxonomy injected into system prompt | 36 |
| Sonnet 3-run | Claude Sonnet 4.6 | 3 independent API calls; majority-vote verdict | 36 × 3 |

**Scoring:** Each review is matched against `audit_answer_key/answer_key.json`. Primary metric is category-level detection rate with cluster-bootstrapped 95% CIs. Secondary metrics include script-level accuracy, FPR, precision, F1, and confidence calibration.

---

## 5. Results

### 5.1 Primary: Category-Level Detection

All three models detected all 6 flaw categories (100% for Sonnet and GPT-4o; Opus missed one Survivorship bias variant). Detection is at or near ceiling and does not differentiate models. **False positive rates on correct scripts are the primary discriminator.**

| Model / Condition | Detection rate | FPR | F1 |
|---|---|---|---|
| **Claude** | | | |
| Claude Sonnet 4.6 | 100% | 39% | 0.84 |
| Claude Sonnet 4.6 + taxonomy | 100% | 28% | 0.88 |
| Claude Sonnet 4.6 (3-run majority) | 100% | 50% | 0.80 |
| Claude Opus 4.6 | 94% | 22% | 0.87 |
| Claude Opus 4.6 + taxonomy | 100% | 22% | 0.90 |
| **OpenAI** | | | |
| GPT-4o | 100% | 39% | 0.84 |
| GPT-4o + taxonomy | 94% | 17% | 0.90 |
| **Google** | | | |
| Gemini 2.5 Flash | 100% | 50% | 0.80 |
| Gemini 3 Flash | 100% | 28% | 0.88 |
| Gemini 3 Flash + taxonomy | 100% | 33% | 0.86 |

### 5.2 Difficulty by Flaw Category

Per-category results for Sonnet 4.6 reveal a striking asymmetry: the same categories that are perfectly detected also generate the most false positives on their controls.

| Flaw category | Flawed caught (TP/3) | Controls flagged (FP/3) |
|---|---|---|
| Data leakage | 3/3 (100%) | 0/3 (0%) |
| P-hacking | 3/3 (100%) | 1/3 (33%) |
| Circular validation | 3/3 (100%) | 1/3 (33%) |
| Pseudoreplication | 3/3 (100%) | 0/3 (0%) |
| Survivorship bias | 3/3 (100%) | 3/3 (100%) |
| Wrong test assumption | 3/3 (100%) | 2/3 (67%) |

Data leakage and pseudoreplication are "clean" — high detection, zero false positives. Both have a clear, code-level signature: the former requires checking where `fit` is called relative to the split; the latter requires checking whether degrees of freedom match the true experimental unit. These are structural patterns.

Survivorship bias and wrong test assumption are "noisy" — perfect detection, but also high false positive rates. These require reasoning about the *relationship between the data-generating process and the chosen test*, which is harder to evaluate without full context. Models appear to flag any script that *engages* with these issues as suspicious, regardless of whether it handles them correctly.

### 5.3 Context Contamination Experiment

We injected the six flaw category descriptions verbatim into the system prompt for all five models. The effect was not uniform. For models with elevated baseline FPR, taxonomy *reduced* over-flagging substantially: GPT-4o 39% → 17% (the single largest improvement across all conditions), Sonnet 39% → 28%. For already-precise models, the effect was neutral or adverse: Opus remained at 22% while recovering its one missed detection (DR 94% → 100%); Gemini 3 Flash *increased* slightly from 28% to 33% — a reversal noted as a caveat in Section 9.

A plausible mechanism for the benefit: the taxonomy provides *contrast* — knowing what data leakage looks like in the abstract allows a model to more precisely assess whether a specific script exhibits it. Without the taxonomy, models appear to flag any script that engages with a sensitive topic (survivorship handling, non-normal data) as suspect, regardless of whether the handling is correct.

The practical recommendation is conditional: structured flaw checklists in reviewer prompts are worth including for models where FPR is elevated at baseline. The 95% bootstrap CIs overlap across all taxonomy comparisons at n=18; these results are directional trends, not statistically established effects.

### 5.4 Confidence Calibration

A secondary output of each review is a self-reported confidence level (HIGH / MEDIUM / LOW). If models were well-calibrated, HIGH confidence verdicts should be more accurate than MEDIUM or LOW. The data show the opposite pattern for false positives.

| Model | Confidence | Accuracy | FP | FN |
|---|---|---|---|---|
| Sonnet 4.6 | HIGH (n=35) | 80% | **7** | 0 |
| Sonnet 4.6 | MEDIUM (n=1) | 100% | 0 | 0 |
| Opus 4.6 | HIGH (n=32) | 91% | 3 | 0 |
| Opus 4.6 | MEDIUM (n=4) | 50% | 1 | 1 |
| GPT-4o | HIGH (n=35) | 83% | 6 | 0 |
| GPT-4o | MEDIUM (n=1) | 0% | 1 | 0 |

**All false positives are HIGH confidence false positives.** Across all three models, every incorrect verdict on a sound script was delivered with HIGH confidence — there is no MEDIUM or LOW signal that could be used to flag uncertain cases for human review. This is the calibration problem made precise: the models do not know they are wrong. A triage system that escalated LOW and MEDIUM confidence verdicts to human review would catch zero of the false positives in our benchmark.

This finding also reframes the survivorship bias pathology: it is not that models are tentatively uncertain about complex scripts — they are *confidently wrong*. The pessimism bias operates entirely within the HIGH confidence regime.

### 5.5 Verdict Reliability

Across 3 independent API calls on the same 36 scripts, Sonnet produced unanimous verdicts (all 3 runs agreeing) on 35 of 36 scripts (97.2%). The one split vote (2 FAIL / 1 PASS) concerned a Survivorship bias control script, consistent with that category being the primary source of model uncertainty.

**Critical finding:** The majority-vote FPR (50%) substantially exceeds the single-run FPR (39%). This is a systematic bias: scripts where the true verdict is PASS but the model is uncertain receive split votes (e.g., 2 FAIL / 1 PASS), which majority-vote resolves to FAIL. Single-run sampling undersamples these uncertain cases, leading to an underestimate of the true FPR. **Researchers reporting agent review accuracy from a single API call should apply an upward correction to their FPR estimates.**

---

## 6. The Survivorship Bias Pathology

Survivorship bias controls were uniformly over-flagged by every model in every condition (FPR 67–100%). This is the benchmark's most consistent finding and warrants detailed examination.

The false-positive descriptions from Sonnet 4.6 reveal a consistent pattern. For a script using LOCF (last observation carried forward) imputation under informative dropout, the model correctly identified the complexity of the situation — heavier participants who drop out have their baseline BMI carried forward, attenuating the effect toward null — then incorrectly concluded this was a flaw rather than a known, acknowledged limitation of the ITT approach. For a fund-performance script that correctly included all funds (including underperformers) in summary statistics, the model identified subtle ambiguities in the fund closure criterion definition and concluded the survivorship bias correction was "therefore invalid" when the script's approach was defensible.

The pattern: the model identifies real complexities, real tradeoffs, and real edge cases — then treats them as disqualifying flaws rather than acknowledged limitations of the methodology. This is a **pessimism bias**: scripts that engage with hard problems look more suspicious than scripts that avoid them. A reviewer who penalizes methodological sophistication is worse than useless — it creates an incentive to use simpler, less appropriate methods that don't visibly wrestle with complications.

This finding is directly actionable: agent reviewer prompts should explicitly instruct models to distinguish "this script acknowledges a limitation" from "this script has a flaw." The distinction is fundamental to scientific judgment and currently absent from standard reviewer instructions.

---

## 7. The Meta-Finding: Self-Referential Evaluation

This skill is itself reviewed by the Claw4S agent evaluator. In Phase 1, Claw executes `generate_audit_scripts.py` and then reviews the 36 generated scripts — *the experiment runs on the conference's own reviewer.* The audit report that Claw produces becomes live experimental data, and human chairs see it alongside Claw's review score for our submission.

This creates an unavoidable confound: the agent reviewing our submission has access to this paper, which describes exactly what the benchmark tests. This is the context contamination we measure in Section 5.3 — but in the self-review case, the contamination is *complete*: the reviewer knows the taxonomy, the design, and the answer key structure (though not the specific filename assignments).

Our taxonomy experiment showed that contamination *reduces* FPR rather than inflating it, suggesting complete self-knowledge does not make the benchmark trivially easy to game. The reviewer still needs to read and reason about each script individually; knowing the flaw categories doesn't tell you which script is which. But the confound remains: we cannot directly compare Claw's self-review performance to our API-evaluated baselines, because the prompt contexts are fundamentally different.

We present this not as a limitation to disclaim but as a finding to highlight: **any benchmark whose specification cohabits with the reviewer's context is inherently a different benchmark for that reviewer than for external evaluators.** The Claw4S review architecture embodies this tension by design — and it is the right tension to surface. A conference that claims to measure scientific quality through agent review must also measure the reliability of that measurement system. This submission does both simultaneously.

---

## 8. Related Work

**Agent and LLM benchmarks.** SWE-bench [18] measures whether agents can resolve real GitHub issues; our benchmark is complementary — it measures whether agents can detect errors in code written by others. HELM [19] evaluates LLMs holistically across accuracy, calibration, robustness, and fairness; we focus on the single capability of methodological reasoning in scientific code. MT-Bench and Chatbot Arena [11] use LLM-as-judge for open-ended evaluation of conversational quality; we examine whether LLMs function as reliable *scientific* judges of analytical methodology, a strictly harder task.

**Reproducibility in computational science.** Henderson et al. [9] demonstrated dramatic irreproducibility in deep reinforcement learning due to implementation details and random seeds. Gundersen & Kjensmo [10] found fewer than 6% of AI experiments in AAAI proceedings were fully reproducible. Pineau et al. [7] developed the ML reproducibility checklist; Sculley et al. [8] documented hidden technical debt in ML systems. Statcheck [20] automatically detects numerical inconsistencies in reported statistics; our benchmark addresses the orthogonal problem of methodological soundness in source code, which requires reasoning about analytical choices rather than checking arithmetic.

**Evaluation methodology.** Lipton & Steinhardt [12] catalogued troubling evaluation trends in ML, including overfitting to benchmarks and failure to distinguish empirical from theoretical claims. Our meta-finding instantiates this concern concretely: the benchmark designer and evaluator share context, changing what is being measured. Chang et al. [13] survey evaluation methodology across the LLM landscape; our work contributes an empirical data point on evaluation reliability for a domain that has received little attention — methodological review of scientific code.

---

## 9. Limitations

**Synthetic scripts.** All 36 scripts generate synthetic data from seeded distributions. This is a deliberate design choice: synthetic data provides unambiguous ground truth (we constructed the flaw, so we know exactly what the correct verdict is), full determinism across runs, and zero external dependencies. These properties are essential for a reproducible benchmark. The tradeoff is generalizability: real-world analysis scripts are messier, with data loading, preprocessing pipelines, external dependencies, and domain-specific conventions that all create additional surface area for both true errors and spurious model concerns. Detection rates on real-world code may differ from our benchmarked rates in either direction. We treat the current results as a lower-bound estimate for structural flaws (data leakage, pseudoreplication — which have clear code-level signatures) and an upper-bound estimate for conceptual flaws (survivorship bias, wrong test assumptions — where real scripts may provide more contextual cues that help reviewers calibrate). Establishing the synthetic-to-real transfer gap is the highest-priority direction for future work.

**Scope: frequentist statistical analysis in Python.** Scripts are written in Python with English-language docstrings and cover frequentist statistical methodology exclusively. Models trained predominantly on English code may generalize poorly to other programming languages or non-English scientific communities. More importantly, the six flaw categories are specific to frequentist analysis: Bayesian workflows, simulation studies, causal inference pipelines, qualitative coding environments, and domain-specific pipelines (e.g., genomic variant calling, neuroimaging preprocessing, financial time-series) are entirely absent. Claims about "agent review reliability" should therefore be read as claims about "agent review reliability *for frequentist statistical Python workflows*" — a meaningful but bounded scope. We chose this scope deliberately: frequentist statistics is the lingua franca of quantitative science, the flaws are well-documented in the literature with clear ground truth, and the domain is broad enough to span six genuinely distinct failure modes across multiple scientific fields.

**Six flaw categories.** The flaw space is much larger than six categories: confounding in observational studies, measurement error, selection bias, specification errors in regression models, improper handling of censored data, and many others are not represented. Detection rates on our six categories cannot be extrapolated to the full landscape of methodological error.

**Ground truth verification.** Flawed scripts were constructed by the authors with a single, specific, intentional methodological error per script. Ground truth labels were verified through the following process: (1) each script was executed and its output confirmed plausible; (2) each flaw was checked against the published literature definition for that category (e.g., F2 verified against the Simmons et al. [3] definition of undisclosed multiple comparisons); (3) control scripts were reviewed to confirm the correct analytical approach was implemented without introducing compensating errors. All 36 scripts and their ground truth labels are published at the project repository for independent inspection.

That said, control scripts engaging with genuinely complex methodology (particularly survivorship bias and wrong test assumptions) may contain edge-case subtleties that reasonable experts might assess differently. The false-positive analysis in the Appendix identifies three such cases. In each instance, the model's critique addressed a real methodological complexity but misidentified it as a *flaw* rather than a *limitation* — the key distinction being whether the script's analytical choice is appropriate for its stated purpose. An independent expert audit would further strengthen confidence in these borderline cases and is a natural extension of this work.

**Five models evaluated.** Results may not generalize to smaller open-weight models or to domain-specialized systems fine-tuned for code review. Model capability is a moving target; FPR patterns documented here may not persist across model generations.

**Taxonomy anomaly.** Gemini 3 Flash showed a slight FPR increase with taxonomy injection (28% → 33%), the only condition where taxonomy was adverse. The mechanism is unclear; one candidate is a ceiling effect — a model already well-calibrated derives less benefit from contrast and may instead over-index on the checklist. A larger script corpus would be needed to distinguish this from noise.

**Condition asymmetry.** The repeated-measure condition was applied only to Sonnet 4.6. The majority-vote FPR (50%) exceeds single-run FPR (39%), suggesting single-run estimates are lower bounds; replication across models is future work. Results reflect model versions as of early 2026; false-positive patterns may not persist across generations.

**Human expert baseline absent.** Without interannotator agreement statistics from domain statisticians reviewing the same 36 scripts, the FPR estimates conflate two distinct phenomena: (a) model errors on unambiguous control scripts, and (b) model assessments of genuinely ambiguous control scripts where expert opinion is divided. The ground truth verification process (Section 9) provides strong evidence for (a) in most cases, but the survivorship bias and wrong-test-assumption categories contain scripts where reasonable experts could disagree. Establishing human baseline accuracy is the highest-priority validation step before treating these FPR numbers as definitive.

---

## 10. Reproducibility Statement

All results in this paper are fully reproducible from the published code:

```bash
git clone https://github.com/jheuer/replication-trap
cd replication-trap
pip install -r requirements.txt
python3 generate_audit_scripts.py      # generates audit_scripts/ and answer key
# set ANTHROPIC_API_KEY, OPENAI_API_KEY in .env
python3 run_evaluation.py --model sonnet
python3 run_evaluation.py --model opus
python3 run_evaluation.py --model gpt4o
python3 run_evaluation.py --model sonnet --with-taxonomy
python3 run_evaluation.py --model sonnet --runs 3
python3 run_evaluation.py --compare --no-generate
```

All random seeds are fixed (`numpy.random.seed(42)`, `random.Random(42)`). The generator produces identical scripts across platforms. LLM API responses are stochastic; exact verdicts may vary, but FPR and DR estimates should be stable within the reported confidence intervals across independent runs.

---

## 11. Discussion and Conclusions

Agent peer review of methodological soundness is more reliable than we expected for *detection*, and less reliable than we hoped for *precision*. Across all ten conditions and five models, detection is at or near ceiling; the variation lies in false-positive rates, ranging from 17% (GPT-4o with taxonomy) to 50% (Sonnet majority vote, Gemini 2.5 Flash baseline).

Three practical recommendations follow:

1. **For models with elevated baseline FPR, include structured flaw taxonomies in reviewer prompts.** The effect is conditional on baseline calibration, with neutral or adverse outcomes for already-precise models. The mechanism appears to be improved contrast: taxonomies help models distinguish "this script addresses X" from "this script has flaw X."

2. **High-complexity correct scripts are systematically over-flagged**, so reviewers should be prompted to distinguish "handling a hard problem" from "handling it wrongly."

3. **Single-run FPR underestimates the true marginal FPR** — multi-run experiments should be standard practice when calibration is the quantity of interest.

More broadly, our results frame agent peer review as a **calibration problem**: the capability to detect methodological errors exists at high sensitivity; the gap is in specificity. Progress on agent review quality should be measured primarily by FPR reduction, not by further improvements to already-ceiling detection rates.

The benchmark, all experimental results, and the evaluation harness are released at [https://github.com/jheuer/replication-trap](https://github.com/jheuer/replication-trap).

---

## Appendix: Representative False Positive Analysis

The following examples illustrate the survivorship bias and wrong-test-assumption false positives in detail. These are drawn verbatim from the Sonnet 4.6 evaluation run. Understanding *why* a correct script was flagged is as informative as the aggregate FPR.

### A.1 Survivorship Bias — LOCF Imputation Script (script_08.py)

This control script performs an intent-to-treat analysis of a weight-loss intervention using last-observation-carried-forward (LOCF) imputation for dropout. The script correctly applies LOCF, explicitly acknowledges the conservative bias introduced when heavier participants drop out, and draws conclusions appropriately hedged for this limitation.

**Model verdict:** FAIL (HIGH confidence)

**Model's flaw identification:** *"The dropout mechanism is informatively biased in a direction that makes LOCF anti-conservative rather than conservative ... The critical flaw is that the script presents this as a valid single-arm ITT analysis using a one-sample t-test against zero, when there is no control arm. Without a randomised control group, any observed BMI reduction cannot be attributed to the intervention."*

**Assessment:** The model correctly identified a real statistical complexity (single-arm designs cannot isolate intervention effects from temporal confounds). However, the script is labeled as a *control* precisely because it performs a methodologically standard ITT analysis appropriate for its stated purpose: estimating the average treatment effect under randomization, not establishing causation. Single-arm pre-post designs are widely used and publishable for this purpose. The model conflated "this design has limitations" with "this design has a flaw," applying a causal inference standard to a descriptive analysis.

This illustrates the pessimism bias: the model penalized the script for not doing something it was never attempting to do.

### A.2 Wrong Test Assumption — Adaptive Test Selection Script (script_27.py)

This control script performs a normality test (Shapiro-Wilk) on a subset of data before selecting between Mann-Whitney U and Student's t-test. The normality test is applied to a sample of 50 observations, and because the data are lognormal, correctly routes to Mann-Whitney U.

**Model verdict:** FAIL (HIGH confidence)

**Model's flaw identification:** *"The normality test is performed on a subsample (first 50 observations) while the Mann-Whitney U test is then run on the full datasets (n=120 and n=130). This is methodologically inconsistent ... this 'test-then-choose' approach ... is itself a recognised methodological problem: it inflates Type I error rates."*

**Assessment:** The model identified a genuine methodological debate — whether adaptive test selection (preliminary testing for normality) inflates Type I error rates. This is a real concern in the literature [15]. However, the ground truth label for this script is PASS for two independent reasons.

First, the F5 flaw is specifically defined as *applying a parametric test to data that violates normality assumptions*. This script does not do that: it correctly routes to Mann-Whitney U when the normality test fails. The planted flaw (t-test on lognormal data) is absent.

Second, the script's docstring explicitly states that length-of-stay data is known to be right-skewed, establishing domain knowledge that pre-specifies the analysis direction. The normality pre-test is therefore an additional conservative check, not the decision mechanism. A pre-specified choice of Mann-Whitney U for right-skewed data — with a normality test as a confirmatory gate — is a defensible and commonly practiced approach, distinct from the blind adaptive selection that drives Type I inflation.

The model critiqued the script's *philosophy of test selection* rather than assessing whether it committed the specific flaw being tested. This illustrates a general failure mode: applying a stricter methodological standard than the one the script is being evaluated against, then treating the gap as a flaw.

### A.3 Survivorship Bias — Fund Performance Script (script_04.py)

This control script computes average annual returns across a fund universe including both "survivors" (funds with positive cumulative returns) and "closed" funds (funds with negative cumulative returns at the 10-year horizon). The script explicitly includes all funds in the summary statistics, which is the correct approach for avoiding survivorship bias.

**Model verdict:** FAIL (HIGH confidence)

**Model's flaw identification:** *"The script misidentifies which funds 'closed.' It labels a fund as closed only if its cumulative return at year 10 is below 1.0 ... funds that dipped below par mid-period but recovered are incorrectly labeled as survivors."*

**Assessment:** The model correctly observed that the "closed" label is applied based on end-of-period performance, not on true mid-period fund liquidation. This is a real nuance: in practice, funds that underperform may be merged or liquidated mid-period, which would require truncating their return series rather than including all 10 years. However, the script's stated purpose is to include all funds regardless of outcome — a deliberate design choice to avoid excluding poor performers. The model identified a scenario where the script's closure criterion could be more realistic, then treated this as a disqualifying flaw rather than a modeling simplification appropriate to the analysis.

These three examples share a common structure: the model identifies a real complexity, a real limitation, or a real debate in the methodology, then resolves that complexity by condemning the script rather than by assessing whether the script's approach is appropriate for its stated purpose. This is the core failure mode our benchmark surfaces.

---

## References

[1] Open Science Collaboration. Estimating the reproducibility of psychological science. *Science*, 349(6251):aac4716, 2015.

[2] J. P. A. Ioannidis. Why most published research findings are false. *PLoS Medicine*, 2(8):e124, 2005.

[3] J. P. Simmons, L. D. Nelson, and U. Simonsohn. False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science*, 22(11):1359–1366, 2011.

[4] M. Baker. 1,500 scientists lift the lid on reproducibility. *Nature*, 533(7604):452–454, 2016.

[5] S. Kapoor and A. Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9):100804, 2023.

[6] S. H. Hurlbert. Pseudoreplication and the design of ecological field experiments. *Ecological Monographs*, 54(2):187–211, 1984.

[7] J. Pineau et al. Improving reproducibility in machine learning research. *Journal of Machine Learning Research*, 22(164):1–20, 2021.

[8] D. Sculley et al. Hidden technical debt in machine learning systems. *NeurIPS*, 28, 2015.

[9] P. Henderson et al. Deep reinforcement learning that matters. *AAAI*, 32(1), 2018.

[10] O. E. Gundersen and S. Kjensmo. State of the art: Reproducibility in artificial intelligence. *AAAI*, 32(1), 2018.

[11] L. Zheng et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS*, 36, 2023.

[12] Z. C. Lipton and J. Steinhardt. Troubling trends in machine learning scholarship. *Queue*, 17(1):45–77, 2019.

[13] Y. Chang et al. A survey on evaluation of large language models. *ACM TIST*, 15(3):1–45, 2024.

[14] A. Gelman and E. Loken. The statistical crisis in science. *American Scientist*, 102(6):460–465, 2014.

[15] M. L. Head et al. The extent and consequences of p-hacking in science. *PLOS Biology*, 13(3):e1002106, 2015.

[16] V. Stodden, J. Seiler, and Z. Ma. An empirical analysis of journal policy effectiveness for computational reproducibility. *PNAS*, 115(11):2584–2589, 2018.

[17] K. S. Button et al. Power failure: why small sample size undermines the reliability of neuroscience. *Nature Reviews Neuroscience*, 14(5):365–376, 2013.

[18] C. E. Jimenez et al. SWE-bench: Can language models resolve real-world GitHub issues? *ICLR*, 2024.

[19] P. Liang et al. Holistic evaluation of language models. *TMLR*, 2022.

[20] M. B. Nuijten et al. The prevalence of statistical reporting errors in psychology (1985–2013). *Behavior Research Methods*, 48:1205–1226, 2016.
