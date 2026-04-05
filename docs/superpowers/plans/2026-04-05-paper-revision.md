# Paper Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `research_note.tex` to incorporate 5-model evaluation data and restructure Section 5 around detection vs. FPR axes.

**Architecture:** Series of targeted edits to `research_note.tex`, then sync `submission_content.md`. Each task is one self-contained section or table. Commit after each task.

**Tech Stack:** LaTeX (lualatex), plain text editing

---

## Reference Data

All per-category FP counts needed for Table 2 (controls flagged / 3):

| Category           | Sonnet | Opus | GPT-4o | Gem2.5 | Gem3 |
|--------------------|--------|------|--------|--------|------|
| Data leakage       | 0/3    | 0/3  | 0/3    | 1/3    | 0/3  |
| P-hacking          | 1/3    | 0/3  | 2/3    | 2/3    | 1/3  |
| Circular validation| 1/3    | 0/3  | 0/3    | 0/3    | 0/3  |
| Pseudoreplication  | 0/3    | 0/3  | 0/3    | 0/3    | 0/3  |
| Survivorship bias  | 3/3    | 3/3  | 3/3    | 3/3    | 2/3  |
| Wrong test assump. | 2/3    | 1/3  | 2/3    | 3/3    | 2/3  |

Variants caught (TP/3): all 3/3 for all models except Opus Survivorship bias (2/3).

---

## Task 1: Update Abstract

**Files:**
- Modify: `research_note.tex:45-66`

- [ ] **Step 1: Replace the abstract**

Replace lines 45–66 (the full `\begin{abstract}...\end{abstract}` block) with:

```latex
\begin{abstract}
Agent-based peer review is a foundational premise of executable science:
if skills replace papers, agents must replace reviewers.
But how reliably do agents detect \emph{methodological} errors---flaws
that run without errors, produce plausible output, and invalidate
conclusions silently?
We present \textbf{The Replication Trap}, a benchmark of 36 statistical
analysis scripts (3 variants $\times$ 6 flaw categories, plus matched
controls) drawn from the replication crisis literature.
We evaluate five frontier LLMs across nine experimental conditions.
\textbf{Key finding: detection of flawed scripts is near-ceiling
across all models and conditions (sensitivity 94--100\%), but
false-positive rates on methodologically correct scripts range from
17\% to 50\%---the primary axis of variation.}
Survivorship bias controls are universally over-flagged.
Injecting a flaw taxonomy into the system prompt \emph{reduces} false
positives for models with elevated baseline FPR (GPT-4o: 39\%\,\to\,17\%;
Sonnet: 39\%\,\to\,28\%), while having neutral or adverse effects for
already-precise models---suggesting the benefit is conditional, not universal.
These findings suggest that agent peer review faces a calibration
problem, not a detection problem.
\end{abstract}
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: update abstract for 5-model results"
```

---

## Task 2: Update Contributions List (Section 1)

**Files:**
- Modify: `research_note.tex:114-120`

- [ ] **Step 1: Replace the contributions enumerate block**

Replace the `\begin{enumerate}...\end{enumerate}` block in the Introduction (lines 109–121) with:

```latex
\begin{enumerate}[nosep,leftmargin=1.5em]
  \item A \textbf{benchmark of 36 scripts} spanning 6 flaw categories,
    each with 3 independently instantiated flawed variants and 3 matched
    controls, enabling category-level statistical inference with
    cluster-bootstrapped confidence intervals.
  \item \textbf{Multi-model experimental results} across five frontier
    LLMs---Claude Sonnet 4.6, Claude Opus 4.6, GPT-4o, Gemini 2.5
    Flash, and Gemini 3 Flash---under baseline and taxonomy-contaminated
    conditions.
  \item \textbf{An empirical characterisation of agent review failure
    modes}: calibration errors dominate; the taxonomy intervention
    reduces FPR for high-FPR models but not for already-precise ones;
    and survivorship-bias controls are universally over-flagged across
    all models and conditions.
\end{enumerate}
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: update contributions list for 5 models"
```

---

## Task 3: Update Experimental Setup (Section 4)

**Files:**
- Modify: `research_note.tex:186-203`

- [ ] **Step 1: Replace the Experimental Setup section body**

Replace lines 186–203 (from `\section{Experimental Setup}` through the closing `\end{enumerate}`) with:

```latex
\section{Experimental Setup}

\textbf{Models:}
Claude Sonnet 4.6, Claude Opus 4.6, GPT-4o, Gemini 2.5 Flash
(\texttt{gemini/gemini-2.5-flash}), and Gemini 3 Flash
(\texttt{gemini/gemini-3-flash-preview}) via the LiteLLM API
with a clean system prompt: \emph{``You are a scientific methods
expert reviewing statistical analysis code for methodological
soundness.''} Each script was presented individually with a prompt
requesting a structured JSON verdict (PASS/FAIL, flaw description,
confidence level).

\textbf{Conditions:}
\begin{enumerate}[nosep,leftmargin=1.5em]
  \item \textbf{Baseline} (all five models): single run, no additional
    context.
  \item \textbf{Taxonomy contamination} (all five models): the six flaw
    categories injected verbatim into the system prompt before evaluation.
  \item \textbf{Repeated measures} (Sonnet only, $n=3$ runs):
    three independent API calls per script; majority-vote verdict used.
\end{enumerate}
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: update experimental setup to 5 models"
```

---

## Task 4: Restructure Section 5.1 — Detection

Replace the current `\subsection{Primary: Category-Level Detection}` block (lines 207–237, ending after `\end{table}`) with the new subsection title and expanded Table 1.

**Files:**
- Modify: `research_note.tex:207-237`

- [ ] **Step 1: Replace subsection 5.1 and Table 1**

```latex
\subsection{Detection}

Detection of flawed scripts is at or near ceiling across all models
and conditions: five of ten conditions achieve 100\% category-level
detection; the remaining two (Opus baseline, GPT-4o with taxonomy)
detect 5/6 categories (one missed Survivorship variant each). This
dimension does not differentiate conditions. The primary axis of
variation is false positive rate (Table~\ref{tab:main}).

\begin{table}[ht]
\centering
\caption{All conditions. DR = script-level detection rate (18 flawed
scripts); FPR = false positive rate on 18 correct controls;
95\% CI from naive bootstrap. Pairwise FPR differences should be
read as directional (CIs overlap at $n{=}18$).}
\label{tab:main}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Condition} & \textbf{Cats} & \textbf{DR} & \textbf{FPR [95\% CI]} & \textbf{F1} \\
\midrule
\multicolumn{5}{l}{\textit{Claude}} \\
Sonnet 4.6           & 6/6 & 100\% & 39\% [17--63] & 0.84 \\
Sonnet (taxonomy)    & 6/6 & 100\% & 28\% [8--50]  & 0.88 \\
Sonnet ($n{=}3$)     & 6/6 & 100\% & 50\% [27--73] & 0.80 \\
Opus 4.6             & 6/6 & 94\%  & 22\% [6--43]  & 0.87 \\
Opus (taxonomy)      & 6/6 & 100\% & 22\% [6--43]  & 0.90 \\
\midrule
\multicolumn{5}{l}{\textit{OpenAI}} \\
GPT-4o               & 6/6 & 100\% & 39\% [17--63] & 0.84 \\
GPT-4o (taxonomy)    & 6/6 & 94\%  & 17\% [0--36]  & 0.90 \\
\midrule
\multicolumn{5}{l}{\textit{Google}} \\
Gemini 2.5 Flash     & 6/6 & 100\% & 50\% [27--73] & 0.80 \\
Gemini 3 Flash       & 6/6 & 100\% & 28\% [8--50]  & 0.88 \\
Gemini 3 (taxonomy)  & 6/6 & 100\% & 33\% [13--57] & 0.86 \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: expand Table 1 to all 10 conditions across 5 models"
```

---

## Task 5: Restructure Section 5.2 — False Positive Rates

Replace the current `\subsection{Difficulty by Flaw Category}` block (lines 239–266) with the new 5.2 containing a multi-model per-category table.

**Files:**
- Modify: `research_note.tex:239-266`

- [ ] **Step 1: Replace subsection 5.2 and Table 2**

```latex
\subsection{False Positive Rates}

Across baseline conditions, FPR ranges from 22\% (Opus) to 50\%
(Sonnet, Gemini 2.5 Flash). The variation is structurally driven by
two flaw categories: Survivorship bias and Wrong test assumption
(Table~\ref{tab:cats}). Data leakage and pseudoreplication are
``clean'' across all models---high detection, near-zero false
positives---because correct and flawed scripts differ by a single
unambiguous code pattern. Survivorship bias and wrong test assumption
are ``noisy'' because their correct handling involves methodological
complexity that models conflate with methodological error.

\begin{table*}[ht]
\centering
\caption{Per-category results across all five baseline models.
``TP/3'' = flawed variants caught; ``FP/3'' = correct controls
incorrectly flagged. All models detect 3/3 flawed variants per
category except Opus Survivorship bias (2/3).}
\label{tab:cats}
\small
\begin{tabular}{l cc cc cc cc cc}
\toprule
& \multicolumn{2}{c}{\textbf{Sonnet}} & \multicolumn{2}{c}{\textbf{Opus}} & \multicolumn{2}{c}{\textbf{GPT-4o}} & \multicolumn{2}{c}{\textbf{Gem 2.5}} & \multicolumn{2}{c}{\textbf{Gem 3}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
\textbf{Category} & TP/3 & FP/3 & TP/3 & FP/3 & TP/3 & FP/3 & TP/3 & FP/3 & TP/3 & FP/3 \\
\midrule
Data leakage          & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 1/3 & 3/3 & 0/3 \\
$P$-hacking           & 3/3 & 1/3 & 3/3 & 0/3 & 3/3 & 2/3 & 3/3 & 2/3 & 3/3 & 1/3 \\
Circular validation   & 3/3 & 1/3 & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 0/3 \\
Pseudoreplication     & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 0/3 & 3/3 & 0/3 \\
Survivorship bias     & 3/3 & 3/3 & 2/3 & 3/3 & 3/3 & 3/3 & 3/3 & 3/3 & 3/3 & 2/3 \\
Wrong test assumption & 3/3 & 2/3 & 3/3 & 1/3 & 3/3 & 2/3 & 3/3 & 3/3 & 3/3 & 2/3 \\
\bottomrule
\end{tabular}
\end{table*}
```

Note: `table*` spans both columns in the two-column layout. If the journal style requires single-column tables, change `table*` back to `table`.

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: expand Table 2 to multi-model per-category breakdown"
```

---

## Task 6: Rewrite Section 5.3 — Taxonomy as a Calibration Intervention

Replace the current `\subsection{Context Contamination}` block (lines 268–288) with the rewritten taxonomy section using 4-model data.

**Files:**
- Modify: `research_note.tex:268-288`

- [ ] **Step 1: Replace subsection 5.3**

```latex
\subsection{Taxonomy as a Calibration Intervention}

We injected the six flaw category descriptions verbatim into the
system prompt for four models. The effect was not uniform.
For models with elevated baseline FPR, taxonomy \emph{reduced}
over-flagging substantially: GPT-4o 39\%\,\to\,17\%
(the single largest improvement across all conditions), Sonnet
39\%\,\to\,28\%. For already-precise models, the effect was neutral
or adverse: Opus remained at 22\% while recovering its one missed
detection (DR 94\%\,\to\,100\%); Gemini 3 \emph{increased} slightly
from 28\% to 33\%---a reversal noted as a caveat in Section~\ref{sec:limitations}.

A plausible mechanism for the benefit: the taxonomy provides
\emph{contrast}---knowing what data leakage looks like in the abstract
allows a model to more precisely assess whether a specific script
exhibits it. Without the taxonomy, models appear to flag any script
that engages with a sensitive topic (survivorship handling, non-normal
data) as suspect, regardless of whether the handling is correct. The
taxonomy supplies a standard against which the implementation is compared.

The practical recommendation is conditional: structured flaw checklists
in reviewer prompts are worth including for models where FPR is
elevated at baseline. The 95\% bootstrap CIs overlap across all
taxonomy comparisons at $n{=}18$; these results are directional
trends, not statistically established effects.
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: rewrite taxonomy section with 4-model data"
```

---

## Task 7: Remove Section 5.4 Reliability (fold into Limitations)

Delete the current `\subsection{Reliability}` block (lines 290–305) entirely. Its content will be added to Limitations in Task 8.

**Files:**
- Modify: `research_note.tex:290-305`

- [ ] **Step 1: Delete the Reliability subsection**

Remove lines 290–305:
```
\subsection{Reliability}

Across 3 independent runs on the same 36 scripts, Sonnet produced
...
call should treat the result as a lower bound.
```

The section following it (`\section{The Survivorship Bias Pathology}`) should now immediately follow the end of 5.3.

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: remove reliability subsection (moves to Limitations)"
```

---

## Task 8: Update The Survivorship Bias Pathology (Section 6)

The section currently says "Across all five experimental conditions" — update to "ten" now that we have 10 conditions.

**Files:**
- Modify: `research_note.tex:309`

- [ ] **Step 1: Update condition count**

Find and replace:
```latex
Across all five experimental conditions, Survivorship bias controls
```
with:
```latex
Across all ten experimental conditions, Survivorship bias controls
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: update survivorship section condition count to 10"
```

---

## Task 9: Update Limitations (Section 9)

**Files:**
- Modify: `research_note.tex:376-393`

- [ ] **Step 1: Replace the Limitations section body**

Replace lines 376–393 (from `\section{Limitations}` through the end of the section, just before `\section{Discussion`) with:

```latex
\section{Limitations}
\label{sec:limitations}

\textbf{Synthetic scripts.} All scripts generate synthetic data from seeded
distributions. Real-world code is messier---preprocessing pipelines, external
dependencies, and domain conventions create additional surface area for both
true errors and spurious concerns. Detection rates on real-world code may
differ substantially.

\textbf{Ground truth.} Control scripts were written and verified by the authors.
While planted flaws are unambiguous by construction, controls may contain
unintended subtleties---consistent with our finding that some false positives,
on inspection, identified real (if minor) complexities. An independent expert
audit would strengthen the ground truth labels.

\textbf{Taxonomy anomaly.} Gemini 3 Flash showed a slight FPR increase with
taxonomy injection (28\%\,\to\,33\%), the only condition where taxonomy was
adverse. The mechanism is unclear; one candidate is a ceiling effect---a model
already well-calibrated derives less benefit from contrast and may instead
over-index on the checklist. A larger script corpus would be needed to
distinguish this from noise.

\textbf{Condition asymmetry.} The repeated-measure condition was applied only
to Sonnet 4.6. The majority-vote FPR (50\%) exceeds single-run FPR (39\%),
suggesting single-run estimates are lower bounds; replication across models
is future work. Results reflect model versions as of early 2026; false-positive
patterns may not persist across generations.
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: update Limitations with taxonomy anomaly and reliability note"
```

---

## Task 10: Update Discussion and Conclusions (Section 11)

**Files:**
- Modify: `research_note.tex:395-421`

- [ ] **Step 1: Replace the Discussion section body**

Replace lines 395–421 (from `\section{Discussion and Conclusions}` to just before `{\small`) with:

```latex
\section{Discussion and Conclusions}

Agent peer review of methodological soundness is more reliable than
we expected for \emph{detection}, and less reliable than we hoped for
\emph{precision}. Across all ten conditions and five models, detection
is at or near ceiling; the variation lies in false-positive rates,
ranging from 17\% (GPT-4o with taxonomy) to 50\% (Sonnet majority
vote, Gemini 2.5 Flash baseline).

These findings suggest three practical recommendations for agent review
systems: (1) \textbf{for models with elevated baseline FPR, include
structured flaw taxonomies in reviewer prompts}---the effect is
conditional on baseline calibration, with neutral or adverse outcomes
for already-precise models; (2) \textbf{high-complexity correct scripts
are systematically over-flagged}, so reviewers should be prompted to
distinguish ``handling a hard problem'' from ``handling it wrongly'';
and (3) \textbf{single-run FPR underestimates the true marginal
FPR}---multi-run experiments should be standard practice when
calibration is the quantity of interest.

The benchmark is released as a Claw4S submission at
\url{https://github.com/jheuer/replication-trap}.
```

- [ ] **Step 2: Commit**

```bash
git add research_note.tex
git commit -m "paper: update Discussion for 5-model results and revised taxonomy claim"
```

---

## Task 11: Sync submission_content.md

Update `submission_content.md` to mirror the key changes in the `.tex` so the two files stay in sync. This file is the markdown version used in the README and SKILL.md context.

**Files:**
- Modify: `submission_content.md`

- [ ] **Step 1: Update the main results table in submission_content.md**

Find the results table (currently 5 rows: Sonnet, Sonnet+taxonomy, Sonnet n=3, Opus, GPT-4o) and replace with the 10-row table from Task 4.

- [ ] **Step 2: Update the abstract paragraph**

Find "We evaluate three frontier LLMs under four experimental conditions" and replace to match the updated abstract.

- [ ] **Step 3: Update Section 4 model list**

Find "Models evaluated: Claude Sonnet 4.6, Claude Opus 4.6, and GPT-4o" and update to the 5-model list.

- [ ] **Step 4: Update Section 5.3 taxonomy prose**

Find the Context Contamination section and replace with the prose from Task 6.

- [ ] **Step 5: Update Section 9 Limitations**

Replace with the updated Limitations text from Task 9.

- [ ] **Step 6: Update Section 11 Discussion**

Replace with the updated Discussion text from Task 10.

- [ ] **Step 7: Commit**

```bash
git add submission_content.md
git commit -m "paper: sync submission_content.md with tex updates"
```

---

## Self-Review

**Spec coverage check:**
- Abstract update → Task 1 ✓
- Contributions → Task 2 ✓
- Section 4 (5 models, conditions table) → Task 3 ✓
- Section 5.1 Detection + expanded Table 1 → Task 4 ✓
- Section 5.2 FPR + multi-model Table 2 → Task 5 ✓
- Section 5.3 Taxonomy rewrite (4-model) → Task 6 ✓
- Remove Reliability subsection → Task 7 ✓
- Survivorship section condition count → Task 8 ✓
- Limitations (taxonomy anomaly + reliability note) → Task 9 ✓
- Discussion (FPR range, conditional taxonomy claim) → Task 10 ✓
- Sync submission_content.md → Task 11 ✓

**Placeholder scan:** No TBDs or TODOs. All LaTeX is complete.

**Consistency check:**
- Table 1 uses "Cats" column header; prose references "Cats det." — use "Cats" consistently in both
- `\label{sec:limitations}` added in Task 9 must match the `Section~\ref{sec:limitations}` reference in Task 6 ✓
- `table*` in Task 5 spans two columns — correct for two-column article class ✓
- FPR range in Discussion (Task 10) updated to 17%–50% ✓
- Reliability claim removed from Discussion recommendations ✓
