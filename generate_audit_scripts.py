#!/usr/bin/env python3
"""
generate_audit_scripts.py

Generates 12 self-contained statistical analysis scripts in audit_scripts/
with opaque filenames (script_01.py through script_12.py).
An answer key is written to audit_answer_key/answer_key.json.
"""

import json
import os
import random
from script_variants import (
    _VA1, _VA2, _VB1, _VB2,
    _VC1, _VC2, _VD1, _VD2,
    _VE1, _VE2, _VF1, _VF2,
    _VG1, _VG2, _VH1, _VH2,
    _VI1, _VI2, _VJ1, _VJ2,
    _VK1, _VK2, _VL1, _VL2,
)

OUTPUT_DIR = "audit_scripts"
ANSWER_KEY_DIR = "audit_answer_key"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANSWER_KEY_DIR, exist_ok=True)

_SE = '''\
"""
Study: Predicting patient readmission risk from clinical features.
Method: Logistic regression with standardized features, 80/20 train/test split.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Simulate 500 patients, 10 clinical features
n, p = 500, 10
X = np.random.randn(n, p)
true_coef = np.array([0.8, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0.6])
prob = 1 / (1 + np.exp(-X @ true_coef))
y = (np.random.rand(n) < prob).astype(int)

# Standardize features for numerical stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Conclusion: Model achieves {test_acc:.1%} accuracy on held-out data.")
'''

_SK = '''\
"""
Study: Predicting patient readmission risk from clinical features.
Method: Logistic regression with standardized features, 80/20 train/test split.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Simulate 500 patients, 10 clinical features
n, p = 500, 10
X = np.random.randn(n, p)
true_coef = np.array([0.8, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0.6])
prob = 1 / (1 + np.exp(-X @ true_coef))
y = (np.random.rand(n) < prob).astype(int)

# Split into train and test FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features — fit only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Conclusion: Model achieves {test_acc:.1%} accuracy on held-out data.")
'''

_SA = '''\
"""
Study: Does a dietary supplement affect any measured biomarker?
Method: Pearson correlation between supplement dosage and 20 biomarkers,
        reporting the statistically significant finding.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n = 80  # participants
dosage = np.random.uniform(0, 100, n)

# 20 biomarker measurements (all independent of dosage in truth)
biomarkers = {f"biomarker_{i+1:02d}": np.random.randn(n) for i in range(20)}

# Test each biomarker for correlation with dosage
significant_results = []
for name, values in biomarkers.items():
    r, p = stats.pearsonr(dosage, values)
    if p < 0.05:
        significant_results.append((name, r, p))

# Report the most significant finding
if significant_results:
    best = min(significant_results, key=lambda x: x[2])
    print(f"Result: {best[0]} is significantly correlated with dosage")
    print(f"  Pearson r = {best[1]:.3f}, p = {best[2]:.4f}")
    print(f"Conclusion: The supplement significantly affects {best[0]} (p < 0.05).")
else:
    print("No significant correlations found.")
'''

_SG = '''\
"""
Study: Does a dietary supplement affect any measured biomarker?
Method: Pearson correlation between supplement dosage and 20 biomarkers,
        with Bonferroni correction for multiple comparisons.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n = 80  # participants
dosage = np.random.uniform(0, 100, n)

# 20 biomarker measurements (all independent of dosage in truth)
biomarkers = {f"biomarker_{i+1:02d}": np.random.randn(n) for i in range(20)}

# Test each biomarker for correlation with dosage
n_tests = len(biomarkers)
alpha = 0.05
corrected_alpha = alpha / n_tests  # Bonferroni correction

results = []
for name, values in biomarkers.items():
    r, p = stats.pearsonr(dosage, values)
    results.append((name, r, p))

# Apply multiple comparison correction
significant = [(name, r, p) for name, r, p in results if p < corrected_alpha]

print(f"Tested {n_tests} biomarkers against dosage.")
print(f"Bonferroni-corrected alpha: {corrected_alpha:.4f}")
if significant:
    for name, r, p in significant:
        print(f"  {name}: r = {r:.3f}, p = {p:.4f} (significant after correction)")
else:
    print("No biomarkers survived correction for multiple comparisons.")
print(f"Conclusion: After correcting for {n_tests} tests, "
      f"{'no significant' if not significant else len(significant)} "
      f"association(s) found.")
'''

_SJ = '''\
"""
Study: Selecting optimal regularization for a ridge regression model
       predicting gene expression from methylation features.
Method: Grid search over alpha values, evaluated on held-out test set.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Simulated methylation (features) and expression (target)
n, p = 300, 50
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:5] = [2.0, -1.5, 1.0, -0.5, 0.8]
y = X @ true_w + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Grid search: find the alpha that gives the best TEST performance
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
best_alpha, best_mse = None, float("inf")

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

# Retrain with best alpha and report test performance
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)
final_mse = mean_squared_error(y_test, final_model.predict(X_test))

print(f"Best alpha: {best_alpha}")
print(f"Test MSE: {final_mse:.4f}")
print(f"Conclusion: Ridge regression with alpha={best_alpha} achieves "
      f"MSE={final_mse:.4f} on held-out data.")
'''

_SC = '''\
"""
Study: Selecting optimal regularization for a ridge regression model
       predicting gene expression from methylation features.
Method: Grid search over alpha values using cross-validation on training data,
        then final evaluation on held-out test set.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Simulated methylation (features) and expression (target)
n, p = 300, 50
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:5] = [2.0, -1.5, 1.0, -0.5, 0.8]
y = X @ true_w + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Grid search using cross-validation on TRAINING data only
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_

# Evaluate on held-out test set (never used for model selection)
test_mse = mean_squared_error(y_test, ridge_cv.predict(X_test))

print(f"Best alpha (via 5-fold CV on train): {best_alpha}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Conclusion: Ridge regression with alpha={best_alpha} (selected via CV) "
      f"achieves MSE={test_mse:.4f} on held-out data.")
'''

_SF = '''\
"""
Study: Measuring mean reaction yield across 100 experimental batches.
Method: Compute mean and 95% CI from batch yield measurements.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate 100 batch yields — some batches fail (yield = NaN or 0)
n_batches = 100
raw_yields = np.random.normal(loc=65, scale=15, size=n_batches)

# Some batches fail completely (set to NaN)
fail_mask = np.random.rand(n_batches) < 0.15  # ~15% failure rate
raw_yields[fail_mask] = np.nan

# Clean data: remove failed batches before analysis
clean_yields = raw_yields[~np.isnan(raw_yields)]

mean_yield = np.mean(clean_yields)
se = stats.sem(clean_yields)
ci = stats.t.interval(0.95, len(clean_yields)-1, loc=mean_yield, scale=se)

print(f"Batches analyzed: {len(clean_yields)} of {n_batches}")
print(f"Mean yield: {mean_yield:.1f}%")
print(f"95% CI: [{ci[0]:.1f}%, {ci[1]:.1f}%]")
print(f"Conclusion: The process achieves a mean yield of {mean_yield:.1f}% "
      f"(95% CI: {ci[0]:.1f}–{ci[1]:.1f}%).")
'''

_SB = '''\
"""
Study: Measuring mean reaction yield across 100 experimental batches.
Method: Compute mean and 95% CI from batch yield measurements,
        treating failed batches as zero yield (intent-to-treat principle).
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate 100 batch yields — some batches fail (yield = NaN or 0)
n_batches = 100
raw_yields = np.random.normal(loc=65, scale=15, size=n_batches)

# Some batches fail completely (set to NaN)
fail_mask = np.random.rand(n_batches) < 0.15  # ~15% failure rate
raw_yields[fail_mask] = np.nan

# Intent-to-treat: failed batches count as 0 yield
itt_yields = np.where(np.isnan(raw_yields), 0.0, raw_yields)

n_failed = np.sum(fail_mask)
mean_yield = np.mean(itt_yields)
se = stats.sem(itt_yields)
ci = stats.t.interval(0.95, len(itt_yields)-1, loc=mean_yield, scale=se)

print(f"Total batches: {n_batches} ({n_failed} failed, counted as 0% yield)")
print(f"Mean yield (ITT): {mean_yield:.1f}%")
print(f"95% CI: [{ci[0]:.1f}%, {ci[1]:.1f}%]")
print(f"Conclusion: Including all batches (ITT), mean yield is {mean_yield:.1f}% "
      f"(95% CI: {ci[0]:.1f}–{ci[1]:.1f}%). "
      f"{n_failed} of {n_batches} batches failed.")
'''

_SL = '''\
"""
Study: Comparing hospital length-of-stay between two treatment groups.
Method: Independent samples t-test.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# Length-of-stay data is typically right-skewed (lognormal)
# Group A: standard care, Group B: new intervention
n_a, n_b = 60, 55
group_a = np.random.lognormal(mean=2.0, sigma=0.8, size=n_a)
group_b = np.random.lognormal(mean=1.8, sigma=0.9, size=n_b)

# Compare groups using t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)

print(f"Group A (standard): mean = {np.mean(group_a):.1f} days, "
      f"median = {np.median(group_a):.1f} days, n = {n_a}")
print(f"Group B (new tx):    mean = {np.mean(group_b):.1f} days, "
      f"median = {np.median(group_b):.1f} days, n = {n_b}")
print(f"t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print(f"Conclusion: Significant difference in length-of-stay (p = {p_value:.4f}).")
else:
    print(f"Conclusion: No significant difference (p = {p_value:.4f}).")
'''

_SH = '''\
"""
Study: Comparing hospital length-of-stay between two treatment groups.
Method: Check normality, then use appropriate test (Mann-Whitney U for
        non-normal data).
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# Length-of-stay data is typically right-skewed (lognormal)
# Group A: standard care, Group B: new intervention
n_a, n_b = 60, 55
group_a = np.random.lognormal(mean=2.0, sigma=0.8, size=n_a)
group_b = np.random.lognormal(mean=1.8, sigma=0.9, size=n_b)

# Check normality assumption
_, p_norm_a = stats.shapiro(group_a)
_, p_norm_b = stats.shapiro(group_b)

print(f"Group A (standard): mean = {np.mean(group_a):.1f} days, "
      f"median = {np.median(group_a):.1f} days, n = {n_a}")
print(f"Group B (new tx):    mean = {np.mean(group_b):.1f} days, "
      f"median = {np.median(group_b):.1f} days, n = {n_b}")
print(f"Shapiro-Wilk normality: A p = {p_norm_a:.4f}, B p = {p_norm_b:.4f}")

if p_norm_a < 0.05 or p_norm_b < 0.05:
    print("Data is non-normal; using Mann-Whitney U test.")
    u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    test_name = "Mann-Whitney U"
    print(f"{test_name}: U = {u_stat:.1f}, p = {p_value:.4f}")
else:
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    test_name = "t-test"
    print(f"{test_name}: t = {t_stat:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    print(f"Conclusion: Significant difference in length-of-stay "
          f"({test_name}, p = {p_value:.4f}).")
else:
    print(f"Conclusion: No significant difference "
          f"({test_name}, p = {p_value:.4f}).")
'''

_SD = '''\
"""
Study: Does a training program improve reaction time?
Method: Compare pre and post reaction times using independent samples t-test
        across all measurements.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# 20 subjects, each measured 5 times before and 5 times after training
n_subjects = 20
n_measurements = 5

# Each subject has a baseline reaction time (between-subject variability)
subject_baselines = np.random.normal(loc=350, scale=40, size=n_subjects)
improvement = 15  # true mean improvement in ms

pre_all = []
post_all = []
for i in range(n_subjects):
    # Within-subject variability around their baseline
    pre = np.random.normal(loc=subject_baselines[i], scale=10, size=n_measurements)
    post = np.random.normal(
        loc=subject_baselines[i] - improvement, scale=10, size=n_measurements
    )
    pre_all.extend(pre)
    post_all.extend(post)

pre_all = np.array(pre_all)
post_all = np.array(post_all)

# Treat all 100 measurements as independent observations
t_stat, p_value = stats.ttest_ind(pre_all, post_all)

print(f"Pre-training:  n = {len(pre_all)}, mean = {np.mean(pre_all):.1f} ms")
print(f"Post-training: n = {len(post_all)}, mean = {np.mean(post_all):.1f} ms")
print(f"t-test: t = {t_stat:.3f}, p = {p_value:.6f}")
print(f"Conclusion: Training significantly improved reaction time "
      f"(t({len(pre_all)+len(post_all)-2}) = {t_stat:.2f}, p = {p_value:.6f}).")
'''

_SI = '''\
"""
Study: Does a training program improve reaction time?
Method: Average each subject's measurements, then compare pre vs post
        using a paired t-test (correct unit of analysis: subject, not measurement).
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# 20 subjects, each measured 5 times before and 5 times after training
n_subjects = 20
n_measurements = 5

# Each subject has a baseline reaction time (between-subject variability)
subject_baselines = np.random.normal(loc=350, scale=40, size=n_subjects)
improvement = 15  # true mean improvement in ms

subject_pre_means = []
subject_post_means = []
for i in range(n_subjects):
    pre = np.random.normal(loc=subject_baselines[i], scale=10, size=n_measurements)
    post = np.random.normal(
        loc=subject_baselines[i] - improvement, scale=10, size=n_measurements
    )
    subject_pre_means.append(np.mean(pre))
    subject_post_means.append(np.mean(post))

pre_means = np.array(subject_pre_means)
post_means = np.array(subject_post_means)

# Paired t-test on subject-level means (correct unit of analysis)
t_stat, p_value = stats.ttest_rel(pre_means, post_means)

print(f"Subjects: {n_subjects}, measurements per subject: {n_measurements}")
print(f"Pre-training mean:  {np.mean(pre_means):.1f} ms")
print(f"Post-training mean: {np.mean(post_means):.1f} ms")
print(f"Mean improvement:   {np.mean(pre_means - post_means):.1f} ms")
print(f"Paired t-test: t({n_subjects-1}) = {t_stat:.3f}, p = {p_value:.6f}")
if p_value < 0.05:
    print(f"Conclusion: Training significantly improved reaction time "
          f"(paired t({n_subjects-1}) = {t_stat:.2f}, p = {p_value:.6f}).")
else:
    print(f"Conclusion: No significant improvement detected "
          f"(paired t({n_subjects-1}) = {t_stat:.2f}, p = {p_value:.6f}).")
'''

# Registry: (content, is_flawed, category, description)
_registry = [
    (_SA, True,  "P-hacking",
     "Tests 20 hypotheses, reports only significant one without correction"),
    (_SB, False, "Survivorship bias",
     "Control: intent-to-treat, failed runs counted as 0"),
    (_SC, False, "Circular validation",
     "Control: cross-validation on training set for model selection"),
    (_SD, True,  "Pseudoreplication",
     "Treats repeated measures as independent (N=100 instead of N=20)"),
    (_SE, True,  "Data leakage",
     "StandardScaler fit on full dataset before train/test split"),
    (_SF, True,  "Survivorship bias",
     "Drops NaN/failed runs before computing summary statistics"),
    (_SG, False, "P-hacking",
     "Control: Bonferroni correction applied"),
    (_SH, False, "Wrong test assumption",
     "Control: normality check, Mann-Whitney U for non-normal data"),
    (_SI, False, "Pseudoreplication",
     "Control: paired t-test on subject-level means"),
    (_SJ, True,  "Circular validation",
     "Hyperparameters selected by maximizing test-set performance"),
    (_SK, False, "Data leakage",
     "Control: scaler fit on training data only"),
    (_SL, True,  "Wrong test assumption",
     "Parametric t-test on heavily skewed lognormal data"),
    # Variant set (2 flawed + 2 control per category)
    (_VA1, True,  "Data leakage",
     "MinMaxScaler fit on full dataset before split (credit scoring)"),
    (_VB1, False, "Data leakage",
     "Control: split first, scaler fit on training data only (credit scoring)"),
    (_VA2, True,  "Data leakage",
     "StandardScaler fit on full dataset before split (house price regression)"),
    (_VB2, False, "Data leakage",
     "Control: split first, scaler fit on training data only (house price)"),
    (_VC1, True,  "P-hacking",
     "13 engagement metrics tested, only significant one reported (A/B test)"),
    (_VD1, False, "P-hacking",
     "Control: Bonferroni correction across 13 engagement metrics"),
    (_VC2, True,  "P-hacking",
     "18 health outcomes tested, lowest p-value reported without correction"),
    (_VD2, False, "P-hacking",
     "Control: Bonferroni correction across 18 health outcomes"),
    (_VE1, True,  "Circular validation",
     "MLP architecture selected by test-set accuracy (tumour classification)"),
    (_VF1, False, "Circular validation",
     "Control: architecture selected via 5-fold CV on training data"),
    (_VE2, True,  "Circular validation",
     "Decision tree depth selected by test-set accuracy (employee attrition)"),
    (_VF2, False, "Circular validation",
     "Control: depth selected via cross-validation on training data"),
    (_VG1, True,  "Survivorship bias",
     "Clinical trial analyzes completers only, dropouts excluded"),
    (_VH1, False, "Survivorship bias",
     "Control: intent-to-treat with last-observation-carried-forward"),
    (_VG2, True,  "Survivorship bias",
     "Investment fund returns computed on surviving funds only"),
    (_VH2, False, "Survivorship bias",
     "Control: all funds included, including those that closed"),
    (_VI1, True,  "Wrong test assumption",
     "t-test on right-skewed social media engagement data"),
    (_VJ1, False, "Wrong test assumption",
     "Control: normality check, Mann-Whitney U for skewed engagement data"),
    (_VI2, True,  "Wrong test assumption",
     "t-test on raw lognormal antibody titres without log transform"),
    (_VJ2, False, "Wrong test assumption",
     "Control: log-transform titres, verify normality, t-test on log scale"),
    (_VK1, True,  "Pseudoreplication",
     "48 plants in 6 pots treated as independent (pseudoreplication)"),
    (_VL1, False, "Pseudoreplication",
     "Control: pot-level means used as unit of analysis"),
    (_VK2, True,  "Pseudoreplication",
     "250 students in 10 classes treated as independent"),
    (_VL2, False, "Pseudoreplication",
     "Control: class-level means used as unit of analysis"),
]

# Deterministic shuffle to assign opaque names
_rng = random.Random(42)
_indices = list(range(len(_registry)))
_rng.shuffle(_indices)

answer_key = {}
for new_idx, orig_idx in enumerate(_indices):
    content, is_flawed, category, description = _registry[orig_idx]
    opaque_name = f"script_{new_idx + 1:02d}.py"
    path = os.path.join(OUTPUT_DIR, opaque_name)
    with open(path, "w") as f:
        f.write(content.strip() + "\n")
    answer_key[opaque_name] = {
        "is_flawed": is_flawed,
        "category": category,
        "description": description,
    }

key_path = os.path.join(ANSWER_KEY_DIR, "answer_key.json")
with open(key_path, "w") as f:
    json.dump(answer_key, f, indent=2)

print(f"Generated {len(_registry)} scripts in {OUTPUT_DIR}/:")
for name in sorted(answer_key.keys()):
    print(f"  {name}")
print(f"\nAnswer key written to {key_path} (DO NOT read before reviewing)")
print(f"Scripts are in randomized order with opaque names.")
