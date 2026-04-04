# script_variants.py
# 24 new script templates for the statistical methodology benchmark.
# Each is a triple-quoted string variable containing a complete, runnable Python script.
#
# Naming convention:
#   _VA1, _VA2  — F1 flawed variants   (Data leakage)
#   _VB1, _VB2  — F1 control variants  (Data leakage)
#   _VC1, _VC2  — F2 flawed variants   (P-hacking)
#   _VD1, _VD2  — F2 control variants  (P-hacking)
#   _VE1, _VE2  — F3 flawed variants   (Circular validation)
#   _VF1, _VF2  — F3 control variants  (Circular validation)
#   _VG1, _VG2  — F4 flawed variants   (Survivorship bias)
#   _VH1, _VH2  — F4 control variants  (Survivorship bias)
#   _VI1, _VI2  — F5 flawed variants   (Wrong test assumption)
#   _VJ1, _VJ2  — F5 control variants  (Wrong test assumption)
#   _VK1, _VK2  — F6 flawed variants   (Pseudoreplication)
#   _VL1, _VL2  — F6 control variants  (Pseudoreplication)

# ---------------------------------------------------------------------------
# F1  DATA LEAKAGE
# ---------------------------------------------------------------------------

_VA1 = '''\
"""
Study: Predicting credit default risk from applicant financial features.
Method: Random forest classifier with MinMaxScaler normalization, 80/20 split.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(42)

# Simulate 800 loan applicants with 12 financial features
# (income, debt ratio, credit utilization, etc.)
n, p = 800, 12
X = np.random.randn(n, p)
true_coef = np.array([1.2, -0.8, 0.5, 0.4, -0.3, 0, 0, 0, 0, 0, 0, 0.7])
log_odds = X @ true_coef
prob = 1 / (1 + np.exp(-log_odds))
y = (np.random.rand(n) < prob).astype(int)

# Normalize all features to [0, 1] for the classifier
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # fit on full dataset before splitting

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, rf.predict(X_train))
test_acc  = accuracy_score(y_test,  rf.predict(X_test))
auc       = roc_auc_score(y_test,   rf.predict_proba(X_test)[:, 1])

print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Test AUC:       {auc:.3f}")
print(f"Conclusion: Random forest achieves {test_acc:.1%} accuracy "
      f"(AUC = {auc:.3f}) on held-out applicants.")
'''

_VA2 = '''\
"""
Study: Predicting residential property sale price from structural features.
Method: Linear regression with StandardScaler normalization; R² reported on test set.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

# Simulate 600 houses: square footage, bedrooms, bathrooms, age, lot size, etc.
n, p = 600, 8
X = np.random.randn(n, p)
true_coef = np.array([50000, -3000, 8000, -400, 12000, 2000, -500, 1500])
noise = np.random.randn(n) * 20000
y = 250000 + X @ true_coef + noise  # sale price in USD

# Standardize full feature matrix before any splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit on full dataset

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred_test = model.predict(X_test)
r2   = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Test R²:   {r2:.4f}")
print(f"Test RMSE: ${rmse:,.0f}")
print(f"Conclusion: The linear model explains {r2:.1%} of sale-price variance "
      f"on held-out properties (RMSE = ${rmse:,.0f}).")
'''

_VB1 = '''\
"""
Study: Predicting credit default risk from applicant financial features.
Method: Random forest classifier with MinMaxScaler normalization;
        scaler fit only on training data to prevent leakage.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(42)

# Simulate 800 loan applicants with 12 financial features
n, p = 800, 12
X = np.random.randn(n, p)
true_coef = np.array([1.2, -0.8, 0.5, 0.4, -0.3, 0, 0, 0, 0, 0, 0, 0.7])
log_odds = X @ true_coef
prob = 1 / (1 + np.exp(-log_odds))
y = (np.random.rand(n) < prob).astype(int)

# Split FIRST — before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler only on training data; apply to test without refitting
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Fit random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluate
train_acc = accuracy_score(y_train, rf.predict(X_train_scaled))
test_acc  = accuracy_score(y_test,  rf.predict(X_test_scaled))
auc       = roc_auc_score(y_test,   rf.predict_proba(X_test_scaled)[:, 1])

print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Test AUC:       {auc:.3f}")
print(f"Conclusion: Random forest achieves {test_acc:.1%} accuracy "
      f"(AUC = {auc:.3f}) on held-out applicants.")
'''

_VB2 = '''\
"""
Study: Predicting residential property sale price from structural features.
Method: Linear regression with StandardScaler normalization;
        scaler fit only on training fold to prevent data leakage.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

# Simulate 600 houses: square footage, bedrooms, bathrooms, age, lot size, etc.
n, p = 600, 8
X = np.random.randn(n, p)
true_coef = np.array([50000, -3000, 8000, -400, 12000, 2000, -500, 1500])
noise = np.random.randn(n) * 20000
y = 250000 + X @ true_coef + noise  # sale price in USD

# Split FIRST — before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler on training data only; apply transform to both sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Fit linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate on held-out test set
y_pred_test = model.predict(X_test_scaled)
r2   = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Test R²:   {r2:.4f}")
print(f"Test RMSE: ${rmse:,.0f}")
print(f"Conclusion: The linear model explains {r2:.1%} of sale-price variance "
      f"on held-out properties (RMSE = ${rmse:,.0f}).")
'''

# ---------------------------------------------------------------------------
# F2  P-HACKING
# ---------------------------------------------------------------------------

_VC1 = '''\
"""
Study: A/B test of a new mobile-app onboarding feature on user engagement.
Method: Independent-samples t-test across 13 engagement metrics;
        significant result reported.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_per_group = 200  # users per condition

# 13 engagement metrics (all generated independently of group membership)
metric_names = [
    "session_duration_sec", "pages_viewed", "clicks", "scroll_depth_pct",
    "return_visits_7d", "feature_uses", "share_events", "notifications_opened",
    "items_saved", "search_queries", "video_plays", "checkout_starts",
    "time_to_first_action_sec",
]

control = {m: np.random.exponential(scale=10, size=n_per_group) for m in metric_names}
treatment = {m: np.random.exponential(scale=10, size=n_per_group) for m in metric_names}

# Test all 13 metrics, collect significant ones
significant = []
for m in metric_names:
    t_stat, p = stats.ttest_ind(control[m], treatment[m])
    if p < 0.05:
        significant.append((m, t_stat, p))

# Report only the most significant finding
if significant:
    best = min(significant, key=lambda x: x[2])
    m, t_stat, p = best
    print(f"Metric: {m}")
    print(f"  Control mean:   {np.mean(control[m]):.2f}")
    print(f"  Treatment mean: {np.mean(treatment[m]):.2f}")
    print(f"  t = {t_stat:.3f}, p = {p:.4f}")
    print(f"Conclusion: The new onboarding feature significantly improved "
          f"{m} (p = {p:.4f}).")
else:
    print("No significant differences found.")
'''

_VC2 = '''\
"""
Study: Association between industrial air pollution index and health outcomes
       across 150 census tracts.
Method: Pearson correlation between PM2.5 levels and 18 health outcome rates;
        lowest p-value reported.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n = 150  # census tracts

pm25 = np.random.gamma(shape=3, scale=4, size=n)  # PM2.5 µg/m³

health_outcomes = [
    "asthma_rate", "copd_rate", "ischemic_heart_rate", "stroke_rate",
    "lung_cancer_rate", "hypertension_rate", "diabetes_rate", "obesity_rate",
    "preterm_birth_rate", "low_birthweight_rate", "infant_mortality_rate",
    "all_cause_mortality_rate", "respiratory_er_visits", "cardiovascular_er_visits",
    "pneumonia_rate", "allergy_rate", "sleep_disorder_rate", "anxiety_rate",
]

# All outcomes generated independently of PM2.5 (null is true for all)
outcomes = {name: np.random.normal(loc=50, scale=10, size=n)
            for name in health_outcomes}

results = []
for name, vals in outcomes.items():
    r, p = stats.pearsonr(pm25, vals)
    results.append((name, r, p))

# Report the most significant association without correction
best = min(results, key=lambda x: x[2])
name, r, p = best
print(f"PM2.5 vs {name}:")
print(f"  Pearson r = {r:.3f}, p = {p:.4f}")
print(f"  ({len(health_outcomes)} outcomes tested)")
print(f"Conclusion: Higher PM2.5 is significantly associated with {name} "
      f"(r = {r:.3f}, p = {p:.4f}).")
'''

_VD1 = '''\
"""
Study: A/B test of a new mobile-app onboarding feature on user engagement.
Method: Independent-samples t-test across 13 engagement metrics
        with Bonferroni correction for multiple comparisons.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_per_group = 200  # users per condition

metric_names = [
    "session_duration_sec", "pages_viewed", "clicks", "scroll_depth_pct",
    "return_visits_7d", "feature_uses", "share_events", "notifications_opened",
    "items_saved", "search_queries", "video_plays", "checkout_starts",
    "time_to_first_action_sec",
]

control = {m: np.random.exponential(scale=10, size=n_per_group) for m in metric_names}
treatment = {m: np.random.exponential(scale=10, size=n_per_group) for m in metric_names}

n_tests = len(metric_names)
alpha_corrected = 0.05 / n_tests  # Bonferroni correction

print(f"Testing {n_tests} metrics; Bonferroni-corrected alpha = {alpha_corrected:.5f}")
print()

significant = []
for m in metric_names:
    t_stat, p = stats.ttest_ind(control[m], treatment[m])
    flag = " *" if p < alpha_corrected else ""
    print(f"  {m:30s}: t = {t_stat:6.3f}, p = {p:.4f}{flag}")
    if p < alpha_corrected:
        significant.append((m, t_stat, p))

print()
if significant:
    print(f"Conclusion: {len(significant)} metric(s) significant after Bonferroni correction.")
else:
    print(f"Conclusion: No engagement metric showed a significant difference after "
          f"correcting for {n_tests} comparisons (Bonferroni alpha = {alpha_corrected:.5f}).")
'''

_VD2 = '''\
"""
Study: Association between industrial air pollution index and health outcomes
       across 150 census tracts.
Method: Pearson correlation between PM2.5 levels and 18 health outcome rates,
        with Bonferroni correction for multiple comparisons.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n = 150  # census tracts

pm25 = np.random.gamma(shape=3, scale=4, size=n)  # PM2.5 µg/m³

health_outcomes = [
    "asthma_rate", "copd_rate", "ischemic_heart_rate", "stroke_rate",
    "lung_cancer_rate", "hypertension_rate", "diabetes_rate", "obesity_rate",
    "preterm_birth_rate", "low_birthweight_rate", "infant_mortality_rate",
    "all_cause_mortality_rate", "respiratory_er_visits", "cardiovascular_er_visits",
    "pneumonia_rate", "allergy_rate", "sleep_disorder_rate", "anxiety_rate",
]

outcomes = {name: np.random.normal(loc=50, scale=10, size=n)
            for name in health_outcomes}

n_tests = len(health_outcomes)
alpha_corrected = 0.05 / n_tests  # Bonferroni correction

print(f"Testing {n_tests} health outcomes; Bonferroni-corrected alpha = {alpha_corrected:.5f}")
print()

significant = []
for name, vals in outcomes.items():
    r, p = stats.pearsonr(pm25, vals)
    flag = " *" if p < alpha_corrected else ""
    print(f"  {name:30s}: r = {r:6.3f}, p = {p:.4f}{flag}")
    if p < alpha_corrected:
        significant.append((name, r, p))

print()
if significant:
    print(f"Conclusion: {len(significant)} health outcome(s) significantly associated "
          f"with PM2.5 after Bonferroni correction.")
else:
    print(f"Conclusion: No health outcome was significantly associated with PM2.5 "
          f"after correcting for {n_tests} comparisons (Bonferroni alpha = {alpha_corrected:.5f}).")
'''

# ---------------------------------------------------------------------------
# F3  CIRCULAR VALIDATION
# ---------------------------------------------------------------------------

_VE1 = '''\
"""
Study: Classifying tumour malignancy from imaging-derived radiomic features.
Method: MLP neural network; architecture (hidden layer size) selected by
        test-set accuracy, then test accuracy reported as final performance.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Simulate 400 patients, 30 radiomic features
n, p = 400, 30
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:6] = [1.5, -1.2, 0.8, -0.6, 1.0, 0.4]
prob = 1 / (1 + np.exp(-(X @ true_w)))
y = (np.random.rand(n) < prob).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Evaluate candidate architectures on the test set to pick the best one
architectures = [
    (32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32),
]

best_arch, best_acc = None, -1
for arch in architectures:
    clf = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    if acc > best_acc:
        best_acc = acc
        best_arch = arch

# Report test accuracy of the architecture selected on the test set
final_clf = MLPClassifier(hidden_layer_sizes=best_arch, max_iter=500, random_state=42)
final_clf.fit(X_train, y_train)
final_acc = accuracy_score(y_test, final_clf.predict(X_test))

print(f"Best architecture: {best_arch}")
print(f"Test accuracy: {final_acc:.3f}")
print(f"Conclusion: MLP with architecture {best_arch} achieves {final_acc:.1%} "
      f"accuracy classifying tumour malignancy on held-out imaging data.")
'''

_VE2 = '''\
"""
Study: Predicting employee attrition from HR survey and performance features.
Method: Decision tree; max_depth selected by test-set accuracy,
        then test accuracy reported as final performance.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Simulate 500 employees, 15 features (job satisfaction, overtime, tenure, etc.)
n, p = 500, 15
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:5] = [1.1, -0.9, 0.6, 0.5, -0.4]
prob = 1 / (1 + np.exp(-(X @ true_w)))
y = (np.random.rand(n) < prob).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scan over max_depth values; pick depth with best test-set accuracy
depths = [2, 3, 4, 5, 6, 7, 8, 10, None]
best_depth, best_acc = None, -1
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    if acc > best_acc:
        best_acc = acc
        best_depth = d

# Retrain with selected depth and report test performance
final_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_clf.fit(X_train, y_train)
test_acc = accuracy_score(y_test, final_clf.predict(X_test))

print(f"Selected max_depth: {best_depth}")
print(f"Test accuracy:      {test_acc:.3f}")
print(f"Conclusion: Decision tree (max_depth={best_depth}) achieves {test_acc:.1%} "
      f"accuracy predicting employee attrition on held-out data.")
'''

_VF1 = '''\
"""
Study: Classifying tumour malignancy from imaging-derived radiomic features.
Method: MLP neural network; architecture selected via 5-fold cross-validation
        on training data; final performance reported on held-out test set.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Simulate 400 patients, 30 radiomic features
n, p = 400, 30
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:6] = [1.5, -1.2, 0.8, -0.6, 1.0, 0.4]
prob = 1 / (1 + np.exp(-(X @ true_w)))
y = (np.random.rand(n) < prob).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Hyperparameter search via 5-fold CV on training data only
param_grid = {
    "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
}
base_clf = MLPClassifier(max_iter=500, random_state=42)
gs = GridSearchCV(base_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)

best_arch = gs.best_params_["hidden_layer_sizes"]
cv_acc = gs.best_score_

# Evaluate on test set — never used during selection
test_acc = accuracy_score(y_test, gs.predict(X_test))

print(f"Best architecture (via 5-fold CV): {best_arch}")
print(f"CV accuracy (train):  {cv_acc:.3f}")
print(f"Test accuracy:        {test_acc:.3f}")
print(f"Conclusion: MLP with architecture {best_arch} (selected by CV) "
      f"achieves {test_acc:.1%} accuracy on held-out imaging data.")
'''

_VF2 = '''\
"""
Study: Predicting employee attrition from HR survey and performance features.
Method: Decision tree; max_depth selected via cross-validation on training data;
        final evaluation on a held-out test set.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Simulate 500 employees, 15 features
n, p = 500, 15
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:5] = [1.1, -0.9, 0.6, 0.5, -0.4]
prob = 1 / (1 + np.exp(-(X @ true_w)))
y = (np.random.rand(n) < prob).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Select max_depth using cross-validation on TRAINING data only
depths = [2, 3, 4, 5, 6, 7, 8, 10, None]
best_depth, best_cv_acc = None, -1
print("Depth selection via 5-fold CV on training data:")
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    mean_cv = cv_scores.mean()
    print(f"  max_depth={str(d):4s}: CV accuracy = {mean_cv:.3f} ± {cv_scores.std():.3f}")
    if mean_cv > best_cv_acc:
        best_cv_acc = mean_cv
        best_depth = d

# Retrain on full training set, evaluate on held-out test set
final_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_clf.fit(X_train, y_train)
test_acc = accuracy_score(y_test, final_clf.predict(X_test))

print(f"Selected max_depth: {best_depth} (CV accuracy = {best_cv_acc:.3f})")
print(f"Test accuracy:      {test_acc:.3f}")
print(f"Conclusion: Decision tree (max_depth={best_depth}, selected by CV) "
      f"achieves {test_acc:.1%} accuracy on held-out employee data.")
'''

# ---------------------------------------------------------------------------
# F4  SURVIVORSHIP BIAS
# ---------------------------------------------------------------------------

_VG1 = '''\
"""
Study: Efficacy of a 12-week weight-loss intervention in overweight adults.
Method: Compare baseline and follow-up BMI; analysis restricted to
        participants who completed all assessments.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_enrolled = 200  # participants randomised

# Baseline BMI (overweight/obese population)
bmi_baseline = np.random.normal(loc=32, scale=4, size=n_enrolled)

# True intervention effect: –1.5 BMI units on average
true_effect = -1.5
noise = np.random.randn(n_enrolled) * 2.0
bmi_followup = bmi_baseline + true_effect + noise

# 20% dropout — participants who drop out tend to be heavier (informative)
dropout_prob = 0.10 + 0.02 * (bmi_baseline - bmi_baseline.mean()) / bmi_baseline.std()
dropout_prob = np.clip(dropout_prob, 0, 1)
dropped = np.random.rand(n_enrolled) < dropout_prob
bmi_followup[dropped] = np.nan  # outcome missing for dropouts

# Analyse completers only (survivorship bias)
completers = ~dropped
bmi_base_c = bmi_baseline[completers]
bmi_fu_c   = bmi_followup[completers]

change = bmi_fu_c - bmi_base_c
t_stat, p = stats.ttest_1samp(change, popmean=0)

print(f"Enrolled: {n_enrolled}  |  Completers: {completers.sum()}  |  "
      f"Dropouts: {dropped.sum()}")
print(f"Mean BMI change (completers): {change.mean():.2f} units")
print(f"One-sample t-test: t = {t_stat:.3f}, p = {p:.4f}")
print(f"Conclusion: Among completers, the intervention reduced BMI by "
      f"{abs(change.mean()):.2f} units (p = {p:.4f}).")
'''

_VG2 = '''\
"""
Study: Ten-year performance of actively managed equity funds.
Method: Report mean annualised return across funds still operating at year 10.
"""
import numpy as np

np.random.seed(42)

n_funds = 50
n_years = 10

# Annual returns: mean ~6%, sd ~15% (equity-like)
annual_returns = np.random.normal(loc=0.06, scale=0.15, size=(n_funds, n_years))

# Cumulative return after each year
cumulative = np.cumprod(1 + annual_returns, axis=1)

# Funds with cumulative return < 1 at any point are "closed" (dropped from sample)
# In practice we observe only funds still open at year 10
survived = cumulative[:, -1] >= 1.0  # survived if still above water at end

surviving_annual = annual_returns[survived]
mean_annual_surviving = surviving_annual.mean()
mean_total_surviving  = (cumulative[survived, -1] - 1).mean()

print(f"Funds launched: {n_funds}")
print(f"Funds surviving to year 10: {survived.sum()}")
print(f"Mean annualised return (surviving funds): {mean_annual_surviving:.2%}")
print(f"Mean total 10-year return (surviving funds): {mean_total_surviving:.2%}")
print(f"Conclusion: Active equity funds delivered an average annualised return of "
      f"{mean_annual_surviving:.2%} over the decade.")
'''

_VH1 = '''\
"""
Study: Efficacy of a 12-week weight-loss intervention in overweight adults.
Method: Intent-to-treat analysis — last observation carried forward (LOCF)
        for participants who drop out.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_enrolled = 200  # participants randomised

# Baseline BMI
bmi_baseline = np.random.normal(loc=32, scale=4, size=n_enrolled)

# True intervention effect: –1.5 BMI units on average
true_effect = -1.5
noise = np.random.randn(n_enrolled) * 2.0
bmi_followup = bmi_baseline + true_effect + noise

# 20% informative dropout (heavier participants more likely to drop out)
dropout_prob = 0.10 + 0.02 * (bmi_baseline - bmi_baseline.mean()) / bmi_baseline.std()
dropout_prob = np.clip(dropout_prob, 0, 1)
dropped = np.random.rand(n_enrolled) < dropout_prob
bmi_followup[dropped] = np.nan

# Intent-to-treat: carry forward baseline BMI for dropouts (LOCF)
bmi_followup_itt = np.where(np.isnan(bmi_followup), bmi_baseline, bmi_followup)

change_itt = bmi_followup_itt - bmi_baseline
t_stat, p = stats.ttest_1samp(change_itt, popmean=0)

print(f"Enrolled: {n_enrolled}  |  Completers: {(~dropped).sum()}  |  "
      f"Dropouts (LOCF): {dropped.sum()}")
print(f"Mean BMI change (ITT): {change_itt.mean():.2f} units")
print(f"One-sample t-test: t = {t_stat:.3f}, p = {p:.4f}")
print(f"Conclusion: In the intent-to-treat population (n={n_enrolled}), "
      f"the intervention changed BMI by {change_itt.mean():.2f} units "
      f"(p = {p:.4f}).")
'''

_VH2 = '''\
"""
Study: Ten-year performance of actively managed equity funds.
Method: Report mean annualised return across ALL funds including those
        that closed during the period (no survivorship bias).
"""
import numpy as np

np.random.seed(42)

n_funds = 50
n_years = 10

# Annual returns: mean ~6%, sd ~15%
annual_returns = np.random.normal(loc=0.06, scale=0.15, size=(n_funds, n_years))

# Cumulative return after each year
cumulative = np.cumprod(1 + annual_returns, axis=1)

# Identify funds that "closed" (cumulative return fell below 1 at some point)
survived = cumulative[:, -1] >= 1.0
closed    = ~survived

# For closed funds, use their final 10-year cumulative return (which is < 0 net)
total_returns_all = cumulative[:, -1] - 1  # all funds, no filtering
mean_annual_all   = annual_returns.mean()   # includes all fund-years

print(f"Funds launched: {n_funds}")
print(f"Funds surviving to year 10: {survived.sum()}")
print(f"Funds closed before year 10: {closed.sum()}")
print(f"Mean 10-year return — surviving funds only: "
      f"{total_returns_all[survived].mean():.2%}")
print(f"Mean 10-year return — ALL funds (no survivorship filter): "
      f"{total_returns_all.mean():.2%}")
print(f"Mean annualised return — ALL funds: {mean_annual_all:.2%}")
print(f"Conclusion: Including all {n_funds} funds (incl. {closed.sum()} closed), "
      f"the average annualised return was {mean_annual_all:.2%}, "
      f"substantially lower than the survivors-only figure.")
'''

# ---------------------------------------------------------------------------
# F5  WRONG TEST ASSUMPTION
# ---------------------------------------------------------------------------

_VI1 = '''\
"""
Study: Comparing user engagement (post interactions) between two content
       formats (video vs. static image) on a social media platform.
Method: Independent-samples t-test on raw interaction counts.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_video  = 120  # video posts
n_static = 130  # static image posts

# Interaction counts are right-skewed (lognormal)
video_interactions  = np.random.lognormal(mean=3.5, sigma=1.2, size=n_video)
static_interactions = np.random.lognormal(mean=3.2, sigma=1.1, size=n_static)

t_stat, p_value = stats.ttest_ind(video_interactions, static_interactions)

print(f"Video posts   (n={n_video}):  mean={np.mean(video_interactions):.1f}, "
      f"median={np.median(video_interactions):.1f}, "
      f"SD={np.std(video_interactions):.1f}")
print(f"Static posts  (n={n_static}): mean={np.mean(static_interactions):.1f}, "
      f"median={np.median(static_interactions):.1f}, "
      f"SD={np.std(static_interactions):.1f}")
print(f"Independent t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print(f"Conclusion: Video posts generated significantly more interactions "
          f"than static images (t = {t_stat:.3f}, p = {p_value:.4f}).")
else:
    print(f"Conclusion: No significant difference in interactions between "
          f"content formats (p = {p_value:.4f}).")
'''

_VI2 = '''\
"""
Study: Comparing serum antibody titres between vaccinated and unvaccinated
       control groups four weeks post-injection.
Method: Independent-samples t-test on raw antibody titre measurements.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_vacc  = 80  # vaccinated participants
n_ctrl  = 75  # unvaccinated controls

# Antibody titres follow a lognormal distribution
titres_vacc = np.random.lognormal(mean=5.0, sigma=0.9, size=n_vacc)
titres_ctrl = np.random.lognormal(mean=3.5, sigma=0.8, size=n_ctrl)

t_stat, p_value = stats.ttest_ind(titres_vacc, titres_ctrl)

print(f"Vaccinated  (n={n_vacc}): mean titre = {np.mean(titres_vacc):.1f}, "
      f"SD = {np.std(titres_vacc):.1f}")
print(f"Controls    (n={n_ctrl}): mean titre = {np.mean(titres_ctrl):.1f}, "
      f"SD = {np.std(titres_ctrl):.1f}")
print(f"Independent t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print(f"Conclusion: Vaccinated participants had significantly higher antibody "
          f"titres than controls (t = {t_stat:.3f}, p = {p_value:.4f}).")
else:
    print(f"Conclusion: No significant titre difference between groups "
          f"(p = {p_value:.4f}).")
'''

_VJ1 = '''\
"""
Study: Comparing user engagement (post interactions) between two content
       formats (video vs. static image) on a social media platform.
Method: Shapiro-Wilk normality check on each group; Mann-Whitney U test
        used because interaction counts are right-skewed.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_video  = 120
n_static = 130

# Interaction counts are right-skewed (lognormal)
video_interactions  = np.random.lognormal(mean=3.5, sigma=1.2, size=n_video)
static_interactions = np.random.lognormal(mean=3.2, sigma=1.1, size=n_static)

# Check normality
_, p_norm_video  = stats.shapiro(video_interactions[:50])   # Shapiro limited to n≤5000
_, p_norm_static = stats.shapiro(static_interactions[:50])

print(f"Video posts   (n={n_video}):  mean={np.mean(video_interactions):.1f}, "
      f"median={np.median(video_interactions):.1f}")
print(f"Static posts  (n={n_static}): mean={np.mean(static_interactions):.1f}, "
      f"median={np.median(static_interactions):.1f}")
print(f"Shapiro-Wilk (subsample n=50): video p={p_norm_video:.4f}, "
      f"static p={p_norm_static:.4f}")

if p_norm_video < 0.05 or p_norm_static < 0.05:
    print("Normality assumption violated; using Mann-Whitney U test.")
    u_stat, p_value = stats.mannwhitneyu(
        video_interactions, static_interactions, alternative="two-sided"
    )
    print(f"Mann-Whitney U: U = {u_stat:.1f}, p = {p_value:.4f}")
else:
    _, p_value = stats.ttest_ind(video_interactions, static_interactions)
    print(f"t-test: p = {p_value:.4f}")

if p_value < 0.05:
    print(f"Conclusion: Video posts generated significantly more interactions "
          f"than static images (Mann-Whitney U, p = {p_value:.4f}).")
else:
    print(f"Conclusion: No significant difference in interactions between "
          f"content formats (p = {p_value:.4f}).")
'''

_VJ2 = '''\
"""
Study: Comparing serum antibody titres between vaccinated and unvaccinated
       control groups four weeks post-injection.
Method: Log-transform titres, verify normality, then apply t-test on log scale.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_vacc  = 80
n_ctrl  = 75

# Antibody titres are lognormally distributed
titres_vacc = np.random.lognormal(mean=5.0, sigma=0.9, size=n_vacc)
titres_ctrl = np.random.lognormal(mean=3.5, sigma=0.8, size=n_ctrl)

print(f"Raw titres — vaccinated: mean={np.mean(titres_vacc):.1f}, "
      f"median={np.median(titres_vacc):.1f}")
print(f"Raw titres — controls:   mean={np.mean(titres_ctrl):.1f}, "
      f"median={np.median(titres_ctrl):.1f}")

# Log-transform (standard for titre data)
log_vacc = np.log(titres_vacc)
log_ctrl = np.log(titres_ctrl)

# Verify normality after transformation
_, p_norm_vacc = stats.shapiro(log_vacc)
_, p_norm_ctrl = stats.shapiro(log_ctrl)
print(f"Shapiro-Wilk on log titres: vaccinated p={p_norm_vacc:.4f}, "
      f"controls p={p_norm_ctrl:.4f}")
print(f"Log scale — vaccinated: mean={log_vacc.mean():.3f}, "
      f"SD={log_vacc.std():.3f}")
print(f"Log scale — controls:   mean={log_ctrl.mean():.3f}, "
      f"SD={log_ctrl.std():.3f}")

# t-test on log-transformed titres
t_stat, p_value = stats.ttest_ind(log_vacc, log_ctrl)
fold_change = np.exp(log_vacc.mean() - log_ctrl.mean())

print(f"t-test on log titres: t = {t_stat:.3f}, p = {p_value:.4f}")
print(f"Geometric mean fold-change: {fold_change:.2f}x")
if p_value < 0.05:
    print(f"Conclusion: Vaccinated participants had significantly higher antibody "
          f"titres than controls ({fold_change:.1f}-fold, log-scale t-test "
          f"p = {p_value:.4f}).")
else:
    print(f"Conclusion: No significant titre difference (log-scale t-test, "
          f"p = {p_value:.4f}).")
'''

# ---------------------------------------------------------------------------
# F6  PSEUDOREPLICATION
# ---------------------------------------------------------------------------

_VK1 = '''\
"""
Study: Effect of a soil amendment on wheat seedling growth in controlled
       greenhouse conditions.
Method: Compare shoot length between treatment and control plants;
        treat all individual plant measurements as independent observations.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_pots_per_group = 6  # pots per condition
n_plants_per_pot = 8  # plants per pot

# Pot-level random effect (shared environment within a pot)
control_pot_means   = np.random.normal(loc=18.0, scale=2.5, size=n_pots_per_group)
treatment_pot_means = np.random.normal(loc=20.5, scale=2.5, size=n_pots_per_group)

# Plant-level measurements within each pot
control_plants   = []
treatment_plants = []
for i in range(n_pots_per_group):
    control_plants.extend(
        np.random.normal(loc=control_pot_means[i], scale=0.8, size=n_plants_per_pot)
    )
    treatment_plants.extend(
        np.random.normal(loc=treatment_pot_means[i], scale=0.8, size=n_plants_per_pot)
    )

control_plants   = np.array(control_plants)
treatment_plants = np.array(treatment_plants)

# Incorrectly treat 48 plants per group as independent observations
t_stat, p_value = stats.ttest_ind(control_plants, treatment_plants)

print(f"Control   (n={len(control_plants)} plants): "
      f"mean = {control_plants.mean():.2f} cm, SD = {control_plants.std():.2f}")
print(f"Treatment (n={len(treatment_plants)} plants): "
      f"mean = {treatment_plants.mean():.2f} cm, SD = {treatment_plants.std():.2f}")
print(f"t-test (plants as unit): t = {t_stat:.3f}, p = {p_value:.4f}")
print(f"Conclusion: The soil amendment significantly increased shoot length "
      f"(t = {t_stat:.3f}, p = {p_value:.4f}, n = {len(control_plants)} plants "
      f"per group).")
'''

_VK2 = '''\
"""
Study: Effect of a peer-tutoring intervention on maths test scores in
       secondary school students.
Method: Compare test scores between intervention and control classes;
        treat all individual student scores as independent observations.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_classes_per_group = 10   # classes per condition
n_students_per_class = 25  # students per class

# Class-level random effect (shared teacher, environment)
control_class_means      = np.random.normal(loc=65, scale=6, size=n_classes_per_group)
intervention_class_means = np.random.normal(loc=68, scale=6, size=n_classes_per_group)

# Student scores within each class
control_scores      = []
intervention_scores = []
for i in range(n_classes_per_group):
    control_scores.extend(
        np.random.normal(loc=control_class_means[i], scale=8, size=n_students_per_class)
    )
    intervention_scores.extend(
        np.random.normal(loc=intervention_class_means[i], scale=8,
                         size=n_students_per_class)
    )

control_scores      = np.array(control_scores)
intervention_scores = np.array(intervention_scores)

# Incorrectly treat 250 students per group as independent observations
t_stat, p_value = stats.ttest_ind(control_scores, intervention_scores)

print(f"Control      (n={len(control_scores)} students): "
      f"mean = {control_scores.mean():.1f}, SD = {control_scores.std():.1f}")
print(f"Intervention (n={len(intervention_scores)} students): "
      f"mean = {intervention_scores.mean():.1f}, SD = {intervention_scores.std():.1f}")
print(f"t-test (students as unit): t = {t_stat:.3f}, p = {p_value:.4f}")
print(f"Conclusion: The peer-tutoring intervention significantly improved "
      f"maths scores (t = {t_stat:.3f}, p = {p_value:.4f}, "
      f"n = {len(control_scores)} students per group).")
'''

_VL1 = '''\
"""
Study: Effect of a soil amendment on wheat seedling growth in controlled
       greenhouse conditions.
Method: Average shoot length per pot, then compare pot-level means between
        treatment and control (correct unit of analysis: the pot).
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_pots_per_group = 6  # pots per condition
n_plants_per_pot = 8  # plants per pot

# Pot-level random effect
control_pot_means   = np.random.normal(loc=18.0, scale=2.5, size=n_pots_per_group)
treatment_pot_means = np.random.normal(loc=20.5, scale=2.5, size=n_pots_per_group)

# Plant-level measurements within each pot
control_pot_avgs   = []
treatment_pot_avgs = []
for i in range(n_pots_per_group):
    ctrl_plants = np.random.normal(
        loc=control_pot_means[i], scale=0.8, size=n_plants_per_pot
    )
    trt_plants = np.random.normal(
        loc=treatment_pot_means[i], scale=0.8, size=n_plants_per_pot
    )
    control_pot_avgs.append(ctrl_plants.mean())
    treatment_pot_avgs.append(trt_plants.mean())

control_pot_avgs   = np.array(control_pot_avgs)
treatment_pot_avgs = np.array(treatment_pot_avgs)

# Correct unit of analysis: pot means (n=6 per group)
t_stat, p_value = stats.ttest_ind(control_pot_avgs, treatment_pot_avgs)

print(f"Control   ({n_pots_per_group} pots, {n_plants_per_pot} plants/pot): "
      f"pot mean = {control_pot_avgs.mean():.2f} cm ± {control_pot_avgs.std():.2f}")
print(f"Treatment ({n_pots_per_group} pots, {n_plants_per_pot} plants/pot): "
      f"pot mean = {treatment_pot_avgs.mean():.2f} cm ± {treatment_pot_avgs.std():.2f}")
print(f"t-test on pot means (n={n_pots_per_group}/group): "
      f"t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print(f"Conclusion: The soil amendment significantly increased shoot length "
          f"(t({2*n_pots_per_group-2}) = {t_stat:.3f}, p = {p_value:.4f}, "
          f"n = {n_pots_per_group} pots per group).")
else:
    print(f"Conclusion: No significant effect of soil amendment on shoot length "
          f"(t({2*n_pots_per_group-2}) = {t_stat:.3f}, p = {p_value:.4f}, "
          f"n = {n_pots_per_group} pots per group).")
'''

_VL2 = '''\
"""
Study: Effect of a peer-tutoring intervention on maths test scores in
       secondary school students.
Method: Average test score per class, then compare class-level means between
        conditions (correct unit of analysis: the class, not the student).
"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_classes_per_group = 10   # classes per condition
n_students_per_class = 25  # students per class

# Class-level random effect
control_class_means      = np.random.normal(loc=65, scale=6, size=n_classes_per_group)
intervention_class_means = np.random.normal(loc=68, scale=6, size=n_classes_per_group)

# Compute class-average scores (correct aggregation)
control_class_avgs      = []
intervention_class_avgs = []
for i in range(n_classes_per_group):
    ctrl_students = np.random.normal(
        loc=control_class_means[i], scale=8, size=n_students_per_class
    )
    int_students = np.random.normal(
        loc=intervention_class_means[i], scale=8, size=n_students_per_class
    )
    control_class_avgs.append(ctrl_students.mean())
    intervention_class_avgs.append(int_students.mean())

control_class_avgs      = np.array(control_class_avgs)
intervention_class_avgs = np.array(intervention_class_avgs)

# Correct unit of analysis: class means (n=10 per group)
t_stat, p_value = stats.ttest_ind(control_class_avgs, intervention_class_avgs)

print(f"Control      ({n_classes_per_group} classes, {n_students_per_class} students/class): "
      f"class mean = {control_class_avgs.mean():.1f} ± {control_class_avgs.std():.1f}")
print(f"Intervention ({n_classes_per_group} classes, {n_students_per_class} students/class): "
      f"class mean = {intervention_class_avgs.mean():.1f} ± {intervention_class_avgs.std():.1f}")
print(f"t-test on class means (n={n_classes_per_group}/group): "
      f"t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print(f"Conclusion: The peer-tutoring intervention significantly improved "
          f"maths scores (t({2*n_classes_per_group-2}) = {t_stat:.3f}, "
          f"p = {p_value:.4f}, n = {n_classes_per_group} classes per group).")
else:
    print(f"Conclusion: No significant effect of peer tutoring on maths scores "
          f"(t({2*n_classes_per_group-2}) = {t_stat:.3f}, p = {p_value:.4f}, "
          f"n = {n_classes_per_group} classes per group).")
'''
