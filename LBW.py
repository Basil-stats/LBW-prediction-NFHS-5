
# ============ SETTINGS (edit these) ============
DATA_PATH    = r"C:\path\to\data.csv"      # exported by codes.do
FAST         = False         
RANDOM_STATE = 42
TEST_SIZE    = 0.30
OUT_DIR      = r"C:\path\to\lbw_outputs"
# ===============================================

N_ITER = 4 if FAST else 10     # random-search candidates per model
N_FOLD = 3 if FAST else 5      # CV folds
N_BOOT = 100 if FAST else 1000 # bootstrap resamples for CIs

# %%
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             confusion_matrix, brier_score_loss, f1_score)
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline            
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)

# %% 
# ## 1. Load data and build the analytical sample

# %%
df = pd.read_csv(DATA_PATH, na_values=[".", ".a", ".b", ".c"])   # Stata extended missing codes -> NaN
print(f"Rows in file: {len(df):,}")

# Analytical sample: complete cases on all Model B variables + plausible anthropometry.
NEEDED = ["bwt", "aom", "education", "wealth", "religion", "residence", "marsta", "bo", "bi", "pregint",
          "anc", "morbidity", "smoking", "height", "weight", "anemia", "sex", "tod", "pod", "region"]
df = df.dropna(subset=NEEDED)
df = df[df["height"].between(130, 200) & df["weight"].between(25, 150)].reset_index(drop=True)
print(f"Analytical sample after complete-case + plausibility filters: {len(df):,}")
y  = df["bwt"].astype(int)
has_wt = "wt" in df.columns

n, n_lbw = len(df), int(y.sum())
print(f"Analytical sample: {n:,} births | LBW: {n_lbw:,}")
print(f"Unweighted LBW prevalence: {n_lbw} / {n} = {100*n_lbw/n:.2f}%")
if has_wt:
    w = df["wt"]
    print(f"Weighted LBW prevalence (v005): {100*np.average(y, weights=w):.2f}%")
else:
    print("No 'wt' column -> weighted prevalence not computed. Add v005/1e6 as 'wt' to report it.")

# summary of maternal anthropometry
print(df[["height", "weight"]].describe().round(1))

# %% 
# ## 2. Feature sets: Model A (antenatal, primary) and Model B (full, comparison)

# %%
ANTENATAL = ["aom", "education", "wealth", "religion", "residence", "marsta",
             "bo", "bi", "pregint", "anc", "morbidity", "smoking", "height", "region"]
POST_EXTRA = ["weight", "anemia", "sex", "tod", "pod"]
FULL = ANTENATAL + POST_EXTRA

NOMINAL = ["religion", "marsta", "pod", "region"]   # unordered categories -> one-hot
FEATURE_SETS = {"A_antenatal": ANTENATAL, "B_full": FULL}

missing = [c for c in FULL if c not in df.columns]
assert not missing, f"Columns not found in data: {missing}"

def make_preprocessor(features):
    nom = [c for c in features if c in NOMINAL]
    num = [c for c in features if c not in NOMINAL]
    return ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), nom),
        ("num", StandardScaler(), num),
    ])

# %% 
# ## 3. Split once: 70% training (all CV, tuning and SMOTE happen here) / 30% untouched test

# %%
idx_train, idx_test = train_test_split(df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
y_train, y_test = y.loc[idx_train], y.loc[idx_test]
print(f"Train: {len(idx_train):,} (LBW {100*y_train.mean():.2f}%) | Test: {len(idx_test):,} (LBW {100*y_test.mean():.2f}%)")
print("The test set keeps the ORIGINAL class distribution and is never resampled.")

# %% 
# ## 4. Models, imbalance strategies and hyperparameter search spaces

# %%
spw = (y_train == 0).sum() / (y_train == 1).sum()   # for XGBoost class weighting

def get_models(strategy):
    # strategy = 'smote' (SMOTE inside folds) or 'weighted' (class weights, no resampling)
    cw = "balanced" if strategy == "weighted" else None
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight=cw),
                                {"clf__C": loguniform(1e-3, 10)}),
        "Decision Tree": (DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_STATE),
                          {"clf__max_depth": [3, 4, 5, 6, 8, 10], "clf__min_samples_leaf": [20, 50, 100, 200]}),
        "Random Forest": (RandomForestClassifier(class_weight=cw, n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
                          {"clf__max_depth": [4, 6, 8, 12, None], "clf__min_samples_leaf": [5, 10, 25, 50],
                           "clf__max_features": ["sqrt", 0.5]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
                                  scale_pos_weight=(spw if strategy == "weighted" else 1)),
                    {"clf__n_estimators": [200, 400], "clf__max_depth": [2, 3, 4, 6],
                     "clf__learning_rate": [0.01, 0.03, 0.1], "clf__subsample": [0.7, 0.9],
                     "clf__colsample_bytree": [0.7, 0.9], "clf__min_child_weight": [1, 5, 10]}),
        # MLP has no class_weight option -> under 'weighted' it is trained unweighted (threshold tuning handles imbalance)
        "Neural Network": (MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300,
                                         random_state=RANDOM_STATE),
                           {"clf__alpha": loguniform(1e-5, 1e-1), "clf__learning_rate_init": [1e-3, 3e-3, 1e-2],
                            "clf__hidden_layer_sizes": [(32,), (64, 32)]}),
    }

def build_pipeline(features, clf, strategy):
    steps = [("prep", make_preprocessor(features))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))   # applied to training folds only
    steps.append(("clf", clf))
    return Pipeline(steps)

# %% 
# ## 5. Metrics helpers (bootstrap CIs, threshold from training out-of-fold predictions)

# %%
def youden_threshold(y_true, prob):
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])

def boot_ci(y_true, prob, fn, n_boot=N_BOOT):
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        stats.append(fn(y_true[i], prob[i]))
    return np.percentile(stats, [2.5, 97.5])

def evaluate(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y_true, pred).ravel()
    auc, pr = roc_auc_score(y_true, prob), average_precision_score(y_true, prob)
    auc_lo, auc_hi = boot_ci(y_true, prob, roc_auc_score)
    pr_lo, pr_hi = boot_ci(y_true, prob, average_precision_score)
    return {
        "Threshold": thr,
        "ROC-AUC": auc, "ROC-AUC 95% CI": f"{auc_lo:.3f}-{auc_hi:.3f}",
        "PR-AUC": pr, "PR-AUC 95% CI": f"{pr_lo:.3f}-{pr_hi:.3f}",
        "Sensitivity": tp / (tp + fn_), "Specificity": tn / (tn + fp),
        "PPV (Precision)": tp / (tp + fp) if (tp + fp) else np.nan, "NPV": tn / (tn + fn_),
        "F1": f1_score(y_true, pred), "Accuracy": (tp + tn) / len(y_true),
        "Brier": brier_score_loss(y_true, prob),
        "TP": tp, "FP": fp, "FN": fn_, "TN": tn,
    }

# %% 
# ## 6. Train: tuning + 5-fold CV inside the training set, then ONE evaluation on the test set

# %%
cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_STATE)
results, fitted, test_probs = [], {}, {}

for fs_name, features in FEATURE_SETS.items():
    X_train, X_test = df.loc[idx_train, features], df.loc[idx_test, features]
    for strategy in ["smote", "weighted"]:
        for m_name, (clf, space) in get_models(strategy).items():
            pipe = build_pipeline(features, clf, strategy)
            search = RandomizedSearchCV(pipe, space, n_iter=N_ITER, scoring="roc_auc", cv=cv,
                                        random_state=RANDOM_STATE, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            b = search.best_index_
            cv_auc_mean = search.cv_results_["mean_test_score"][b]
            cv_auc_sd = search.cv_results_["std_test_score"][b]

            # out-of-fold probabilities on TRAINING data -> choose threshold (test set not touched)
            oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            thr = youden_threshold(y_train, oof)

            prob = best.predict_proba(X_test)[:, 1]
            row = {"Model set": fs_name, "Imbalance": strategy, "Algorithm": m_name,
                   "CV ROC-AUC (mean)": cv_auc_mean, "CV ROC-AUC (SD)": cv_auc_sd}
            row.update(evaluate(y_test, prob, thr))
            results.append(row)
            key = (fs_name, strategy, m_name)
            fitted[key], test_probs[key] = best, prob
            print(f"{fs_name:12s} | {strategy:8s} | {m_name:20s} | CV AUC {cv_auc_mean:.3f}±{cv_auc_sd:.3f} "
                  f"| Test AUC {row['ROC-AUC']:.3f} | Sens {row['Sensitivity']:.2f} | Spec {row['Specificity']:.2f}")

res = pd.DataFrame(results)
res.to_csv(f"{OUT_DIR}/performance_all_models.csv", index=False)

# %%
# Model A (antenatal), both imbalance strategies
show = ["Imbalance", "Algorithm", "CV ROC-AUC (mean)", "CV ROC-AUC (SD)", "ROC-AUC", "ROC-AUC 95% CI", "PR-AUC",
        "PR-AUC 95% CI", "Sensitivity", "Specificity", "PPV (Precision)", "NPV", "F1", "Brier", "TP", "FP", "FN", "TN"]
for fs in FEATURE_SETS:
    print(f"\n===== {fs} — evaluated on the ORIGINAL (unbalanced) test set =====")
    display(res.loc[res["Model set"] == fs, show].round(3))
print(f"\nReference: PR-AUC of a no-skill model = LBW prevalence in test set = {y_test.mean():.3f}")

# %% 
# ## 7. ROC, precision–recall and calibration curves (test set)

# %%
def plot_curves(fs_name, strategy):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for (fs, st, m), prob in test_probs.items():
        if fs != fs_name or st != strategy:
            continue
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax[0].plot(fpr, tpr, label=f"{m} (AUC {roc_auc_score(y_test, prob):.2f})")
        p, r, _ = precision_recall_curve(y_test, prob)
        ax[1].plot(r, p, label=f"{m} (AP {average_precision_score(y_test, prob):.2f})")
        fr, mp = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
        ax[2].plot(mp, fr, marker="o", label=m)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC")
    ax[1].axhline(y_test.mean(), color="k", ls="--"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[2].plot([0, 1], [0, 1], "k--"); ax[2].set(xlabel="Predicted probability", ylabel="Observed LBW proportion", title="Calibration")
    for a in ax: a.legend(fontsize=8)
    fig.suptitle(f"{fs_name} | {strategy} | original test set")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/curves_{fs_name}_{strategy}.png", dpi=300); plt.show()

for fs in FEATURE_SETS:
    for st in ["smote", "weighted"]:
        plot_curves(fs, st)

# %%
# Confusion matrices for Model A (antenatal) - class-weighted 
plt.close("all")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, m in zip(axes, get_models("weighted").keys()):
    r = res[(res["Model set"] == "A_antenatal") & (res["Imbalance"] == "weighted") & (res["Algorithm"] == m)].iloc[0]
    cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred NBW", "Pred LBW"], yticklabels=["NBW", "LBW"], title=m)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/confusion_A_antenatal_weighted.png", dpi=300); plt.close(fig)



# %% 
# ## 8. SHAP for the best antenatal model (chosen by **CV** AUC, not test AUC, to avoid test-set selection)

# %%
import shap
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import brier_score_loss as _brier

subA = res[res["Model set"] == "A_antenatal"]
best_row = subA.loc[subA["CV ROC-AUC (mean)"].idxmax()]
key = ("A_antenatal", best_row["Imbalance"], best_row["Algorithm"])
print("Selected:", key)
pipe = fitted[key]

# ---- recalibration (Platt scaling on out-of-fold TRAINING predictions; test set untouched) ----
X_tr_A, X_te_A = df.loc[idx_train, ANTENATAL], df.loc[idx_test, ANTENATAL]
oof = cross_val_predict(pipe, X_tr_A, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
_logit = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
platt = _LR().fit(_logit(oof).reshape(-1, 1), y_train)
p_raw = test_probs[key]
p_cal = platt.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
print(f"Brier raw {_brier(y_test, p_raw):.3f} | recalibrated {_brier(y_test, p_cal):.3f} | "
      f"no-skill {_brier(y_test, np.full(len(y_test), y_train.mean())):.3f}")

plt.close("all")
fig, ax = plt.subplots(figsize=(5.5, 5))
for p, lab in [(p_raw, "Raw"), (p_cal, "Recalibrated (Platt)")]:
    fr, mp = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, marker="o", label=lab)
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlim=(0, .7), ylim=(0, .7),
       xlabel="Predicted probability of LBW", ylabel="Observed proportion LBW")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT_DIR}/calibration_final.png", dpi=300); plt.close(fig)

# ---- SHAP with readable labels ----
LABELS = {"aom": "Maternal age at birth", "education": "Maternal education", "wealth": "Wealth index",
          "religion_2": "Religion: Muslim", "religion_3": "Religion: Christian", "religion_4": "Religion: Other",
          "religion_2.0": "Religion: Muslim", "religion_3.0": "Religion: Christian", "religion_4.0": "Religion: Other",
          "residence": "Urban residence", "marsta_2": "Widowed", "marsta_3": "Divorced/separated/other",
          "marsta_2.0": "Widowed", "marsta_3.0": "Divorced/separated/other",
          "bo": "Birth order", "bi": "Preceding birth interval", "pregint": "Intended pregnancy",
          "anc": "Antenatal care visits", "morbidity": "Pregnancy complications",
          "smoking": "Tobacco use", "height": "Maternal height",
          "region_2": "Region: Central", "region_3": "Region: East", "region_4": "Region: Northeast",
          "region_5": "Region: West", "region_6": "Region: South",
          "region_2.0": "Region: Central", "region_3.0": "Region: East", "region_4.0": "Region: Northeast",
          "region_5.0": "Region: West", "region_6.0": "Region: South"}
prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
X_test_t = pd.DataFrame(prep.transform(X_te_A), columns=[c.split("__", 1)[1] for c in prep.get_feature_names_out()])
X_test_t.columns = [LABELS.get(c, c) for c in X_test_t.columns]
X_shap = X_test_t.sample(min(5000, len(X_test_t)), random_state=RANDOM_STATE)

if best_row["Algorithm"] in ["Random Forest", "XGBoost", "Decision Tree"]:
    sv = shap.TreeExplainer(clf).shap_values(X_shap)
    if isinstance(sv, list): sv = sv[1]
    if sv.ndim == 3: sv = sv[:, :, 1]
else:
    X_shap = X_shap.iloc[:300]
    sv = shap.KernelExplainer(lambda d: clf.predict_proba(d)[:, 1],
                              shap.sample(X_test_t, 200, random_state=RANDOM_STATE)).shap_values(X_shap)

# bar plot of mean |SHAP|
plt.close("all")
shap.summary_plot(sv, X_shap, plot_type="bar", show=False, max_display=15)
plt.xlabel("Mean |SHAP value| (log-odds scale)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/shap_bar_A_antenatal.png", dpi=300, bbox_inches="tight"); plt.close("all")

# beeswarm plot
shap.summary_plot(sv, X_shap, show=False, max_display=15)
plt.xlabel("SHAP value (log-odds of LBW)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/shap_beeswarm_A_antenatal.png", dpi=300, bbox_inches="tight"); plt.close("all")

pd.Series(np.abs(sv).mean(0), index=X_shap.columns).sort_values(ascending=False) \
  .round(4).to_csv(f"{OUT_DIR}/shap_importance.csv", header=["mean_abs_shap"])

# %% 
# ## 9. Survey-weighted sensitivity check 

# %%
if has_wt:
    w_test = df.loc[idx_test, "wt"]
    rows = []
    for (fs, st, m), prob in test_probs.items():
        rows.append({"Model set": fs, "Imbalance": st, "Algorithm": m,
                     "Unweighted AUC": roc_auc_score(y_test, prob),
                     "Weighted AUC": roc_auc_score(y_test, prob, sample_weight=w_test)})
    display(pd.DataFrame(rows).round(3))
else:
    print("Skipped: add 'wt' (= v005/1e6) to data.csv to run this.")

# %% 
# ## 10. Table 1 check 

# %%
def table1(data, vars_):
    out = []
    for v in vars_:
        for cat, g in data.groupby(v):
            n_l, n_n = int(g.bwt.sum()), int((g.bwt == 0).sum())
            row = {"Variable": v, "Category (code)": cat, "LBW n": n_l, "NBW n": n_n,
                   "LBW row % (unweighted)": 100 * n_l / (n_l + n_n)}
            row["NBW row % (unweighted)"] = 100 - row["LBW row % (unweighted)"]
            if has_wt:
                row["LBW row % (weighted)"] = 100 * np.average(g.bwt, weights=g.wt)
            out.append(row)
    t = pd.DataFrame(out)
    # check that categories of each variable sum to the sample size
    for v in vars_:
        s = t.loc[t.Variable == v, ["LBW n", "NBW n"]].sum().sum()
        if s != len(data):
            print(f"WARNING: '{v}' categories sum to {s:,}, not {len(data):,} (missing codes?)")
    return t.round(2)

t1 = table1(df, ["aom", "sex", "marsta", "religion", "residence", "region", "education", "wealth", "bo", "bi",
                 "smoking", "morbidity", "anc", "tod", "pod", "anemia", "pregint"])
t1.to_csv(f"{OUT_DIR}/table1_check.csv", index=False)
display(t1)
print("LBW% + NBW% = 100 in every row by construction.")
