"""Stage 2 - Tasks 2/3/4: Random Forest pipeline + TreeExplainer, force plot, summary plot."""
import json
import os
import platform
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from _paths import DATA, OUT, FIG
os.makedirs(FIG, exist_ok=True)
SEED = 2021

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

# ---------------------------------------------------------------- data & split
raw = pd.read_csv(DATA)
df = raw.drop(columns=["id"])
X = df.drop(columns=["stroke"])
y = df["stroke"]
NUM = ["age", "avg_glucose_level", "bmi"]
CAT = [c for c in X.columns if c not in NUM]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
])
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.5,
                            class_weight=None, random_state=SEED, n_jobs=1)
pipe = Pipeline([("pre", pre), ("rf", rf)])
t0 = time.time()
pipe.fit(X_tr, y_tr)
fit_seconds = time.time() - t0

classes = pipe.named_steps["rf"].classes_          # [0, 1]
POS_IDX = int(np.where(classes == 1)[0][0])

# ------------------------------------------------------- threshold on validation
p_val = pipe.predict_proba(X_val)[:, POS_IDX]
grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
f1_grid = np.array([f1_score(y_val, (p_val >= t).astype(int)) for t in grid])
THR = float(grid[int(np.argmax(f1_grid))])

p_te = pipe.predict_proba(X_te)[:, POS_IDX]
y_hat = (p_te >= THR).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_hat).ravel()

metrics = {
    "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "n_test": int(len(y_te)),
    "pos_train": int(y_tr.sum()), "pos_val": int(y_val.sum()), "pos_test": int(y_te.sum()),
    "prevalence_test": round(float(y_te.mean()), 4),
    "threshold": THR,
    "accuracy": round(float(accuracy_score(y_te, y_hat)), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(y_te, y_hat)), 4),
    "precision": round(float(precision_score(y_te, y_hat, zero_division=0)), 4),
    "recall": round(float(recall_score(y_te, y_hat)), 4),
    "f1": round(float(f1_score(y_te, y_hat)), 4),
    "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
    "average_precision": round(float(average_precision_score(y_te, p_te)), 4),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "fit_seconds": round(fit_seconds, 1),
    "n_features_encoded": int(len(pipe.named_steps["pre"].get_feature_names_out())),
}
# accuracy of the naive majority rule, for context
metrics["majority_accuracy"] = round(float(1 - y_te.mean()), 4)

# -------------------------------------------------------------- SHAP (Task 2)
X_tr_t = pipe.named_steps["pre"].transform(X_tr)
X_te_t = pipe.named_steps["pre"].transform(X_te)
raw_names = [n.replace("num__", "").replace("cat__", "") for n in pipe.named_steps["pre"].get_feature_names_out()]


VAR_LABEL = {"age": "Age", "avg_glucose_level": "Average glucose", "bmi": "BMI",
             "gender": "Gender", "hypertension": "Hypertension", "heart_disease": "Heart disease",
             "ever_married": "Ever married", "work_type": "Work type",
             "Residence_type": "Residence", "smoking_status": "Smoking status"}


def split_name(name):
    for c in sorted(CAT, key=len, reverse=True):
        if name.startswith(c + "_"):
            return c, name[len(c) + 1:]
    return name, None


def disp(name):
    var, level = split_name(name)
    return f"{VAR_LABEL.get(var, var)} = {level}" if level is not None else VAR_LABEL.get(var, var)


feat_names = raw_names                    # raw encoded names (used for lookups)
feat_vars = [split_name(n)[0] for n in raw_names]
display_names = [disp(n) for n in raw_names]
X_te_df = pd.DataFrame(X_te_t, columns=display_names)

explainer = shap.TreeExplainer(pipe.named_steps["rf"])
shap_all = np.asarray(explainer.shap_values(X_te_t))
if shap_all.ndim == 3:
    sv = shap_all[:, :, POS_IDX]
else:
    sv = shap_all
base = float(np.ravel(explainer.expected_value)[POS_IDX])

recon = base + sv.sum(axis=1)
resid = float(np.max(np.abs(recon - p_te)))
check_add = bool(explainer.check_additivity(X_te_df.values if False else X_te_t, pipe.named_steps["rf"].predict_proba(X_te_t)[:, POS_IDX], sv) is not False) if hasattr(explainer, "check_additivity") else None

shap_metrics = {
    "explainer": "shap.TreeExplainer(model_output='raw' -> class probability for sklearn RF)",
    "base_value": round(base, 6),
    "expected_value_raw": [float(v) for v in np.ravel(explainer.expected_value)],
    "max_abs_additivity_residual": resid,
    "mean_reconstructed": round(float(recon.mean()), 4),
    "mean_model_probability": round(float(p_te.mean()), 4),
    "shap_version": shap.__version__,
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "n_explained": int(len(y_te)),
}

# --------------------------------------------------- Task 3: pick a held-out case
tp_mask = (y_te.values == 1) & (y_hat == 1)
tp_positions = np.where(tp_mask)[0]
tp_probs = p_te[tp_positions]
case_pos = int(tp_positions[int(np.argmin(np.abs(tp_probs - np.median(tp_probs))))])

case_row = X_te.iloc[case_pos]
case_prob = float(p_te[case_pos])
case_shap = sv[case_pos]
order = np.argsort(case_shap)
top_neg = [(feat_names[i], float(case_shap[i])) for i in order[:6] if case_shap[i] < 0][:4]
top_pos = [(feat_names[i], float(case_shap[i])) for i in order[::-1][:6] if case_shap[i] > 0][:4]

case = {
    "position_in_test": case_pos,
    "patient_label": f"held-out case #{case_pos + 1:04d}",
    "actual": int(y_te.iloc[case_pos]),
    "predicted": int(y_hat[case_pos]),
    "probability": round(case_prob, 4),
    "case_type": "true positive",
    "median_tp_probability": round(float(np.median(tp_probs)), 4),
    "n_true_positives": int(len(tp_positions)),
    "features": {k: (None if pd.isna(v) else (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else str(v)))
                 for k, v in case_row.items()},
    "top_positive": top_pos,
    "top_negative": top_neg,
    "shap_sum": round(float(case_shap.sum()), 4),
    "reconstruction": round(base + float(case_shap.sum()), 4),
}

# force plot (static)
shap.force_plot(base, case_shap, X_te_df.iloc[case_pos], matplotlib=True, show=False,
                figsize=(26, 5.2), text_rotation=60, contribution_threshold=0.001)
fig = plt.gcf()
fig.savefig(os.path.join(FIG, "figS1_raw_force_plot.png"), bbox_inches="tight", dpi=200)
plt.close(fig)

# --------------------------------------------------- Task 4: global summary plot
shap.summary_plot(sv, X_te_df, plot_type="dot", max_display=15, show=False,
                  color_bar_label="Feature value (blue = low, red = high)")
fig = plt.gcf()
fig.set_size_inches(9.6, 6.4)
fig.axes[-1].set_ylabel("Feature value\n(blue = low, red = high)", fontsize=8)
fig.axes[0].set_title("Global SHAP summary - Random Forest, held-out test set (n = %d)" % len(y_te),
                      fontsize=10, loc="left")
fig.savefig(os.path.join(FIG, "fig3_summary_beeswarm.png"), bbox_inches="tight", dpi=220)
plt.close(fig)

mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=display_names).sort_values(ascending=False)

# grouped by original variable: sum signed contributions per case, then mean |.|
def group_of(name):
    if " = " in name:
        return name.split(" = ")[0]
    return name

groups = {}
for j, n in enumerate(display_names):
    groups.setdefault(group_of(n), []).append(j)
grouped = {}
for g, idxs in groups.items():
    grouped[g] = float(np.abs(sv[:, idxs].sum(axis=1)).mean())
grouped = pd.Series(grouped).sort_values(ascending=False)

# supplementary bar chart
fig, ax = plt.subplots(figsize=(7.2, 3.6))
sel = mean_abs.head(12)[::-1]
ax.barh(sel.index, sel.values, color="#4C72B0")
ax.set_xlabel("Mean |SHAP value| (probability units)")
ax.set_title("Top encoded features by mean absolute SHAP (held-out, n = %d)" % len(y_te), fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3b_mean_abs_bar.png"), bbox_inches="tight", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
sel = grouped.head(10)[::-1]
ax.barh(sel.index, sel.values, color="#55A868")
ax.set_xlabel("Mean |summed SHAP| per patient (probability units)")
ax.set_title("Grouped by original variable (one-hot categories recombined)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3c_grouped_bar.png"), bbox_inches="tight", dpi=220)
plt.close(fig)

global_res = {
    "mean_abs_top12": [{"feature": k, "mean_abs_shap": round(float(v), 5)} for k, v in mean_abs.head(12).items()],
    "grouped_by_variable": [{"variable": k, "mean_abs_summed_shap": round(float(v), 5)} for k, v in grouped.items()],
    "sign_summary_top": {},
}
for f in mean_abs.head(6).index:
    j = display_names.index(f)
    vals = X_te_df[f].values
    corr = float(np.corrcoef(vals, sv[:, j])[0, 1]) if np.std(vals) > 0 else float("nan")
    global_res["sign_summary_top"][f] = {
        "corr_feature_vs_shap": round(corr, 3),
        "mean_shap": round(float(sv[:, j].mean()), 5),
        "share_positive": round(float((sv[:, j] > 0).mean()), 3),
    }

# ---------------------------------------------------------------- persistence
with open(os.path.join(OUT, "task2_results.json"), "w", encoding="utf-8") as f:
    json.dump({"metrics": metrics, "shap": shap_metrics, "case": case, "global": global_res},
              f, indent=2, ensure_ascii=False)
np.savez_compressed(os.path.join(OUT, "shap_rf.npz"), sv=sv, p_te=p_te, base=base,
                    y_te=y_te.values, case_pos=case_pos, X_te_t=X_te_t,
                    feat_names=np.array(feat_names), display_names=np.array(display_names),
                    feat_vars=np.array(feat_vars))
with open(os.path.join(OUT, "rf_pipeline.pkl"), "wb") as f:
    pickle.dump({"pipeline": pipe, "threshold": THR, "feature_names": feat_names,
                 "display_names": display_names,
                 "num": NUM, "cat": CAT}, f)

print(json.dumps({"metrics": metrics, "shap": shap_metrics}, indent=2, ensure_ascii=False))
print("\nCASE", json.dumps(case, indent=2, ensure_ascii=False)[:1200])
print("\nTOP12\n", mean_abs.head(12).to_string())
print("\nGROUPED\n", grouped.head(10).to_string())
