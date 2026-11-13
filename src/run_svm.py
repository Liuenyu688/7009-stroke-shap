"""Stage 3 - Task 6: RBF SVM + Kernel SHAP (local + global), compared with the Task 2 model."""
import json
import os
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

from _paths import DATA, OUT, FIG
SEED = 2021
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

rf_res = json.load(open(os.path.join(OUT, "task2_results.json"), encoding="utf-8"))
CASE_POS = rf_res["case"]["position_in_test"]
display_names = [str(x) for x in np.load(os.path.join(OUT, "shap_rf.npz"), allow_pickle=True)["display_names"]]

# ------------------------------------------------------------------ data
df = pd.read_csv(DATA).drop(columns=["id"])
X, y = df.drop(columns=["stroke"]), df["stroke"]
NUM = ["age", "avg_glucose_level", "bmi"]
CAT = [c for c in X.columns if c not in NUM]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
])
svc = SVC(kernel="rbf", C=1.0, gamma=0.02, probability=True,
          class_weight="balanced", random_state=SEED)
pipe = Pipeline([("pre", pre), ("svc", svc)])
t0 = time.time()
pipe.fit(X_tr, y_tr)
fit_seconds = time.time() - t0

# ------------------------------------------------------------ threshold + metrics
p_val = pipe.predict_proba(X_val)[:, 1]
grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
THR = float(grid[int(np.argmax([f1_score(y_val, (p_val >= t).astype(int)) for t in grid]))])
p_te = pipe.predict_proba(X_te)[:, 1]
y_hat = (p_te >= THR).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_hat).ravel()

metrics = {
    "threshold": THR,
    "accuracy": round(float(accuracy_score(y_te, y_hat)), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(y_te, y_hat)), 4),
    "precision": round(float(precision_score(y_te, y_hat, zero_division=0)), 4),
    "recall": round(float(recall_score(y_te, y_hat)), 4),
    "f1": round(float(f1_score(y_te, y_hat)), 4),
    "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
    "average_precision": round(float(average_precision_score(y_te, p_te)), 4),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "n_test": int(len(y_te)),
    "n_support_vectors": int(len(pipe.named_steps["svc"].support_)),
    "fit_seconds": round(fit_seconds, 1),
    "params": {"kernel": "rbf", "C": 1.0, "gamma": 0.02, "class_weight": "balanced"},
}

# ------------------------------------------------------------------ Kernel SHAP
X_tr_t = pipe.named_steps["pre"].transform(X_tr)
X_te_t = pipe.named_steps["pre"].transform(X_te)
X_te_df = pd.DataFrame(X_te_t, columns=display_names)

BG_SIZE = 100
BG = shap.sample(X_tr_t, BG_SIZE, random_state=SEED)
f = lambda z: pipe.named_steps["svc"].predict_proba(z)[:, 1]
explainer = shap.KernelExplainer(f, BG)

t0 = time.time()
sv_case = np.asarray(explainer.shap_values(X_te_t[CASE_POS:CASE_POS + 1], nsamples=2048,
                                              l1_reg=0.0, silent=True))[0]
t_local = time.time() - t0
base = float(np.ravel(explainer.expected_value)[0])
case_prob = float(p_te[CASE_POS])
resid_local = float(abs(base + sv_case.sum() - case_prob))

rng = np.random.default_rng(SEED)
GLOBAL_N, NSAMPLES = 150, 512
g_idx = np.sort(rng.choice(len(y_te), size=GLOBAL_N, replace=False))
t0 = time.time()
sv_global = np.asarray(explainer.shap_values(X_te_t[g_idx], nsamples=NSAMPLES, l1_reg=0.0, silent=True))
t_global = time.time() - t0
resid_global = float(np.max(np.abs(base + sv_global.sum(axis=1) - p_te[g_idx])))

# ------------------------------------------------------------ figures 4 and 5
exp = shap.Explanation(values=sv_case, base_values=base, data=X_te_t[CASE_POS],
                       feature_names=display_names)
plt.figure()
shap.plots.waterfall(exp, max_display=12, show=False)
fig = plt.gcf()
fig.set_size_inches(9.2, 5.0)
fig.axes[0].set_title("Kernel SHAP - SVM, same held-out case #%04d (probability of stroke = 1)"
                      % (CASE_POS + 1), fontsize=10, loc="left")
fig.savefig(os.path.join(FIG, "fig4_svm_waterfall.png"), bbox_inches="tight", dpi=230)
plt.close(fig)

shap.summary_plot(sv_global, X_te_df.iloc[g_idx], plot_type="dot", max_display=15, show=False)
fig = plt.gcf()
fig.set_size_inches(9.6, 6.4)
fig.axes[-1].set_ylabel("Feature value\n(blue = low, red = high)", fontsize=8)
fig.axes[0].set_title("Kernel SHAP summary - SVM, %d held-out cases" % GLOBAL_N, fontsize=10, loc="left")
fig.savefig(os.path.join(FIG, "fig5_svm_beeswarm.png"), bbox_inches="tight", dpi=230)
plt.close(fig)

mean_abs = pd.Series(np.abs(sv_global).mean(axis=0), index=display_names).sort_values(ascending=False)

# --------------------------------------------------------- grouped contributions
LABELS = [n.split(" = ")[0] if " = " in n else n for n in display_names]
grp = {}
for j, l in enumerate(LABELS):
    grp.setdefault(l, []).append(j)
case_grouped = {k: round(float(sv_case[v].sum()), 5) for k, v in grp.items()}
glob_grouped = {k: round(float(np.abs(sv_global[:, v].sum(axis=1)).mean()), 5) for k, v in grp.items()}

out = {
    "metrics": metrics,
    "shap": {
        "explainer": "shap.KernelExplainer on predict_proba[:, 1] (probability scale)",
        "background_size": BG_SIZE,
        "background_sampling": "shap.sample(X_train, 100, random_state=2021)",
        "nsamples_local": 2048, "nsamples_global": NSAMPLES, "l1_reg": 0.0,
        "global_sample_size": GLOBAL_N, "global_seed": SEED,
        "base_value": round(base, 6),
        "local_reconstruction_residual": resid_local,
        "global_max_residual": resid_global,
        "seconds_local_case": round(t_local, 1),
        "seconds_global_sample": round(t_global, 1),
        "shap_version": shap.__version__,
    },
    "case": {
        "position_in_test": CASE_POS,
        "probability": round(case_prob, 4),
        "predicted": int(case_prob >= THR),
        "actual": int(y_te.iloc[CASE_POS]),
        "contributions_grouped": dict(sorted(case_grouped.items(), key=lambda kv: -abs(kv[1]))),
        "top_positive": sorted([(k, v) for k, v in case_grouped.items() if v > 0], key=lambda kv: -kv[1])[:3],
        "top_negative": sorted([(k, v) for k, v in case_grouped.items() if v < 0], key=lambda kv: kv[1])[:3],
    },
    "global": {
        "mean_abs_top10": [{"feature": k, "mean_abs_shap": round(float(v), 5)} for k, v in mean_abs.head(10).items()],
        "grouped_by_variable": [{"variable": k, "mean_abs_summed_shap": v}
                                for k, v in sorted(glob_grouped.items(), key=lambda kv: -kv[1])],
    },
}

json.dump(out, open(os.path.join(OUT, "task6_results.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
np.savez_compressed(os.path.join(OUT, "shap_svm.npz"), sv_case=sv_case, sv_global=sv_global,
                    base=base, p_te=p_te, case_pos=CASE_POS, g_idx=g_idx, X_te_t=X_te_t)
with open(os.path.join(OUT, "svm_pipeline.pkl"), "wb") as fh:
    pickle.dump({"pipeline": pipe, "threshold": THR}, fh)

print(json.dumps({"metrics": metrics, "shap": out["shap"], "case": out["case"]}, indent=2, ensure_ascii=False))
print("\nTOP10\n", mean_abs.head(10).to_string())
