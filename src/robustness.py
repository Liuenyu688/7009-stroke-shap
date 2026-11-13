"""Robustness check for Section 8.2: repeat the SVM explanations with a different background sample."""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from _paths import DATA, OUT
SEED = 2021

m = pickle.load(open(os.path.join(OUT, "svm_pipeline.pkl"), "rb"))
pipe, THR = m["pipeline"], m["threshold"]
rf = np.load(os.path.join(OUT, "shap_rf.npz"), allow_pickle=True)
display_names = [str(x) for x in rf["display_names"]]
feat_vars = [str(x) for x in rf["feat_vars"]]

df = pd.read_csv(DATA).drop(columns=["id"])
X, y = df.drop(columns=["stroke"]), df["stroke"]
from sklearn.model_selection import train_test_split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

pre = pipe.named_steps["pre"]
X_tr_t = pre.transform(X_tr)
X_te_t = pre.transform(X_te)
f = lambda z: pipe.named_steps["svc"].predict_proba(z)[:, 1]

groups = {}
for j, v in enumerate(feat_vars):
    groups.setdefault(v, []).append(j)
label_of = {v: display_names[idxs[0]].split(" = ")[0] for v, idxs in groups.items()}

cp = 334
N_GLOBAL = 40
rng = np.random.default_rng(SEED)
g_idx = np.sort(rng.choice(len(y_te), size=N_GLOBAL, replace=False))

out = {}
for tag, seed in [("bg2021", 2021), ("bg7", 7)]:
    bg = shap.sample(X_tr_t, 100, random_state=seed)
    ex = shap.KernelExplainer(f, bg)
    t0 = time.time()
    svc_case = np.asarray(ex.shap_values(X_te_t[cp:cp + 1], nsamples=2048, l1_reg=0.0, silent=True))[0]
    svc_glob = np.asarray(ex.shap_values(X_te_t[g_idx], nsamples=512, l1_reg=0.0, silent=True))
    out[tag] = {
        "base": float(np.ravel(ex.expected_value)[0]),
        "case_grouped": {label_of[v]: round(float(svc_case[idxs].sum()), 5) for v, idxs in groups.items()},
        "global_grouped": {label_of[v]: round(float(np.abs(svc_glob[:, idxs].sum(axis=1)).mean()), 5)
                           for v, idxs in groups.items()},
        "seconds": round(time.time() - t0, 1),
    }
    print(tag, "done in", out[tag]["seconds"], "s  base =", round(out[tag]["base"], 5))

a, b = out["bg2021"], out["bg7"]
diffs = {k: round(b["case_grouped"][k] - a["case_grouped"][k], 5) for k in a["case_grouped"]}
vars_ = list(a["global_grouped"])
rho = float(spearmanr([a["global_grouped"][k] for k in vars_],
                      [b["global_grouped"][k] for k in vars_]).statistic)
top3_a = sorted(vars_, key=lambda k: -a["global_grouped"][k])[:3]
top3_b = sorted(vars_, key=lambda k: -b["global_grouped"][k])[:3]

out["comparison"] = {
    "max_abs_case_diff": round(max(abs(v) for v in diffs.values()), 5),
    "case_sign_agreement": round(float(np.mean([np.sign(a["case_grouped"][k]) == np.sign(b["case_grouped"][k])
                                                for k in vars_])), 3),
    "case_diffs": diffs,
    "spearman_global_rank": round(rho, 3),
    "top3_bg2021": top3_a,
    "top3_bg7": top3_b,
    "top3_overlap": len(set(top3_a[:2]) & set(top3_b[:2])),
}
json.dump(out, open(os.path.join(OUT, "robustness.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(out["comparison"], indent=2, ensure_ascii=False))
