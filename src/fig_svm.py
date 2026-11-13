"""Redraw Figure 4 (grouped waterfall for the SVM case) and Figure 5 (SVM beeswarm)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from _paths import OUT, FIG
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

d = np.load(os.path.join(OUT, "shap_svm.npz"), allow_pickle=True)
sv_case, sv_global = d["sv_case"], d["sv_global"]
base, cp, g_idx = float(d["base"]), int(d["case_pos"]), d["g_idx"]
X_te_t = d["X_te_t"]
rf = np.load(os.path.join(OUT, "shap_rf.npz"), allow_pickle=True)
display_names = [str(x) for x in rf["display_names"]]
feat_vars = [str(x) for x in rf["feat_vars"]]
res6 = json.load(open(os.path.join(OUT, "task6_results.json"), encoding="utf-8"))
res2 = json.load(open(os.path.join(OUT, "task2_results.json"), encoding="utf-8"))
case = res2["case"]

# ---- group encoded columns into original variables
groups = {}
for j, v in enumerate(feat_vars):
    groups.setdefault(v, []).append(j)
label_of = {v: display_names[idxs[0]].split(" = ")[0] for v, idxs in groups.items()}

rows, vals = [], []
for v, idxs in groups.items():
    c = float(sv_case[idxs].sum())
    raw = case["features"].get(v)
    if v in ("age", "bmi", "avg_glucose_level"):
        fv = "missing" if raw is None else f"{raw:g}"
    else:
        fv = str(raw)
    rows.append((label_of[v], fv, c))
    vals.append(c)

order = np.argsort(vals)
labels = [f"{rows[i][0]} = {rows[i][1]}" for i in order]
values = np.array([vals[i] for i in order])

exp = shap.Explanation(values=values, base_values=base, data=None, feature_names=labels)
plt.figure()
shap.plots.waterfall(exp, max_display=11, show=False)
fig = plt.gcf()
fig.set_size_inches(9.4, 5.2)
fig.axes[0].set_title("Kernel SHAP (SVM) - held-out case #%04d, probability of stroke = 1"
                      % (cp + 1), fontsize=10, loc="left")
fig.savefig(os.path.join(FIG, "fig4_svm_waterfall.png"), bbox_inches="tight", dpi=230)
plt.close("all")
print("grouped case contributions (SVM)")
for i in np.argsort(vals)[::-1]:
    print(f"  {rows[i][0]:18s} {rows[i][1]:18s} {vals[i]:+.5f}")
print("sum %.5f + base %.5f = %.5f (prob %.5f)" % (sum(vals), base, base + sum(vals), res6["case"]["probability"]))

# ---- beeswarm (encoded level, top 12)
X_te_df = pd.DataFrame(X_te_t, columns=display_names)
shap.summary_plot(sv_global, X_te_df.iloc[g_idx], plot_type="dot", max_display=12, show=False)
fig = plt.gcf()
fig.set_size_inches(9.6, 5.8)
fig.axes[-1].set_ylabel("Feature value\n(blue = low, red = high)", fontsize=8)
fig.axes[0].set_title("Kernel SHAP summary - SVM, %d held-out cases" % len(g_idx), fontsize=10, loc="left")
fig.savefig(os.path.join(FIG, "fig5_svm_beeswarm.png"), bbox_inches="tight", dpi=230)
plt.close("all")
print("figures written")
