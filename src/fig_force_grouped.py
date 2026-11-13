"""Figure 2 - local force plot for the selected held-out case (grouped by original variable)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from _paths import OUT
d = np.load(os.path.join(OUT, "shap_rf.npz"), allow_pickle=True)
sv, base, cp = d["sv"], float(d["base"]), int(d["case_pos"])
display_names = [str(x) for x in d["display_names"]]
feat_vars = [str(x) for x in d["feat_vars"]]
res = json.load(open(os.path.join(OUT, "task2_results.json"), encoding="utf-8"))
case = res["case"]

# group encoded columns back to their original variable, summing signed contributions
groups = {}
for j, v in enumerate(feat_vars):
    groups.setdefault(v, []).append(j)

rows, vals = [], []
for v, idxs in groups.items():
    c = float(sv[cp, idxs].sum())
    label = display_names[idxs[0]].split(" = ")[0]
    raw = case["features"].get(v)
    if v in ("age", "bmi", "avg_glucose_level"):
        fv = "missing" if raw is None else f"{raw:g}"
    else:
        fv = str(raw)
    rows.append((label, fv, c))
    vals.append(c)

print("per-variable contributions for the selected case")
for i in np.argsort(vals)[::-1]:
    print(f"  {rows[i][0]:18s} {rows[i][1]:20s} {vals[i]:+.4f}")

keep = [i for i, v in enumerate(vals) if abs(v) >= 0.002]
drop = [i for i, v in enumerate(vals) if abs(v) < 0.002]
residual = float(sum(vals[i] for i in drop))
kept = [(rows[i][0], rows[i][1], vals[i]) for i in keep]
if drop:
    kept.append(("all other variables", "-", residual))
kept.sort(key=lambda t: t[2])

labels = [k[0] for k in kept]
featvals = [k[1] for k in kept]
values = np.array([k[2] for k in kept])
feat_df = pd.DataFrame([featvals], columns=labels)

w, h = 11, 4.2
fig = plt.figure(figsize=(w, h))
shap.force_plot(base, values, feat_df.iloc[0], matplotlib=True, show=False,
                text_rotation=18, figsize=(w, h))
plt.savefig(os.path.join(OUT, "figures", "fig2_force_plot.png"), bbox_inches="tight", dpi=240)
plt.close("all")

json.dump({"base": round(base, 5),
           "contributions": [{"feature": l, "patient_value": v, "shap": round(float(s), 5)} for l, v, s in kept],
           "residual": round(residual, 5), "n_merged": len(drop),
           "sum": round(float(sum(vals)), 5), "probability": case["probability"]},
          open(os.path.join(OUT, "case_grouped.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("\nbase %.4f + sum %.4f = %.4f (model probability %.4f)"
      % (base, sum(vals), base + sum(vals), case["probability"]))
