"""Stage 1 - Task 1 EDA: descriptive tables + Figure 1 panel."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _paths import DATA, OUT, FIG
os.makedirs(FIG, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv(DATA)
res = {}

res["shape"] = list(df.shape)
res["n_dup_ids"] = int(df.id.duplicated().sum())
res["n_dup_rows"] = int(df.duplicated().sum())
res["missing"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
res["missing_pct"] = {k: round(float(v) / len(df) * 100, 2) for k, v in df.isna().sum().items() if v > 0}
res["stroke_counts"] = {int(k): int(v) for k, v in df.stroke.value_counts().items()}
res["prevalence"] = round(float(df.stroke.mean()), 4)

num = ["age", "avg_glucose_level", "bmi"]
res["numeric_summary"] = {
    c: {
        "mean": round(float(df[c].mean()), 2),
        "sd": round(float(df[c].std()), 2),
        "median": round(float(df[c].median()), 2),
        "q1": round(float(df[c].quantile(0.25)), 2),
        "q3": round(float(df[c].quantile(0.75)), 2),
        "min": round(float(df[c].min()), 2),
        "max": round(float(df[c].max()), 2),
    }
    for c in num
}
res["numeric_by_stroke_median"] = {
    c: {int(k): round(float(v), 2) for k, v in df.groupby("stroke")[c].median().items()} for c in num
}

cats = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status",
        "hypertension", "heart_disease"]
res["categorical"] = {}
for c in cats:
    g = df.groupby(c, observed=True).stroke.agg(["count", "sum", "mean"])
    res["categorical"][c] = {
        str(idx): {"n": int(r["count"]), "stroke_n": int(r["sum"]), "rate": round(float(r["mean"]), 4)}
        for idx, r in g.iterrows()
    }

res["stroke_rate_by_age_band"] = {
    str(k): {"n": int(v["count"]), "stroke_n": int(v["sum"]), "rate": round(float(v["mean"]), 4)}
    for k, v in df.groupby(pd.cut(df.age, [0, 20, 40, 60, 80], right=False), observed=True)
    .stroke.agg(["count", "sum", "mean"]).iterrows()
}
res["glucose_gt200_by_stroke"] = {
    int(k): round(float(v), 4) for k, v in df.groupby("stroke").avg_glucose_level.apply(lambda s: (s > 200).mean()).items()
}

# ---------------- Figure 1: 2x3 EDA panel ----------------
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))

# (a) class balance
ax = axes[0, 0]
counts = df.stroke.value_counts().sort_index()
bars = ax.bar(["No stroke (0)", "Stroke (1)"], counts.values, color=["#4C72B0", "#C44E52"], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v}\n({v/len(df)*100:.2f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, counts.max() * 1.22)
ax.set_ylabel("Patients (n)")
ax.set_title("(a) Class imbalance in the target", fontsize=10, loc="left")

# (b) age distribution by stroke
ax = axes[0, 1]
sns.kdeplot(data=df, x="age", hue="stroke", common_norm=False, fill=True,
            palette={0: "#4C72B0", 1: "#C44E52"}, ax=ax, legend=False, alpha=0.45)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
ax.set_title("(b) Age distribution by outcome", fontsize=10, loc="left")

# (c) glucose by stroke
ax = axes[0, 2]
sns.boxplot(data=df, x="stroke", y="avg_glucose_level", hue="stroke",
            palette={0: "#4C72B0", 1: "#C44E52"}, ax=ax, legend=False, width=0.5,
            fliersize=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No stroke", "Stroke"])
ax.set_xlabel(""); ax.set_ylabel("Average glucose (mg/dL)")
ax.set_title("(c) Glucose level by outcome", fontsize=10, loc="left")

# (d) bmi: missingness + distribution
ax = axes[1, 0]
bmi_plot = df.assign(bmi_group=np.where(df.bmi.isna(), "missing", "observed"))
sns.boxplot(data=bmi_plot[bmi_plot.bmi_group == "observed"], x="stroke", y="bmi",
            hue="stroke", palette={0: "#4C72B0", 1: "#C44E52"}, ax=ax, legend=False,
            width=0.5, fliersize=2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No stroke", "Stroke"])
ax.set_xlabel(""); ax.set_ylabel("BMI (kg/m$^2$)")
n_miss = int(df.bmi.isna().sum())
ax.set_title(f"(d) BMI by outcome (BMI missing for {n_miss}, {n_miss/len(df)*100:.1f}%)",
             fontsize=10, loc="left")

# (e) stroke rate by age band
ax = axes[1, 1]
band = df.groupby(pd.cut(df.age, [0, 20, 40, 60, 80], right=False), observed=True).stroke.agg(["count", "mean"])
labels = ["<20", "20-39", "40-59", "60-79"]
vals = band["mean"].values * 100
bars = ax.bar(labels, vals, color="#DD8452", width=0.6)
for b, v, n in zip(bars, vals, band["count"].values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Stroke prevalence (%)")
ax.set_xlabel("Age band (years)")
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("(e) Prevalence rises steeply with age", fontsize=10, loc="left")

# (f) stroke rate by comorbidity
ax = axes[1, 2]
groups = {
    "No HTN\nno HD": df[(df.hypertension == 0) & (df.heart_disease == 0)],
    "HTN only": df[(df.hypertension == 1) & (df.heart_disease == 0)],
    "HD only": df[(df.hypertension == 0) & (df.heart_disease == 1)],
    "HTN + HD": df[(df.hypertension == 1) & (df.heart_disease == 1)],
}
vals = [g.stroke.mean() * 100 for g in groups.values()]
ns = [len(g) for g in groups.values()]
bars = ax.bar(list(groups.keys()), vals, color="#55A868", width=0.6)
for b, v, n in zip(bars, vals, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}%\n(n={n})", ha="center", fontsize=8)
ax.set_ylabel("Stroke prevalence (%)")
ax.set_ylim(0, max(vals) * 1.4)
ax.set_title("(f) Prevalence by comorbidity", fontsize=10, loc="left")

fig.suptitle("Exploratory analysis of the stroke dataset (n = 5,110)", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(os.path.join(FIG, "fig1_eda_panel.png"), bbox_inches="tight")
plt.close(fig)

with open(os.path.join(OUT, "eda_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print(json.dumps({k: res[k] for k in ["shape", "n_dup_rows", "missing", "stroke_counts", "prevalence"]},
                 indent=2, ensure_ascii=False))
print("figure 1 written")
