# 7009 Interpretable Machine Learning — Assignment 1

**Explaining a stroke-risk classifier with feature attribution.**
A Random Forest and an RBF SVM are both trained on the same stroke dataset and explained
with SHAP, using `TreeExplainer` (exact) for the tree model and `KernelExplainer` for the
opaque kernel model. The repository contains the full pipeline, the generated figures, the
machine-readable results and the submitted report.

---

## 1. Headline results

Held-out test set, n = 1,022, single operating point taken from the validation split.

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.863 | 0.719 | 0.192 | 0.560 | 0.286 | 0.856 | 0.234 |
| RBF SVM | 0.873 | 0.734 | 0.210 | 0.580 | 0.309 | 0.859 | 0.219 |

Majority-class baseline accuracy is **0.951**, so accuracy alone is not a meaningful
signal here; balanced accuracy, recall and PR-AUC are the metrics that matter.

Global importance (mean |SHAP| over 1,022 test patients, one-hot levels summed back to the
original variable):

| Rank | Random Forest | RBF SVM |
|---|---|---|
| 1 | Age (0.0463) | Age (0.0438) |
| 2 | Glucose (0.0117) | Smoking status (0.0057) |
| 3 | BMI (0.0082) | Glucose (0.0041) |
| 4 | Hypertension (0.0055) | Work type (0.0036) |
| 5 | Heart disease (0.0045) | Hypertension (0.0035) |

The two models agree that age dominates but disagree sharply below the top rank — the SVM
promotes smoking status to second place and pushes heart disease to last. **The
explanation is a property of the model, not of the patient population.**

---

## 2. Repository layout

```
.
├── data/
│   └── healthcare-dataset-stroke-data.csv     # 5,110 × 12, MD5 3f25a4c2f2963d969e1a156c08968577
├── src/
│   ├── _paths.py               # all paths resolved relative to the repo root
│   ├── run_eda.py              # Task 1 — data checks, Figure 1
│   ├── run_models.py           # Task 2–4 — RF pipeline, metrics, TreeSHAP, local case
│   ├── fig_force_grouped.py    # Figure 2 — grouped force plot + case_grouped.json
│   ├── run_svm.py              # Task 6 — SVM pipeline, KernelSHAP
│   ├── fig_svm.py              # Figures 4–5 — grouped waterfall and beeswarm
│   ├── robustness.py           # second background sample, agreement statistics
│   ├── build_report.py         # assembles the .docx from the template
│   └── build_slides.py         # Task 5 — three audience slides
├── outputs/
│   ├── figures/                # fig1 … fig5, fig3b/3c, figS1
│   └── *.json                  # all numbers quoted in the report
├── report/
│   ├── Stroke_SHAP_Report.docx # submitted report, 20 pages
│   └── task5_slides.pptx       # three slides, one per audience
└── requirements.txt
```

## 3. Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/run_eda.py
python src/run_models.py          # ~2 min, fits the Random Forest and the TreeExplainer
python src/fig_force_grouped.py
python src/run_svm.py             # ~3 min, KernelExplainer on 1,022 test patients
python src/fig_svm.py
python src/robustness.py          # ~13 min, re-runs KernelSHAP with a second background
python src/build_report.py        # needs the course template, see section 5
python src/build_slides.py
```

Everything is deterministic: `random_state = 2021` everywhere and `n_jobs = 1` for the
Random Forest. `n_jobs = -1` was tested and rejected — it changes the fitted trees between
runs, which silently moves the selected local case, so single-threaded fitting is a
deliberate reproducibility decision rather than an oversight.

## 4. Decisions worth knowing about

- **Split** — stratified 60/20/20 (3,066 / 1,022 / 1,022), seed 2021.
- **No leakage** — imputation, encoding and scaling live inside a `ColumnTransformer` +
  `Pipeline` that is fitted on the training split only. `bmi` ships as the literal string
  `"N/A"` and is coerced to `NaN` before median imputation.
- **No resampling** — SMOTE and random oversampling were rejected. Duplicated minority
  rows would leak into the SHAP background sample and flatter the attributions.
- **Operating point** — the decision threshold (0.14) is chosen on the validation split by
  maximum F1 and then frozen. It is never tuned on the test set.
- **Explainer scale** — SHAP values are returned on the probability scale, not in
  log-odds. Additivity holds to 2.0 × 10⁻¹⁵ for the Random Forest.
- **One-hot grouping** — a variable with several encoded levels contributes several SHAP
  values. Levels are summed **with their signs** before any aggregation; averaging
  absolute values per level would overstate multi-level categoricals.
- **KernelSHAP regularisation** — `l1_reg="auto"` (the default AIC rule) zeroed four
  attributions with 100 background samples, so it is set to `l1_reg=0.0`.
- **Background size** — 100 training rows, `nsamples = 2048` for the local case and
  `512` for the 150-case global estimate.

## 5. Note on the report

`build_report.py` is a document assembler, not an analysis step: it copies the course
template and fills in numbers, tables and figures. The template
(`Stroke_SHAP_Report_Template.docx`) is course material and is therefore **not** committed.
Drop it into `report/` if you want to rebuild the .docx; the committed
`report/Stroke_SHAP_Report.docx` is the finished artefact and needs nothing.

## 6. Data

`healthcare-dataset-stroke-data.csv` — the public *Stroke Prediction Dataset*
(fedesoriano, Kaggle), 5,110 patients × 12 columns. Redistributed here under its original
terms so the results can be reproduced exactly; MD5
`3f25a4c2f2963d969e1a156c08968577`.

Known quirks that the pipeline handles: 201 missing `bmi` values stored as `"N/A"`
(3.93 %), 30.22 % `"Unknown"` smoking status, one `"Other"` gender row, and a slight
spelling difference between the assignment brief (`Govt_jov`) and the actual column
(`Govt_job`).
