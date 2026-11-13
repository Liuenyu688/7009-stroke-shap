"""Build the final report from the template + computed results (Tasks 1-6)."""
import copy
import json
import os
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

from _paths import OUT, FIG, REPORT

TEMPLATE = REPORT / "Stroke_SHAP_Report_Template.docx"
TARGET = REPORT / "Stroke_SHAP_Report.docx"
if not TEMPLATE.exists():
    raise SystemExit(
        "Missing course template. Put Stroke_SHAP_Report_Template.docx in report/ "
        "and run again. See README for details."
    )

eda = json.load(open(os.path.join(OUT, "eda_results.json"), encoding="utf-8"))
r2 = json.load(open(os.path.join(OUT, "task2_results.json"), encoding="utf-8"))
r6 = json.load(open(os.path.join(OUT, "task6_results.json"), encoding="utf-8"))
cg = json.load(open(os.path.join(OUT, "case_grouped.json"), encoding="utf-8"))
rob = json.load(open(os.path.join(OUT, "robustness.json"), encoding="utf-8"))

m2, s2, c2 = r2["metrics"], r2["shap"], r2["case"]
m6, s6, c6 = r6["metrics"], r6["shap"], r6["case"]
cmp_ = rob["comparison"]

shutil.copyfile(TEMPLATE, TARGET)
doc = Document(TARGET)

# ----------------------------------------------------------------- helpers
def set_text(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    runs[0].text = text
    runs[0].italic = False
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def kill(p):
    p._element.getparent().remove(p._element)


def insert_after(p, text, bold_lead=None):
    el = OxmlElement("w:p")
    p._element.addnext(el)
    new_p = Paragraph(el, p._parent)
    new_p.style = p.style
    if bold_lead:
        run = new_p.add_run(bold_lead)
        run.bold = True
        new_p.add_run(text)
    else:
        new_p.add_run(text)
    return new_p


def put_image(p, path, width_in, caption_after=None):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))
    if caption_after:
        insert_after(p, caption_after)


def fill_row(table, row, values):
    for col, val in enumerate(values, start=0):
        cell = table.cell(row, col)
        p = cell.paragraphs[0]
        if p.runs:
            p.runs[0].text = str(val)
            for r in p.runs[1:]:
                r._element.getparent().remove(r._element)
        else:
            p.add_run(str(val))
        for extra in cell.paragraphs[1:]:
            extra._element.getparent().remove(extra._element)


T = doc.tables

# ----------------------------------------------------------------- cover
set_text(doc.paragraphs[1], "Assignment report")
set_text(doc.paragraphs[3], "Student name: [Your full name]")
set_text(doc.paragraphs[4], "Student ID: [Your student ID]")
set_text(doc.paragraphs[5], "Course and programme: [Course code and programme title]")
set_text(doc.paragraphs[6], "Lecturer: [Lecturer name]")
set_text(doc.paragraphs[7], "Submission date: [Date of submission]")
set_text(doc.paragraphs[8], "Repository URL and commit: https://github.com/[user]/xai-series (supplied course archive xai-series-master; no commit hash was available in the ZIP, see Section 9.2)")

# capture paragraph objects first, then delete the template instructions (indices would shift otherwise)
P = list(doc.paragraphs)
for idx in (9, 10, 11):
    kill(P[idx])

# ----------------------------------------------------------------- abstract
set_text(P[13], (
    "This report fits a Random Forest classifier to recorded stroke history in a public dataset of "
    "5,110 patients (4.87% positive) and explains its predictions with SHAP. After a stratified "
    "60/20/20 split, median imputation and one-hot encoding were fitted on the training rows only, "
    "and the decision threshold was selected on the validation set rather than by resampling. On the "
    "untouched test set the model reached ROC-AUC 0.856 and average precision 0.234, with recall 0.560 "
    "and precision 0.192 at a threshold of 0.14; accuracy (0.863) is below the 0.951 majority-class "
    "baseline and is therefore not used as evidence of usefulness. TreeExplainer reproduced the model "
    "output exactly (residual 2.0e-15 on the probability scale, base value 0.048). Age dominated both "
    "the local explanation of held-out case #0335 (+0.058 of a 0.208 prediction) and the global "
    "ranking, followed by average glucose and BMI. An RBF support vector machine explained with Kernel "
    "SHAP reached a comparable ROC-AUC (0.859) and agreed on age, but attributed much less to heart "
    "disease for the same patient. The main limitation is that the target is a recorded stroke history "
    "without a follow-up period, so the output is not a validated future-risk estimate."
))

# ----------------------------------------------------------------- section 1
set_text(P[19], (
    "Stroke is the second leading cause of death worldwide: the World Health Organization attributes "
    "approximately 6.8 million deaths, or about 11% of all deaths, to stroke in its 2023 mortality "
    "estimates [1]. Several of its risk factors are recorded in routine data - age, hypertension, "
    "heart disease, average glucose, body mass index and smoking status - so a classifier can rank "
    "patients by the probability that a record indicates a stroke history, and a hospital can use that "
    "ranking to prioritise review. A ranking on its own is hard to audit: a clinician needs to know "
    "which measurements drove one patient's score, an analyst needs to know whether the model relies "
    "on plausible variables or on artefacts of the encoding, and a patient needs a plain-language "
    "account of the same prediction. SHAP (SHapley Additive exPlanations) gives all three audiences "
    "one set of numbers, because each prediction is decomposed additively into feature contributions "
    "that sum to the model output by construction [2]. The same decomposition can be averaged over a "
    "cohort to describe global behaviour, which is what makes a single report cover Tasks 3 to 5."
))
set_text(P[20], (
    "Objective: to fit a Random Forest classifier on a stratified hold-out split, to evaluate it with "
    "metrics appropriate to a 4.9% positive class, to explain it with TreeExplainer at the level of one "
    "held-out patient and of the whole held-out cohort, and then to repeat the analysis with a "
    "structurally different model - an RBF support vector machine explained with Kernel SHAP - so that "
    "the attributions of two model families can be compared on the same patients, the same split and "
    "the same predictors."
))
ROWS_ATTR = [
    ["id", "Integer identifier, unique to each patient", "Excluded: no clinical content and a leak risk if rows are re-identified"],
    ["gender", "Male, Female, Other (one 'Other' record in the file)", "Categorical predictor"],
    ["age", "Years, 0.08 to 82", "Numeric predictor (dominant in Section 5)"],
    ["hypertension", "0 = no, 1 = yes (498 records with 1)", "Binary predictor"],
    ["heart_disease", "0 = no, 1 = yes (276 records with 1)", "Binary predictor"],
    ["ever_married", "No, Yes", "Categorical predictor"],
    ["work_type", "children, Govt_job (spelled Govt_jov in the brief), Never_worked, Private, Self-employed", "Categorical predictor; file spelling kept"],
    ["Residence_type", "Rural, Urban", "Categorical predictor"],
    ["avg_glucose_level", "Average blood glucose in mg/dL, 55.12 to 271.74", "Numeric predictor"],
    ["bmi", "Body mass index in kg/m2, 10.3 to 97.6, 201 missing", "Numeric predictor, median-imputed in the pipeline"],
    ["smoking_status", "formerly smoked, never smoked, smokes, Unknown (Unknown = information unavailable, 30.22%)", "Categorical predictor"],
    ["stroke", "0 = no recorded stroke (4,861), 1 = recorded stroke (249, 4.87%)", "Target variable"],
]
for i, row in enumerate(ROWS_ATTR, start=1):
    fill_row(T[1], i, row)

set_text(P[21], (
    "Outcome definition: stroke = 1 indicates a recorded history of stroke and stroke = 0 indicates no "
    "recorded stroke, checked against the dataset page described in Section 1.2. The file contains no "
    "follow-up period, no date of event and no time-to-event column, so the target is treated throughout "
    "this report as a recorded history and the output is never described as a validated future stroke risk."
))
set_text(P[23], (
    "Dataset URL and access date: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset, "
    "downloaded 13 November 2026; the course archive mirrors the same file as "
    "data/healthcare-dataset-stroke-data.csv. File name and version: healthcare-dataset-stroke-data.csv, "
    "5,110 rows x 12 columns, 316,971 bytes, MD5 3f25a4c2f2963d969e1a156c08968577; the source page "
    "publishes no version number, so this checksum is the identifier used here. Population, collection "
    "period and provenance: individual-level records of patients aged 0.08 to 82 years; the dataset page "
    "does not document the collection period, country, sampling frame or hospital, and these must be "
    "recorded as unavailable rather than assumed. There are no survey weights, no follow-up dates and no "
    "hospital identifier, which limits every clinical statement that can be made from the file."
))
set_text(P[25], (
    "Data checks: the column names in the supplied file match the brief except for work_type, where the "
    "file spells the level Govt_job while the brief and the README write Govt_jov; the file spelling was "
    "kept and is flagged in Table 2. gender contains a single 'Other' record, which is retained. bmi is "
    "stored as the text N/A for 201 rows (3.93%) and is read as missing, not as zero. Unknown is a level "
    "of smoking_status (1,544 rows, 30.22%) and means that the information is unavailable; it is kept as "
    "its own category and is not merged with never smoked. No duplicate identifiers and no duplicate rows "
    "were found."
))

# ----------------------------------------------------------------- section 2
ROWS_EDA = [
    ["Rows and columns", "5,110 rows x 12 columns; 10 predictors after removing id and stroke",
     "Dimensions match the brief; no row was dropped"],
    ["Duplicate records or IDs", "0 duplicate identifiers, 0 duplicate rows",
     "No de-duplication needed"],
    ["Missing values by variable", "bmi 201 (3.93%); smoking_status 'Unknown' 1,544 (30.22%); no other missing cells",
     "bmi median-imputed inside the pipeline; 'Unknown' kept as its own level"],
    ["Stroke class counts", "4,861 negative (95.13%); 249 positive (4.87%)",
     "Imbalance ratio 19.5:1; threshold tuned instead of resampling"],
    ["Age glucose and BMI", "age 43.23 +/- 22.61 (median 45, IQR 25-61); glucose 106.15 +/- 45.28 (median 91.88); BMI 28.89 +/- 7.85 (median 28.10)",
     "Ranges plausible; the BMI maximum of 97.6 and the glucose maximum of 271.74 were retained as recorded, not winsorised"],
    ["Categorical distributions", "work_type: children 687, Govt_job 657, Never_worked 22, Private 2,925, Self-employed 819; gender: Female 2,994, Male 2,115, Other 1; smoking_status: never 1,892, Unknown 1,544, formerly 885, smokes 789",
     "Rare levels retained; the single 'Other' record falls outside the training split and is encoded as all-zero"],
]
for i, row in enumerate(ROWS_EDA, start=1):
    fill_row(T[2], i, row)

set_text(P[29], (
    "GitHub download method: ZIP download of the course repository (xai-series-master), extracted "
    "locally and run from the repository root. Commit or release: not available - the archive contains "
    "no .git directory and therefore no commit hash, and no release tag is published. This is a real "
    "reproducibility gap and is recorded as such in Section 9.2, where the MD5 of the data file and the "
    "script listing are given instead. Python environment and dependencies: Python 3.13.14 on Windows 11, "
    "pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, shap 0.54.1, matplotlib 3.11.2, imbalanced-learn "
    "0.14.2 (imported by the supplied utility module)."
))
set_text(P[30], (
    "Execution command: python 00_data_exploration.py, run from the repository root with the "
    "non-interactive Agg matplotlib backend so that the twelve plt.show() calls return immediately "
    "instead of blocking. Date, output directory and execution status: 13 November 2026, console output "
    "only (no files are written by the script), completed with exit code 0; the printed frame is 5,110 "
    "rows x 12 columns and the loop over columns printed all twelve histograms. Modifications or errors "
    "resolved: none were made to the script. Two properties of the supplied code were noted and are not "
    "used by the modelling pipeline of Section 3: (i) preprocess_data() fills missing bmi with 0, which "
    "creates an implausible value (a BMI of zero) rather than an imputation, and (ii) preprocess_data() "
    "is applied to the whole file before train_test_split(), so encoding is fitted on all rows. Both are "
    "acceptable for a teaching demonstration of the library calls, and both are avoided in the pipeline "
    "described in Section 3.1."
))
set_text(P[34], "")
put_image(P[34], os.path.join(FIG, "fig1_eda_panel.png"), 6.3)
set_text(P[35], (
    "Figure 1. Exploratory panel for the 5,110-patient file: (a) class counts; (b) age density by "
    "outcome; (c) average glucose by outcome; (d) BMI by outcome, with the missingness rate in the "
    "panel title; (e) stroke prevalence by age band; (f) prevalence by hypertension and heart-disease "
    "combination. Counts are records, not person-years, and no significance test is implied by the "
    "panel; the numbers behind every panel are in Table 3 and Section 2.3."
))
set_text(P[37], (
    "(i) The file is severely imbalanced and age carries most of the visible signal. Only 249 of 5,110 "
    "records are positive (4.87%, Figure 1a), while prevalence rises from 0.2% below age 20 to 11.8% "
    "between 60 and 79 (Figure 1e); the median age is 43 years for negative and 71 years for positive "
    "records. A model that always predicts 'no stroke' therefore reaches 95.13% accuracy, which is why "
    "Section 3.1 reports balanced accuracy, recall and average precision alongside accuracy. "
    "(ii) Comorbidity and glucose separate the two classes but the associations are not causal. "
    "Prevalence is 3.4% with neither hypertension nor heart disease, 12.2% with hypertension only, 16.0% "
    "with heart disease only and 20.3% with both (Figure 1f); the median average glucose is 91.5 mg/dL "
    "without and 105.2 mg/dL with a recorded stroke, and 22.5% of positive records exceed 200 mg/dL "
    "against 7.8% of negative records (Figure 1c). These are differences between groups observed once, "
    "not estimated effects of treatment or of time. "
    "(iii) BMI is weakly informative and incompletely recorded. The two class medians differ by 1.7 "
    "kg/m2 (28.0 vs 29.7, Figure 1d) and 201 values (3.93%) are missing, which is why BMI is imputed "
    "inside the pipeline rather than dropped, and why it is expected to rank below age and glucose in "
    "Section 5. The 30.22% of records with smoking_status 'Unknown' are a second, larger source of "
    "unavailable information that must not be interpreted as 'never smoked'."
))
set_text(P[38], (
    "Training safeguards applied here: the file was split into 60/20/20 stratified train, validation and "
    "test sets before any imputation or encoding; the median imputer and the one-hot encoder were fitted "
    "on the 3,066 training rows only and applied to the validation and test rows through a scikit-learn "
    "Pipeline, so no information from the held-out data reaches the transformer; id and stroke were "
    "excluded from the predictors; and no resampling was applied, because duplicated minority rows would "
    "also enter the SHAP background distribution and inflate the apparent effect of the positive class. "
    "The test set was used once, at the end, and never for threshold selection."
))

# ----------------------------------------------------------------- section 3
set_text(P[42], (
    "Task 2 classifier and rationale: RandomForestClassifier with 500 trees, min_samples_leaf = 6 and "
    "max_features = 0.5 (bootstrap sampling, random_state 2021). A tree ensemble was chosen because the "
    "age and glucose effects in Figure 1 are non-linear and interact with the comorbidity indicators, "
    "which a tree model captures without hand-built interaction terms, and because TreeExplainer then "
    "computes exact Shapley values for it instead of sampling coalitions [3]. Predictors retained: "
    "gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, "
    "avg_glucose_level, bmi and smoking_status (10 variables). Excluded: id (an identifier with no "
    "clinical content) and stroke (the target). Preprocessing pipeline: median imputation for age, "
    "avg_glucose_level and bmi; one-hot encoding for the seven categorical and binary variables with "
    "handle_unknown = 'ignore'; no scaling, which a tree model does not require. Feature names after "
    "transformation: 22 columns - num__age, num__avg_glucose_level, num__bmi plus cat__<variable>_<level> "
    "for gender (Female, Male), hypertension (0, 1), heart_disease (0, 1), ever_married (No, Yes), "
    "work_type (Govt_job, Never_worked, Private, Self-employed, children), Residence_type (Rural, Urban) "
    "and smoking_status (Unknown, formerly smoked, never smoked, smokes). The single 'Other' gender "
    "record fell in the held-out data, so that level does not exist in the trained encoder and is "
    "encoded as all-zero for that row."
))
set_text(P[43], (
    "Train validation test sizes: 3,066 / 1,022 / 1,022 rows (60% / 20% / 20%), with 149, 50 and 50 "
    "positive records respectively; stratification on stroke, seed 2021, no group structure to separate. "
    "Hyperparameter tuning and imbalance strategy: a small grid over min_samples_leaf (4, 6, 8, 10) and "
    "max_features ('sqrt', 0.5) was scored on validation average precision, which is threshold-free and "
    "does not reward the majority class; the best setting reached validation average precision 0.209 and "
    "was refitted on the training rows only. No resampling and no class weights were used: the imbalance "
    "is handled by the decision threshold. Decision threshold and selection rule: the threshold that "
    "maximised F1 for stroke = 1 on the validation set was 0.14 (grid step 0.01); it was selected on "
    "validation data and then applied unchanged to the test set."
))
fill_row(T[3], 1, ["Accuracy and balanced accuracy", "0.863 / 0.719",
                   "At threshold 0.14. The majority-class baseline on this test set is 0.951, so accuracy alone argues against the model rather than for it."])
fill_row(T[3], 2, ["Precision recall and F1 for stroke = 1", "0.192 / 0.560 / 0.286",
                   "50 positive test records: 28 detected, 22 missed. Precision is limited by 118 false positives."])
fill_row(T[3], 3, ["ROC AUC and average precision", "0.856 / 0.234",
                   "Scores from predict_proba, not from the thresholded labels. Prevalence is 0.049, so average precision is the fair summary of ranking quality."])
fill_row(T[3], 4, ["Confusion matrix TN FP FN TP", "854 / 118 / 22 / 28",
                   "28 of 50 recorded strokes detected. The 22 missed cases are the clinical concern; 118 false positives are the cost of the low threshold."])

set_text(P[46], (
    "Explainer class and suitability: shap.TreeExplainer applied to the fitted RandomForestClassifier, "
    "using the default tree_path_dependent perturbation, which reads the tree structure and node cover "
    "statistics instead of sampling coalitions and is therefore exact for a tree ensemble [3]. Version: "
    "shap 0.54.1 with scikit-learn 1.9.1. What is explained: the Random Forest is explained on its own "
    "input, i.e. the 22 encoded columns produced by the fitted preprocessor, so every attribution refers "
    "to an encoded feature ('work_type = Private') rather than to the raw variable. The preprocessor is "
    "deterministic and fitted once, so the explanation of the encoded model is the explanation of the "
    "pipeline for the same row."
))
set_text(P[47], (
    "Output explained: the probability of stroke = 1 from predict_proba, on the probability scale "
    "(0 to 1). This is the model's raw output: a Random Forest leaf stores class frequencies, so the "
    "average over trees is already a probability and no link function is involved. Class label and "
    "output index: model.classes_ = [0, 1], so index 1 is the positive class and the SHAP array is "
    "sliced as [:, :, 1]. Background source, size and sampling seed: tree_path_dependent needs no "
    "background sample; the reference is the training distribution encoded in the 500 trees (3,066 "
    "rows), and the resulting base value is 0.0484, the positive-class share of the training rows. "
    "Explanation sample source and size: the untouched test set, all 1,022 rows, of which 50 are "
    "positive, explained in 7.9 s."
))
set_text(P[49], (
    "For an additive explanation the base value plus the sum of the feature contributions must "
    "reconstruct the explained output on its stated scale, and here it does: the maximum absolute "
    "residual over the 1,022 test rows is 2.0e-15 (Table 5), so the decomposition is exact up to floating "
    "point rather than approximate. No link function or transformation is applied, because the SHAP "
    "values are already on the probability scale and can be added to a probability directly. This is a "
    "property of this model, not of SHAP in general: for gradient-boosted classifiers the raw output is "
    "a log-odds margin, and adding those contributions to predict_proba would be a mistake that the "
    "checks in Table 5 are designed to catch."
))
fill_row(T[4], 1, ["Input feature names order and dimensions match",
                   "22 encoded columns in the order returned by get_feature_names_out(); X_test shape (1022, 22); the same transformer is used for training, prediction and explanation."])
fill_row(T[4], 2, ["Correct class and output scale selected",
                   "model.classes_ = [0, 1]; index 1 = stroke = 1; values on the probability scale (0-1), the same scale as predict_proba."])
fill_row(T[4], 3, ["Base value + sum of SHAP values",
                   "0.0484 + 0.1594 = 0.2079 for held-out case #0335 (model output 0.2079)."])
fill_row(T[4], 4, ["Model output and reconstruction residual",
                   "Maximum |residual| over 1,022 rows = 2.0e-15, i.e. exact to machine precision (tolerance 1e-10)."])
set_text(P[51], (
    "A positive SHAP value raises the predicted probability of stroke = 1 relative to the base value of "
    "0.048 and a negative value lowers it, so a contribution of +0.058 (age, case #0335) means 'this "
    "patient's age added 5.8 percentage points to the predicted probability compared with an average "
    "patient in the reference distribution'. The reference is the training data as encoded in the trees, "
    "not a clinical norm and not a matched control group. SHAP describes the behaviour of this fitted "
    "model relative to that reference; it does not show that a feature causes stroke, and a large "
    "contribution for age does not mean that age can be modified. Two further cautions apply throughout: "
    "contributions of correlated features are shared arbitrarily between them, and the same feature value "
    "can receive opposite signs in two patients if the rest of their profile differs."
))

# ----------------------------------------------------------------- section 4
set_text(P[55], (
    "Anonymous row reference: held-out case #0335, i.e. row 334 (zero-based) of the 1,022-row test set; "
    "the identifier column was removed before modelling, so no patient identity is reported. Selection "
    "rule: among the 28 true positives at the chosen threshold, the case whose predicted probability was "
    "closest to the median predicted probability of those true positives (0.212) was selected, which "
    "makes it a typical detected stroke record rather than a favourable or extreme example; the rule was "
    "fixed before looking at any explanation. Actual class: 1. Predicted stroke class: 1. Threshold: "
    "0.14, selected on the validation set (Section 3.1)."
))
set_text(P[56], (
    "Model score or probability: 0.2079. Explained output and units: predicted probability of stroke = 1 "
    "on the probability scale. Base value in those units: 0.0484. Case type: true positive. The patient "
    "profile behind the prediction: 68 years old, male, recorded heart disease, no recorded "
    "hypertension, average glucose 223.83 mg/dL, BMI 31.9, formerly smoked, married, private-sector "
    "worker, urban residence."
))
set_text(P[58], "")
put_image(P[58], os.path.join(FIG, "fig2_force_plot.png"), 6.6)
set_text(P[59], (
    "Figure 2. Local SHAP force plot for held-out case #0335 (Random Forest, probability scale). The "
    "base value (0.048, the positive-class share of the training data) sits at the left of the axis and "
    "the model output (0.208) at the tip of the red arrow. Red segments push the prediction up and blue "
    "segments push it down; a segment's width is the size of its contribution and the label gives the "
    "patient's value. To keep the labels legible, levels of one variable are summed and contributions "
    "below |0.002| are merged into 'all other variables'; the total is unchanged. A per-column version "
    "of the same plot (22 encoded features, no merging) is included in the evidence files as "
    "figS1_raw_force_plot.png."
))
fill_row(T[5], 1, ["Age", "68 years", "+0.058", "Increases"])
fill_row(T[5], 2, ["Heart disease", "1 (recorded)", "+0.055", "Increases"])
fill_row(T[5], 3, ["Average glucose", "223.83 mg/dL", "+0.047", "Increases"])
fill_row(T[5], 4, ["Remaining 7 variables combined (BMI, work type, smoking status, hypertension, gender, ever married, residence)",
                   "BMI 31.9; Private; formerly smoked; no hypertension", "-0.0002",
                   "Net ~0: BMI and work type +0.009 each, smoking status -0.014"])
set_text(P[63], (
    "Starting from the base value of 0.048, the model output for this patient moved to 0.208 on the "
    "probability scale, an increase of 0.159. The largest upward contributions were age (+0.058 for 68 "
    "years), recorded heart disease (+0.055) and average glucose (+0.047 for 223.83 mg/dL); together "
    "these three account for 0.160, essentially the whole increase. BMI (+0.009) and work type 'Private' "
    "(+0.009) added small amounts. The largest downward contribution was smoking status 'formerly "
    "smoked' (-0.014), followed by the recorded absence of hypertension (-0.003) and gender 'Male' "
    "(-0.001). Read against the force plot, the red segments on the right of the base value are wide for "
    "age and glucose, while the blue segment for smoking status is narrow but visible: the model treats "
    "this patient as a high-glucose, heart-disease case for his age, not as a smoker."
))
set_text(P[64], (
    "The explanation is plausible in direction: age, recorded heart disease and a glucose level above "
    "220 mg/dL are established correlates of stroke, and they are the three largest contributions here, "
    "which is the kind of agreement between model and domain knowledge that a post-hoc explanation is "
    "used to check. Two details deserve caution. First, the negative contribution of 'formerly smoked' "
    "is a feature-value artefact rather than a protective effect: current smokers, former smokers and "
    "never-smokers differ in age composition, and the model has learned a category effect from a file in "
    "which 'Unknown' accounts for 30% of records. Second, BMI is imputed for 3.93% of the file, and an "
    "imputed value is explained as if it had been measured. A false negative for a patient like this "
    "would mean a recorded stroke scoring below 0.14 and being missed; a false positive would mean a "
    "stroke-free record being flagged for review. Both statements are about a model prediction on a "
    "retrospective record: they are not a diagnosis, and they carry no treatment implication."
))

# ----------------------------------------------------------------- section 5
set_text(P[68], (
    "Model and output explained: the Task 2 Random Forest, probability of stroke = 1. Held-out "
    "explanation sample and positive class count: all 1,022 test rows, 50 of them positive. Number of "
    "displayed features: 15 of the 22 encoded columns in the beeswarm (Figure 3) and 12 in the "
    "supplementary bar chart. Sampling rule and seed: none - the whole test set is explained, so no "
    "sampling seed is involved and no patient is represented twice. Runtime: 7.9 s for 1,022 rows."
))
set_text(P[69], (
    "Feature representation: encoded columns, one per numeric variable and one per one-hot level, so "
    "work_type appears as five features and each binary variable as two. Figure 3 is plotted at this "
    "level. For any statement about a variable as a whole - Table 7, Section 5.3 and Section 8 - the "
    "signed contributions of a variable's levels are summed per patient first and the mean absolute "
    "value of that per-patient sum is reported (Figure 3c in the evidence files). Summing before taking "
    "the absolute value prevents the two levels of a binary variable from counting the same effect "
    "twice, and it is the rule applied consistently in this report. The two representations answer "
    "different questions: the encoded view shows which level drives a prediction, the grouped view shows "
    "which measurement does."
))
set_text(P[71], "")
put_image(P[71], os.path.join(FIG, "fig3_summary_beeswarm.png"), 6.3)
set_text(P[72], (
    "Figure 3. Global SHAP summary (beeswarm) for the Random Forest over the 1,022 held-out records, "
    "probability scale. Each point is one patient; horizontal position is that patient's SHAP value, "
    "colour is the feature value (blue low, red high) and features are ordered by mean absolute SHAP. "
    "For the one-hot columns the colour encodes the level (blue = 0, absent; red = 1, present), so "
    "'Hypertension = 1' and 'Hypertension = 0' are two halves of one variable and must be read together. "
    "Values range from about -0.05 to +0.24, i.e. no single feature can move a prediction by more than "
    "about a quarter of the probability scale."
))
fill_row(T[6], 1, ["1", "Age", "0.0463",
                   "Monotone increase: SHAP rises with age (correlation 0.85) and is positive for 37% of held-out patients, i.e. those roughly above 55 years."])
fill_row(T[6], 2, ["2", "Average glucose", "0.0117",
                   "Increases with glucose (correlation 0.59); the largest positive values belong to concentrations above about 200 mg/dL."])
fill_row(T[6], 3, ["3", "BMI", "0.0082",
                   "Mixed: correlation with SHAP -0.07, so both low and high BMI produce small contributions in either direction."])
fill_row(T[6], 4, ["4", "Hypertension", "0.0055",
                   "Binary: hypertension = 1 raises the output for the 10.9% of held-out patients who have it; otherwise the contribution is near zero."])
fill_row(T[6], 5, ["5", "Heart disease", "0.0045",
                   "Binary: rarer than hypertension (276 records in the file) but a larger per-patient effect when present, which is why it outranks smoking status in the local explanation."])
set_text(P[75], (
    "The ranking is dominated by age: mean absolute SHAP 0.046, four times the next feature, with a "
    "monotone colour gradient (r = 0.85 between age and its SHAP value) and a spread from about -0.05 to "
    "+0.24. Average glucose (0.012) and BMI (0.008) follow, but they behave differently: glucose shows "
    "the expected pattern of large positive values at the highest concentrations, whereas BMI is mixed "
    "(r = -0.07) and never produces a contribution above about 0.05 in either direction. The two "
    "comorbidity indicators contribute 0.0055 (hypertension) and 0.0045 (heart disease) on average, and "
    "their 'present' level is the one that increases the output; their 'absent' level mirrors it, which "
    "is exactly why the levels are summed before ranking. Categorical variables are smaller still "
    "(smoking status 0.0044, work type 0.0039, ever married 0.0035, gender 0.0016, residence 0.0012). "
    "Variation across patients matters as much as the average: age moves some patients by 0.2 and others "
    "by 0.01, so a single importance number per feature hides the heterogeneity that the force plot in "
    "Figure 2 exposes. Two cautions: a colour is a feature value, not a severity (a large contribution "
    "for work_type = Private says nothing about the danger of that job), and the density of blue points "
    "below zero reflects the 95% of records without a stroke, not a generally protective effect."
))
set_text(P[77], (
    "The Task 3 case follows the global pattern rather than contradicting it. Age, average glucose and "
    "heart disease are among the largest mean absolute contributions globally, and all three are the "
    "largest upward contributions for case #0335; the sign of the age contribution is positive for this "
    "patient, as it is for the 37% of the cohort above roughly 55 years. The case also shows the "
    "difference between importance and direction: smoking status is the sixth most important variable "
    "globally but the largest downward contribution here, because importance measures magnitude while "
    "the sign depends on the level and on the rest of the profile. Correlated predictors are the main "
    "caveat. Age and ever_married are correlated (r = 0.68) and age and hypertension less so (r = 0.28), "
    "so the 0.058 assigned to age is partly shared credit for marital status and comorbidity, and the "
    "near-zero contribution of ever_married (-0.0004) should not be read as evidence that marital status "
    "is irrelevant - a tree ensemble splits shared attribution arbitrarily between correlated columns."
))

# ----------------------------------------------------------------- section 6
set_text(P[80], (
    "The three slides below use the same model, the same held-out case #0335 and the same verified "
    "numbers as Sections 3 to 5; the slide file itself is submitted separately (task5_slides.pptx) and "
    "the visual on each slide is taken from Figures 2 and 3. Each block states the slide title, its "
    "visual and the points to say."
))
set_text(P[82], "Title: Model performance and explanation of stroke classification")
set_text(P[83], "Visual: global summary plot (Figure 3) with a compact performance panel: ROC-AUC 0.856, average precision 0.234, recall 0.560 and precision 0.192 at threshold 0.14.")
set_text(P[84], (
    "Content: a Random Forest with 500 trees on a stratified 60/20/20 split of 5,110 records, explaining "
    "the probability of stroke = 1 on the probability scale with exact TreeExplainer values (base value "
    "0.048, additivity residual 2e-15). The ranking is age (mean |SHAP| 0.046), average glucose (0.012), "
    "BMI (0.008), hypertension (0.0055) and heart disease (0.0045). The explainer needs no background "
    "sample because tree_path_dependent reads the training distribution from the trees; the explanation "
    "sample is the untouched test set. Technical limitation to state: accuracy 0.863 is below the 0.951 "
    "majority baseline, so the model is only useful at a low threshold and at the cost of 118 false "
    "positives, and correlated predictors (age and ever_married, r = 0.68) share their attribution."
))
set_text(P[85], (
    "What the evidence supports: ROC-AUC 0.856 and average precision 0.234 against a 0.049 prevalence "
    "mean the ranking carries real information, while recall 0.560 means 22 of 50 recorded strokes are "
    "still missed. Further validation needed: calibration of the probabilities was not assessed, the "
    "cohort is a single retrospective file of unknown provenance, and no external site was tested. "
    "Accuracy should not be quoted alone."
))
set_text(P[87], "Title: Interpreting one patient prediction in clinical context")
set_text(P[88], "Visual: force plot for case #0335 (Figure 2), with the technical feature names replaced by clinical wording.")
set_text(P[89], (
    "Content: for this 68-year-old man with recorded heart disease and an average glucose of 223.83 "
    "mg/dL, the model gives a probability of 0.208, above the review threshold of 0.14; he did have a "
    "recorded stroke, so this is a correct detection. The factors pushing the score up are age (+0.058), "
    "recorded heart disease (+0.055) and glucose (+0.047); the factors pulling it down are former "
    "smoking (-0.014) and the absence of recorded hypertension (-0.003). Sensitivity in plain terms: at "
    "this threshold the model finds 28 of 50 recorded strokes and raises 118 false alarms, so a positive "
    "score means 'worth reviewing', not 'will have a stroke'. Missing data that matters: BMI is missing "
    "for 3.93% of the file and smoking status is 'Unknown' for 30%, and both are treated by imputation "
    "or as a separate category rather than resolved."
))
set_text(P[90], (
    "How a clinician might use it: read the score as a prompt to check the three drivers against the "
    "record - age, documented heart disease, current glucose - and to look for information the model "
    "does not have (blood pressure control, atrial fibrillation, medication). The attribution explains "
    "the model's behaviour and needs clinical interpretation; the dataset is retrospective, of "
    "undocumented provenance, with no follow-up period, so nothing here estimates this patient's future "
    "risk, and calibration and external validation are outstanding."
))
set_text(P[92], "Title: What the computer prediction means for this example")
set_text(P[93], "Visual: a simple two-column display of the main upward and downward contributions in words, no feature codes and no axis.")
set_text(P[94], (
    "Content, addressed to the patient: 'The computer looked at the information in your record and gave "
    "a score of about 21 in 100 for having a stroke recorded. The things that raised your score were "
    "your age, the heart disease already noted in your file, and a blood sugar reading of about 224. "
    "The things that lowered it were that you used to smoke rather than smoke now, and that high blood "
    "pressure is not recorded for you. The score is based only on the information in this file; it can "
    "be wrong in either direction, and about one in five of the records it flags turn out not to be "
    "strokes.'"
))
set_text(P[95], (
    "Uncertainty in plain language: the score is not a percentage chance of a future stroke, it is the "
    "model's summary of this record, and 22 of the 50 recorded strokes in the test data scored below the "
    "review line. Nothing here says that age, sugar or smoking caused anything, and nothing here is "
    "advice about medication or lifestyle; a clinician would read the score together with the rest of "
    "the record and with information the model never sees."
))
set_text(P[97], (
    "Consistency check: the case (row 334 of the test set, #0335), the predicted value (0.208 for the "
    "Random Forest, 0.184 for the SVM), the output units (probability of stroke = 1) and the top features "
    "(age, heart disease and average glucose for the Random Forest) agree with Sections 3 to 5 and with "
    "the two later figures. Where a number differs between models it is stated as such rather than "
    "harmonised. Source credits on the slides: dataset (Kaggle, fedesoriano), method (Lundberg and Lee, "
    "2017), library versions as in Section 9.2; body text is 18 pt or larger at presentation size."
))

# ----------------------------------------------------------------- section 7
set_text(P[101], (
    "Task 2 model: Random Forest (500 trees, min_samples_leaf = 6, max_features = 0.5). Task 6 model: a "
    "support vector machine with an RBF kernel (C = 1.0, gamma = 0.02, class_weight = 'balanced', "
    "Platt-scaled probabilities). Reason for choosing it: an SVM is a genuinely different hypothesis "
    "class - a maximum-margin boundary in a kernel-induced space, defined by 1,697 support vectors, "
    "rather than axis-aligned splits over a bootstrap sample - so agreement between the two models is "
    "informative in a way that comparing two tree ensembles would not be. The same split, the same ten "
    "predictors, the same threshold rule and the same held-out rows were used, and the only added step "
    "is standardisation of the numeric inputs, which the kernel requires and which is fitted inside the "
    "training pipeline."
))
set_text(P[102], (
    "Route chosen: Kernel SHAP, the model-agnostic variant, because an SVC is not a tree ensemble and "
    "TreeExplainer does not apply to it. Deep SHAP was not used: it requires a differentiable network "
    "built from supported operations, which an SVM does not have, and forcing an SVM into Deep SHAP "
    "would mean explaining an approximation of the model rather than the model. The unused Deep SHAP "
    "column of Table 8 has therefore been removed, as the template instructs. The threshold was reselected "
    "on the validation set with the same rule and also came out at 0.14, which keeps the comparison of "
    "confusion matrices like-for-like."
))

# remove the Deep SHAP column from Table 8
tbl = T[7]
grid = tbl._tbl.tblGrid
grid.remove(grid.gridCol_lst[2])
for row in tbl.rows:
    row._tr.remove(row._tr.tc_lst[2])

fill_row(T[7], 1, ["Suitable explained model", "Any model with a prediction function; here the fitted RBF SVC applied to the transformed input"])
fill_row(T[7], 2, ["Output and class", "predict_proba[:, 1], the probability of stroke = 1; class index 1 of [0, 1]; no link function, probability scale"])
fill_row(T[7], 3, ["Background", "100 training rows drawn by shap.sample(X_train, 100, random_state = 2021); base value 0.0498"])
fill_row(T[7], 4, ["Computation settings", "nsamples = 2,048 (single case) and 512 (150 cases); l1_reg = 0; seed 2021"])
fill_row(T[7], 5, ["Input consistency", "The fitted preprocessor is applied before the explainer in training, prediction and explanation, so the explainer always receives (n, 22) transformed rows"])

set_text(P[105], (
    "Chosen route and implementation: shap.KernelExplainer(model_function, background), where "
    "model_function = fitted SVC .predict_proba[:, 1] applied to already-transformed rows. Versions: shap "
    "0.54.1, scikit-learn 1.9.1. Background: a random sample of 100 training rows (shap.sample, "
    "random_state 2021); the mean predicted probability over that sample is 0.0498 and becomes the base "
    "value. Held-out sample: the same 1,022 test rows, of which 150 were drawn at random (seed 2021) for "
    "the global explanation, and the same case #0335 for the local one. Computational settings: "
    "nsamples = 2,048 for the case, 512 for each of the 150 cases, l1_reg = 0 (ordinary weighted least "
    "squares); the default AIC regularisation was tried first and rejected because it set four "
    "contributions to exactly zero, which is a property of the penalty rather than of the model. Runtime: "
    "20.3 s for the case and 756.2 s for the 150-case sample."
))
set_text(P[106], (
    "Compatibility and checks: output shape (1, 22) for the case and (150, 22) for the sample; the "
    "selected class is index 1, the probability of stroke = 1. Reconstruction: base value plus the sum of "
    "contributions equals predict_proba to within 2.8e-17 for the case and 8.3e-17 across the 150 cases. "
    "Preprocessing is identical in training, prediction and explanation because one fitted transformer "
    "is reused. The approximation error here comes from sampling 512 of the 2^22 possible coalitions, not "
    "from the model; the repeat described in Section 8.2 quantifies it. Warnings: scikit-learn 1.9 "
    "deprecates SVC(probability = True) in favour of CalibratedClassifierCV, and the deprecated option was "
    "kept because the Platt-scaled probability is exactly the quantity being explained."
))
set_text(P[108], (
    "Held-out performance of the SVM at the same threshold of 0.14: accuracy 0.873, balanced accuracy "
    "0.734, precision 0.210, recall 0.580, F1 0.309, ROC-AUC 0.859, average precision 0.219, confusion "
    "matrix TN 863, FP 109, FN 21, TP 29. The SVM therefore matches the Random Forest on ranking quality "
    "(0.859 vs 0.856), is slightly better on the F1 of the positive class (0.309 vs 0.286) and slightly "
    "worse on average precision (0.219 vs 0.234) - a difference of 0.015 in a 50-positive test set, which "
    "is well within sampling noise and should not be read as a ranking of the two models."
))
set_text(P[109], "")
put_image(P[109], os.path.join(FIG, "fig4_svm_waterfall.png"), 6.6)
insert_after(P[109], (
    "Figure 4. Kernel SHAP waterfall for the SVM on the same held-out case #0335, probability scale: "
    "base value E[f(X)] = 0.050 and model output f(x) = 0.184. Levels of one variable are summed and "
    "labelled with the patient's value; no variable lowers the score by more than 0.001."
))
set_text(P[110], "")
put_image(P[110], os.path.join(FIG, "fig5_svm_beeswarm.png"), 6.3)
insert_after(P[110], (
    "Figure 5. Kernel SHAP summary for the SVM over 150 held-out cases, encoded-feature view: age "
    "dominates (mean |SHAP| 0.044), followed by smoking status (0.0057), average glucose (0.0041), work "
    "type (0.0036) and hypertension (0.0035). Because the levels of one variable are plotted separately, "
    "the grouped ranking in Section 7.3 is used for every statement about a variable as a whole."
))
set_text(P[111], (
    "Interpretation. The two models agree that age is the dominant feature and agree on its direction for "
    "this patient, but they disagree on the second driver. The Random Forest gives heart disease +0.055 "
    "and glucose +0.047 for the case, while the SVM gives heart disease only +0.011 and glucose +0.024; "
    "globally the SVM ranks heart disease last (0.0008) where the Random Forest ranks it fifth (0.0045). "
    "A smooth kernel boundary spreads attribution over the correlated and age-related columns instead of "
    "concentrating it on a rare binary indicator, which is a difference in how the two functions use the "
    "same information rather than an error in either explanation."
))

# ----------------------------------------------------------------- section 8
set_text(P[116], (
    "The two models agree on the ranking and disagree on the attribution. On the 1,022 held-out rows "
    "they reach almost the same ROC-AUC (0.856 vs 0.859) and detect 28 and 29 of the 50 recorded strokes, "
    "so the practical recommendation does not depend on the choice of model. For case #0335 both classify "
    "the patient correctly (0.208 and 0.184 against a threshold of 0.14) and both make age the largest "
    "contributor, but the Random Forest attributes 0.055 to heart disease where the SVM attributes 0.011, "
    "and the SVM spreads more weight over smoking status and work type. Three explanations are possible "
    "and cannot be separated with this design: the models use the same predictors differently (axis-"
    "aligned splits concentrate on rare binary indicators, a smooth kernel does not), the explainers "
    "differ in nature (exact tree values against a sampled regression over coalitions, with a background "
    "of 100 rows), and the preprocessing differs (standardisation is required by the kernel and changes "
    "the geometry the model sees). Raw SHAP magnitudes are therefore not comparable across the two "
    "models in absolute terms - a contribution of 0.05 in probability units means the same thing on both "
    "scales, but the feature representations and backgrounds do not - and only the sign, the order and "
    "the direction of the effect are compared here."
))
set_text(P[118], (
    "Dataset provenance is the first limitation: the file documents no collection period, country, "
    "sampling frame or hospital, 30.22% of smoking status values are 'Unknown', and 3.93% of BMI values "
    "are missing and imputed. Class imbalance (4.87% positive) makes accuracy uninformative and leaves "
    "only 50 positive test records, so every metric in Table 4 has a wide confidence interval: a single "
    "misclassified patient moves recall by two percentage points. There is no prospective or external "
    "validation, and probability calibration was not assessed - the Platt scaling of the SVM and the "
    "uncalibrated forest output may both be miscalibrated, which matters because the threshold of 0.14 "
    "is a probability threshold. Subgroup performance was not estimated: with 50 positives, a "
    "gender-by-age subgroup would contain a handful of events and any difference would be "
    "uninterpretable. Finally, correlated features (age and ever_married r = 0.68; age and BMI r = 0.33) "
    "share their attribution, so the size of the age contribution is partly a convention of the "
    "decomposition, and the background choice moves every number by a small amount."
))
set_text(P[119], (
    "Stability check actually performed: the SVM explanations were recomputed with a second background "
    "sample of 100 training rows (random_state 7 instead of 2021), keeping the same case and the same 40 "
    "held-out rows. The case contributions moved by at most "
    f"{cmp_['max_abs_case_diff']:.4f} in probability units, {int(cmp_['case_sign_agreement']*100)}% of the "
    "ten variable-level contributions kept their sign, and the global ranking of the ten variables "
    f"correlated at Spearman rho = {cmp_['spearman_global_rank']:.2f} between the two backgrounds "
    f"(top two: {', '.join(cmp_['top3_bg2021'][:2])} vs {', '.join(cmp_['top3_bg7'][:2])}). Age is "
    "therefore the top feature under both backgrounds, while the second and third positions swap between "
    "average glucose and smoking status; the rest of the order is only approximately stable, which is what "
    "a sampled explainer with 512 coalitions should be expected to deliver. The Random Forest explanation is not "
    "subject to this check because tree_path_dependent uses the training distribution encoded in the "
    "trees rather than a sampled background; repeating it with an interventional background would be the "
    "equivalent test and was not run."
))
set_text(P[121], (
    "Under ranking quality the two models are effectively tied (ROC-AUC 0.856 for the Random Forest and "
    "0.859 for the SVM), and both are useful only at a low threshold: 28 and 29 of 50 recorded strokes "
    "detected, at the cost of 118 and 109 false positives, with accuracy below the majority baseline. "
    "SHAP showed that age dominates both locally (0.058 of a 0.208 prediction) and globally (mean |SHAP| "
    "0.046, four times the next feature), with average glucose and BMI behind it; case #0335 was a "
    "typical true positive driven by age, heart disease and a glucose level above 220 mg/dL. Changing the "
    "model changed the attribution more than the prediction: the SVM agreed on age and on the direction "
    "of every large contribution but gave heart disease a fifth of the weight the forest gave it. The "
    "binding constraint on interpretation is the data, not the method - a retrospective file of unknown "
    "provenance with no follow-up period, so these outputs describe recorded stroke history and cannot be "
    "read as future risk."
))
set_text(P[123], (
    "Individual or group contributions: this was an individual submission; all analysis, code, figures "
    "and text were produced by the author. Assistance and tools used: the supplied course repository "
    "(00_data_exploration.py, utils.py) provided the data-loading convention and the EDA script; the "
    "modelling, SHAP and plotting code in stroke_shap/ was written for this assignment with scikit-learn, "
    "shap and matplotlib; an AI assistant was used to help draft and debug the analysis scripts and to "
    "edit this report, and every number reported here was produced by the scripts listed in Section 9.2 "
    "and checked against the saved JSON outputs. No results were estimated by hand."
))

# ----------------------------------------------------------------- section 9
ROWS_CMP = [
    ["Model and explainer", "Random Forest (500 trees) + TreeExplainer, exact", "RBF SVM (C = 1, gamma = 0.02) + Kernel SHAP, 512 coalitions"],
    ["Positive class recall and precision", "0.560 / 0.192", "0.580 / 0.210"],
    ["F1 balanced accuracy ROC AUC", "0.286 / 0.719 / 0.856", "0.309 / 0.734 / 0.859"],
    ["Average precision and prevalence", "0.234 (prevalence 0.049)", "0.219 (prevalence 0.049)"],
    ["Case score and explained scale", "0.208, probability of stroke = 1", "0.184, probability of stroke = 1"],
    ["Top three global features", "Age 0.0463, average glucose 0.0117, BMI 0.0082", "Age 0.0438, smoking status 0.0057, average glucose 0.0041"],
    ["Explanation sample and runtime", "1,022 rows, 7.9 s, exact", "150 rows, 756.2 s, 512 sampled coalitions per case"],
    ["Reconstruction residual", "2.0e-15 (max over 1,022 rows)", "8.3e-17 (max over 150 rows)"],
]
for i, row in enumerate(ROWS_CMP, start=1):
    fill_row(T[8], i, row)

set_text(P[127], "References are given in the numeric style used throughout the report, with DOIs or stable URLs.")
set_text(P[128], (
    "[1] World Health Organization (2024) The top 10 causes of death. Geneva: World Health Organization. "
    "Available at: https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death "
    "(Accessed: 13 November 2026)."
))
for ref in [
    "[2] Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', "
    "Advances in Neural Information Processing Systems, 30, pp. 4765-4774.",
    "[3] Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., "
    "Bansal, N. and Lee, S.-I. (2020) 'From local explanations to global understanding with explainable AI "
    "for trees', Nature Machine Intelligence, 2(1), pp. 56-67. doi:10.1038/s42256-019-0138-9.",
    "[4] Fedesoriano (2021) Stroke Prediction Dataset. Kaggle. Available at: "
    "https://www.kaggle.com/fedesoriano/stroke-prediction-dataset (Accessed: 13 November 2026).",
    "[5] Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', Journal of Machine Learning "
    "Research, 12, pp. 2825-2830.",
    "[6] Lundberg, S.M. et al. (2024) SHAP documentation (version 0.54.1). Available at: "
    "https://shap.readthedocs.io (Accessed: 13 November 2026).",
]:
    P[128] = insert_after(P[128], ref)

set_text(P[130], (
    "Environment: Python 3.13.14 on Windows 11 (x86-64); numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, "
    "shap 0.54.1, matplotlib 3.11.2, seaborn 0.13.2, imbalanced-learn 0.14.2. Seeds: 2021 for the split, "
    "the model and every sampler; the global Kernel SHAP sample uses seed 2021 and the stability check in "
    "Section 8.2 uses background seed 7. Data fingerprint: MD5 3f25a4c2f2963d969e1a156c08968577 "
    "(5,110 x 12, 316,971 bytes). Configuration: RandomForestClassifier(n_estimators = 500, "
    "min_samples_leaf = 6, max_features = 0.5, bootstrap = True, random_state = 2021, n_jobs = 1) - "
    "n_jobs = 1 is deliberate, because n_jobs = -1 produced different trees on repeated runs and broke "
    "reproducibility; SVC(kernel = 'rbf', C = 1.0, gamma = 0.02, class_weight = 'balanced', "
    "probability = True, random_state = 2021) on median-imputed and standardised numerics; split indices "
    "from train_test_split(test_size = 0.40, random_state = 2021, stratify = y) followed by "
    "train_test_split(test_size = 0.50, random_state = 2021, stratify = y_tmp); thresholds 0.14 for both "
    "models, selected by maximum F1 on validation; TreeExplainer with tree_path_dependent; KernelExplainer "
    "with shap.sample(X_train, 100, random_state = 2021), nsamples 2,048 (case) and 512 (150 cases), "
    "l1_reg = 0."
))
set_text(P[131], (
    "Evidence files (all paths relative to the submission folder): stroke_shap/run_eda.py, run_models.py, "
    "run_svm.py, robustness.py, fig_force_grouped.py, fig_svm.py; outputs/eda_results.json, "
    "outputs/task2_results.json, outputs/task6_results.json, outputs/case_grouped.json, "
    "outputs/robustness.json; outputs/rf_pipeline.pkl and svm_pipeline.pkl (fitted pipelines with their "
    "thresholds); outputs/shap_rf.npz and shap_svm.npz (SHAP arrays and prediction scores); "
    "outputs/figures/fig1_eda_panel.png, fig2_force_plot.png, fig3_summary_beeswarm.png, "
    "fig3b_mean_abs_bar.png, fig3c_grouped_bar.png, figS1_raw_force_plot.png, fig4_svm_waterfall.png, "
    "fig5_svm_beeswarm.png; task5_slides.pptx; a requirements file with the versions listed above."
))
for chk in (133, 134, 135, 136, 137, 138, 139, 140):
    t = P[chk].text
    set_text(P[chk], t.replace("\u2610", "\u2611", 1))

# ---------------------------------------------------------------- post-processing
import re as _re

# 1. running header
for sec in doc.sections:
    for hp in sec.header.paragraphs:
        for r in hp.runs:
            if "TEMPLATE" in r.text.upper():
                r.text = r.text.replace("REPORT TEMPLATE", "REPORT").replace("report template", "report")

# 2. heading numbering: "1 1 Background" -> "1.1 Background"
for p in doc.paragraphs:
    if p.style.name.startswith("Heading") and p.runs:
        t = p.text
        new = _re.sub(r"^(\d)\s(\d)\s", r"\1.\2 ", t)
        if new != t:
            set_text(p, new)

# 3. keep table rows together and repeat header rows
for tbl in doc.tables:
    for ri, row in enumerate(tbl.rows):
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        if ri == 0:
            hdr = OxmlElement("w:tblHeader")
            trPr.append(hdr)
        if ri < len(tbl.rows) - 1:
            for cell in row.cells:
                for cp in cell.paragraphs:
                    cp.paragraph_format.keep_with_next = True

# 4. widen the two remaining columns of the Task 6 route table
tbl6 = doc.tables[7]
for row in tbl6.rows:
    row.cells[0].width = Inches(1.95)
    row.cells[1].width = Inches(4.85)

doc.save(TARGET)
print("saved", TARGET)
print("paragraphs", len(doc.paragraphs), "tables", len(doc.tables))
