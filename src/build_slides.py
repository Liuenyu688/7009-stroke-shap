"""Task 5 - three slides for the data scientist, the doctor and the patient."""
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from _paths import OUT, FIG, REPORT

TARGET = REPORT / "task5_slides.pptx"

r2 = json.load(open(os.path.join(OUT, "task2_results.json"), encoding="utf-8"))
r6 = json.load(open(os.path.join(OUT, "task6_results.json"), encoding="utf-8"))
c2, m2 = r2["case"], r2["metrics"]

INK = RGBColor(0x1F, 0x2A, 0x37)
MUTED = RGBColor(0x5A, 0x6B, 0x7B)
ACCENT = RGBColor(0x1F, 0x6F, 0x8B)
WARN = RGBColor(0xB0, 0x3A, 0x2E)
POS = RGBColor(0xC0, 0x39, 0x2B)
NEG = RGBColor(0x2A, 0x6F, 0xB5)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def box(slide, x, y, w, h, fill=None, line=None):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill if fill else WHITE
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, space=6, line_spacing=1.05):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(space)
        p.line_spacing = line_spacing
        if isinstance(item, str):
            item = [(item, {})]
        for chunk, fmt in item:
            r = p.add_run()
            r.text = chunk
            f = r.font
            f.size = Pt(fmt.get("size", 16))
            f.bold = fmt.get("bold", False)
            f.italic = fmt.get("italic", False)
            f.color.rgb = fmt.get("color", INK)
            f.name = "Calibri"
    return tb


def header(slide, title, audience, k=1):
    box(slide, 0, 0, 13.333, 1.32, fill=LIGHT)
    box(slide, 0, 1.32, 13.333, 0.06, fill=ACCENT)
    text(slide, 0.62, 0.24, 10.4, 0.6, [[(title, {"size": 27, "bold": True, "color": INK})]])
    text(slide, 0.62, 0.86, 10.4, 0.35, [[(audience, {"size": 13, "color": MUTED})]])
    text(slide, 11.4, 0.3, 1.4, 0.4, [[(f"{k}/3", {"size": 13, "color": MUTED})]], align=PP_ALIGN.RIGHT)


def footer(slide, src):
    text(slide, 0.62, 7.02, 12.1, 0.32, [[(src, {"size": 10, "color": MUTED})]])


# ----------------------------------------------------------------- slide 1
s1 = prs.slides.add_slide(BLANK)
header(s1, "Model performance and explanation of stroke classification",
       "Audience 1 of 3 - data scientist: how well does it rank, and what is it using?", 1)

box(s1, 0.62, 1.72, 5.55, 3.05, fill=WHITE, line=RGBColor(0xD8, 0xDF, 0xE6))
text(s1, 0.85, 1.88, 5.1, 2.8, [
    [("Model  ", {"size": 14, "bold": True, "color": ACCENT}),
     ("Random Forest, 500 trees, 10 predictors, 22 encoded columns", {"size": 14})],
    [("Split  ", {"size": 14, "bold": True, "color": ACCENT}),
     ("3,066 / 1,022 / 1,022 stratified; threshold 0.14 chosen on validation", {"size": 14})],
    [("Explainer  ", {"size": 14, "bold": True, "color": ACCENT}),
     ("exact TreeExplainer, probability of stroke = 1, base value 0.048, residual 2e-15", {"size": 14})],
    [("Top features  ", {"size": 14, "bold": True, "color": ACCENT}),
     ("age 0.046, glucose 0.012, BMI 0.008, hypertension 0.0055, heart disease 0.0045", {"size": 14})],
])

metrics = [("ROC-AUC", "0.856"), ("Avg precision", "0.234"), ("Recall", "0.560"), ("Precision", "0.192")]
for i, (label, value) in enumerate(metrics):
    x = 0.62 + i * 1.42
    box(s1, x, 5.02, 1.28, 1.05, fill=LIGHT)
    text(s1, x, 5.12, 1.28, 0.4, [[(value, {"size": 22, "bold": True, "color": ACCENT})]], align=PP_ALIGN.CENTER)
    text(s1, x, 5.55, 1.28, 0.4, [[(label, {"size": 11, "color": MUTED})]], align=PP_ALIGN.CENTER)

text(s1, 0.62, 6.22, 5.55, 0.7, [
    [("Accuracy is 0.863 - below the 0.951 majority baseline. Quote AUC, AP and recall, never accuracy alone.",
      {"size": 13, "italic": True, "color": WARN})],
])

s1.shapes.add_picture(os.path.join(FIG, "fig3_summary_beeswarm.png"), Inches(6.80), Inches(1.50), width=Inches(5.70))
text(s1, 6.80, 5.55, 5.75, 0.9, [
    [("Global SHAP summary, 1,022 held-out records. Age dominates; colour is the feature value, "
      "position is the contribution to the probability.", {"size": 12, "color": MUTED})],
])
footer(s1, "Source: task2_results.json, fig3_summary_beeswarm.png | Random Forest + TreeExplainer, shap 0.54.1")

# ----------------------------------------------------------------- slide 2
s2 = prs.slides.add_slide(BLANK)
header(s2, "Interpreting one patient prediction in clinical context",
       "Audience 2 of 3 - doctor: what drove this score, and what it does not tell you", 2)

text(s2, 0.62, 1.62, 6.0, 1.5, [
    [("68-year-old man, recorded heart disease, average glucose 224 mg/dL, BMI 31.9, former smoker, "
      "no recorded hypertension.", {"size": 16})],
    [("Model score 0.208", {"size": 22, "bold": True, "color": POS}),
     ("   review threshold 0.14 - correct detection (true positive)", {"size": 15, "color": MUTED})],
])

text(s2, 0.62, 3.02, 6.0, 2.4, [
    [("Pushes the score up", {"size": 15, "bold": True, "color": POS})],
    [("Age 68   +0.058", {"size": 15})],
    [("Recorded heart disease   +0.055", {"size": 15})],
    [("Average glucose 224 mg/dL   +0.047", {"size": 15})],
    [("Pulls it down", {"size": 15, "bold": True, "color": NEG})],
    [("Former smoker -0.014  |  no recorded hypertension -0.003", {"size": 15})],
])

text(s2, 0.62, 5.42, 6.0, 1.4, [
    [("At this threshold the model detects 28 of 50 recorded strokes and raises 118 false alarms: "
      "a high score means 'review', not 'will have a stroke'.", {"size": 13, "color": MUTED})],
    [("BMI missing in 3.9% of records, smoking status 'Unknown' in 30% - check the record before acting.",
      {"size": 13, "color": MUTED})],
])

s2.shapes.add_picture(os.path.join(FIG, "fig2_force_plot.png"), Inches(6.75), Inches(1.72), width=Inches(6.05))
text(s2, 6.75, 4.42, 6.05, 1.5, [
    [("Each bar is one item from the record. Red pushes the score up, blue pulls it down; the width "
      "is the size of the effect.", {"size": 12, "color": MUTED})],
    [("The attribution explains the model's behaviour and still needs clinical interpretation. The data "
      "are retrospective, with no follow-up, so this is not a future-risk estimate.",
      {"size": 12, "color": WARN})],
])
footer(s2, "Source: fig2_force_plot.png, task2_results.json | case #0335, held-out test set, probability scale")

# ----------------------------------------------------------------- slide 3
s3 = prs.slides.add_slide(BLANK)
header(s3, "What the computer prediction means for this example",
       "Audience 3 of 3 - patient: the same numbers, in plain words", 3)

box(s3, 0.62, 1.72, 4.1, 2.35, fill=LIGHT)
text(s3, 0.85, 1.9, 3.7, 2.0, [
    [("Your score", {"size": 15, "color": MUTED})],
    [("about 21 in 100", {"size": 30, "bold": True, "color": POS})],
    [("That is the computer's summary of the information in your record.", {"size": 12, "color": MUTED})],
])

text(s3, 5.05, 1.72, 3.75, 3.1, [
    [("What raised your score", {"size": 17, "bold": True, "color": POS})],
    [("Your age", {"size": 16})],
    [("Heart disease already noted in your file", {"size": 16})],
    [("A blood sugar reading of about 224", {"size": 16})],
])
text(s3, 9.0, 1.72, 3.75, 3.1, [
    [("What lowered it", {"size": 17, "bold": True, "color": NEG})],
    [("You used to smoke, but do not smoke now", {"size": 16})],
    [("High blood pressure is not recorded for you", {"size": 16})],
])

# score track
TRACK_X, TRACK_W, TRACK_Y = 0.62, 12.09, 4.32
box(s3, TRACK_X, TRACK_Y, TRACK_W, 0.30, fill=RGBColor(0xE3, 0xE8, 0xEE))
frac = 0.208
box(s3, TRACK_X, TRACK_Y, TRACK_W * frac, 0.30, fill=POS)
marker_x = TRACK_X + TRACK_W * frac
box(s3, marker_x - 0.03, TRACK_Y - 0.12, 0.06, 0.54, fill=INK)
text(s3, TRACK_X, TRACK_Y + 0.34, 2.0, 0.3, [[("lower score", {"size": 11, "color": MUTED})]])
text(s3, TRACK_X + TRACK_W - 2.0, TRACK_Y + 0.34, 2.0, 0.3,
     [[("higher score", {"size": 11, "color": MUTED})]], align=PP_ALIGN.RIGHT)
text(s3, TRACK_X + TRACK_W * 0.14 - 0.05, TRACK_Y + 0.58, 6.4, 0.3,
     [[("^ the line where the model starts flagging a record for review (14 in 100)",
        {"size": 11, "color": MUTED})]])

box(s3, 0.62, 5.25, 12.1, 1.62, fill=WHITE, line=RGBColor(0xD8, 0xDF, 0xE6))
text(s3, 0.9, 5.38, 11.5, 1.45, [
    [("This score can be wrong in either direction - about one in five of the records it flags turn out "
      "not to be strokes, and some strokes score below the line.", {"size": 14})],
    [("It is not a diagnosis, it is not a percentage chance of a future stroke, and it is not advice "
      "about medicines or lifestyle. Your clinician will read it together with the rest of your record.",
      {"size": 14, "color": MUTED})],
])
footer(s3, "Source: case #0335, Random Forest, probability 0.208 | wording checked against the report, Section 6")

prs.save(TARGET)
print("saved", TARGET, "slides:", len(prs.slides))
