"""Model-performance scoring (inference.metrics), against hand-labelled expected values."""

from __future__ import annotations

import pytest

from inference.metrics import (
    EvalRecord,
    classify_alert,
    classify_outcome,
    format_summary,
    summarize,
    wilson,
)

VT, VF, AF, NSR = "VENTRICULAR_TACHYCARDIA", "VENTRICULAR_FIBRILLATION", "ATRIAL_FIBRILLATION", "NORMAL_SINUS"


@pytest.mark.parametrize("pred,truth,kind,match,label", [
    (VT, VT, "TP", True, "TP ✓"),
    (VF, VT, "TP", False, "TP ✗class"),   # alarm right, class wrong
    (VT, NSR, "FP", False, "FP"),
    (NSR, VT, "FN", False, "FN"),
    (NSR, NSR, "TN", True, "TN"),
])
def test_outcome_per_event(pred, truth, kind, match, label):
    o = classify_outcome(pred, truth)
    assert (o.kind, o.class_match, o.label) == (kind, match, label)
    assert o.code == ("TP_WRONG_CLASS" if label == "TP ✗class" else kind)


@pytest.mark.parametrize("truth,model_classes,why", [
    (None, None, "no ground truth"),
    ("OTHER", None, "not a model class"),
    ("ST_ELEVATION", [VT, NSR], "outside this model's 2 classes"),
])
def test_unscorable(truth, model_classes, why):
    o = classify_outcome(VT, truth, model_classes)
    assert o.kind == "UNSCORABLE" and why in o.reason and o.label == "—"


@pytest.mark.parametrize("dispatched,truth,expect", [
    (True, VT, "alert_correct"), (False, VT, "alert_missed"),
    (True, NSR, "alert_false"), (False, NSR, "silent_correct"), (True, "OTHER", None),
])
def test_alert_outcome(dispatched, truth, expect):
    assert classify_alert(dispatched, truth) == expect


def test_wilson_known_value():
    # 9/12: p=0.75, Wilson 95% CI ≈ [0.468, 0.911]
    w = wilson(9, 12)
    assert w["value"] == 0.75 and w["k"] == 9 and w["n"] == 12
    assert (w["low"], w["high"]) == pytest.approx((0.468, 0.911), abs=0.002)
    assert wilson(0, 0) is None
    assert wilson(5, 5)["high"] == 1.0  # never above 1


_RECORDS = [
    EvalRecord(VT, VT, 0.9, dispatched=True),     # TP ✓, alert correct
    EvalRecord(VF, VT, 0.5, dispatched=True),     # TP ✗class, alert correct
    EvalRecord(AF, NSR, 0.6, dispatched=True),    # FP, false alert
    EvalRecord(NSR, AF, 0.7, dispatched=True),    # FN at model level, but alerted (vitals): correct
    EvalRecord(NSR, NSR, 0.8, dispatched=False),  # TN, silent correct
    EvalRecord(NSR, VT, 0.55, dispatched=False),  # FN, missed alert
    EvalRecord(VT, "OTHER", 0.4, dispatched=True),  # unscorable
    EvalRecord(AF, None, 0.9),                    # unscorable (no ground truth)
]


def test_summary_counts_and_rates():
    s = summarize(_RECORDS)
    assert (s["events"], s["scored"], s["unscorable"]) == (8, 6, 2)
    assert s["counts"] == {"TP": 2, "FP": 1, "FN": 2, "TN": 1, "TP_wrong_class": 1}
    # sensitivity TP/(TP+FN)=2/4, specificity TN/(TN+FP)=1/2, PPV 2/3, NPV 1/3
    assert (s["sensitivity"]["k"], s["sensitivity"]["n"]) == (2, 4)
    assert (s["specificity"]["k"], s["specificity"]["n"]) == (1, 2)
    assert (s["ppv"]["k"], s["ppv"]["n"]) == (2, 3)
    assert (s["npv"]["k"], s["npv"]["n"]) == (1, 3)
    # exact class right: VT→VT and NSR→NSR = 2 of 6; binary (TP+TN) = 3 of 6
    assert s["accuracy_exact"]["k"] == 2 and s["accuracy_binary"]["k"] == 3
    assert s["unscorable_reasons"] == {"label OTHER is not a model class": 1, "no ground truth": 1}


def test_summary_per_class_and_confusion():
    s = summarize(_RECORDS)
    assert s["confusion"] == {AF: {NSR: 1}, NSR: {AF: 1, NSR: 1}, VT: {VT: 1, VF: 1, NSR: 1}}
    vt = s["per_class"][VT]
    assert (vt["support"], vt["tp"], vt["fp"], vt["fn"]) == (3, 1, 0, 2)
    assert vt["precision"]["value"] == 1.0 and vt["recall"]["k"] == 1
    assert s["per_class"][VF]["support"] == 0  # predicted, never true: excluded from macro-F1


def test_summary_alert_level():
    a = summarize(_RECORDS)["alert"]
    assert a["counts"] == {"alert_correct": 3, "alert_missed": 1, "alert_false": 1,
                           "silent_correct": 1}
    assert (a["sensitivity"]["k"], a["sensitivity"]["n"]) == (3, 4)
    assert (a["false_alert_rate"]["k"], a["false_alert_rate"]["n"]) == (1, 2)


def test_model_classes_make_out_of_head_labels_unscorable():
    s = summarize([EvalRecord(VT, "ST_ELEVATION"), EvalRecord(VT, VT)], model_classes=[VT, NSR])
    assert s["scored"] == 1 and s["unscorable"] == 1


def test_mean_confidence_split_by_correctness():
    s = summarize(_RECORDS)
    # exact-correct: 0.9 (VT) + 0.8 (NSR); wrong: 0.5, 0.6, 0.7, 0.55
    assert s["mean_confidence"] == {"correct": 0.85, "wrong": 0.588}


def test_no_alert_block_without_dispatch_info():
    assert "alert" not in summarize([EvalRecord(VT, VT)])


def test_empty_and_all_unscorable():
    s = summarize([EvalRecord(VT, None)])
    assert s["scored"] == 0 and s["sensitivity"] is None and s["macro_f1"] is None
    assert "0 scored of 1 events" in format_summary(s)


def test_format_summary_is_readable():
    text = format_summary(summarize(_RECORDS))
    assert "TP 2 (wrong class 1) · FP 1 · FN 2 · TN 1" in text
    assert "sensitivity  50% (2/4, 95% CI" in text
    assert "alerts       correct 3 · missed 1 · false 1" in text
    assert "2 unscorable" in text
