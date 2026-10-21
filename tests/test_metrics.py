import numpy as np

from jev_eval.metrics import calibration, field_metrics


def _row(item, gold, pred, conf, repeat=0):
    return {
        "item_id": item,
        "repeat": repeat,
        "gold": {"intent": gold},
        "valid": pred is not None,
        "answers": {"intent": {"label": pred, "confidence": conf}},
    }


def test_perfect_calibration_has_zero_ece():
    conf = np.array([0.9] * 10)
    correct = np.array([1.0] * 9 + [0.0])
    assert calibration(conf, correct)["ece"] < 1e-9


def test_overconfidence_shows_in_ece():
    out = calibration(np.array([0.95] * 10), np.array([1.0] * 5 + [0.0] * 5))
    assert abs(out["ece"] - 0.45) < 1e-9
    assert out["bins"][0]["n"] == 10


def test_accuracy_counts_invalid_as_wrong():
    rows = [_row("a", "x", "x", 0.9), _row("b", "y", None, None), _row("c", "x", "y", 0.6), _row("d", "y", "y", 0.8)]
    m = field_metrics(rows, "intent", ["x", "y"], None)
    assert m["accuracy"] == 0.5
    assert m["confidence_coverage"] == 0.75
    lo, hi = m["accuracy_ci"]
    assert lo <= 0.5 <= hi


def test_consistency_uses_all_repeats():
    rows = [
        _row("a", "x", "x", 0.9, 0),
        _row("a", "x", "x", 0.9, 1),
        _row("b", "x", "x", 0.9, 0),
        _row("b", "x", "y", 0.9, 1),
    ]
    assert field_metrics(rows, "intent", ["x", "y"], None)["consistency"] == 0.5


def test_out_of_scope_precision_recall():
    rows = [_row("a", "oos", "oos", 0.9), _row("b", "x", "oos", 0.9), _row("c", "oos", "x", 0.9)]
    oos = field_metrics(rows, "intent", ["x", "oos"], "oos")["out_of_scope"]
    assert oos == {"precision": 0.5, "recall": 0.5}
