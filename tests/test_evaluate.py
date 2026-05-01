import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluate import build_evaluation_payload, compute_metrics


def test_compute_metrics_correctness():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["true_human_pred_human"] == 1
    assert metrics["true_human_pred_bot"] == 1
    assert metrics["true_bot_pred_human"] == 0
    assert metrics["true_bot_pred_bot"] == 2


def test_baseline_comparison_exists():
    y_true = [0, 0, 1, 1]
    model_predictions = [0, 1, 1, 1]

    payload = build_evaluation_payload(y_true, model_predictions)

    assert "model" in payload
    assert "baseline" in payload
    assert "comparison" in payload
    assert "balanced_accuracy_lift" in payload["comparison"]
    assert payload["baseline"]["balanced_accuracy"] == pytest.approx(0.5)
    assert payload["comparison"]["balanced_accuracy_lift"] == pytest.approx(0.25)
