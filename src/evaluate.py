import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from train_final import TARGET_COLUMN, add_features
from utils import data_path, models_path, project_root, reports_path, write_json


DEFAULT_DATASET = data_path("botwiki_verified_2019_balanced.csv")
DEFAULT_MODEL = models_path("train_best_model.joblib")
DEFAULT_OUTPUT = reports_path("evaluation_metrics.json")


def compute_metrics(y_true, y_pred) -> dict[str, object]:
    """Compute core classification metrics in a serializable format."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "true_human_pred_human": int(matrix[0, 0]),
        "true_human_pred_bot": int(matrix[0, 1]),
        "true_bot_pred_human": int(matrix[1, 0]),
        "true_bot_pred_bot": int(matrix[1, 1]),
    }


def majority_baseline_predictions(y_true):
    """Return predictions from a most-frequent-class baseline."""
    baseline = DummyClassifier(strategy="most_frequent")
    empty_features = [[0] for _ in range(len(y_true))]
    baseline.fit(empty_features, y_true)
    return baseline.predict(empty_features)


def build_evaluation_payload(y_true, model_predictions) -> dict[str, object]:
    """Build model metrics plus a dummy-baseline comparison."""
    baseline_predictions = majority_baseline_predictions(y_true)
    model_metrics = compute_metrics(y_true, model_predictions)
    baseline_metrics = compute_metrics(y_true, baseline_predictions)

    return {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "comparison": {
            "balanced_accuracy_lift": (
                model_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
            ),
            "accuracy_lift": model_metrics["accuracy"] - baseline_metrics["accuracy"],
        },
    }


def display_path(path: Path) -> str:
    """Return a project-relative path for reports when possible."""
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for saved-model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved bot detection model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Evaluate the saved model against a CSV dataset and save JSON metrics."""
    args = parse_args()
    df = pd.read_csv(args.dataset)
    df = add_features(df)

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN, "User ID"], errors="ignore")
    model = joblib.load(args.model)
    predictions = model.predict(X)

    payload = build_evaluation_payload(y, predictions)
    payload["dataset"] = display_path(args.dataset)
    payload["model_path"] = display_path(args.model)
    payload["rows"] = len(df)

    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    print(f"Saved evaluation metrics to: {args.output}")


if __name__ == "__main__":
    main()
