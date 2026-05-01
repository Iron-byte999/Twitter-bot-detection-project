import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import data_path, project_root, reports_path


RANDOM_STATE = 42
TARGET_COLUMN = "Bot Label"
NUMERIC_FEATURES = [
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "follower_ratio",
    "retweet_per_mention",
]

def default_dataset() -> Path:
    """Return the final bias-reduced dataset used by the project."""
    return data_path("botwiki_verified_2019_balanced.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for baseline training."""
    parser = argparse.ArgumentParser(description="Train simple baseline bot detectors.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset(),
        help="CSV path. Defaults to data/botwiki_verified_2019_balanced.csv.",
    )
    return parser.parse_args()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small baseline feature set."""
    df = df.copy()
    df["Tweet"] = df["Tweet"].fillna("")
    df["follower_ratio"] = df["Follower Count"] / (df["Mention Count"] + 1)
    df["retweet_per_mention"] = df["Retweet Count"] / (df["Mention Count"] + 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a simple numeric-plus-text preprocessor."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("text", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "Tweet"),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    """Create intentionally simple baseline models."""
    return {
        "dummy_majority": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression_baseline": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def format_results(name: str, y_true: pd.Series, y_pred) -> str:
    """Format baseline metrics for one model."""
    return (
        f"\n=== {name} ===\n"
        f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
        f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}\n"
        f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}\n\n"
        f"{classification_report(y_true, y_pred, target_names=['human', 'bot'], zero_division=0)}"
    )


def main() -> None:
    """Train simple baseline models and save a text report."""
    args = parse_args()

    df = pd.read_csv(args.dataset)
    df = add_features(df)

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN, "User ID"], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = []
    for name, model in make_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results.append(format_results(name, y_test, model.predict(X_test)))

    report = (
        f"Dataset: {args.dataset}\n"
        f"Rows: {len(df)}\n"
        f"Class balance:\n{y.value_counts().sort_index().to_string()}\n"
        + "\n".join(results)
        + "\n"
    )
    report_path = reports_path("train_baseline_results.txt")
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
