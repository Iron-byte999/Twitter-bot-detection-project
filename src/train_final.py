import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import data_path, models_path, project_root, reports_path, write_json


RANDOM_STATE = 42
TARGET_COLUMN = "Bot Label"
NUMERIC_FEATURES = [
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "Verified",
    "retweet_count_log",
    "mention_count_log",
    "follower_count_log",
    "follower_to_mentions",
    "retweets_to_mentions",
    "total_engagement",
    "engagement_per_follower",
    "tweet_length",
    "tweet_word_count",
    "avg_word_length",
    "unique_word_ratio",
    "hashtag_count",
    "hashtag_symbol_count",
    "mention_symbol_count",
    "url_count",
    "has_url",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "digit_ratio",
    "username_length",
    "username_digit_count",
    "username_has_digit",
    "username_underscore_count",
    "username_alpha_ratio",
    "location_length",
    "has_location",
    "days_since_first_post",
    "created_year",
    "created_month",
    "created_hour",
    "created_dayofweek",
]

def find_dataset(root: Path) -> Path:
    """Find the default prepared training CSV."""
    candidates = [
        data_path("botwiki_verified_2019_balanced.csv"),
        data_path("botwiki_verified_2019.csv"),
        Path.cwd() / "data" / "botwiki_verified_2019_balanced.csv",
        Path.cwd() / "data" / "botwiki_verified_2019.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a training CSV. Run src/preprocess_final.py to create "
        "data/botwiki_verified_2019_balanced.csv."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the training script."""
    parser = argparse.ArgumentParser(description="Train bot detection models.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to data/botwiki_verified_2019_balanced.csv.",
    )
    return parser.parse_args()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric, text, profile, and date-derived model features."""
    df = df.copy()

    df["Tweet"] = df["Tweet"].fillna("")
    df["Hashtags"] = df["Hashtags"].fillna("")
    df["Username"] = df["Username"].fillna("")
    df["Location"] = df["Location"].fillna("")

    df["Verified"] = df["Verified"].astype(int)
    df["retweet_count_log"] = np.log1p(df["Retweet Count"])
    df["mention_count_log"] = np.log1p(df["Mention Count"])
    df["follower_count_log"] = np.log1p(df["Follower Count"])
    df["follower_to_mentions"] = df["Follower Count"] / (df["Mention Count"] + 1)
    df["retweets_to_mentions"] = df["Retweet Count"] / (df["Mention Count"] + 1)
    df["total_engagement"] = df["Retweet Count"] + df["Mention Count"]
    df["engagement_per_follower"] = df["total_engagement"] / (df["Follower Count"] + 1)

    df["tweet_length"] = df["Tweet"].str.len()
    df["tweet_word_count"] = df["Tweet"].str.split().str.len()
    df["avg_word_length"] = df["tweet_length"] / df["tweet_word_count"].replace(0, np.nan)
    df["unique_word_ratio"] = df["Tweet"].apply(unique_word_ratio)
    df["hashtag_count"] = df["Hashtags"].str.split().str.len()
    df["hashtag_symbol_count"] = df["Tweet"].str.count("#")
    df["mention_symbol_count"] = df["Tweet"].str.count("@")
    df["url_count"] = df["Tweet"].str.count(r"https?://|www\.")
    df["has_url"] = (df["url_count"] > 0).astype(int)
    df["exclamation_count"] = df["Tweet"].str.count("!")
    df["question_count"] = df["Tweet"].str.count(r"\?")
    df["uppercase_ratio"] = df["Tweet"].apply(uppercase_ratio)
    df["digit_ratio"] = df["Tweet"].apply(digit_ratio)

    df["username_length"] = df["Username"].str.len()
    df["username_digit_count"] = df["Username"].str.count(r"\d")
    df["username_has_digit"] = (df["username_digit_count"] > 0).astype(int)
    df["username_underscore_count"] = df["Username"].str.count("_")
    df["username_alpha_ratio"] = df["Username"].apply(alpha_ratio)
    df["location_length"] = df["Location"].str.len()
    df["has_location"] = (df["location_length"] > 0).astype(int)

    created_at = pd.to_datetime(df["Created At"], format="mixed", errors="coerce", utc=True)
    newest_date = created_at.max()
    df["days_since_first_post"] = (newest_date - created_at).dt.days
    df["created_year"] = created_at.dt.year
    df["created_month"] = created_at.dt.month
    df["created_hour"] = created_at.dt.hour
    df["created_dayofweek"] = created_at.dt.dayofweek

    return df


def unique_word_ratio(text: str) -> float:
    """Return the fraction of unique words in a text value."""
    words = str(text).lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def uppercase_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters that are uppercase."""
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    uppercase_letters = [char for char in letters if char.isupper()]
    return len(uppercase_letters) / len(letters)


def digit_ratio(text: str) -> float:
    """Return the fraction of characters that are digits."""
    text = str(text)
    if not text:
        return 0.0
    digits = [char for char in text if char.isdigit()]
    return len(digits) / len(text)


def alpha_ratio(text: str) -> float:
    """Return the fraction of characters that are alphabetic."""
    text = str(text)
    if not text:
        return 0.0
    letters = [char for char in text if char.isalpha()]
    return len(letters) / len(text)


def build_preprocessor(include_text: bool = True) -> ColumnTransformer:
    """Build the sklearn column transformer for numeric and optional text features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_pipeline, NUMERIC_FEATURES)]

    if include_text:
        transformers.extend(
            [
                (
                    "tweet_text",
                    TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
                    "Tweet",
                ),
                (
                    "hashtags",
                    TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2),
                    "Hashtags",
                ),
            ]
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    """Create the candidate model pipelines evaluated by training."""
    return {
        "dummy_majority": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LinearSVC(
                        C=0.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting_numeric": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(include_text=False)),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "rbf_svm_numeric": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(include_text=False)),
                (
                    "rbf_features",
                    Nystroem(
                        kernel="rbf",
                        gamma=0.1,
                        n_components=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=0.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "mlp_neural_network_numeric": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(include_text=False)),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.001,
                        batch_size=256,
                        early_stopping=True,
                        max_iter=60,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def format_results(name: str, y_true: pd.Series, y_pred: np.ndarray) -> str:
    """Format human-readable metrics for one trained model."""
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    return (
        f"\n=== {name} ===\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Balanced accuracy: {balanced_accuracy:.4f}\n"
        f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}\n\n"
        f"{classification_report(y_true, y_pred, target_names=['human', 'bot'], zero_division=0)}"
    )


def metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    """Return machine-readable metrics for one trained model."""
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "true_human_pred_human": int(matrix[0, 0]),
        "true_human_pred_bot": int(matrix[0, 1]),
        "true_bot_pred_human": int(matrix[1, 0]),
        "true_bot_pred_bot": int(matrix[1, 1]),
    }


def main() -> None:
    """Train all candidate models, save reports, metrics, and the best model."""
    args = parse_args()
    root = project_root()

    dataset_path = args.dataset if args.dataset is not None else find_dataset(root)
    df = pd.read_csv(dataset_path)
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
    metric_rows = []
    scores = {}
    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in make_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = balanced_accuracy_score(y_test, predictions)
        scores[name] = score
        results.append(format_results(name, y_test, predictions))
        metric_rows.append(metric_row(name, y_test, predictions))

        if name != "dummy_majority" and score > best_score:
            best_name = name
            best_model = model
            best_score = score

    dummy_score = scores["dummy_majority"]
    signal_warning = ""
    if best_score < dummy_score + 0.03:
        signal_warning = (
            "\n\nWARNING: The best real model is less than 3 percentage points "
            "above the dummy baseline. This usually means the dataset features "
            "do not contain enough useful signal, or the labels were generated "
            "almost randomly. A different dataset or more bot-specific features "
            "will be needed for high accuracy."
        )

    report = (
        f"Dataset: {dataset_path}\n"
        f"Rows: {len(df)}\n"
        f"Class balance:\n{y.value_counts().sort_index().to_string()}\n"
        + "\n".join(results)
        + f"\n\nBest model: {best_name} "
        f"(balanced accuracy={best_score:.4f})"
        + signal_warning
        + "\n"
    )

    report_path = reports_path("train_results.txt")
    report_path.write_text(report, encoding="utf-8")

    metrics_csv_path = reports_path("train_metrics.csv")
    metrics_json_path = reports_path("train_metrics.json")
    pd.DataFrame(metric_rows).to_csv(metrics_csv_path, index=False)
    metrics_payload = {
        "dataset": str(dataset_path),
        "rows": len(df),
        "class_balance": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "best_model": best_name,
        "best_balanced_accuracy": best_score,
        "metrics": metric_rows,
    }
    write_json(metrics_json_path, metrics_payload)

    if best_model is not None:
        model_path = models_path("train_best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"\nSaved best model to: {model_path}")

    print(report)
    print(f"Saved report to: {report_path}")
    print(f"Saved metrics CSV to: {metrics_csv_path}")
    print(f"Saved metrics JSON to: {metrics_json_path}")


if __name__ == "__main__":
    main()
