import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import train_final


def make_small_training_csv(path):
    rows = []
    for index in range(24):
        label = index % 2
        rows.append(
            {
                "User ID": index,
                "Username": "account",
                "Tweet": "profile",
                "Retweet Count": 100 + label * 1000 + index,
                "Mention Count": 20 + label * 100 + index,
                "Follower Count": 50 + label * 500 + index,
                "Verified": False,
                "Bot Label": label,
                "Location": "",
                "Created At": "Wed Jun 05 00:00:00 +0000 2019",
                "Hashtags": "none",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def small_model_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", train_final.build_preprocessor(include_text=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=train_final.RANDOM_STATE,
                ),
            ),
        ]
    )


def test_training_produces_model_artifact(monkeypatch):
    test_root = Path(__file__).resolve().parents[1] / "tests" / "_tmp_model_artifact"
    if test_root.exists():
        shutil.rmtree(test_root)
    dataset_path = test_root / "data" / "tiny.csv"
    dataset_path.parent.mkdir(parents=True)

    try:
        make_small_training_csv(dataset_path)

        monkeypatch.setattr(train_final, "project_root", lambda: test_root)
        def test_reports_path(filename):
            path = test_root / "reports" / filename
            path.parent.mkdir(exist_ok=True)
            return path

        def test_models_path(filename):
            path = test_root / "models" / filename
            path.parent.mkdir(exist_ok=True)
            return path

        monkeypatch.setattr(train_final, "reports_path", test_reports_path)
        monkeypatch.setattr(train_final, "models_path", test_models_path)
        monkeypatch.setattr(train_final, "parse_args", lambda: argparse.Namespace(dataset=dataset_path))
        monkeypatch.setattr(
            train_final,
            "make_models",
            lambda: {
                "dummy_majority": Pipeline(
                    steps=[
                        ("preprocessor", train_final.build_preprocessor(include_text=False)),
                        ("classifier", train_final.DummyClassifier(strategy="most_frequent")),
                    ]
                ),
                "logistic_regression_test": small_model_pipeline(),
            },
        )

        train_final.main()

        model_path = test_root / "models" / "train_best_model.joblib"
        assert model_path.exists()
        assert hasattr(joblib.load(model_path), "predict")
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)


def test_prediction_output_shape_and_type(final_df):
    featured = train_final.add_features(final_df.head(80))
    y = featured[train_final.TARGET_COLUMN].astype(int)
    X = featured.drop(columns=[train_final.TARGET_COLUMN, "User ID"], errors="ignore")

    model = small_model_pipeline()
    model.fit(X, y)
    predictions = model.predict(X.head(10))

    assert len(predictions) == 10
    assert predictions.dtype.kind in {"i", "u", "b"}
    assert set(predictions).issubset({0, 1})
