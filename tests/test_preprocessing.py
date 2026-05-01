import numpy as np
from sklearn.model_selection import train_test_split

from train_final import RANDOM_STATE, TARGET_COLUMN, add_features, build_preprocessor


def test_final_preprocessing_neutralizes_obvious_source_columns(final_df):
    assert set(final_df["Username"].unique()) == {"account"}
    assert set(final_df["Tweet"].unique()) == {"profile"}
    assert set(final_df["Hashtags"].unique()) == {"none"}
    assert set(final_df["Location"].fillna("").unique()) == {""}
    assert set(final_df["Created At"].unique()) == {"Wed Jun 05 00:00:00 +0000 2019"}
    assert set(final_df["Verified"].unique()) == {False}


def test_numeric_scaling_is_deterministic(final_df):
    featured = add_features(final_df.head(80))
    preprocessor_one = build_preprocessor(include_text=False)
    preprocessor_two = build_preprocessor(include_text=False)

    transformed_one = preprocessor_one.fit_transform(featured)
    transformed_two = preprocessor_two.fit_transform(featured)

    assert np.allclose(transformed_one, transformed_two, equal_nan=True)


def test_train_test_split_has_no_user_id_leakage(final_df):
    train_df, test_df = train_test_split(
        final_df,
        test_size=0.2,
        stratify=final_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    train_ids = set(train_df["User ID"])
    test_ids = set(test_df["User ID"])
    assert train_ids.isdisjoint(test_ids)


def test_strict_leakage_dataset_pairs_identical_inputs(strict_df):
    feature_columns = [column for column in strict_df.columns if column not in {"User ID", "Bot Label"}]
    pair_sizes = strict_df.groupby(feature_columns, dropna=False)["Bot Label"].agg(["size", "nunique"])

    assert pair_sizes["size"].eq(2).all()
    assert pair_sizes["nunique"].eq(2).all()
