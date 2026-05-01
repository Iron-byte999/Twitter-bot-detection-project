EXPECTED_COLUMNS = {
    "User ID",
    "Username",
    "Tweet",
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "Verified",
    "Bot Label",
    "Location",
    "Created At",
    "Hashtags",
}


def test_final_dataset_schema(final_df):
    assert set(final_df.columns) == EXPECTED_COLUMNS


def test_final_dataset_required_values_present(final_df):
    required_columns = EXPECTED_COLUMNS - {"Location"}
    assert not final_df[list(required_columns)].isna().any().any()


def test_final_dataset_ranges_and_labels(final_df):
    assert final_df["Bot Label"].isin([0, 1]).all()
    assert final_df["Retweet Count"].ge(0).all()
    assert final_df["Mention Count"].ge(0).all()
    assert final_df["Follower Count"].ge(0).all()
    assert final_df["Verified"].isin([True, False]).all()


def test_final_dataset_has_no_duplicate_user_ids(final_df):
    assert final_df["User ID"].is_unique
