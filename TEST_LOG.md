# Test Log

Command used:

```powershell
.\.venv\Scripts\python.exe -m pytest twitter-bot-detection\tests -q -p no:cacheprovider --basetemp=twitter-bot-detection\.pytest_tmp
```

Overall result:

```text
12 passed in 1.64s
```

## Summary

| Category | Count | Coverage |
|---|---:|---|
| Data validation | 4 | Schema, missing values, ranges/labels, duplicates |
| Preprocessing | 4 | Neutralization, scaling consistency, train/test leakage, strict leakage pairs |
| Model | 2 | Model artifact creation, prediction output shape/type |
| Evaluation | 2 | Metric correctness, baseline comparison |

## Data Validation

### DV-001: Dataset Schema

- **Input/condition:** Load `data/botwiki_verified_2019_balanced.csv` and inspect its columns.
- **Expected outcome:** Columns exactly match the required schema: `User ID`, `Username`, `Tweet`, `Retweet Count`, `Mention Count`, `Follower Count`, `Verified`, `Bot Label`, `Location`, `Created At`, `Hashtags`.
- **Actual outcome/evidence:** `test_final_dataset_schema` passed.
- **Pass/Fail:** Pass

### DV-002: Missing Required Values

- **Input/condition:** Check required fields in the final dataset for missing values. `Location` is allowed to be blank because preprocessing intentionally neutralizes it.
- **Expected outcome:** Required columns contain no missing values.
- **Actual outcome/evidence:** `test_final_dataset_required_values_present` passed.
- **Pass/Fail:** Pass

### DV-003: Ranges And Labels

- **Input/condition:** Check label, count, and boolean value ranges in the final dataset.
- **Expected outcome:** `Bot Label` only contains `0` and `1`; count columns are non-negative; `Verified` is boolean.
- **Actual outcome/evidence:** `test_final_dataset_ranges_and_labels` passed.
- **Pass/Fail:** Pass

### DV-004: Duplicate User IDs

- **Input/condition:** Check `User ID` values in the final dataset.
- **Expected outcome:** No duplicate `User ID` values.
- **Actual outcome/evidence:** `test_final_dataset_has_no_duplicate_user_ids` passed.
- **Pass/Fail:** Pass

## Preprocessing

### PP-001: Source Field Neutralization

- **Input/condition:** Inspect final preprocessed fields that could reveal source bias.
- **Expected outcome:** `Username`, `Tweet`, `Hashtags`, `Location`, `Created At`, and `Verified` are neutralized to constant values.
- **Actual outcome/evidence:** `test_final_preprocessing_neutralizes_obvious_source_columns` passed.
- **Pass/Fail:** Pass

### PP-002: Scaling Consistency

- **Input/condition:** Run numeric preprocessing twice on the same sample with `build_preprocessor(include_text=False)`.
- **Expected outcome:** Scaled/transformed numeric output is deterministic and equal across repeated runs.
- **Actual outcome/evidence:** `test_numeric_scaling_is_deterministic` passed.
- **Pass/Fail:** Pass

### PP-003: Train/Test Leakage Check

- **Input/condition:** Recreate the train/test split using the project `RANDOM_STATE` and final labels.
- **Expected outcome:** No `User ID` appears in both train and test sets.
- **Actual outcome/evidence:** `test_train_test_split_has_no_user_id_leakage` passed.
- **Pass/Fail:** Pass

### PP-004: Strict Leakage Pair Check

- **Input/condition:** Inspect strict leakage-check dataset `data/botwiki_verified_2019_strict.csv`.
- **Expected outcome:** Each pair has identical model inputs but opposite labels, proving it can test label leakage.
- **Actual outcome/evidence:** `test_strict_leakage_dataset_pairs_identical_inputs` passed.
- **Pass/Fail:** Pass

## Model

### MD-001: Model Artifact Creation

- **Input/condition:** Run a small controlled training job through `train_final.main()` using a temporary tiny dataset and lightweight model list.
- **Expected outcome:** Training creates a saved model artifact at `models/train_best_model.joblib`, and the loaded artifact has a `predict` method.
- **Actual outcome/evidence:** `test_training_produces_model_artifact` passed.
- **Pass/Fail:** Pass

### MD-002: Prediction Output Shape And Type

- **Input/condition:** Fit a small pipeline and predict on 10 rows from the final dataset.
- **Expected outcome:** Prediction output length is 10, output type is numeric/binary, and all predictions are in `{0, 1}`.
- **Actual outcome/evidence:** `test_prediction_output_shape_and_type` passed.
- **Pass/Fail:** Pass

## Evaluation

### EV-001: Metric Computation Correctness

- **Input/condition:** Call `compute_metrics()` with known labels `y_true = [0, 0, 1, 1]` and predictions `y_pred = [0, 1, 1, 1]`.
- **Expected outcome:** Accuracy is `0.75`, balanced accuracy is `0.75`, and confusion matrix cells are `[1, 1, 0, 2]`.
- **Actual outcome/evidence:** `test_compute_metrics_correctness` passed.
- **Pass/Fail:** Pass

### EV-002: Baseline Comparison Exists

- **Input/condition:** Call `build_evaluation_payload()` with known labels and predictions.
- **Expected outcome:** Output includes `model`, `baseline`, and `comparison`; baseline balanced accuracy is `0.5`; balanced accuracy lift is `0.25`.
- **Actual outcome/evidence:** `test_baseline_comparison_exists` passed.
- **Pass/Fail:** Pass

