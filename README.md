# Twitter Bot Detection

## Project Goal

This project builds and evaluates machine-learning models that classify Twitter/X accounts as `human` or `bot`.

The current dataset combines two 2019 account collections:

- `verified-2019.tar.gz`: verified accounts labeled as humans
- `botwiki-2019.tar.gz`: Botwiki accounts labeled as bots

Because these two classes come from different source populations, the project includes multiple preprocessing scripts. The final preprocessing code used for the project is `preprocess_final.py`. The original baseline is kept as `preprocess_baseline.py`, and the leakage check is kept as `preprocess_leakage_check.py`.Results are based on performance on Botwiki-vs-verified dataset.

## Dependencies

This project uses Python with pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn, and pytest.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the virtual environment already exists, only run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data Files

Raw archives are expected under:

```text
twitter-bot-detection/data/verified-2019.tar.gz
twitter-bot-detection/data/botwiki-2019.tar.gz
```

Prepared CSV outputs are written to:

```text
twitter-bot-detection/data/botwiki_verified_2019.csv
twitter-bot-detection/data/botwiki_verified_2019_balanced.csv
twitter-bot-detection/data/botwiki_verified_2019_strict.csv
```

## Code Organization

The code is split into modules under `twitter-bot-detection/src/`:

- `preprocess_baseline.py`: original preprocessing baseline. It converts the raw archives into one CSV, but it has a bias problem because the human and bot examples come from different source populations.
- `preprocess_final.py`: final preprocessing code used for the project. It creates a bias-reduced balanced dataset by neutralizing obvious source clues and matching humans to bots on account statistics.
- `preprocess_leakage_check.py`: leakage-check experiment only. It creates paired bot/human rows with identical model inputs to verify that the training pipeline is not accidentally leaking labels.
- `evaluate.py`: evaluates the saved final model and compares it with a majority-class baseline.
- `train_final.py`: final training script. It builds features, trains multiple sklearn models, saves reports, saves metrics, and writes the best model.
- `train_baseline.py`: simple baseline training script using the final bias-reduced dataset.
- `utils.py`: shared path and JSON helpers used by preprocessing, training, and evaluation scripts.

Main functions in the active scripts have docstrings, and editable configuration values are kept as constants near the top of each file. The training dataset can also be changed with `--dataset`.

## Preprocessing

Original biased baseline preprocessing:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\preprocess_baseline.py
```

Final bias-reduced preprocessing used for the project:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\preprocess_final.py
```

Leakage-check preprocessing only:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\preprocess_leakage_check.py
```

## Training and Evaluation

Train the simple baseline on the final bias-reduced dataset:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_baseline.py
```

Train on the final bias-reduced dataset:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py
```

Train on the original biased baseline dataset:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py --dataset twitter-bot-detection\data\botwiki_verified_2019.csv
```

Train on the strict leakage-check dataset:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py --dataset twitter-bot-detection\data\botwiki_verified_2019_strict.csv
```

Evaluate the saved final model:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\evaluate.py
```

`train_final.py` evaluates these models:

- dummy majority baseline
- logistic regression
- linear SVM
- random forest
- gradient boosting using numeric features
- RBF-kernel approximation SVM using numeric features
- MLP neural network using numeric features

Evaluation is done with a stratified 80/20 train/test split using `RANDOM_STATE = 42`.

## Reproducing the Report

To reproduce the final project report from the raw archives:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\preprocess_final.py
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py --dataset twitter-bot-detection\data\botwiki_verified_2019_balanced.csv
```

The latest final bias-reduced run produced:

```text
Dataset: twitter-bot-detection/data/botwiki_verified_2019_balanced.csv
Rows: 1396
Humans: 698
Bots: 698
Best model: mlp_neural_network_numeric
Best balanced accuracy: 0.9679
```

To reproduce the original biased baseline:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\preprocess_baseline.py
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py --dataset twitter-bot-detection\data\botwiki_verified_2019.csv
```

To reproduce the leakage-check experiment:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\preprocess_leakage_check.py
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py --dataset twitter-bot-detection\data\botwiki_verified_2019_strict.csv
```

The strict leakage-check should score near random guessing. In the latest run, the best balanced accuracy was about `0.4964`, which supports that the training pipeline is not leaking the label directly.

## Saved Outputs

Training writes outputs under `twitter-bot-detection/reports/` and `twitter-bot-detection/models/`:

```text
reports/train_baseline_results.txt
reports/train_results.txt
reports/train_metrics.csv
reports/train_metrics.json
reports/evaluation_metrics.json
models/train_best_model.joblib
```

The text report is easiest to read. The CSV and JSON metrics are intended for reproducibility, tables, and later plotting. `models/train_best_model.joblib` is the saved trained sklearn pipeline, including fitted preprocessing steps and the best classifier. It is a binary model artifact, not a human-readable source-code file. No plots are generated by the current training script.

## Tests

Automated tests are under `twitter-bot-detection/tests/`:

```powershell
.\.venv\Scripts\python.exe -m pytest twitter-bot-detection\tests -q -p no:cacheprovider --basetemp=twitter-bot-detection\.pytest_tmp
```

The current automated suite has 12 tests:

- 4 data validation tests covering schema, missing values, ranges/labels, and duplicate user IDs
- 4 preprocessing tests covering neutralized source fields, deterministic scaling, train/test ID leakage, and strict leakage-pair checks
- 2 model tests covering saved model artifact creation and prediction output shape/type
- 2 evaluation tests covering metric computation correctness and baseline comparison output

Latest test log:

```text
12 passed in 1.64s
```

Detailed pass/fail evidence for each test is documented in `TEST_LOG.md`.

## Configuration

Common configuration is editable in the source files:

- `RANDOM_STATE` in `train_final.py`, `preprocess_baseline.py`, `preprocess_final.py`, and `preprocess_leakage_check.py`
- `NUMERIC_FEATURES` in `train_final.py`
- archive/output paths such as `DEFAULT_VERIFIED`, `DEFAULT_BOTWIKI`, and `DEFAULT_OUTPUT`

You can also override the training CSV without editing code:

```powershell
.\.venv\Scripts\python.exe twitter-bot-detection\src\train_final.py --dataset path\to\dataset.csv
```

## Notes on Bias

The original `preprocess_baseline.py` dataset is useful as a baseline, but it has an important bias limitation: bots come from Botwiki and humans come from verified accounts. High accuracy may reflect differences between those source populations.

The final project uses `preprocess_final.py` because it reduces the most obvious bias by balancing the classes, neutralizing profile text, hashtags, username, location, created date, and verified status, and matching human accounts to bot accounts on account-count features.

`preprocess_leakage_check.py` is not intended as the final training data. It is a leakage test: each bot row is paired with a human row that has identical model inputs. Accuracy drops near random guessing on that dataset, which supports that the training code is not leaking labels directly.
