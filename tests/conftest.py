import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"

sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def final_dataset_path() -> Path:
    return DATA_DIR / "botwiki_verified_2019_balanced.csv"


@pytest.fixture(scope="session")
def strict_dataset_path() -> Path:
    return DATA_DIR / "botwiki_verified_2019_strict.csv"


@pytest.fixture(scope="session")
def final_df(final_dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(final_dataset_path)


@pytest.fixture(scope="session")
def strict_df(strict_dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(strict_dataset_path)
