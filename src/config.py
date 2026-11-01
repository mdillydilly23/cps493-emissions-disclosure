from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

FULL_DATA = DATA_DIR / "carbon_emission.csv"
SAMPLE_DATA = DATA_DIR / "carbon_emission_sample.csv"

COMPANY_COL = "Company_ID"
DATE_COL = "Date"

REGRESSION_TARGET = "Carbon_Emission_tCO2e"
CLASSIFICATION_TARGET = "High_Emitter_Flag"

# Labels derived from emissions; never use as features.
LEAKAGE_COLS = [
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
]

# High_Emitter_Flag = 1 when emissions exceed this quantile on the training split.
HIGH_EMITTER_TRAIN_QUANTILE = 0.75

RANDOM_STATE = 42
TEST_COMPANY_FRACTION = 0.2
