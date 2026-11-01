from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

FULL_DATA = DATA_DIR / "green_miles.csv"
SAMPLE_DATA = DATA_DIR / "green_miles_sample.csv"

COMPANY_COL = "Company_ID"
REGRESSION_TARGET = "Emissions_Discrepancy"
CLASSIFICATION_TARGET = "Greenwashing_Flag"

# Leakage: do not use these as features for either task.
LEAKAGE_COLS = [
    "Reported_CO2_Emissions",
    "Calculated_CO2_Emissions",
    "Emissions_Discrepancy",
    "Greenwashing_Flag",
]

NUMERIC_FEATURES = [
    "Year",
    "Fleet_Size",
    "Avg_Vehicle_Age",
    "Total_Distance_km",
    "Payload_ton_km",
    "Carbon_Credits_Purchased",
    "Delivery_Efficiency",
    "Avg_Fuel_Consumption",
    "EU_ETS_Compliance",
    "Weather_Impact_Index",
    "ESG_Score",
    "Sustainability_Budget",
]

CATEGORICAL_FEATURES = ["Country", "Industry_Sector", "Fleet_Type"]

RANDOM_STATE = 42
TEST_COMPANY_FRACTION = 0.2
