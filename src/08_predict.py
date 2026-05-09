# src/08_predict.py
"""
WeCare - Interactive Clinical Prediction Terminal.
Run as:  python src/08_predict.py
"""

# 'json' module is used to parse configuration files and save diagnostic payloads.
import json

# 'logging' captures background warnings and system statuses during execution.
import logging

# 'os' module allows the script to interact with the underlying operating system to check file paths.
import os

# 'sys' module provides access to variables used or maintained by the interpreter, like the system path.
import sys

# 'warnings' module is used to suppress non-critical library alerts that might clutter the terminal.
import warnings

# 'numpy' (np) provides support for large, multi-dimensional arrays and mathematical functions.
import numpy as np

# 'pandas' (pd) offers data structures and operations for manipulating numerical tables and time series.
import pandas as pd

# Suppress version warnings if the statistical models were saved using an older version of scikit-learn.
from sklearn.exceptions import InconsistentVersionWarning

# Add the current directory to the system path to ensure relative imports resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Attempt to import our central configuration settings and utility functions.
try:
    # DEFAULTS: Baseline clinical values to use if data is missing.
    # EMBEDDINGS_CSV, FEATURES_CSV: Paths to historical clinical datasets.
    # FEATURE_METADATA_JSON: Path to the file containing column names and importance rankings.
    # MODELS_DIR, RESULTS_DIR: Output folder locations.
    # THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK: The operational thresholds for triggering clinical alerts.
    from config import (
        DEFAULTS,
        EMBEDDINGS_CSV,
        FEATURES_CSV,
        FEATURE_METADATA_JSON,
        MODELS_DIR,
        RESULTS_DIR,
        THRESHOLD_HIGH_RISK,
        THRESHOLD_MEDIUM_RISK,
    )
    # get_embedding: Function to convert clinical notes into mathematical representations.
    # get_model_container: Function to load the saved expert system weights from disk.
    from embedding_utils import get_embedding, get_model_container
except ImportError:
    # Fallback to relative imports if the primary import fails depending on how the script was invoked.
    from .config import (
        DEFAULTS,
        EMBEDDINGS_CSV,
        FEATURES_CSV,
        FEATURE_METADATA_JSON,
        MODELS_DIR,
        RESULTS_DIR,
        THRESHOLD_HIGH_RISK,
        THRESHOLD_MEDIUM_RISK,
    )
    from .embedding_utils import get_embedding, get_model_container

# Configure the logging system to only show serious warnings, keeping the terminal clean.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Explicitly ignore version warnings from scikit-learn.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Ignore missing feature name warnings that commonly happen when passing numpy arrays instead of pandas DataFrames.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the core predictive system into memory immediately when the script starts to avoid delays later.
model_container = get_model_container()

# Initialize a global cache variable to store calculated average values for clinical metrics, preventing redundant calculations.
_REFERENCE_MEANS_CACHE = None

# Initialize a global cache variable for storing the pool of historical patient profiles used for missing data imputation.
_TEMPLATE_POOL_CACHE = None

# Define the absolute path to the CSV file containing the ranked importance of all clinical features.
FEATURE_IMPORTANCE_CSV = os.path.join(MODELS_DIR, "feature_importance_report.csv")


# Helper functions

# Define a function to isolate the most actionable clinical metrics for the user to input.
def _load_major_features(model_features: list) -> list:
    """
    Load important clinical factors from metadata and keep only practical,
    non-derived fields for user prompts.
    """
    # Define a set of 'derived' features which are calculated automatically and shouldn't be requested from the user.
    derived = {
        "los_hours", "log_los_days", "age_los", "is_weekend", "is_night",
        "proc_per_day", "dx_per_day", "med_per_day", "icu_los_ratio",
        "prev_readmits", "readmit_age", "dx_proc", "los_transfer",
    }
    
    # Define features that are always explicitly requested first, so we don't ask for them twice.
    already_prompted = {"anchor_age", "gender", "los_days", "prev_admissions", "admission_type"}

    # Initialize an empty list to hold the top features.
    top = []
    try:
        # Check if the feature importance report exists on disk.
        if os.path.exists(FEATURE_IMPORTANCE_CSV):
            # Load the report using pandas.
            imp = pd.read_csv(FEATURE_IMPORTANCE_CSV, low_memory=False)
            
            # Check if the report contains the required 'feature' column.
            if "feature" in imp.columns:
                # Determine which scoring column to use for sorting.
                score_col = "combined_score" if "combined_score" in imp.columns else None
                
                # Sort the features by importance if a score column is available.
                if score_col:
                    imp = imp.sort_values(score_col, ascending=False)
                    
                # Extract the sorted feature names into a python list.
                top = imp["feature"].dropna().astype(str).tolist()
    except Exception:
        # If any error occurs reading the file, fallback to an empty list.
        top = []

    try:
        # If the CSV report failed, try reading the fallback JSON metadata file.
        if not top and os.path.exists(FEATURE_METADATA_JSON):
            with open(FEATURE_METADATA_JSON, "r", encoding="utf-8") as f:
                # Parse the JSON data into a dictionary.
                meta = json.load(f)
            # Extract the top 20 list, defaulting to an empty list if missing.
            top = meta.get("top_20_important", []) or []
    except Exception:
        # Again, fallback to an empty list on failure.
        top = []

    # If both files failed or were empty, just use the first 30 features the model expects.
    if not top:
        top = model_features[:30]

    # Initialize a list to hold the final filtered features.
    filtered = []
    
    # Iterate through our ranked top features.
    for feat in top:
        # Ignore the feature if the model itself doesn't actually use it.
        if feat not in model_features:
            continue
            
        # Ignore NLP features (prefix 'ct5_') as they are extracted automatically from the text note.
        if feat.startswith("ct5_"):
            continue
            
        # Ignore the feature if it's auto-calculated or already prompted for.
        if feat in derived or feat in already_prompted:
            continue
            
        # Add the valid, actionable feature to our final list.
        filtered.append(feat)

    # Return exactly the top 12 most important actionable features.
    return filtered[:12]

# A dictionary mapping internal variable names to user-friendly labels and acceptable numerical ranges.
FRIENDLY_MAJOR_FEATURES = {
    "bmi": {
        "label": "Body Mass Index (BMI)",
        "min": 10.0,
        "max": 60.0,
    },
    "days_since_last": {
        "label": "Days since last hospital discharge",
        "min": 0,
        "max": 3650,
    },
    "prev_readmit_rate": {
        "label": "Past readmission rate (0.0 to 1.0)",
        "min": 0.0,
        "max": 1.0,
    },
    "prev_los_mean": {
        "label": "Average past stay length in days",
        "min": 0.0,
        "max": 60.0,
    },
    "ed_time_hours": {
        "label": "ER wait time in hours",
        "min": 0.0,
        "max": 72.0,
    },
    "transfer_count": {
        "label": "Number of ward/room transfers",
        "min": 0,
        "max": 20,
    },
    "proc_count": {
        "label": "Number of procedures this stay",
        "min": 0,
        "max": 50,
    },
    "dx_count": {
        "label": "Number of diagnosed conditions",
        "min": 1,
        "max": 80,
    },
    "had_ed": {
        "label": "Came via Emergency Room? (0=no, 1=yes)",
        "min": 0,
        "max": 1,
    },
    "insurance": {
        "label": "Insurance category code",
        "min": 0,
        "max": 10,
    },
    "race_enc": {
        "label": "Race encoded value",
        "min": 0.0,
        "max": 1.0,
    },
    "language_enc": {
        "label": "Language encoded value",
        "min": 0.0,
        "max": 1.0,
    },
    "marital_enc": {
        "label": "Marital-status encoded value",
        "min": 0.0,
        "max": 1.0,
    },
    "admission_location": {
        "label": "Admission location code",
        "min": 0,
        "max": 20,
    },
    "discharge_location": {
        "label": "Discharge location code",
        "min": 0,
        "max": 20,
    },
    "admission_hour": {
        "label": "Admission hour (0-23)",
        "min": 0,
        "max": 23,
    },
    "admission_dow": {
        "label": "Admission day of week (0=Mon ... 6=Sun)",
        "min": 0,
        "max": 6,
    },
    "rx_count": {
        "label": "Medication orders count",
        "min": 0,
        "max": 500,
    },
    "med_admin_count": {
        "label": "Medication administrations count",
        "min": 0,
        "max": 2000,
    },
    "lab_abnormal_count": {
        "label": "Abnormal lab results count",
        "min": 0,
        "max": 1000,
    },
    "lab_abnormal_rate": {
        "label": "Abnormal lab ratio (0.0 to 1.0)",
        "min": 0.0,
        "max": 1.0,
    },
    "poe_count": {
        "label": "Provider order entries count",
        "min": 0,
        "max": 1000,
    },
    "primary_dx_freq": {
        "label": "Primary diagnosis frequency",
        "min": 0.0,
        "max": 1.0,
    },
}

# Define a standard subset of demographic inputs requested for every patient.
COMMON_PROFILE_FEATURES = [
    "insurance",
    "race_enc",
    "language_enc",
    "marital_enc",
    "admission_location",
    "discharge_location",
    "admission_dow",
    "admission_hour",
]

# Utility function to pull a safe default value if the user skips a question.
def _default_for_feature(name: str, feature_means: dict):
    # If a calculated statistical average exists, use it.
    if name in feature_means:
        return float(feature_means[name])
    # Otherwise, fallback to our hardcoded clinical baseline in config.py.
    if name in DEFAULTS:
        return DEFAULTS[name]
    # Absolute zero fallback.
    return 0.0

# Function to calculate or load average values for all inputs based on historic hospital data.
def _load_reference_feature_means(model_features: list) -> dict:
    """
    Build means from raw feature files, which are better defaults
    than internal means computed after statistical rebalancing.
    """
    # Use the global cache to avoid recalculating this every single time the user runs a prediction.
    global _REFERENCE_MEANS_CACHE
    if _REFERENCE_MEANS_CACHE is not None:
        return _REFERENCE_MEANS_CACHE

    # Initialize empty dictionary for means.
    means = {}
    
    # Split the required model features into regular EHR data and text note data.
    non_ct5 = [f for f in model_features if not f.startswith("ct5_")]
    ct5 = [f for f in model_features if f.startswith("ct5_")]

    try:
        # Identify the best CSV file to read regular features from.
        feat_path = FEATURES_CSV.replace(".csv", "_pruned.csv")
        if not os.path.exists(feat_path):
            feat_path = FEATURES_CSV
            
        # If the file exists, process it.
        if os.path.exists(feat_path):
            # Read just the header row to get column names quickly.
            cols = list(pd.read_csv(feat_path, nrows=0).columns)
            
            # Filter down to only columns the model actually uses.
            usecols = [c for c in non_ct5 if c in cols]
            if usecols:
                # Read the specific columns for all rows.
                df = pd.read_csv(feat_path, usecols=usecols, low_memory=False)
                
                # Calculate the numerical average of each column and store it in our dictionary.
                means.update(df.mean(numeric_only=True).to_dict())
    except Exception:
        # Fail silently if files are missing, as we have multiple fallback mechanisms.
        pass

    try:
        # Repeat the average calculation process for the text note embedding file.
        if os.path.exists(EMBEDDINGS_CSV) and ct5:
            cols = list(pd.read_csv(EMBEDDINGS_CSV, nrows=0).columns)
            usecols = [c for c in ct5 if c in cols]
            if usecols:
                df = pd.read_csv(EMBEDDINGS_CSV, usecols=usecols, low_memory=False)
                means.update(df.mean(numeric_only=True).to_dict())
    except Exception:
        pass

    # Save the computed averages into the global cache to speed up subsequent requests.
    _REFERENCE_MEANS_CACHE = {k: float(v) for k, v in means.items()}
    return _REFERENCE_MEANS_CACHE

# Function to load a "pool" of historical patients to help impute missing values intelligently.
def _load_template_pool(model_features: list, max_rows: int = 30000) -> pd.DataFrame:
    """
    Load a limited pool of real patient records for nearest-neighbor baseline fill.
    This keeps predictions highly realistic when data is missing.
    """
    # Use the global cache.
    global _TEMPLATE_POOL_CACHE
    if _TEMPLATE_POOL_CACHE is not None:
        return _TEMPLATE_POOL_CACHE

    try:
        # Identify the primary feature file.
        feat_path = FEATURES_CSV.replace(".csv", "_pruned.csv")
        if not os.path.exists(feat_path):
            feat_path = FEATURES_CSV
            
        # If no file exists, return an empty DataFrame immediately.
        if not os.path.exists(feat_path):
            _TEMPLATE_POOL_CACHE = pd.DataFrame()
            return _TEMPLATE_POOL_CACHE

        # Read the top N rows into memory.
        tab = pd.read_csv(feat_path, nrows=max_rows, low_memory=False)
        
        # If the embeddings file also exists, merge it into the main table.
        if os.path.exists(EMBEDDINGS_CSV):
            emb = pd.read_csv(EMBEDDINGS_CSV, nrows=max_rows, low_memory=False)
            if "hadm_id" in tab.columns and "hadm_id" in emb.columns:
                # Merge based on Hospital Admission ID.
                df = tab.merge(emb, on="hadm_id", how="left")
            else:
                df = tab
        else:
            df = tab

        # Filter the merged table down strictly to the features required by the system.
        keep = [c for c in model_features if c in df.columns]
        
        # Replace any lingering NaN values with 0 and save to cache.
        _TEMPLATE_POOL_CACHE = df[keep].fillna(0)
        return _TEMPLATE_POOL_CACHE
    except Exception:
        # Return an empty DataFrame on any crash.
        _TEMPLATE_POOL_CACHE = pd.DataFrame()
        return _TEMPLATE_POOL_CACHE

# Function to search the historical pool for the patient that most closely matches the user's inputs.
def _nearest_template_baseline(user_data: dict, model_features: list) -> dict:
    """Pick a realistic baseline row nearest to user-entered core fields."""
    # Load the historical patient pool.
    pool = _load_template_pool(model_features)
    
    # Abort if the pool couldn't load.
    if pool.empty:
        return {}

    # Define the key metrics we will use to find a "matching" patient.
    keys = [k for k in [
        "anchor_age", "los_days", "prev_admissions", "admission_type",
        "days_since_last", "prev_readmit_rate", "prev_los_mean"
    ] if k in pool.columns and k in user_data]
    
    if not keys:
        return {}

    # Define scaling factors to ensure one metric (like age) doesn't completely overwhelm another (like length of stay) during the distance calculation.
    scales = {
        "anchor_age": 20.0,
        "los_days": 5.0,
        "prev_admissions": 5.0,
        "admission_type": 1.0,
        "days_since_last": 365.0,
        "prev_readmit_rate": 0.2,
        "prev_los_mean": 3.0,
    }

    # Initialize a zeroed distance array for every patient in the pool.
    dist = np.zeros(len(pool), dtype=np.float32)
    
    # Loop through each of our key metrics.
    for k in keys:
        # Get the scaling factor.
        s = float(scales.get(k, 1.0))
        
        # Calculate the absolute mathematical difference between the user's value and the pool's value, divided by the scale.
        diff = np.abs(pool[k].astype(float).to_numpy() - float(user_data[k])) / max(s, 1e-6)
        
        # Categorical mismatch for admission_type is heavily penalized to ensure we don't mix urgent vs planned visits.
        if k == "admission_type":
            diff = (pool[k].astype(float).to_numpy() != float(user_data[k])).astype(np.float32) * 2.0
            
        # Accumulate the distance score.
        dist += diff.astype(np.float32)

    # Find the index of the patient in the pool with the smallest accumulated distance (the most similar patient).
    idx = int(np.argmin(dist))
    
    # Return that specific historical patient's full data record as a dictionary to be used for imputing missing values.
    return {c: float(pool.iloc[idx][c]) for c in pool.columns}

# Utility function to infer whether a default value should be cast as an integer or a decimal.
def _infer_cast(default):
    if isinstance(default, bool):
        return int
    if isinstance(default, int):
        return int
    return float

# Utility function to round decimal default values to exactly two places for cleaner display.
def _round_default(value):
    if isinstance(value, float):
        return round(value, 2)
    return value

# Utility function to ensure user input does not exceed logical clinical boundaries (e.g., negative age).
def _coerce_range(value, min_val, max_val):
    if min_val is not None and value < min_val:
        return min_val
    if max_val is not None and value > max_val:
        return max_val
    return value

# Function to recalculate all derived variables based on the final user inputs.
def _recompute_engineered_features(full: dict) -> None:
    """Recompute core engineered features for consistency with user inputs."""
    # Extract raw base values safely.
    age = float(full.get("anchor_age", 65))
    los = float(full.get("los_days", 3.0))
    prev_adm = float(full.get("prev_admissions", 0))
    prev_rate = float(full.get("prev_readmit_rate", 0))
    proc_count = float(full.get("proc_count", 0))
    dx_count = float(full.get("dx_count", 1))
    rx_count = float(full.get("rx_count", 0))
    transfer_count = float(full.get("transfer_count", 0))
    icu_los_sum = float(full.get("icu_los_sum", 0))
    admission_hour = float(full.get("admission_hour", 12))
    admission_dow = float(full.get("admission_dow", 2))

    # Re-calculate compound variables based strictly on the raw values above.
    full["los_hours"] = los * 24
    full["log_los_days"] = float(np.log1p(max(los, 0.0)))
    full["age_los"] = age * los
    full["prev_readmits"] = max(prev_adm * prev_rate, 0.0)
    full["readmit_age"] = age * prev_rate
    full["dx_proc"] = dx_count * proc_count
    full["proc_per_day"] = proc_count / (los + 1.0)
    full["dx_per_day"] = dx_count / (los + 1.0)
    full["med_per_day"] = rx_count / (los + 1.0)
    full["los_transfer"] = los * transfer_count
    full["icu_los_ratio"] = icu_los_sum / (los + 0.01)
    
    # Binary threshold checks for extreme edge cases.
    full["is_first_visit"] = 1.0 if prev_adm <= 0 else 0.0
    full["high_risk"] = 1.0 if (los > 10 or age >= 80 or full.get("had_icu", 0) > 0) else 0.0
    full["very_high_risk"] = 1.0 if (los > 20 or age >= 90 or full.get("had_icu", 0) > 1) else 0.0
    full["is_weekend"] = 1.0 if int(round(admission_dow)) in (5, 6) else 0.0
    full["is_night"] = 1.0 if (admission_hour < 7 or admission_hour >= 22) else 0.0

    # Discretize age into predefined demographic groups for categorical evaluation.
    if age < 40:
        full["age_group"] = 0
    elif age < 55:
        full["age_group"] = 1
    elif age < 65:
        full["age_group"] = 2
    elif age < 75:
        full["age_group"] = 3
    elif age < 85:
        full["age_group"] = 4
    else:
        full["age_group"] = 5

    # Discretize length of stay into standard time buckets.
    if los <= 1:
        full["los_cat"] = 0
    elif los <= 3:
        full["los_cat"] = 1
    elif los <= 7:
        full["los_cat"] = 2
    elif los <= 14:
        full["los_cat"] = 3
    elif los <= 30:
        full["los_cat"] = 4
    else:
        full["los_cat"] = 5

# Developer function to print out exactly what numbers are being fed into the system for troubleshooting.
def _print_payload_debug(features: list, row: dict) -> None:
    """Print exact data payload in correct order and save to JSON for evaluation auditing."""
    print("\n" + "=" * 52)
    print(f"  DEBUG PAYLOAD (exact clinical inputs): {len(features)} features")
    print("=" * 52)
    
    # Print every single feature and its finalized value to the console.
    for i, feat in enumerate(features, start=1):
        print(f"  {i:03d}. {feat}: {row.get(feat, 0)}")

    try:
        # Also attempt to save this payload to a JSON file on disk for later review.
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "last_prediction_payload.json")
        payload = [{"index": i + 1, "feature": f, "value": row.get(f, 0)} for i, f in enumerate(features)]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n  Payload saved to: {out_path}")
    except Exception as e:
        print(f"\n  Could not save payload JSON: {e}")


# User input collection block for terminal usage.

def get_user_input() -> dict:
    """Interactive CLI prompt for patient data entry."""
    print("\n" + "=" * 52)
    print("  WECARE PREDICTIVE PLATFORM - TERMINAL")
    print("=" * 52)

    # Helper function to display a question, validate the answer, and enforce clinical bounds.
    def _prompt(prompt: str, default, cast=int, min_val=None, max_val=None):
        try:
            # Request input from the user.
            raw = input(prompt).strip()
            # If the user typed nothing, use the default. Otherwise, try to cast their string to a number.
            val = cast(raw) if raw else default
            
            # If bounds are defined, enforce them.
            if min_val is not None or max_val is not None:
                clipped = _coerce_range(val, min_val, max_val)
                if clipped != val:
                    print(f"  Value out of range, using nearest valid value: {clipped}")
                return clipped
            return val
        except (ValueError, TypeError):
            # If the cast failed (e.g., user typed "hello" instead of a number), fallback to the default.
            print(f"  Invalid input, using default: {default}")
            return default

    # Extract all necessary statistical parameters to drive smart defaults.
    model_data = model_container.model_data or {}
    model_features = model_data.get("features", [])
    feature_means = model_data.get("feature_means", {})
    reference_means = _load_reference_feature_means(model_features)
    prompt_means = reference_means or feature_means
    major_features = _load_major_features(model_features)

    try:
        # Block 1: Mandatory core vitals.
        print("  Basic details:")
        age = _prompt("  Age in years [65]: ", 65, int, 18, 120)
        gender = _prompt("  Sex (0=female, 1=male) [0]: ", 0, int, 0, 1)
        los = _prompt("  Current hospital stay length in days [3.0]: ", 3.0, float, 0.0, 120.0)
        prev = _prompt("  Number of previous hospital admissions [0]: ", 0, int, 0, 50)
        adm_t = _prompt(
            "  Admission type (1=emergency, 2=urgent, 3=planned) [1]: ",
            1, int, 1, 3
        )

        profile_values = {}
        # Block 2: Optional demographic profile inputs based on historical importance.
        geo_candidates = [f for f in COMMON_PROFILE_FEATURES if f in model_features]
        if geo_candidates:
            print("\n  Common profile/geographic context (press Enter to keep suggested average value):")
            for feat in geo_candidates:
                meta = FRIENDLY_MAJOR_FEATURES.get(feat, {})
                dflt = _round_default(_default_for_feature(feat, prompt_means))
                cast = _infer_cast(dflt)
                label = meta.get("label", feat.replace("_", " ").title())
                # Prompt the user with the dynamically calculated default.
                profile_values[feat] = _prompt(
                    f"  {label} [{dflt}]: ",
                    dflt,
                    cast,
                    meta.get("min"),
                    meta.get("max"),
                )

        major_values = {}
        # Block 3: Dynamic prompts for the top predictive factors calculated in this specific build.
        if major_features:
            print("\n  Top contributing risk factors (press Enter to keep suggested average value):")
            for feat in major_features:
                # Omit confusing coded variables from the manual terminal.
                if (
                    feat.startswith("proc_")
                    or feat.startswith("dxcat_")
                    or feat.startswith("dx_")
                    or feat.startswith("med_")
                    or feat.startswith("ct5_")
                ):
                    continue
                # Do not ask the same question twice.
                if feat in profile_values:
                    continue
                    
                meta = FRIENDLY_MAJOR_FEATURES.get(feat, {})
                dflt = _round_default(_default_for_feature(feat, prompt_means))
                cast = _infer_cast(dflt)
                label = meta.get("label", feat.replace("_", " ").title())
                # Ask for the top contributing factors.
                major_values[feat] = _prompt(
                    f"  {label} [{dflt}]: ",
                    dflt,
                    cast,
                    meta.get("min"),
                    meta.get("max"),
                )

        # Block 4: Allow the user to paste unstructured text for NLP analysis.
        note = input(
            "  Short clinical summary (optional, plain language is fine):\n"
            "  > "
        ).strip()
        
        # Finally, ask if the user wants to see the raw input array before prediction.
        debug_payload = input(
            f"  Show full {len(model_features)}-feature evaluation debug payload? (y/N): "
        ).strip().lower() == "y"
        
    except KeyboardInterrupt:
        # Graceful exit if the user hits CTRL+C.
        print("\n  Cancelled.")
        return {}

    # Compile all collected inputs into a single dictionary.
    data = {
        "anchor_age": age,
        "gender": gender,
        "los_days": los,
        "los_hours": los * 24,
        "prev_admissions": prev,
        "admission_type": adm_t,
        "clinical_note": note or None,
        "_debug_payload": debug_payload,
    }
    # Merge in the optional answers.
    data.update(profile_values)
    data.update(major_values)
    return data


# Inference Pipeline Function

def run_inference(data: dict) -> None:
    """
    Executes the clinical heuristic pipeline to generate a final risk score.
    """
    # Verify the predictive weights are successfully loaded into memory.
    if not model_container.model_data:
        print("  System not loaded. Ensure configuration files are correct.")
        return

    # Step 1: Intelligent Imputation Pipeline
    # When a user only provides 5 fields, we must accurately guess the other 155 fields.
    # We do this by finding the most statistically similar past patient to use as a baseline.
    feature_means = model_container.model_data.get("feature_means", {})
    model_features = model_container.model_data.get("features", [])
    
    # Get the nearest neighbor patient baseline based on core metrics.
    template_base = _nearest_template_baseline(data, model_features)
    # Get the raw dataset averages as a fallback.
    reference_means = _load_reference_feature_means(model_features)
    
    # Establish priority: Template > Raw Means > Internal Model Means.
    baseline_means = template_base or reference_means or feature_means
    
    if baseline_means:
        # Initialize the 'full' data payload using the baseline means.
        full = {k: float(v) for k, v in baseline_means.items()}
        # Ensure all hardcoded DEFAULTS from config.py are present.
        for k, v in DEFAULTS.items():
            full.setdefault(k, v)
    else:
        # If absolutely everything fails, just use the hardcoded DEFAULTS.
        full = dict(DEFAULTS)
        
    # Overwrite the baseline data with the specific, accurate data the user typed in.
    full.update(data)
    
    # Recalculate intermediate fields based on the new explicit inputs.
    full["los_hours"] = data.get("los_days", 3.0) * 24
    full["log_los_days"] = np.log1p(data.get("los_days", 3.0))
    full["age_los"] = data.get("anchor_age", 65) * data.get("los_days", 3.0)
    
    # Run the comprehensive recalculation for all derived clinical fields.
    _recompute_engineered_features(full)

    # Step 2: Logging Information
    if template_base:
        print("  Baseline initialized from nearest historical patient template.")
    elif reference_means:
        print(f"  Baseline initialized from raw dataset averages ({len(reference_means)} features).")
    elif feature_means:
        print(f"  Baseline initialized from internal average parameters ({len(feature_means)} features).")
    else:
        print("  Note: Using hardcoded emergency baseline.")

    # Step 3: Natural Language Processing Conversion
    clinical_note = data.get("clinical_note")
    if clinical_note:
        # Convert the raw text string into a mathematical vector representation.
        emb = get_embedding(text=clinical_note, features=full)
        print(f"  Note processing: {len(clinical_note)} characters converted to structured format.")
        
        # Inject the resulting vector into our data payload using the 'ct5_' prefix standard.
        for i, val in enumerate(emb):
            full[f"ct5_{i}"] = float(val)
            
        # Update metadata flags indicating a note was present.
        full["ct5_has_note"] = 1.0
        full["ct5_note_len_chars"] = float(np.log1p(len(clinical_note)))
        full["ct5_note_len_tokens"] = float(np.log1p(len(clinical_note.split())))
    else:
        # If no note is provided, we zero out the metadata flags. 
        # Crucially, we do NOT zero out the ct5_ vector features themselves, as the template baseline 
        # already contains the safe, average vector for a patient of this profile.
        print("  No clinical note provided - using historical baseline vector profile.")
        full["ct5_has_note"] = 0.0
        full["ct5_note_len_chars"] = 0.0
        full["ct5_note_len_tokens"] = 0.0

    # Step 4: Final Feature Alignment
    features = model_container.model_data["features"]
    
    # Double check if any required features completely missed imputation.
    missing = [f for f in features if f not in full and not f.startswith("ct5_")]
    if missing:
        print(f"  Warning: {len(missing)} fields missing entirely (defaulting to 0): {missing[:3]} ...")
        
    # Build the final ordered row exactly as the inference system expects it.
    row = {f: full.get(f, 0) for f in features}
    
    # Convert to a pandas DataFrame representing one row of data.
    X = pd.DataFrame([row])[features]

    # Print the full list of variables to terminal if debugging is turned on.
    if data.get("_debug_payload", False):
        _print_payload_debug(features, row)

    # Step 5: Execute Inference
    try:
        # Call the core predict_proba function to generate the final readmission percentage score.
        proba = float(model_container.predict_proba(X)[0])
    except Exception as e:
        print(f"  Execution error: {e}")
        return

    # Step 6: Categorize Results based on Risk Thresholds
    if proba >= THRESHOLD_HIGH_RISK:
        risk = "HIGH"
        color = "!!!"
    elif proba >= THRESHOLD_MEDIUM_RISK:
        risk = "MEDIUM"
        color = "??"
    else:
        risk = "LOW"
        color = "OK"

    # Display the final output to the user.
    print("\n" + "*" * 52)
    print(f"  RESULT: {color} {risk} RISK {color}")
    print(f"  30-day predicted readmission probability: {proba:.1%}")
    print(f"  System Thresholds: High>={THRESHOLD_HIGH_RISK:.0%}  Medium>={THRESHOLD_MEDIUM_RISK:.0%}")
    
    # Provide helpful footer information.
    if data.get("_debug_payload", False):
        print("  Debug payload mode: ON")
    else:
        print("  Debug payload mode: OFF (set to 'y' at prompt to inspect all data variables)")
    print("*" * 52 + "\n")


# Primary execution loop when running strictly from terminal
if __name__ == "__main__":
    # Ensure system is valid before starting loop.
    if not model_container.model_data:
        print("ERROR: System logic not loaded. Check configuration pathways.")
        sys.exit(1)

    # Print a startup message displaying the complexity of the current system instance.
    print(
        f"\nSystem loaded successfully: {len(model_container.model_data.get('features', []))} parameters"
    )

    # Indefinite loop asking for patient data, processing, and repeating until cancelled.
    while True:
        data = get_user_input()
        if data:
            run_inference(data)

        try:
            cont = input("  Analyze another patient? (y/n) [y]: ").strip().lower()
        except KeyboardInterrupt:
            # Catch CTRL+C cleanly.
            break
            
        if cont == "n":
            break

    print("  Terminal session closed.")
