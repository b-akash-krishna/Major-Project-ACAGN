"""
Quick diagnostic: what does the model actually output for extreme patient profiles?
Run: python3 diagnose_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import numpy as np
import pandas as pd

def load_predict_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "08_predict.py")
    spec = importlib.util.spec_from_file_location("predict_module", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["predict_module"] = mod
    spec.loader.exec_module(mod)
    return mod

mod = load_predict_module()
from src.config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK

mc = mod.model_container
model_features = mc.model_data["features"]
feature_means  = mc.model_data.get("feature_means", {})
ref_means      = mod._load_reference_feature_means(model_features)

print(f"\n{'='*60}")
print(f"  MODEL DIAGNOSTICS")
print(f"{'='*60}")
print(f"  Features : {len(model_features)}")
print(f"  Threshold HIGH   >= {THRESHOLD_HIGH_RISK}")
print(f"  Threshold MEDIUM >= {THRESHOLD_MEDIUM_RISK}")

baseline = {k: float(v) for k, v in (ref_means or feature_means or DEFAULTS).items()}
baseline.update({k: float(v) for k, v in DEFAULTS.items()})

# Helper to run a single profile
def predict(label, overrides):
    full = dict(baseline)
    for k, v in overrides.items():
        full[k] = float(v)
    mod._recompute_engineered_features(full)
    full["ct5_has_note"] = 0.0
    full["ct5_note_len_chars"] = 0.0
    full["ct5_note_len_tokens"] = 0.0
    row = {f: full.get(f, 0.0) for f in model_features}
    X = pd.DataFrame([row])[model_features]
    proba = float(mc.predict_proba(X)[0])
    tag = "HIGH" if proba >= THRESHOLD_HIGH_RISK else ("MEDIUM" if proba >= THRESHOLD_MEDIUM_RISK else "LOW")
    bar = "█" * int(proba*40) + "░" * (40 - int(proba*40))
    print(f"  [{bar}] {proba:5.1%}  {tag:<6}  {label}")
    return proba

print()
# Test a sweep
predict("BASELINE (all means)", {})
predict("Young healthy, 1-day stay", dict(anchor_age=22, los_days=1, prev_admissions=0, had_icu=0, prev_readmit_rate=0, lab_abnormal_rate=0, dx_count=1, rx_count=5))
predict("Middle, moderate", dict(anchor_age=58, los_days=5, prev_admissions=2, had_icu=0, prev_readmit_rate=0.2, lab_abnormal_rate=0.35, dx_count=4, rx_count=40))
predict("Elderly COPD, ICU 2d", dict(anchor_age=62, los_days=7, prev_admissions=4, had_icu=1, icu_los_sum=2.0, prev_readmit_rate=0.35, lab_abnormal_rate=0.40, dx_count=6, rx_count=60, lab_abnormal_count=30))
predict("Elderly CHF, 9d, 5 prior", dict(anchor_age=70, los_days=9, prev_admissions=5, had_icu=1, icu_los_sum=3.0, prev_readmit_rate=0.45, lab_abnormal_rate=0.50, dx_count=8, rx_count=100, lab_abnormal_count=45))
predict("SEPSIS, 12d, ICU 5d, DM+CKD", dict(anchor_age=65, los_days=12, prev_admissions=4, had_icu=1, icu_los_sum=5.0, prev_readmit_rate=0.50, lab_abnormal_rate=0.55, dx_count=9, rx_count=120, lab_abnormal_count=60))
predict("ADV CHF EF15%, 18d, 8 admits", dict(anchor_age=76, los_days=18, prev_admissions=8, had_icu=1, icu_los_sum=8.0, prev_readmit_rate=0.9, lab_abnormal_rate=0.68, dx_count=15, rx_count=200, lab_abnormal_count=90, prev_los_mean=14.0, days_since_last=7))
predict("ESRD+SEPSIS, 20d, 12 admits", dict(anchor_age=68, los_days=20, prev_admissions=12, had_icu=2, icu_los_sum=12.0, prev_readmit_rate=0.95, lab_abnormal_rate=0.85, dx_count=18, rx_count=300, lab_abnormal_count=200, prev_los_mean=15.0, days_since_last=10, lab_wbc_last=28.0, lab_wbc_min=2.1, lab_platelets_min=40, lab_bun_last=85))
predict("MOF, 25d, 91yr, palliative", dict(anchor_age=91, los_days=25, prev_admissions=10, had_icu=3, icu_los_sum=15.0, prev_readmit_rate=0.85, lab_abnormal_rate=0.80, dx_count=20, rx_count=180, lab_abnormal_count=150, prev_los_mean=18.0, days_since_last=15))

print(f"\n{'='*60}")
print("  KEY FEATURE VALUES IN BASELINE:")
key_feats = ["prev_readmits", "prev_readmit_rate", "days_since_last", "los_hours",
             "had_icu", "icu_los_sum", "icu_los_ratio", "age_los",
             "lab_abnormal_count", "lab_abnormal_rate", "high_risk", "very_high_risk",
             "prev_admissions", "dx_count", "rx_count"]
for f in key_feats:
    if f in baseline:
        print(f"    {f:30s} = {baseline[f]:.4f}")
print(f"{'='*60}\n")
