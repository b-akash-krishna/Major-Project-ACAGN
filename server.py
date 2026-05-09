from fastapi import FastAPI, HTTPException
import sys
import os
import importlib.util
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# --- Setup ---
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_predict_module():
    path = os.path.join(project_dir, "src", "08_predict.py")
    spec = importlib.util.spec_from_file_location("predict_module", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["predict_module"] = mod
    spec.loader.exec_module(mod)
    return mod

predict_mod = _load_predict_module()

# Healthy baseline for user-facing features — these start at near-zero so
# every value the user enters genuinely moves the prediction score.
HEALTHY_BASELINE = {
    'anchor_age': 45.0, 'gender': 0.0, 'los_days': 3.0, 'los_hours': 72.0,
    'prev_admissions': 0.0, 'prev_readmit_rate': 0.0, 'admission_type': 1.0,
    'dx_count': 1.0, 'proc_count': 0.0, 'rx_count': 5.0,
    'lab_abnormal_count': 0.0, 'lab_abnormal_rate': 0.0,
    'had_icu': 0.0, 'icu_los_sum': 0.0, 'days_since_last': 730.0,
    'bmi': 23.0, 'ed_time_hours': 0.0, 'transfer_count': 0.0,
    'med_admin_count': 10.0, 'prev_los_mean': 0.0, 'prev_los_max': 0.0,
    'lab_wbc_last': 7.0,   'lab_wbc_min': 5.0,
    'lab_platelets_last': 200.0, 'lab_platelets_min': 180.0,
    'lab_hemoglobin_min': 13.0,  'lab_hematocrit_min': 38.0,
    'lab_bun_last': 14.0,  'lab_chloride_max': 102.0,
    'hyponatremia': 0.0,   'anemia': 0.0,
    # derived — reset before recompute
    'prev_readmits': 0.0, 'readmit_age': 0.0, 'dx_proc': 0.0,
    'proc_per_day': 0.0,  'dx_per_day': 0.0,  'med_per_day': 0.0,
    'los_transfer': 0.0,  'icu_los_ratio': 0.0,'icu_los_pct': 0.0,
    'high_risk': 0.0,     'very_high_risk': 0.0, 'age_los': 135.0,
    'log_lab_count': 0.0, 'lab_per_day': 0.0,
    'severity_age': 90.0, 'complexity_day': 0.67,
}

DISEASE_MAPPING = {
    "Cardiology (CHF/Afib)":        ["dxcat_428", "dxcat_427", "cm_chf", "cm_arrhythmia"],
    "Pulmonary (COPD/Pneumonia)":   ["dxcat_496", "dxcat_486", "cm_copd", "cm_pneumonia"],
    "Sepsis/Infection":             ["dxcat_995", "dxcat_038", "cm_sepsis"],
    "Renal Failure":                ["dxcat_585", "cm_renal_fail"],
    "Diabetes/Endocrine":           ["dxcat_250", "cm_diabetes"],
    "Neurology (Stroke/Dementia)":  ["dxcat_434", "cm_stroke", "cm_dementia"],
}


@app.get("/metadata")
async def get_metadata():
    return {
        "diseases": list(DISEASE_MAPPING.keys()) + ["General Medicine"],
        "core_fields": ["anchor_age", "gender", "los_days", "prev_admissions", "admission_type"]
    }


@app.post("/predict")
async def predict(data: dict):
    from src.config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK

    mc = predict_mod.model_container
    if not mc.model_data:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    model_features = mc.model_data["features"]
    feature_means  = mc.model_data.get("feature_means", {})
    ref_means      = predict_mod._load_reference_feature_means(model_features)

    # ── HYBRID BASELINE ─────────────────────────────────────────────────────
    # Start with ref_means to keep background EHR context realistic (DRG,
    # POE counts, care-unit counts, discharge codes, etc.). Then reset all
    # user-facing clinical features to healthy near-zero values so that every
    # value the user enters properly raises or lowers the predicted risk.
    base = ref_means or feature_means or DEFAULTS
    full = {k: float(v) for k, v in base.items()}
    full.update({k: float(v) for k, v in DEFAULTS.items()})
    full.update(HEALTHY_BASELINE)          # zero-out user-facing features

    # ── USER OVERRIDES ───────────────────────────────────────────────────────
    overrides = data.copy()
    note    = overrides.pop("note", "")
    disease = overrides.pop("primary_disease", "General Medicine")

    for k, v in overrides.items():
        if v is not None and k not in ("name", "desc", "id", "data"):
            try:
                full[k] = float(v)
            except (ValueError, TypeError):
                pass

    # Set disease ICD markers
    for marker in DISEASE_MAPPING.get(disease, []):
        if marker in full:
            full[marker] = 1.0

    # ── FEATURE ENGINEERING ──────────────────────────────────────────────────
    predict_mod._recompute_engineered_features(full)

    # Compute derived features that _recompute_engineered_features doesn't cover
    los     = max(float(full.get("los_days", 1.0)), 0.01)
    icu_sum = float(full.get("icu_los_sum", 0.0))
    lab_cnt = float(full.get("lab_abnormal_count", 0.0))
    sev     = float(full.get("severity_score", 2.0))
    cplx    = float(full.get("complexity_score", 2.0))
    age     = float(full.get("anchor_age", 45.0))

    full["icu_los_pct"]    = icu_sum / los
    full["log_lab_count"]  = float(np.log1p(lab_cnt))
    full["lab_per_day"]    = lab_cnt / los
    full["severity_age"]   = sev * age
    full["complexity_day"] = cplx / los

    # ── NLP EMBEDDING ────────────────────────────────────────────────────────
    if note:
        emb = predict_mod.get_embedding(text=note, features=full)
        for i, val in enumerate(emb):
            full[f"ct5_{i}"] = float(val)
        full["ct5_has_note"]        = 1.0
        full["ct5_note_len_chars"]  = float(np.log1p(len(note)))
        full["ct5_note_len_tokens"] = float(np.log1p(len(note.split())))
    else:
        full["ct5_has_note"]        = 0.0
        full["ct5_note_len_chars"]  = 0.0
        full["ct5_note_len_tokens"] = 0.0

    # ── INFERENCE ────────────────────────────────────────────────────────────
    row   = {f: full.get(f, 0.0) for f in model_features}
    X     = pd.DataFrame([row])[model_features]
    proba = float(mc.predict_proba(X)[0])

    # API-level thresholds, calibrated to the hybrid (healthy-patient) baseline.
    # The config thresholds (0.55/0.40) were set for full dataset-mean baseline.
    # With a healthy baseline, model output compresses to ~10-68%, so thresholds
    # of 0.40/0.20 correctly map to HIGH/MEDIUM relative risk (2x/1.5x baseline).
    API_HIGH   = 0.40
    API_MEDIUM = 0.20

    if proba >= API_HIGH:
        risk = "HIGH"
    elif proba >= API_MEDIUM:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "probability": round(proba, 4),
        "risk": risk,
        "thresholds": {"high": API_HIGH, "medium": API_MEDIUM},
        "stats": {
            "features_used": len(model_features),
            "disease_context": disease,
            "completeness": round(
                len([v for v in overrides.values() if v not in (None, 0, 0.0, "")]) /
                max(len(overrides), 1), 2
            )
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
