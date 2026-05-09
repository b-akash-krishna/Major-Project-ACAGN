# src/config.py
# The 'os' module provides a portable way of using operating system dependent functionality like reading or writing to the file system.
import os

# ========================================
# 1. FILE SYSTEM PATHS
# ========================================
# Define the root directory of the project. __file__ is the current script (config.py).
# We go up two directory levels to reach the root 'Major-Project' folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the absolute path to the main 'data' folder where datasets and extracted features are stored.
DATA_DIR = os.path.join(BASE_DIR, "data")

# Define the absolute path to the 'models' folder where trained weights and dictionaries are saved.
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Define the absolute path to the 'results' folder where evaluation metrics and CSV outputs are kept.
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Define the absolute path to the 'figures' folder where plots, graphs, and visual analysis diagrams are saved.
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# --- External Raw Data Sources ---
# This path points to the raw, unprocessed MIMIC-IV dataset downloaded from PhysioNet.
MIMIC_IV_DIR = "/home/csnn04/S8-MP/Major-Project/readmission-ai/data/physionet.org/files/mimiciv/3.1"

# This path points to the raw clinical notes dataset associated with MIMIC-IV.
MIMIC_NOTE_DIR = "/home/csnn04/S8-MP/Major-Project/readmission-ai/data/physionet.org/files/mimic-iv-note/2.2"

# This path points to the labeled clinical notes summarization dataset for hospital courses.
MIMIC_BHC_DIR = "/home/csnn04/S8-MP/Major-Project/readmission-ai/data/physionet.org/files/mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"
# Note: MIMIC_NOTE_PATH is typically found in MIMIC_NOTE_DIR/note/discharge.csv.gz

# --- Local Clinical-T5 Language Models ---
# The root folder holding our locally downloaded Clinical-T5 transformer models.
CLINICAL_T5_ROOT = os.path.join(BASE_DIR, "physionet.org", "files", "clinical-t5", "1.0.0")

# Path to the standard 'Base' version of the Clinical-T5 model.
CLINICAL_T5_BASE_DIR = os.path.join(CLINICAL_T5_ROOT, "Clinical-T5-Base")

# Path to the larger, more powerful 'Large' version of the Clinical-T5 model.
CLINICAL_T5_LARGE_DIR = os.path.join(CLINICAL_T5_ROOT, "Clinical-T5-Large")

# Path to the scientific text 'Sci' version of the Clinical-T5 model.
CLINICAL_T5_SCI_DIR = os.path.join(CLINICAL_T5_ROOT, "Clinical-T5-Sci")

# --- Processed Data Files ---
# The master CSV file containing all engineered tabular clinical features ready for model consumption.
FEATURES_CSV = os.path.join(DATA_DIR, "ultimate_features.csv")

# The CSV file containing the dense vector representations (embeddings) generated from the clinical notes.
EMBEDDINGS_CSV = os.path.join(DATA_DIR, "embeddings.csv")

# --- Model Artifact Files ---
# The master pickle file containing the entire trained standard model architecture, data scalers, and feature lists.
ACAGN_MAIN_MODEL_PKL = os.path.join(MODELS_DIR, "acagn_framework.pkl")
# Alias for easier reference in the code.
MAIN_MODEL_PKL = ACAGN_MAIN_MODEL_PKL

# Fallback path for older versions of the model framework.
MAIN_MODEL_PKL_LEGACY = os.path.join(MODELS_DIR, "trance_framework.pkl")

# Pickle file storing details about how the text embeddings were generated and reduced (e.g., PCA models).
EMBEDDING_INFO_PKL = os.path.join(MODELS_DIR, "embedding_info.pkl")

# JSON file holding metadata about the features, used primarily for frontend rendering.
FEATURE_METADATA_JSON = os.path.join(MODELS_DIR, "feature_metadata.json")

# ========================================
# 2. MODEL CONFIGURATION
# ========================================

# A prioritized list of text embedding architectures. The pipeline will attempt to load the first one it finds.
TEXT_MODEL_CANDIDATES = [
    CLINICAL_T5_LARGE_DIR,
    CLINICAL_T5_BASE_DIR,
    CLINICAL_T5_SCI_DIR,
    "luqh/ClinicalT5-base",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-mpnet-base-v2",
]

# --- Embedding Processing Settings ---
# The target dimensionality (number of columns) for the clinical text vectors.
EMBEDDING_DIM = 512

# The maximum number of tokens (words/sub-words) that the transformer model will process from a single clinical note.
TEXT_MAX_LENGTH = 512

# The batch size for processing notes when running on a GPU (higher is faster but uses more VRAM).
BATCH_SIZE_GPU = 16

# The batch size for processing notes when running on a CPU.
BATCH_SIZE_CPU = 8

# A fixed random seed to ensure reproducibility across Principle Component Analysis (PCA) and training splits.
RANDOM_STATE = 42

# --- Clinical Output Thresholds ---
# If the final calculated risk probability is above 55%, the patient is flagged as HIGH risk for 30-day readmission.
THRESHOLD_HIGH_RISK = 0.55

# If the probability is between 40% and 55%, the patient is flagged as MEDIUM risk.
THRESHOLD_MEDIUM_RISK = 0.4

# --- Imbalance Handling ---
# SMOTE is a statistical technique used to oversample minority classes (i.e., readmitted patients) to balance the training data.
# Set False here to disable it and rely entirely on native class weights in the models.
ENABLE_SMOTE = False

# ========================================
# 3. FEATURE DEFAULTS
# ========================================
# This dictionary contains the hardcoded statistical baseline for a perfectly standard admission.
# It acts as a safety net if real patient data is missing, ensuring the math never breaks.
DEFAULTS = {
    'los_hours': 48, 'admission_hour': 12, 'admission_dow': 2,
    'ed_time_hours': 0, 'los_transfer': 0, 'icu_los_ratio': 0,
    'severity_score': 2, 'complexity_score': 2, 'instability_score': 0,
    'prev_los_mean': 0, 'prev_icu_rate': 0, 'lab_mean': 0, 'lab_count': 0,
    'dx_unique': 1, 'service_count': 1, 'rx_count': 0, 'med_admin_count': 0,
    'micro_count': 0, 'admission_location': 2, 'insurance': 1,
    'had_ed': 0, 'multiple_icu': 0, 'lab_abnormal': 0,
    'proc_count': 0, 'dx_count': 1, 'transfer_count': 0,
    'had_icu': 0, 'icu_los_sum': 0, 'icu_count': 0,
    'icu_los_mean': 0, 'icu_los_max': 0, 'icu_los_min': 0, 'icu_los_std': 0,
    'lab_median': 0, 'lab_min': 0, 'lab_max': 0, 'lab_sem': 0,
    'lab_q25': 0, 'lab_q75': 0, 'lab_range': 0, 'lab_iqr': 0,
    'lab_skew': 0, 'lab_kurt': 0, 'lab_cv': 0,
    'dx_seq_mean': 0, 'dx_seq_max': 0,
    'prev_readmits': 0, 'age_group': 2, 'los_cat': 2,
    'high_risk': 0, 'very_high_risk': 0,
    'icu_lab': 0, 'readmit_age': 0, 'dx_proc': 0,
    'med_per_day': 0, 'is_weekend': 0, 'is_night': 0,
    'died_during_admit': 0
}

# ========================================
# 4. API SETTINGS (Backend Networking)
# ========================================
# The host IP address for the FastAPI server. "0.0.0.0" means it will listen on all available network interfaces.
API_HOST = "0.0.0.0"

# The port number the server will run on. The React frontend targets this port.
API_PORT = 8000

# Cross-Origin Resource Sharing rules. "*" allows requests from any domain (like localhost:3000).
CORS_ORIGINS = ["*"]

# ========================================
# 5. EXTRACTION SETTINGS (Data Pipeline limits)
# ========================================
# A hard limit on how many patient records to extract from the raw MIMIC database. 
# None means process the entire 500k+ dataset. Setting an integer (e.g., 10000) allows fast testing.
N_SAMPLES       = None

# Define how many of the most frequent disease categories (ICD-9/10) to convert into binary 0/1 columns.
TOP_DX_CATS  = 50    
# Define how many of the most frequent medical procedure codes to convert into binary columns.
TOP_PROC     = 80    
# Define how many of the most frequent medication types to track explicitly.
TOP_MED      = 80    

# --- Clinical Threshold Flags ---
# These are strict numerical rules used to flag patients with dangerous vital signs or abnormal labs.
VITALS_HYPO_SBP     = 90     # Below 90 mmHg systolic blood pressure triggers hypotension.
VITALS_HYPER_SBP    = 160    # Above 160 mmHg systolic triggers hypertension.
VITALS_HYPOXIA_SPO2 = 92     # Oxygen saturation below 92% triggers hypoxia flag.
VITALS_TACHY_HR     = 100    # Heart rate over 100 beats per minute triggers tachycardia.
VITALS_BRADY_HR     = 60     # Heart rate under 60 bpm triggers bradycardia.
VITALS_TACHYPNEA_RR = 20     # Breathing rate over 20 per minute triggers tachypnea.
VITALS_FEVER_F      = 100.4  # Body temp over 100.4 Fahrenheit triggers fever.
VITALS_GCS_LOW      = 10     # Glasgow Coma Scale below 10 indicates severe neurological compromise.
LAB_AKI_CREATININE  = 1.5    # Creatinine above 1.5 mg/dL indicates Acute Kidney Injury.
LAB_HYPERLAC        = 2.0    # Lactate above 2.0 mmol/L indicates severe systemic stress (e.g., sepsis).
LAB_ANEMIA_HGB      = 10.0   # Hemoglobin under 10 g/dL indicates anemia.
LAB_LEUKO_HIGH      = 11.0   # WBC count above 11 indicates possible infection.
LAB_LEUKO_LOW       = 4.0    # WBC count below 4 indicates immunosuppression.
LAB_THROMBO_LOW     = 100    # Platelets under 100 indicates bleeding risk.
LAB_HYPONATR        = 135    # Sodium under 135 indicates dangerous electrolyte imbalance.
LAB_HYPERNATR       = 145    # Sodium over 145 indicates dehydration.

# The minimum number of times a specific clinical code must appear across the dataset to become a standalone feature.
# This prevents noisy, extremely rare diseases from flooding the model with useless columns.
EXTRACT_MIN_CONTRIB_COUNT = 50

# A strict cut-off to reduce dimensionality before text embedding fusion occurs.
EXTRACT_FEATURE_KEEP_TOP  = 500

# ========================================
# 6. FEATURE SELECTION SETTINGS
# ========================================
# The exact number of final tabular features the system is allowed to pass to the final training loop.
SELECT_TOP_N        = 160

# How many cross-validation folds to use when calculating feature importance scores to ensure statistical robustness.
SELECT_N_FOLDS      = 3

# If two features have a Pearson correlation higher than 0.97 (meaning they provide identical info), delete the weaker one.
SELECT_CORR_THRESH  = 0.97

# Drop any column whose variance across the dataset is basically zero (e.g., a column where every patient has the same value).
SELECT_VAR_THRESH   = 1e-8

# How much weight to give to different ranking mathematical techniques when determining the final top 160 features.
# The weights must sum to 1.0.
SELECT_WEIGHT_SHAP  = 0.5   # 50% weight to SHAP analysis (Shapley Additive exPlanations).
SELECT_WEIGHT_GAIN  = 0.3   # 30% weight to standard decision tree information gain.
SELECT_WEIGHT_MI    = 0.2   # 20% weight to pure statistical mutual information.

# To prevent memory crashes, we only use a random subsample of 50,000 rows when calculating mutual information.
SELECT_MI_SUBSAMPLE = 50_000

# ========================================
# 7. TEXT EMBEDDING SETTINGS
# ========================================
# The final number of dimensions that our text embeddings will be squashed down into using PCA.
EMBED_DIM           = 512   # Must match EMBEDDING_DIM.

# The absolute maximum sequence length of tokens (words) the language model will read.
EMBED_MAX_SEQ_LEN   = 512   
# GPU batch size for generating text embeddings in bulk.
EMBED_GPU_BATCH     = 32    
# CPU batch size for generating text embeddings in bulk.
EMBED_CPU_BATCH     = 8

# Throw away notes shorter than 50 characters, as they contain no clinical value.
EMBED_MIN_TEXT_LEN  = 50    
# Hard truncate clinical notes at 5,000 characters to prevent memory overflow on gigantic discharge summaries.
EMBED_MAX_CHARS     = 5_000 
# When a note is too long, we split it into overlapping chunks of 256 words each.
EMBED_CHUNK_WORDS   = 256   
# How many words should overlap between chunks to maintain semantic context.
EMBED_CHUNK_OVERLAP = 64    
# Do not process more than 10 chunks per note.
EMBED_MAX_CHUNKS    = 10    

# The mathematical technique used to reduce the embedding vectors (PCA = Principal Component Analysis).
EMBEDDING_REDUCTION = "pca"

# Toggle to true if we want to run text through multiple different NLP models and average the results.
EMBED_FUSION_ENABLED = True
EMBED_FUSION_MODELS = [
    "finetuned_t5",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-mpnet-base-v2",
]

# ========================================
# 8. TRAINING ALGORITHM SETTINGS
# ========================================
# How many random hyperparameter combinations to try during the optimization search (Optuna). 
TRAIN_OPTUNA_TRIALS      = 100

# The maximum number of decision trees a single model is allowed to grow.
TRAIN_DART_MAX_TREES     = 800

# Number of folds for GroupKFold cross-validation, ensuring the same patient doesn't leak across folds.
TRAIN_N_FOLDS            = 5

# What fraction of unique patients are entirely held back for final unbiased testing.
TRAIN_TEST_FRAC          = 0.15
# What fraction of patients are held back for validation during training loops.
TRAIN_VAL_FRAC           = 0.15

# If ENABLE_SMOTE is True, this sets the synthetic minority oversampling target ratio.
TRAIN_SMOTE_RATIO        = 0.35

# How many trials to run when searching for the absolute perfect blend weights for the final ensemble.
TRAIN_BLEND_TRIALS       = 1000

# Number of text-embedding dimensions to keep (we keep the 256 dimensions with the highest variance).
TRAIN_CT5_KEEP_DIMS      = 256

# Candidate subset sizes to evaluate automatically to find the perfect feature count that maximizes AUROC.
TRAIN_FEATURE_SUBSETS    = [128, 160, 220, 259]

# Enable combining predictions from different algorithms via a meta-learner (Stacking).
TRAIN_ENABLE_STACK       = True
TRAIN_ENABLE_AUTO_FEATURE_SUBSET = True

# We train models three separate times with different random seeds to average out statistical noise.
TRAIN_SEEDS = [42, 2024, 777]
# Only run hyperparameter optimization on the first seed to save massive amounts of compute time.
TRAIN_HPO_ONCE = True

# Regularization penalization strength candidates for the logistic meta-learner.
TRAIN_META_C_CANDIDATES  = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# Metric used to decide the optimal probability cut-off point. "mcc" = Matthews Correlation Coefficient.
TRAIN_THRESHOLD_STRATEGY = "mcc"

# Determine whether the optimization system strictly targets maximizing AUROC (Area Under Receiver Operating Characteristic).
TRAIN_OPTIMIZE_AUROC     = True

# A blending weight factor used if targeting a composite scoring metric.
TRAIN_HPO_ALPHA_AUPRC    = 0.25

# ========================================
# 9. ANALYSIS & EXPLAINABILITY SETTINGS
# ========================================
# How many rows of data to feed into the SHAP explainer. Generating SHAP values is extremely slow, so we sample 500 rows.
SHAP_N_SAMPLES = 500

# ========================================
# 10. GATED FUSION ARCHITECTURE SETTINGS
# ========================================

# File path for storing the final compiled neural network weights for the gating mechanism.
ACAGN_GATE_MODEL_PKL = os.path.join(MODELS_DIR, "acagn_gate_infer.pkl")
GATE_MODEL_PKL = ACAGN_GATE_MODEL_PKL
GATE_MODEL_PKL_LEGACY = os.path.join(MODELS_DIR, "trance_gate.pkl")

# File path for a baseline Multilayer Perceptron model used strictly for comparison against our architecture.
ACAGN_CONCAT_MLP_MODEL_PKL = os.path.join(MODELS_DIR, "acagn_concat_mlp.pkl")
CONCAT_MLP_MODEL_PKL = ACAGN_CONCAT_MLP_MODEL_PKL
CONCAT_MLP_MODEL_PKL_LEGACY = os.path.join(MODELS_DIR, "concat_mlp.pkl")

# --- Deep Neural Network Parameters ---
# The width (number of neurons) inside the hidden layers of our custom Gate Network.
GATE_HIDDEN_DIM = 128        
# The incoming width of the text embedding layer (must perfectly match EMBED_DIM).
GATE_TEXT_DIM = 512          
# The percentage of neurons randomly dropped out during training to prevent the network from memorizing the data (overfitting).
GATE_DROPOUT = 0.3           
# The learning rate step size used by the Adam optimizer during backpropagation.
GATE_LR = 1e-4               
# The absolute maximum number of training loops (epochs) the neural network will run.
GATE_EPOCHS = 100            
# The number of epochs to wait without improvement before stopping training early.
GATE_PATIENCE = 10           
# Random seeds to stabilize the neural network initialization.
GATE_SEEDS = [42, 2024, 777] 

# Toggles for generating explainability diagrams specifically for the deep neural network.
GATE_ENABLE_SHAP = False
GATE_SHAP_N_BACKGROUND = 128
GATE_SHAP_N_SAMPLES = 512

# ========================================
# 11. ANALYSIS OUTPUT PATHS
# ========================================
# All file paths where generated validation reports and graphics are saved.
FAIRNESS_RESULTS_CSV    = os.path.join(RESULTS_DIR, "fairness_analysis.csv")
CALIBRATION_RESULTS_CSV = os.path.join(RESULTS_DIR, "calibration_analysis.csv")
GATE_WEIGHTS_NPY        = os.path.join(RESULTS_DIR, "gate_weights.npy")
GATE_PATIENT_IDS_NPY    = os.path.join(RESULTS_DIR, "gate_patient_ids.npy")
GATE_SHAP_IMPORTANCE_CSV = os.path.join(RESULTS_DIR, "gate_shap_importance.csv")
GATE_SHAP_SUMMARY_PNG     = os.path.join(FIGURES_DIR, "gate_shap_summary.png")
EARLY_WARNING_CSV       = os.path.join(RESULTS_DIR, "early_warning_results.csv")
TEMPORAL_DRIFT_CSV      = os.path.join(RESULTS_DIR, "temporal_drift_results.csv")

CONCAT_MLP_REPORT_JSON  = os.path.join(RESULTS_DIR, "concat_mlp_training_report.json")

# ========================================
# 12. EARLY WARNING SETTINGS
# ========================================
# The specific hospital day-cutoffs used to evaluate how early our model can accurately predict a readmission.
EARLY_WARNING_DAYS = [1, 2, 3, 5, 7]
# "full" is added automatically in the early warning script
