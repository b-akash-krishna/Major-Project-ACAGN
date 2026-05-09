"""
WeCare: Clinical Feature Gating and Routing Architecture
======================================================
This module defines the mathematical routing network that determines which 
patient features are most relevant based on the physician's clinical notes.

Architecture Flow:
  - Text Processing: The system reads a mathematical representation of the clinical note.
  - Routing Network: The system calculates importance weights (between 0 and 1) for each vital sign and lab result.
  - Feature Filtering: Patient data is multiplied by these weights to silence irrelevant data.
  - Final Classification: The filtered data is processed to generate a final risk percentage.
"""

# 'os' module provides functions to interact with the file system (creating directories, checking paths).
import os

# 'sys' module allows manipulation of the Python runtime environment, like adding custom module paths.
import sys

# 'gc' is the garbage collector interface, used here to free up computer memory during intensive operations.
import gc

# 'json' module provides functions for reading and writing data in the standard JSON text format.
import json

# 'logging' is used to output status messages to the console while the script is running.
import logging

# 'numpy' (np) provides support for high-performance numerical arrays and mathematical functions.
import numpy as np

# 'pandas' (pd) is an advanced data manipulation library used to read and filter tabular CSV data.
import pandas as pd

# 'joblib' provides lightweight pipelining and is used here to save/load heavy mathematical weights to disk.
import joblib

# 'torch' is the core PyTorch library, a powerful mathematical framework used to define our routing network.
import torch

# 'torch.nn' contains the fundamental building blocks (like linear transformations) for constructing the network.
import torch.nn as nn

# 'typing' provides type hints to make the code more readable and self-documenting.
from typing import Optional

# 'Dataset' and 'DataLoader' are PyTorch utilities that help efficiently stream large amounts of data into the system.
from torch.utils.data import Dataset, DataLoader

# Import industry-standard statistical metrics to evaluate how well our system separates low and high-risk patients.
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

# 'IsotonicRegression' is used to calibrate the raw mathematical outputs so they match real-world probabilities.
from sklearn.calibration import IsotonicRegression

# Ensure the script can locate other custom modules in the current directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Attempt to load centralized configuration settings defining file paths and system parameters.
try:
    from config import (
        FEATURES_CSV, EMBEDDINGS_CSV, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, RESULTS_DIR,
        GATE_HIDDEN_DIM, GATE_TEXT_DIM, GATE_DROPOUT,
        GATE_LR, GATE_EPOCHS, GATE_PATIENCE, GATE_SEEDS,
        GATE_WEIGHTS_NPY, GATE_PATIENT_IDS_NPY,
        GATE_ENABLE_SHAP, GATE_SHAP_N_BACKGROUND, GATE_SHAP_N_SAMPLES,
        GATE_SHAP_IMPORTANCE_CSV, GATE_SHAP_SUMMARY_PNG,
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,
        THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK,
    )
except ImportError:
    # Fallback to relative import if the primary import strategy fails.
    from .config import (
        FEATURES_CSV, EMBEDDINGS_CSV, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, RESULTS_DIR,
        GATE_HIDDEN_DIM, GATE_TEXT_DIM, GATE_DROPOUT,
        GATE_LR, GATE_EPOCHS, GATE_PATIENCE, GATE_SEEDS,
        GATE_WEIGHTS_NPY, GATE_PATIENT_IDS_NPY,
        GATE_ENABLE_SHAP, GATE_SHAP_N_BACKGROUND, GATE_SHAP_N_SAMPLES,
        GATE_SHAP_IMPORTANCE_CSV, GATE_SHAP_SUMMARY_PNG,
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,
        THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK,
    )

# Configure the logger to display informative timestamps and messages.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Instantiate a logger for this specific module.
logger = logging.getLogger(__name__)

# ── Output safety ─────────────────────────────────────────────────────────────

# A helper function to prevent accidentally overwriting important files unless explicitly allowed.
def _ensure_writable(path: str, overwrite: bool) -> None:
    if path is None:
        return
    # If the file already exists and the overwrite flag is False, abort the script to protect the data.
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path} (pass --overwrite to replace)")

# ── Dataset Definition ────────────────────────────────────────────────────────

# Define a custom PyTorch Dataset class to manage the patient data during processing.
class ReadmissionDataset(Dataset):
    """
    Holds structured clinical text representations, standard tabular features, and historical outcomes.
    Returns synchronized tensors for each specific patient when requested.
    """
    def __init__(self, text_emb: np.ndarray, tabular: np.ndarray, labels: np.ndarray):
        # Convert standard numpy arrays into high-performance PyTorch tensors.
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.tabular  = torch.tensor(tabular,  dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        # Return the total number of patients in this dataset split.
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the text representation, the tabular data, and the actual outcome for one patient.
        return self.text_emb[idx], self.tabular[idx], self.labels[idx]

# ── Mathematical Architecture ─────────────────────────────────────────────────

# Define the primary neural network class containing the clinical routing logic.
class TextGuidedGate(nn.Module):
    """
    Full clinical fusion system.
    
    This system examines the physician's note and outputs a personalized "attention score"
    for every other vital sign and lab result. For example, if the note mentions a kidney 
    issue, it mathematically increases the importance of Blood Urea Nitrogen (BUN) 
    and decreases the importance of irrelevant lung metrics.
    """

    def __init__(self, text_dim: int, tabular_dim: int,
                 hidden_dim: int = GATE_HIDDEN_DIM, dropout: float = GATE_DROPOUT,
                 top_k: int = 15):
        # Initialize the parent PyTorch Module class.
        super().__init__()
        
        # Store the maximum number of clinical features we allow to be considered per patient.
        self.top_k = top_k

        # Define the sequential mathematical operations that generate the attention scores.
        self.gate_network = nn.Sequential(
            # First linear transformation mapping text variables to an intermediate size.
            nn.Linear(text_dim, hidden_dim),
            # ReLU (Rectified Linear Unit) zeroes out negative values to introduce necessary non-linearity.
            nn.ReLU(),
            # Second linear transformation mapping to the exact number of tabular features.
            nn.Linear(hidden_dim, tabular_dim),
            # Sigmoid activation squashes all final outputs strictly between 0.0 and 1.0 (a percentage).
            nn.Sigmoid()
        )

        # Define the sequential mathematical operations that calculate the final risk percentage.
        self.classifier = nn.Sequential(
            # Combine the text and the newly filtered tabular data, mapping them to a new dimension.
            nn.Linear(text_dim + tabular_dim, 256),
            nn.ReLU(),
            # Dropout randomly ignores a fraction of connections to prevent overfitting to specific cases.
            nn.Dropout(dropout),
            # Compress the information down to 64 dimensions.
            nn.Linear(256, 64),
            nn.ReLU(),
            # Final compression down to a single dimension representing the outcome.
            nn.Linear(64, 1),
            # Final sigmoid ensures the output risk score is a true probability between 0.0 and 1.0.
            nn.Sigmoid()
        )

    # Define the forward pass function detailing exactly how data moves through the defined networks.
    def forward(self, text_emb: torch.Tensor, x_tab: torch.Tensor):
        # 1. Generate the raw importance scores based purely on the clinical note.
        gate_weights = self.gate_network(text_emb)
        
        # 2. Implement Top-K gating: Only keep the most critical signals.
        if self.top_k > 0 and self.top_k < gate_weights.shape[1]:
            if self.training:
                # Inject a tiny amount of mathematical noise during configuration to prevent stagnation.
                noise = torch.randn_like(gate_weights) * 0.05
                score = gate_weights + noise
            else:
                score = gate_weights
                
            # Find the indices of the 'K' highest scoring features for this specific patient.
            _, topk_indices = torch.topk(score, self.top_k, dim=1)
            
            # Create a blank mask and insert 1s at the indices of the top features.
            mask = torch.zeros_like(gate_weights).scatter_(1, topk_indices, 1.0)
            
            # Multiply the original weights by the mask, turning all non-top features permanently to zero.
            gate_weights = gate_weights * mask

        # 3. Apply the final weights to the actual patient lab values. If a weight is 0, the lab is ignored.
        x_gated = gate_weights * x_tab
        
        # 4. Concatenate the original note representation with the newly filtered clinical variables.
        x_fused = torch.cat([text_emb, x_gated], dim=1)
        
        # 5. Pass the combined dataset into the classifier to get the final probability score.
        prob = self.classifier(x_fused).squeeze(1)
        
        # Return both the final probability and the calculated weights for transparency reporting.
        return prob, gate_weights

# ── Data Loading Logic ────────────────────────────────────────────────────────

# Function to read CSV files and align the text data with the clinical data.
def load_fused_data():
    """
    Loads and merges tabular clinical features with text representations.
    Returns multiple aligned arrays suitable for system consumption.
    """
    # Check if a minimized/pruned version of the dataset exists for faster loading.
    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")
    feat_path = pruned if os.path.exists(pruned) else FEATURES_CSV
    logger.info("Loading clinical features from %s", feat_path)
    
    # Read the data, replacing any missing numerical values with 0.
    tab_df = pd.read_csv(feat_path, low_memory=False).fillna(0)

    # Read the processed text representations.
    logger.info("Loading text representations from %s", EMBEDDINGS_CSV)
    emb_df = pd.read_csv(EMBEDDINGS_CSV, low_memory=False)

    # Merge the two tables horizontally based on the unique hospital admission ID.
    df = tab_df.merge(emb_df, on="hadm_id", how="left").fillna(0)
    logger.info("Merged dataset shape: %s", df.shape)

    # Identify metadata columns that should not be used as predictive inputs.
    id_cols  = {"subject_id", "hadm_id", "readmit_30"}
    
    # Identify columns containing text representation numbers.
    emb_cols = [c for c in emb_df.columns if c.startswith("ct5_")]
    
    # Everything else is considered a standard clinical feature.
    tab_cols = [c for c in df.columns if c not in id_cols and c not in emb_cols]

    # Extract metadata arrays.
    groups   = df["subject_id"].astype(int).values
    hadm_ids = df["hadm_id"].astype(int).values
    labels   = df["readmit_30"].astype(np.float32).values

    # Extract standard float32 numerical arrays for the system.
    text_emb = df[emb_cols].values.astype(np.float32)
    tabular  = df[tab_cols].values.astype(np.float32)

    logger.info("Text variables: %d | Tabular variables: %d", text_emb.shape[1], tabular.shape[1])
    return text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols

# Define a thin wrapper around the system specifically required for external SHAP importance analysis.
class _GateProbWrapper(nn.Module):
    """Wrap the network to return only probabilities, ignoring internal gating weights."""
    def __init__(self, gate_model: nn.Module):
        super().__init__()
        self.gate_model = gate_model

    def forward(self, text_emb: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        prob, _ = self.gate_model(text_emb, x_tab)
        # Add a dimension to satisfy the DeepExplainer tool's structural requirements.
        return prob.unsqueeze(1)

# Function to compute SHapley Additive exPlanations (SHAP) to figure out which features matter most.
def compute_gate_shap(
    model: nn.Module,
    text_emb: np.ndarray,
    tabular: np.ndarray,
    emb_cols: list,
    tab_cols: list,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
    force: bool = False,
) -> None:
    """
    Compute SHAP values to determine global feature importance and generate charts.
    This process is mathematically intensive and may take time.
    """
    try:
        # Abort if the feature is disabled in the configuration and not forced.
        if not (GATE_ENABLE_SHAP or force):
            return

        # Use the 'Agg' backend for matplotlib to generate images without requiring a physical monitor.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        # Create necessary output directories if they don't already exist.
        os.makedirs(os.path.dirname(GATE_SHAP_IMPORTANCE_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(GATE_SHAP_SUMMARY_PNG), exist_ok=True)

        # Initialize random state for reproducibility.
        rng = np.random.RandomState(RANDOM_STATE)

        # Extract numerical indices for patients in the training and testing sets.
        tr_idx = np.where(train_mask)[0]
        te_idx = np.where(test_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            logger.warning("Importance analysis skipped: invalid data split.")
            return

        # Select a random subset of patients to use as the background baseline calculation.
        n_bg = int(min(GATE_SHAP_N_BACKGROUND, len(tr_idx)))
        n_ex = int(min(GATE_SHAP_N_SAMPLES, len(te_idx)))
        bg_sel = rng.choice(tr_idx, n_bg, replace=False)
        ex_sel = rng.choice(te_idx, n_ex, replace=False)

        # Force execution onto the CPU as SHAP deep learning algorithms can encounter memory limits on GPUs.
        shap_device = torch.device("cpu")
        model = model.to(shap_device).eval()

        # Convert all the selected data segments into PyTorch tensors.
        text_bg = torch.tensor(text_emb[bg_sel], dtype=torch.float32, device=shap_device)
        tab_bg  = torch.tensor(tabular[bg_sel],  dtype=torch.float32, device=shap_device)
        text_ex = torch.tensor(text_emb[ex_sel], dtype=torch.float32, device=shap_device)
        tab_ex  = torch.tensor(tabular[ex_sel],  dtype=torch.float32, device=shap_device)

        # Wrap the model for the explainer tool.
        wrapped = _GateProbWrapper(model).to(shap_device).eval()
        logger.info("Computing Feature Importance (background size=%d, test size=%d) ...", n_bg, n_ex)

        # Initialize the DeepExplainer with the background distribution.
        explainer = shap.DeepExplainer(wrapped, [text_bg, tab_bg])
        
        # Calculate the impact values for the actual test set.
        shap_vals = explainer.shap_values([text_ex, tab_ex])

        # Normalize the structural output depending on how the tool returned the arrays.
        if isinstance(shap_vals, list) and len(shap_vals) > 0 and isinstance(shap_vals[0], list):
            shap_vals = shap_vals[0]

        if not isinstance(shap_vals, list) or len(shap_vals) != 2:
            logger.warning("Unexpected format for analysis; skipping plot generation.")
            return

        # Separate the impacts related to text vs clinical features.
        sv_text = np.asarray(shap_vals[0])
        sv_tab  = np.asarray(shap_vals[1])
        
        # Squeeze out unnecessary dimensions.
        if sv_text.ndim == 3:
            sv_text = sv_text.squeeze(-1)
        if sv_tab.ndim == 3:
            sv_tab = sv_tab.squeeze(-1)

        X_text = text_emb[ex_sel]
        X_tab  = tabular[ex_sel]

        # Combine text and tabular impacts back into a single unified array.
        sv = np.concatenate([sv_text, sv_tab], axis=1)
        X  = np.concatenate([X_text, X_tab], axis=1)
        feature_names = list(emb_cols) + list(tab_cols)

        # Calculate the absolute mean impact for each feature across all tested patients.
        imp = np.abs(sv).mean(axis=0)
        
        # Construct a pandas DataFrame sorting the features by highest average impact.
        imp_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
        )
        # Write the resulting report to a CSV file.
        imp_df.to_csv(GATE_SHAP_IMPORTANCE_CSV, index=False)

        # Generate a high-resolution scatter plot visualizing feature distributions.
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X, feature_names=feature_names, max_display=30, show=False)
        plt.tight_layout()
        plt.savefig(GATE_SHAP_SUMMARY_PNG, dpi=150)
        plt.close()
        logger.info("Importance visualizer saved to %s", GATE_SHAP_SUMMARY_PNG)
    except Exception as e:
        logger.warning("Feature importance analysis failed: %s", e)


# Helper function to convert a dictionary of numpy arrays back into PyTorch tensors.
def _state_dict_from_numpy(state: dict) -> dict:
    return {k: torch.tensor(v) for k, v in state.items()}

# Function to strictly partition patient data into training, validation, and testing subgroups.
def make_splits(groups, labels):
    """
    Patient-level partitioning.
    Ensures that a single patient's data is never split across training and testing,
    preventing data leakage and overly optimistic evaluation scores.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    unique_patients = np.unique(groups)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_test = int(n * TRAIN_TEST_FRAC)
    n_val  = int(n * TRAIN_VAL_FRAC)

    # Assign patients based on calculated fractions.
    test_pats  = set(unique_patients[-n_test:])
    val_pats   = set(unique_patients[-(n_test + n_val):-n_test])
    train_pats = set(unique_patients[:-(n_test + n_val)])

    # Generate boolean masks allowing quick filtering of the master array.
    train_mask = np.array([g in train_pats for g in groups])
    val_mask   = np.array([g in val_pats   for g in groups])
    test_mask  = np.array([g in test_pats  for g in groups])

    return train_mask, val_mask, test_mask


def run_gate_shap_only() -> None:
    """
    Utility function to run the SHAP analysis on a pre-existing system configuration without retraining.
    """
    model_path = GATE_MODEL_PKL
    # Check for legacy naming conventions if the primary file doesn't exist.
    if not os.path.exists(model_path) and os.path.exists(GATE_MODEL_PKL_LEGACY):
        logger.warning("Falling back to legacy path %s", GATE_MODEL_PKL_LEGACY)
        model_path = GATE_MODEL_PKL_LEGACY
        
    if not os.path.exists(model_path):
        logger.error("Configuration bundle not found: %s", model_path)
        return

    # Load the compressed system state dictionary.
    bundle = joblib.load(model_path)
    best_state = bundle.get("best_seed_state_dict")
    if best_state is None:
        logger.error("Configuration bundle missing optimal internal state. Full execution required.")
        return

    # Load data and splits.
    text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols = load_fused_data()
    train_mask, _, test_mask = make_splits(groups, labels)

    # Initialize a blank system architecture and populate it with the loaded weights.
    model = TextGuidedGate(text_emb.shape[1], tabular.shape[1]).to(torch.device("cpu"))
    model.load_state_dict(_state_dict_from_numpy(best_state))
    model.eval()

    # Trigger the importance analysis.
    compute_gate_shap(
        model,
        text_emb=text_emb,
        tabular=tabular,
        emb_cols=emb_cols,
        tab_cols=tab_cols,
        train_mask=train_mask,
        test_mask=test_mask,
        device=torch.device("cpu"),
        force=True,
    )

# ── Optimization and Tuning Loop ──────────────────────────────────────────────

def train_one_seed(text_emb, tabular, labels, groups, seed: int, device: torch.device):
    """
    Configures and evaluates one instance of the system using a specific random seed.
    """
    # Fix the random generators to ensure exact reproducibility across multiple runs.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get data partition boundaries.
    train_mask, val_mask, test_mask = make_splits(groups, labels)

    # Define the loss function to quantify the mathematical error during optimization.
    # We use Binary Cross Entropy (BCE) which is standard for binary (yes/no) classification tasks.
    criterion = nn.BCELoss()

    # Instantiate PyTorch Datasets.
    train_ds = ReadmissionDataset(text_emb[train_mask], tabular[train_mask], labels[train_mask])
    val_ds   = ReadmissionDataset(text_emb[val_mask],   tabular[val_mask],   labels[val_mask])
    test_ds  = ReadmissionDataset(text_emb[test_mask],  tabular[test_mask],  labels[test_mask])

    # Instantiate DataLoaders to manage batched throughput to the hardware.
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    # Initialize the core architecture parameters.
    text_dim    = text_emb.shape[1]
    tabular_dim = tabular.shape[1]
    model       = TextGuidedGate(text_dim, tabular_dim).to(device)
    
    # Select Adam as the optimizer strategy to automatically adjust weights.
    optimizer   = torch.optim.Adam(model.parameters(), lr=GATE_LR, weight_decay=1e-5)
    
    # Configure a learning rate scheduler to slowly reduce the step size over time, allowing the system to settle gracefully.
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GATE_EPOCHS)

    # Trackers to save the absolute best performing configuration.
    best_val_auroc = 0.0
    best_state     = None
    patience_count = 0

    # Main iterative optimization loop (epochs).
    for epoch in range(GATE_EPOCHS):
        # Set the model to "train" mode (enabling dropout layers and noise).
        model.train()
        
        # Iterate over batches of patient data.
        for text_b, tab_b, label_b in train_loader:
            text_b, tab_b, label_b = text_b.to(device), tab_b.to(device), label_b.to(device)
            
            # Clear previous calculations.
            optimizer.zero_grad()
            
            # Feed data forward to get probability estimates.
            probs, _ = model(text_b, tab_b)
            
            # Calculate the discrepancy between estimated probabilities and the truth.
            loss = criterion(probs, label_b)
            
            # Compute the mathematical gradient (direction to improve).
            loss.backward()
            
            # Clip extreme gradients to prevent catastrophic mathematical instability.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Take a step and update the system's weights.
            optimizer.step()
            
        # Update the learning rate.
        scheduler.step()

        # Switch to "eval" mode (disabling dropout) to test performance on unseen validation data.
        model.eval()
        val_probs_list = []
        val_labels_list = []
        
        # Inform PyTorch not to waste memory tracking gradients during this phase.
        with torch.no_grad():
            for text_b, tab_b, label_b in val_loader:
                text_b, tab_b = text_b.to(device), tab_b.to(device)
                probs, _ = model(text_b, tab_b)
                # Store the resulting predictions.
                val_probs_list.append(probs.cpu().numpy())
                val_labels_list.append(label_b.numpy())

        # Compile all validation predictions into flat arrays.
        val_probs  = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        
        # Evaluate how well the system discriminates using the Area Under the ROC Curve.
        val_auroc  = roc_auc_score(val_labels, val_probs)

        # If this epoch scored better than any previous, save the entire configuration state.
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Reset the patience counter since we made progress.
            patience_count = 0
        else:
            # If performance didn't improve, increment the patience counter.
            patience_count += 1

        # Periodically log current progress to the console.
        if epoch % 10 == 0:
            logger.info("Seed %d | Epoch %d | Val AUROC: %.4f | Best: %.4f",
                        seed, epoch, val_auroc, best_val_auroc)

        # If we've gone too many epochs without an improvement, stop early to save time and prevent overfitting.
        if patience_count >= GATE_PATIENCE:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # Once loop is finished, forcefully restore the weights from the highest scoring epoch.
    model.load_state_dict(best_state)
    model.eval()

    # Final Evaluation Pass on the holdout Test dataset.
    test_probs_list  = []
    test_labels_list = []
    gate_weights_list = []

    with torch.no_grad():
        for text_b, tab_b, label_b in test_loader:
            text_b, tab_b = text_b.to(device), tab_b.to(device)
            probs, gates = model(text_b, tab_b)
            test_probs_list.append(probs.cpu().numpy())
            test_labels_list.append(label_b.numpy())
            gate_weights_list.append(gates.cpu().numpy())

    # Compile the final test arrays.
    test_probs   = np.concatenate(test_probs_list)
    test_labels  = np.concatenate(test_labels_list)
    gate_weights = np.concatenate(gate_weights_list)

    return model, val_probs, val_labels, test_probs, test_labels, gate_weights, test_mask

# ── Expected Calibration Error ────────────────────────────────────────────────

def compute_ece(probs, labels, n_bins=10):
    """Calculate the Expected Calibration Error to see if percentages match reality."""
    # Define 10 probability segments (0.0-0.1, 0.1-0.2, etc.).
    bins = np.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, len(labels)
    
    # Iterate through segments.
    for i in range(n_bins):
        # Find predictions falling into the current segment.
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        # Compare average expected likelihood against the actual observed reality.
        ece += (mask.sum() / total) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)

# ── Primary Execution Pipeline ────────────────────────────────────────────────

def train_gate_model(
    model_out: str = GATE_MODEL_PKL,
    report_out: Optional[str] = None,
    weights_out: str = GATE_WEIGHTS_NPY,
    ids_out: str = GATE_PATIENT_IDS_NPY,
    overwrite: bool = False,
    save_gate_arrays: bool = True,
):
    """
    Orchestrates the entire execution: runs multiple random seeds, averages their predictions
    for stability, applies statistical calibration, and saves all outputs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if report_out is None:
        report_out = os.path.join(RESULTS_DIR, "gate_training_report.json")

    # Enforce file safety parameters.
    _ensure_writable(model_out, overwrite=overwrite)
    _ensure_writable(report_out, overwrite=overwrite)
    if save_gate_arrays:
        _ensure_writable(weights_out, overwrite=overwrite)
        _ensure_writable(ids_out, overwrite=overwrite)

    # Detect if hardware acceleration (GPU) is available, otherwise default to standard CPU processing.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Computing device selected: %s", device)

    # Load data and static splits.
    text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols = load_fused_data()
    train_mask, _, test_mask = make_splits(groups, labels)

    # Initialize tracking structures to average results across multiple runs.
    all_val_probs  = []
    all_test_probs = []
    all_gate_weights = []
    seed_state_dicts = {}
    
    best_seed = None
    best_seed_state_dict = None
    best_seed_model = None
    best_seed_val_auc = -1.0
    
    test_labels_ref  = None
    val_labels_ref   = None
    test_mask_ref    = None

    # Loop over predefined seeds to run independent configuration experiments.
    for seed in GATE_SEEDS:
        logger.info("=== Starting iteration for seed %d ===", seed)
        
        # Execute the optimization loop for this specific seed.
        model, val_probs, val_labels, test_probs, test_labels, gate_weights, test_mask = \
            train_one_seed(text_emb, tabular, labels, groups, seed, device)

        # Store the outputs for later averaging.
        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)
        all_gate_weights.append(gate_weights)
        
        # Extract the optimized weights from PyTorch into standard numpy dictionaries and save them.
        seed_state_dicts[int(seed)] = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

        try:
            v_auc = float(roc_auc_score(val_labels, val_probs))
        except Exception:
            v_auc = float("nan")
            
        # Determine if this specific seed produced the single best overall configuration.
        if np.isfinite(v_auc) and v_auc > best_seed_val_auc:
            best_seed_val_auc = v_auc
            if best_seed_model is not None:
                del best_seed_model
            best_seed_model = model
            best_seed = int(seed)
            best_seed_state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

        # Keep a master reference to the actual patient outcomes for the test set.
        if test_labels_ref is None:
            test_labels_ref = test_labels
            val_labels_ref  = val_labels
            test_mask_ref   = test_mask

        # Clean up memory to prevent crashes on subsequent loops.
        if best_seed_model is not model:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Once all seeds are finished, run the importance analysis on the absolute best version.
    if best_seed_model is not None:
        compute_gate_shap(
            best_seed_model,
            text_emb=text_emb,
            tabular=tabular,
            emb_cols=emb_cols,
            tab_cols=tab_cols,
            train_mask=train_mask,
            test_mask=test_mask_ref,
            device=device,
        )
        del best_seed_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Calculate the average probabilities across all random seeds for maximum stability.
    avg_val_probs  = np.mean(all_val_probs,  axis=0)
    avg_test_probs = np.mean(all_test_probs, axis=0)
    avg_gate_weights = np.mean(all_gate_weights, axis=0)

    # Instantiate an Isotonic Regression tool to correct probability distributions.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    # Determine the correction curve using validation data.
    calibrator.fit(avg_val_probs, val_labels_ref)
    # Apply the correction curve to the testing data.
    cal_test_probs = calibrator.predict(avg_test_probs).astype(np.float32)

    # Generate final performance metrics for the terminal output.
    auroc_raw = roc_auc_score(test_labels_ref, avg_test_probs)
    auroc_cal = roc_auc_score(test_labels_ref, cal_test_probs)
    auprc     = average_precision_score(test_labels_ref, cal_test_probs)
    brier     = brier_score_loss(test_labels_ref, cal_test_probs)
    ece_before = compute_ece(avg_test_probs, test_labels_ref)
    ece_after  = compute_ece(cal_test_probs,  test_labels_ref)

    # Print the statistical evaluation summary.
    logger.info("=" * 55)
    logger.info("Fusion Architecture Evaluation Results")
    logger.info("  AUROC (raw):        %.4f", auroc_raw)
    logger.info("  AUROC (calibrated): %.4f", auroc_cal)
    logger.info("  AUPRC:              %.4f", auprc)
    logger.info("  Brier score:        %.4f", brier)
    logger.info("  ECE before cal:     %.4f", ece_before)
    logger.info("  ECE after cal:      %.4f", ece_after)
    logger.info("=" * 55)

    # Export specific internal weights representing which features were prioritized for each patient.
    test_hadm_ids = hadm_ids[test_mask_ref]
    if save_gate_arrays:
        np.save(weights_out, avg_gate_weights)
        np.save(ids_out, test_hadm_ids)
        logger.info("Internal priority arrays exported to -> %s", weights_out)

    # Package the final results into a dictionary for JSON reporting.
    results = {
        "auroc_raw":    round(float(auroc_raw), 4),
        "auroc_cal":    round(float(auroc_cal), 4),
        "auprc":        round(float(auprc),     4),
        "brier":        round(float(brier),     4),
        "ece_before":   round(float(ece_before), 4),
        "ece_after":    round(float(ece_after),  4),
        "text_features": emb_cols,
        "tab_features": tab_cols,
        "n_test":       int(len(test_labels_ref)),
        "seeds":        GATE_SEEDS,
    }

    # Bundle all necessary variables, metadata, and mathematical weights into a single file payload.
    joblib.dump({
        "calibrator":      calibrator,
        "tab_cols":        tab_cols,
        "emb_cols":        emb_cols,
        "text_dim":        text_emb.shape[1],
        "tabular_dim":     tabular.shape[1],
        "results":         results,
        "test_probs_raw":  avg_test_probs,
        "test_probs_cal":  cal_test_probs,
        "test_labels":     test_labels_ref,
        "test_hadm_ids":   test_hadm_ids,
        "avg_gate_weights": avg_gate_weights,
        "best_seed":        best_seed,
        "best_seed_state_dict": best_seed_state_dict,
        "seed_state_dicts": seed_state_dicts,
        "gate_hidden_dim": int(GATE_HIDDEN_DIM),
        "gate_dropout": float(GATE_DROPOUT),
        "top_k": 15,
    }, model_out)

    # Save the human-readable summary JSON file.
    with open(report_out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Final system configuration bundle saved to -> %s", model_out)
    return results

# Enable the script to accept optional command line arguments when launched from a terminal.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-only", action="store_true", help="Compute feature importance from saved bundle without reprocessing.")
    parser.add_argument("--model-out", default=GATE_MODEL_PKL, help="Filepath destination for the finalized system bundle (.pkl).")
    parser.add_argument(
        "--report-out",
        default=None,
        help="Filepath destination for the JSON metric summary.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacement of existing files.")
    parser.add_argument(
        "--no-save-gate-arrays",
        action="store_true",
        help="Skip writing the internal priority tracking arrays.",
    )
    args = parser.parse_args()

    # Determine execution flow based on terminal arguments.
    if args.shap_only:
        run_gate_shap_only()
    else:
        train_gate_model(
            model_out=args.model_out,
            report_out=args.report_out,
            overwrite=args.overwrite,
            save_gate_arrays=not args.no_save_gate_arrays,
        )
