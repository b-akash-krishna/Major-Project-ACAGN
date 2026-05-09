"""
Hybrid prediction utilities (ACAGN-Hybrid)
=========================================

ACAGN-Hybrid is a probability-level ensemble:

  p_hybrid = w * p_base + (1 - w) * p_gate

where:
  - p_base is the calibrated probability from the ACAGN base ensemble
  - p_gate is the calibrated probability from ACAGN-Gate

Important:
  The gate model bundle must include saved weights (e.g. `seed_state_dicts` or
  `best_seed_state_dict`). Older bundles that only contain test predictions
  cannot be used for new-patient inference.
"""

# Import the 'annotations' feature from the future to enable modern Python type hinting.
# This ensures forward compatibility with Python 3.7+ type hints.
from __future__ import annotations

# The 'os' module provides functions for interacting with the operating system, like checking file paths.
import os

# 'dataclass' from 'dataclasses' is used to create classes that are primarily used to store data.
# It automatically generates __init__ and other useful methods, saving boilerplate code.
from dataclasses import dataclass

# The 'typing' module provides support for type hints, which helps with code readability and debugging.
from typing import Dict, Iterable, List, Optional, Tuple, Union

# 'joblib' is a library used to efficiently save and load Python objects to disk.
# It is specifically optimized for large numpy arrays, making it ideal for our model weights.
import joblib

# 'numpy' (abbreviated as 'np') is the fundamental package for scientific computing in Python.
# We use it here to perform fast numerical operations and manipulate multi-dimensional arrays.
import numpy as np

# 'torch' is the PyTorch library, an open-source machine learning framework.
# We use it to build and run our deep neural network architectures.
import torch

# Try block to handle imports depending on whether this script is run as a top-level module or imported.
try:
    # Import necessary configuration constants from our custom config module.
    from config import GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, GATE_HIDDEN_DIM, GATE_DROPOUT
    # Import the TextGuidedGate neural network model architecture from our gated_fusion_model module.
    from gated_fusion_model import TextGuidedGate
except ImportError:
    # If the standard import fails (e.g., when imported as a package), use relative imports.
    from .config import GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, GATE_HIDDEN_DIM, GATE_DROPOUT
    from .gated_fusion_model import TextGuidedGate


# Define a function to combine the probabilities from the base model and the gate model.
def hybrid_combine(p_base: float, p_gate: float, w_base: float = 0.5) -> float:
    # Ensure the base probability is a floating-point number.
    p_base = float(p_base)
    # Ensure the gate probability is a floating-point number.
    p_gate = float(p_gate)
    # Ensure the weighting factor is a floating-point number.
    w_base = float(w_base)
    # Return the weighted sum: base probability * weight + gate probability * (1 - weight).
    # This mathematically blends the two model outputs.
    return float(w_base * p_base + (1.0 - w_base) * p_gate)


# Define a helper function to locate the correct path for the gate model pickle file.
def _resolve_gate_bundle_path(path: str = GATE_MODEL_PKL) -> str:
    # Check if the primary path exists on the filesystem.
    if os.path.exists(path):
        # If it does, return this primary path.
        return path
    # If primary path is missing, check if the legacy (older) path exists.
    if os.path.exists(GATE_MODEL_PKL_LEGACY):
        # If legacy path exists, return it as a fallback.
        return GATE_MODEL_PKL_LEGACY
    # If neither exists, return the originally requested path to let the caller handle the error.
    return path


# Define a function to convert a dictionary of weights (state_dict) to PyTorch tensors.
def _state_dict_from_numpy(state: dict) -> dict:
    # Initialize an empty dictionary to hold the converted tensors.
    out = {}
    # Iterate through each key-value pair in the input state dictionary.
    for k, v in state.items():
        # Check if the current value is already a PyTorch Tensor.
        if isinstance(v, torch.Tensor):
            # If so, just copy it to the output dictionary.
            out[k] = v
        # If the value is not a Tensor (e.g., it is a numpy array or standard list).
        else:
            # Convert the value into a numpy array for consistency.
            arr = np.asarray(v)
            # Convert the numpy array into a PyTorch Tensor and store it.
            out[k] = torch.from_numpy(arr)
    # Return the fully converted state dictionary containing only PyTorch Tensors.
    return out


# Define a dataclass to hold the specification details for the Gate Model Bundle.
# 'frozen=True' means the data inside cannot be modified after creation (it is immutable).
@dataclass(frozen=True)
class GateBundleSpec:
    # List of column names corresponding to the tabular features.
    tab_cols: List[str]
    # List of column names corresponding to the embedded features (e.g., text representations).
    emb_cols: List[str]
    # The dimensionality (number of features) of the text input.
    text_dim: int
    # The dimensionality (number of features) of the tabular input.
    tabular_dim: int
    # The calibrator object used to convert raw model scores into true probabilities.
    calibrator: object
    # A dictionary mapping random seed integers to their corresponding model weights (state_dicts).
    seed_state_dicts: Dict[int, dict]
    # The specific state_dict that performed the best overall, serving as a fallback.
    best_seed_state_dict: Optional[dict]
    # The number of neurons in the hidden layers of the gate architecture.
    gate_hidden_dim: int
    # The dropout rate applied to the gate network to prevent overfitting.
    gate_dropout: float
    # The number of top features to select or focus on within the gate logic.
    top_k: int


# Define a function to load the gate model bundle and extract its specification.
def load_gate_bundle(bundle_path: Optional[str] = None) -> Tuple[GateBundleSpec, dict]:
    # Resolve the path using our helper function, defaulting to GATE_MODEL_PKL if none is provided.
    path = _resolve_gate_bundle_path(bundle_path or GATE_MODEL_PKL)
    
    # Verify the file actually exists on the disk.
    if not os.path.exists(path):
        # Raise an error if the file is missing, giving the user instructions on how to generate it.
        raise FileNotFoundError(
            f"Gate model bundle not found at {path}. "
            f"Train an inference-ready bundle with: python3 -m src.gated_fusion_model --model-out {GATE_MODEL_PKL}"
        )

    # Use joblib to safely load the Python object (the bundle) from the disk into memory.
    bundle = joblib.load(path)

    # Extract the tabular feature column names from the loaded bundle.
    tab_cols = bundle.get("tab_cols")
    # Extract the embedding feature column names from the loaded bundle.
    emb_cols = bundle.get("emb_cols")
    # Extract the text dimension size.
    text_dim = bundle.get("text_dim")
    # Extract the tabular dimension size.
    tabular_dim = bundle.get("tabular_dim")
    # Extract the probability calibrator model.
    calibrator = bundle.get("calibrator")

    # Extract the dictionary of seed-based model weights, defaulting to an empty dict if missing.
    seed_state_dicts = bundle.get("seed_state_dicts") or {}
    # Extract the best single model weights.
    best_seed_state_dict = bundle.get("best_seed_state_dict")

    # Perform a safety check to ensure all crucial core parameters are present.
    if tab_cols is None or text_dim is None or tabular_dim is None or calibrator is None:
        # Stop execution if vital metadata is missing, as the model cannot run without them.
        raise RuntimeError(f"Gate bundle at {path} is missing required fields for inference.")

    # Explicitly check for embedding columns, which are required for mapping text features.
    if emb_cols is None:
        # Halt and provide clear feedback if the bundle format is outdated.
        raise RuntimeError(
            f"Gate bundle at {path} does not contain emb_cols; cannot map embeddings. "
            f"Re-train gate with updated code to generate an inference-ready bundle."
        )

    # Check if there are actual model weights to load. We need either multiple seeds or one best seed.
    if not seed_state_dicts and best_seed_state_dict is None:
        # Halt execution because we cannot make predictions without trained neural network weights.
        raise RuntimeError(
            f"Gate bundle at {path} does not contain model weights (seed_state_dicts / best_seed_state_dict). "
            f"It likely only contains test predictions. Re-train gate to enable new-patient inference."
        )

    # Construct the GateBundleSpec data object, casting all extracted values to their correct data types.
    spec = GateBundleSpec(
        tab_cols=list(tab_cols),                                    # Convert tabular columns to a standard Python list.
        emb_cols=list(emb_cols),                                    # Convert embedding columns to a standard Python list.
        text_dim=int(text_dim),                                     # Ensure text_dim is an integer.
        tabular_dim=int(tabular_dim),                               # Ensure tabular_dim is an integer.
        calibrator=calibrator,                                      # Keep calibrator as an object.
        seed_state_dicts={int(k): v for k, v in seed_state_dicts.items()}, # Ensure seed keys are integers.
        best_seed_state_dict=best_seed_state_dict,                  # Pass the best state dict as-is.
        gate_hidden_dim=int(bundle.get("gate_hidden_dim", GATE_HIDDEN_DIM)), # Fetch hidden_dim, fallback to config.
        gate_dropout=float(bundle.get("gate_dropout", GATE_DROPOUT)),        # Fetch dropout, fallback to config.
        top_k=int(bundle.get("top_k", 15)),                         # Fetch top_k, fallback to 15.
    )
    # Return both the typed specification object and the raw bundle dictionary.
    return spec, bundle


# Define the main class that wraps the PyTorch model for making predictions.
class GatePredictor:
    """
    Inference wrapper for ACAGN-Gate.

    Uses:
      - seed_state_dicts (preferred; averages probabilities across seeds), or
      - best_seed_state_dict (fallback; single model)

    Expects a fully-populated feature dict containing:
      - gate tabular features for spec.tab_cols
      - embedding features for spec.emb_cols (ct5_0..ct5_511 + metadata columns)
    """

    # The initialization method runs when a new instance of GatePredictor is created.
    def __init__(
        self,
        bundle_path: Optional[str] = None,          # Optional path to the model bundle.
        device: Optional[Union[str, torch.device]] = None, # Optional hardware device (CPU/GPU) identifier.
    ):
        # Configure PyTorch to use the requested device, or default to the CPU if none is specified.
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        # Load the model specifications and the raw bundle using the helper function.
        self.spec, self._bundle = load_gate_bundle(bundle_path=bundle_path)
        # Internally construct and load the PyTorch neural network architectures.
        self._models = self._build_models()

    # Private method to instantiate PyTorch model(s) and load their saved weights.
    def _build_models(self) -> List[torch.nn.Module]:
        # Initialize an empty list to store the constructed PyTorch models.
        models: List[torch.nn.Module] = []
        
        # Determine which weights to load: prefer the ensemble of seeds if available.
        if self.spec.seed_state_dicts:
            # Grab all the weight dictionaries from the different training seeds.
            items: Iterable[dict] = self.spec.seed_state_dicts.values()
        else:
            # Fallback to the single best model if multiple seeds aren't present.
            items = [self.spec.best_seed_state_dict]  # type: ignore[list-item]

        # Loop through each set of saved weights.
        for state in items:
            # Instantiate the TextGuidedGate neural network architecture with the exact config.
            model = TextGuidedGate(
                text_dim=self.spec.text_dim,              # Dimensions of text inputs.
                tabular_dim=self.spec.tabular_dim,        # Dimensions of tabular inputs.
                hidden_dim=self.spec.gate_hidden_dim,     # Hidden layer width.
                dropout=self.spec.gate_dropout,           # Dropout probability.
                top_k=self.spec.top_k,                    # Number of top features to attend to.
            ).to(self.device)                             # Move the network to the assigned hardware device.
            
            # Load the actual numerical weights into the architecture.
            # We use _state_dict_from_numpy to ensure all arrays are proper PyTorch Tensors.
            model.load_state_dict(_state_dict_from_numpy(state))
            
            # Switch the model into 'eval' (evaluation) mode. 
            # This turns off dropout and batch normalization to ensure deterministic, stable predictions.
            model.eval()
            
            # Add this fully loaded and prepared model to our list.
            models.append(model)

        # Return the list of operational models.
        return models

    # Method to take a raw dictionary of patient features and generate a final risk probability.
    def predict_proba_from_full(self, full: dict) -> float:
        # Extract all text-based embedding features using the exact column names expected by the model.
        # We enforce a float32 data type to match PyTorch's default floating precision.
        text_vec = np.asarray([full.get(c, 0.0) for c in self.spec.emb_cols], dtype=np.float32)
        
        # Extract all tabular/clinical features using the expected column names.
        tab_vec = np.asarray([full.get(c, 0.0) for c in self.spec.tab_cols], dtype=np.float32)

        # Convert the numpy array for text features into a PyTorch Tensor.
        # We use .unsqueeze(0) to add a 'batch' dimension since the model expects inputs in batches (e.g., shape [1, features]).
        text_t = torch.from_numpy(text_vec).unsqueeze(0).to(self.device)
        
        # Convert the numpy array for tabular features into a PyTorch Tensor, adding the batch dimension, and moving it to the target device.
        tab_t = torch.from_numpy(tab_vec).unsqueeze(0).to(self.device)

        # Initialize a list to hold the predictions from each model in the ensemble.
        probs = []
        
        # Use torch.no_grad() to tell PyTorch not to track gradients. 
        # This saves memory and significantly speeds up inference since we are not training the model.
        with torch.no_grad():
            # Iterate through each model in our loaded ensemble.
            for m in self._models:
                # Pass the text and tabular tensors through the neural network.
                # The model returns the probability (p) and attention weights (_), we discard the weights.
                p, _ = m(text_t, tab_t)
                # Convert the PyTorch Tensor result back to a standard Python float and append it.
                probs.append(float(p.item()))

        # Average the probabilities across all models in the ensemble to get the raw ensemble prediction.
        raw = float(np.mean(probs)) if probs else 0.0
        
        # Pass the raw prediction through our probability calibrator (e.g., isotonic regression or Platt scaling).
        # This ensures the output number is a true probability (e.g., 0.10 means a 10% chance).
        cal = float(self.spec.calibrator.predict(np.array([raw], dtype=np.float32))[0])
        
        # Return the final, calibrated hybrid risk probability.
        return cal
