import torch
import json
import json5
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from normalizer import DataNormalizer

MODEL_P = Path("../data/model/best_model_jit.pt")
SAMPLE_P = Path("prof_1.json")
CONFIG_P = Path("../inputs/model_input_params.jsonc")
NORM_METADATA_P = Path("../data/normalized_profiles/normalization_metadata.json")
DEVICE = "cpu"

with open(CONFIG_P, 'r', encoding='utf-8') as f: config = json5.load(f)
with open(NORM_METADATA_P, 'r', encoding='utf-8') as f: norm_metadata = json.load(f)
with open(SAMPLE_P, 'r', encoding='utf-8') as f: sample_data = json5.load(f)

model_inputs = {}
for seq_name, var_names in config["sequence_types"].items():
    if var_names: model_inputs[seq_name] = torch.stack([torch.tensor(sample_data[v], dtype=torch.float32) for v in var_names], dim=1).unsqueeze(0).to(DEVICE)
if config.get("global_variables"):
    model_inputs["global"] = torch.tensor([sample_data[v] for v in config["global_variables"]], dtype=torch.float32).unsqueeze(0).to(DEVICE)

# --- Load Model, Create Input Masks, and Infer ---
model = torch.jit.load(MODEL_P, map_location=DEVICE).eval()
input_masks = {k: torch.ones(t.shape[:2], dtype=torch.bool, device=DEVICE) for k,t in model_inputs.items() if k != "global"}
with torch.no_grad():
    prediction_norm = model(model_inputs, input_masks if input_masks else None)

# --- Denormalize Predictions ---
denormalized_outputs = {}
for i, target_var_name in enumerate(config["target_variables"]):
    tensor_to_denorm = prediction_norm[0, :, i] 
    denormalized_outputs[target_var_name] = DataNormalizer.denormalize(tensor_to_denorm, norm_metadata, target_var_name)

for var_name, values in denormalized_outputs.items():
    display_values = values.tolist() if isinstance(values, torch.Tensor) else values
