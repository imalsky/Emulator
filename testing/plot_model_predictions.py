# File 2: plot_model_predictions.py

import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Relative path to the model directory (assuming script is in 'testing/' and 'models/' is sibling)
model_dir = (
    Path(__file__).parent.parent / "models" / "trained_model_picaso_transformer_v2"
)  # Change 'trained_model_picaso_transformer' if needed
processed_dir = (
    Path(__file__).parent.parent / "data" / "processed"
)  # Assuming 'data/' is also sibling to 'testing/'


class DataNormalizer:
    @staticmethod
    def denormalize_tensor(
        x: torch.Tensor, method: str, stats: Dict[str, Any]
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        if method in ("none", "bool"):
            return x
        if not stats:
            raise ValueError(f"No stats for denormalization with method '{method}'")

        dtype, device = x.dtype, x.device
        eps = stats.get("epsilon", 1e-9)

        def to_t(val: float) -> torch.Tensor:
            return torch.as_tensor(val, dtype=dtype, device=device)

        if method == "standard":
            return x * to_t(stats["std"]) + to_t(stats["mean"])
        elif method == "log-standard":
            return 10 ** (x * to_t(stats["log_std"]) + to_t(stats["log_mean"]))
        elif method == "signed-log":
            unscaled_log = x * to_t(stats["std"]) + to_t(stats["mean"])
            return torch.sign(unscaled_log) * (10 ** torch.abs(unscaled_log) - 1.0)
        elif method == "log-min-max":
            unscaled = torch.clamp(x, 0, 1) * (
                to_t(stats["max"]) - to_t(stats["min"])
            ) + to_t(stats["min"])
            return 10**unscaled
        elif method == "max-out":
            return x * to_t(stats["max_val"])
        elif method == "iqr":
            return x * to_t(stats["iqr"]) + to_t(stats["median"])
        elif method == "scaled_signed_offset_log":
            ytmp = x * to_t(stats["m"])
            return torch.sign(ytmp) * (10 ** torch.abs(ytmp) - 1)
        elif method == "symlog":
            unscaled = x * to_t(stats["scale_factor"])
            abs_unscaled = torch.abs(unscaled)
            linear_mask = abs_unscaled <= 1.0
            thr = to_t(stats["threshold"])
            y = torch.zeros_like(x)
            y[linear_mask] = unscaled[linear_mask] * thr
            y[~linear_mask] = (
                torch.sign(unscaled[~linear_mask])
                * thr
                * (10 ** (abs_unscaled[~linear_mask] - 1.0))
            )
            return y
        else:
            raise ValueError(f"Unsupported denormalization method '{method}'")


def load_config_and_metadata(model_dir: Path):
    config_path = model_dir / "run_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    metadata_path = processed_dir / "normalization_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return config, metadata


def load_best_jit_model(model_dir: Path):
    jit_files = list(model_dir.glob("best_model_epoch_*_jit.pt"))
    if not jit_files:
        raise FileNotFoundError("No JIT model files found in the model directory.")
    # Select the one with the highest epoch
    best_jit_path = max(
        jit_files, key=lambda p: int(p.stem.split("_epoch_")[1].split("_")[0])
    )
    # Load to CPU to avoid device issues (model was saved on CUDA)
    map_location = torch.device("cpu")
    model = torch.jit.load(best_jit_path, map_location=map_location)
    model.eval()
    return model, best_jit_path


def get_random_test_profiles(num_profiles: int = 2):
    test_shards = sorted(processed_dir.glob("test_shard_*.npz"))
    if not test_shards:
        raise FileNotFoundError("No test shards found.")
    shard = random.choice(test_shards)
    data = np.load(shard)
    seq_inputs = data["sequence_inputs"]
    globals_arr = (
        data["globals"] if "globals" in data else np.empty((len(seq_inputs), 0))
    )
    targets = data["targets"]

    indices = random.sample(range(len(seq_inputs)), num_profiles)
    profiles = []
    for idx in indices:
        profiles.append(
            {
                "sequence": torch.from_numpy(seq_inputs[idx]).float(),
                "global": (
                    torch.from_numpy(globals_arr[idx]).float()
                    if globals_arr.shape[1] > 0
                    else None
                ),
                "target": torch.from_numpy(targets[idx]).float(),
            }
        )
    return profiles


def plot_predictions(model, profiles, metadata, config, plots_dir: Path):
    input_vars = config["data_specification"]["input_variables"]
    target_vars = config["data_specification"]["target_variables"]
    norm_methods = metadata["normalization_methods"]
    per_key_stats = metadata["per_key_stats"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey="row")

    for i, profile in enumerate(profiles):
        seq = profile["sequence"].unsqueeze(0)  # Add batch dim
        glb = profile["global"].unsqueeze(0) if profile["global"] is not None else None
        # Assume no padding; create dummy mask
        seq_mask = torch.zeros(seq.shape[:2], dtype=torch.bool)

        with torch.no_grad():
            pred = model(seq, glb, seq_mask).squeeze(0)

        # Denormalize relevant variables
        pressure = DataNormalizer.denormalize_tensor(
            profile["sequence"][:, input_vars.index("pressure_bar")],
            norm_methods["pressure_bar"],
            per_key_stats["pressure_bar"],
        )
        true_thermal = DataNormalizer.denormalize_tensor(
            profile["target"][:, target_vars.index("net_thermal_flux")],
            norm_methods["net_thermal_flux"],
            per_key_stats["net_thermal_flux"],
        )
        pred_thermal = DataNormalizer.denormalize_tensor(
            pred[:, target_vars.index("net_thermal_flux")],
            norm_methods["net_thermal_flux"],
            per_key_stats["net_thermal_flux"],
        )
        true_reflected = DataNormalizer.denormalize_tensor(
            profile["target"][:, target_vars.index("net_reflected_flux")],
            norm_methods["net_reflected_flux"],
            per_key_stats["net_reflected_flux"],
        )
        pred_reflected = DataNormalizer.denormalize_tensor(
            pred[:, target_vars.index("net_reflected_flux")],
            norm_methods["net_reflected_flux"],
            per_key_stats["net_reflected_flux"],
        )

        # Convert to numpy for plotting
        pressure = pressure.numpy()
        true_thermal = true_thermal.numpy()
        pred_thermal = pred_thermal.numpy()
        true_reflected = true_reflected.numpy()
        pred_reflected = pred_reflected.numpy()

        # Print debug for a couple levels (first 3 levels of first profile)
        if i == 0:
            print("Debug: Sample values for Profile 1 (first 3 levels):")
            for lvl in range(min(3, len(pressure))):
                print(f"Level {lvl}: Pressure={pressure[lvl]:.2e}")
                print(
                    f"  Thermal: True={true_thermal[lvl]:.2e}, Pred={pred_thermal[lvl]:.2e}, Frac Err={np.abs((pred_thermal[lvl] - true_thermal[lvl]) / (np.abs(true_thermal[lvl]) + 1e-10)):.2e}"
                )
                print(
                    f"  Reflected: True={true_reflected[lvl]:.2e}, Pred={pred_reflected[lvl]:.2e}, Frac Err={np.abs((pred_reflected[lvl] - true_reflected[lvl]) / (np.abs(true_reflected[lvl]) + 1e-10)):.2e}"
                )

        # Top row: Flux vs Pressure (symlog x, log y inverted, xlim -1e10 to 1e10)
        axs[0, 0].plot(true_thermal, pressure, label=f"True Thermal {i+1}")
        axs[0, 0].plot(pred_thermal, pressure, "--", label=f"Pred Thermal {i+1}")
        axs[0, 0].set_xscale("symlog")
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_ylim(1e2, 1e-5)
        axs[0, 0].set_xlim(-1e10, 1e10)
        axs[0, 0].set_title("Net Thermal Flux")
        axs[0, 0].set_ylabel("Pressure (bar)")
        axs[0, 0].set_xlabel("Flux")

        axs[0, 1].plot(true_reflected, pressure, label=f"True Reflected {i+1}")
        axs[0, 1].plot(pred_reflected, pressure, "--", label=f"Pred Reflected {i+1}")
        axs[0, 1].set_xscale("symlog")
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_ylim(1e2, 1e-5)
        axs[0, 1].set_xlim(-1e10, 1e10)
        axs[0, 1].set_title("Net Reflected Flux")
        axs[0, 1].set_xlabel("Flux")

        # Bottom row: Fractional error vs Pressure (semilogx for error, log y inverted, xlim 1e-3 to 1e2)
        # Fractional error formula: | (pred - true) / |true| | per pressure level (absolute value on denominator to handle negatives/zeros)
        frac_err_thermal = np.abs(
            (pred_thermal - true_thermal) / (np.abs(true_thermal) + 1)
        )
        frac_err_reflected = np.abs(
            (pred_reflected - true_reflected) / (np.abs(true_reflected) + 1)
        )

        axs[1, 0].semilogx(frac_err_thermal * 100, pressure, label=f"Profile {i+1}")
        axs[1, 0].set_yscale("log")
        axs[1, 0].set_xlim(1e-3, 1e2)
        axs[1, 0].set_ylim(1e2, 1e-5)
        axs[1, 0].set_title("Percent Error - Thermal")
        axs[1, 0].set_ylabel("Pressure (bar)")
        axs[1, 0].set_xlabel("Percent Error")

        axs[1, 1].semilogx(frac_err_reflected * 100, pressure, label=f"Profile {i+1}")
        axs[1, 1].set_yscale("log")
        axs[1, 1].set_xlim(1e-3, 1e2)
        axs[1, 1].set_ylim(1e2, 1e-5)
        axs[1, 1].set_title("Percent Error - Reflected")
        axs[1, 1].set_xlabel("Percent Error")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "model_predictions.png")


if __name__ == "__main__":
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config, metadata = load_config_and_metadata(model_dir)
    model, jit_path = load_best_jit_model(model_dir)
    print(f"Loaded best JIT model: {jit_path}")

    profiles = get_random_test_profiles()
    plot_predictions(model, profiles, metadata, config, plots_dir)
