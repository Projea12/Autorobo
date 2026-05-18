"""
export_model.py — Model Export for Deployment (Phase 15-16)

Exports trained RL policies to deployment-ready formats:
  - ONNX: for hardware-accelerated inference on edge devices
  - TorchScript: for C++ / ROS2 node deployment
  - TensorRT: for NVIDIA Jetson deployment (optional)

Also bundles the full config and normalisation stats so the
inference runtime can reconstruct the exact observation pipeline.

Usage:
    python scripts/export_model.py model=navigation
    python scripts/export_model.py model=grasping export.format=onnx
    python scripts/export_model.py model=full_pipeline export.device=cuda
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

ExportFormat = Literal["onnx", "torchscript", "tensorrt"]


def load_policy(model_path: Path, algorithm: str):
    """
    Load a trained SB3 policy for export.

    Args:
        model_path: Path to .zip checkpoint.
        algorithm: "PPO" or "SAC".

    Returns:
        The policy network as a torch.nn.Module.

    TODO: Phase 15 — use SB3 .load(), extract .policy attribute,
    set to eval mode, move to target device.
    """
    raise NotImplementedError(
        f"TODO: Phase 15 — load {algorithm} from {model_path}, "
        "extract policy.mlp_extractor + policy.action_net."
    )


def export_onnx(policy, dummy_input: Any, output_path: Path) -> Path:
    """
    Export policy to ONNX format.

    Args:
        policy: torch.nn.Module in eval mode.
        dummy_input: Representative input tensor(s) for tracing.
        output_path: Where to write the .onnx file.

    Returns:
        Path to the exported file.

    TODO: Phase 15 — torch.onnx.export() with opset_version=17,
    dynamic_axes for batch dimension, then onnx.checker.check_model().
    """
    raise NotImplementedError(
        f"TODO: Phase 15 — torch.onnx.export(policy, dummy_input, {output_path})."
    )


def export_torchscript(policy, dummy_input: Any, output_path: Path) -> Path:
    """
    Export policy to TorchScript via tracing.

    Args:
        policy: torch.nn.Module in eval mode.
        dummy_input: Representative input tensor(s).
        output_path: Where to write the .pt file.

    Returns:
        Path to the exported file.

    TODO: Phase 15 — torch.jit.trace(policy, dummy_input),
    scripted_module.save(output_path). Verify with torch.jit.load().
    """
    raise NotImplementedError(
        f"TODO: Phase 15 — torch.jit.trace and save to {output_path}."
    )


def export_tensorrt(onnx_path: Path, output_path: Path, precision: str = "fp16") -> Path:
    """
    Convert ONNX model to TensorRT engine for Jetson deployment.

    Args:
        onnx_path: Path to the ONNX model.
        output_path: Where to write the .trt engine.
        precision: "fp32", "fp16", or "int8".

    Returns:
        Path to the TensorRT engine.

    TODO: Phase 16 — use tensorrt Python API or trtexec CLI,
    requires NVIDIA GPU and TensorRT installation on target device.
    """
    raise NotImplementedError(
        "TODO: Phase 16 — implement TensorRT export. "
        "Run on target Jetson device: trtexec --onnx=model.onnx --saveEngine=model.trt"
    )


def bundle_deployment_package(
    model_path: Path,
    cfg: DictConfig,
    export_dir: Path,
) -> None:
    """
    Create a self-contained deployment bundle.

    Includes:
      - Exported model (ONNX or TorchScript)
      - Observation normalisation stats (mean, std)
      - Full config YAML
      - Model card (metadata JSON)

    Args:
        model_path: Path to exported model file.
        cfg: Hydra config used during training.
        export_dir: Output directory for the bundle.

    TODO: Phase 15 — copy model, write config.yaml, normalisation.json,
    model_card.json with training metadata.
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = export_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    log.info(f"Config saved to {config_path}")

    # Save model card
    model_card: dict[str, Any] = {
        "project": cfg.project.name,
        "version": cfg.project.version,
        "phase": cfg.project.phase,
        "algorithm": cfg.training.algorithm,
        "total_timesteps": cfg.training.total_timesteps,
        "export_format": "onnx",
    }
    card_path = export_dir / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    log.info(f"Model card saved to {card_path}")

    raise NotImplementedError(
        "TODO: Phase 15 — copy exported model file, extract and save "
        "VecNormalize statistics (obs mean/std) as normalisation.json."
    )


@hydra.main(config_path="../configs/hydra", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main model export entry point.

    Args:
        cfg: Composed Hydra config. Key overrides:
          - export.model: navigation | grasping
          - export.format: onnx | torchscript | tensorrt
          - export.output_dir: where to write exported model
          - export.device: cpu | cuda:0 | mps
    """
    log.info("=== AutoRobo Model Export (Phase 15) ===")

    model_type = cfg.get("export", {}).get("model", "navigation")
    fmt: ExportFormat = cfg.get("export", {}).get("format", "onnx")
    output_dir = Path(cfg.project.model_dir) / "exported" / model_type

    log.info(f"Exporting {model_type} model as {fmt} to {output_dir}")

    raise NotImplementedError(
        f"TODO: Phase 15 — load {model_type} policy, create dummy input, "
        f"call export_{fmt}(), then bundle_deployment_package()."
    )


if __name__ == "__main__":
    main()
