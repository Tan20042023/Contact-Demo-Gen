"""Isolated TPU feasibility configurations for sealed Task 13 inputs."""

from __future__ import annotations

import os
from pathlib import Path

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


INPUT_ROOT = Path(os.environ.get("TASK13_TPU_INPUT_ROOT", "/mnt/task13/input_assets"))
LOCAL_RUNS_ROOT = Path(os.environ.get("TASK13_TPU_LOCAL_RUNS_ROOT", "/home/tanjunhao/task13_local_runs"))
FSDP_DEVICES = int(os.environ.get("TASK13_TPU_FSDP_DEVICES", "4"))
NUM_WORKERS = int(os.environ.get("TASK13_TPU_NUM_WORKERS", "0"))
CONDS = ("nominal_src", "repeat", "visual", "contact", "combined")


def _model(action_dim: int | None):
    kwargs = {
        "pi05": True,
        "action_horizon": 30,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "max_token_len": 250,
    }
    if action_dim is not None:
        kwargs["action_dim"] = action_dim
    return pi0_config.Pi0Config(**kwargs)


def _make_config(task: str, condition: str, *, smoke: bool):
    # Deferred import avoids a circular import during config registration.
    from .config import AssetsConfig, DataConfig, DualArmDataConfig, SingleArmDataConfig, TrainConfig

    dual_arm = task == "bimanual_assembly"
    model = _model(44 if dual_arm else None)
    # The sealed GCS input bundle preserves the Task13 source layout.  Keep
    # these paths explicit so LeRobot opens the local v3 metadata instead of
    # falling back to the Hub for the placeholder repo ID (``local_repo``).
    data_root = INPUT_ROOT / "datasets" / "task13" / "v1" / "lerobot" / task / condition
    assets_root = INPUT_ROOT / "outputs" / "task13_policy_matrix" / "v1" / "assets_full" / task / condition
    base_name = "pi05_base_action_dim_44" if dual_arm else "pi05_base"
    base = INPUT_ROOT / "checkpoints" / base_name / "params"
    phase = "smoke" if smoke else "technical"
    name = f"task13_tpu_{phase}_{task}_{condition}"
    data_common = {
        "root": data_root,
        "repo_id": "local_repo",
        "assets": AssetsConfig(assets_dir=str(assets_root), asset_id=f"{task}/local_repo"),
        "base_config": DataConfig(prompt_from_task=True),
    }
    data = DualArmDataConfig(**data_common) if dual_arm else SingleArmDataConfig(**data_common)
    return TrainConfig(
        name=name,
        exp_name=f"{task}__{condition}__seed42__{phase}",
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(base)),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=str(assets_root),
        checkpoint_base_dir=str(LOCAL_RUNS_ROOT / f"checkpoints_{phase}"),
        seed=42,
        batch_size=32,
        num_workers=NUM_WORKERS,
        num_train_steps=100 if smoke else 30_000,
        log_interval=1 if smoke else 100,
        # Start with a denser Spot-recovery interval; P2 may change this only
        # after P1 measures the all-worker save and upload cost.
        save_interval=100 if smoke else 2_500,
        keep_period=30_000,
        wandb_enabled=False,
        fsdp_devices=FSDP_DEVICES,
    )


def get_task13_tpu_configs():
    return [
        _make_config(task, condition, smoke=smoke)
        for smoke in (True, False)
        for task in ("bimanual_assembly", "hammer_nail")
        for condition in CONDS
    ]
