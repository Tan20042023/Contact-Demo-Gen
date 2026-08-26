"""Isolated TPU feasibility configurations for sealed Task 13 inputs.

Input assets are copied locally from GCS by the spot-VM bootstrap. Checkpoints
write directly to a distinct GCS runs prefix through etils.epath and Orbax.
"""

from __future__ import annotations

import os
from pathlib import Path

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders


INPUT_ROOT = Path(os.environ.get("TASK13_TPU_INPUT_ROOT", "/mnt/task13/input_assets"))
RUNS_ROOT = os.environ.get(
    "TASK13_TPU_RUNS_ROOT",
    "gs://euw4/user/tanjunhao/task13_tpu_feasibility/v1/runs",
)
FSDP_DEVICES = int(os.environ.get("TASK13_TPU_FSDP_DEVICES", "4"))
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
    data_root = INPUT_ROOT / "lerobot" / task / condition
    assets_root = INPUT_ROOT / "assets_full" / task / condition
    base = INPUT_ROOT / "checkpoints" / ("pi05_base_action_dim_44" if dual_arm else "pi05_base") / "params"
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
        checkpoint_base_dir=f"{RUNS_ROOT}/checkpoints_{phase}",
        seed=42,
        batch_size=32,
        num_workers=4,
        num_train_steps=100 if smoke else 30_000,
        log_interval=1 if smoke else 100,
        save_interval=100 if smoke else 5_000,
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
