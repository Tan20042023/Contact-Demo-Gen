from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
from orbax.checkpoint import options as ocp_options

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


class _LocalProcessArrayMetadataValidator:
    """Accept exactly one metadata record in a worker-local TPU shard."""

    def validate_all_array_metadatas(self, array_metadatas):
        if len(array_metadatas) != 1:
            raise ValueError(
                "A worker-local Task13 TPU checkpoint must contain metadata from exactly one process; "
                f"found {len(array_metadatas)}."
            )


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tpu_multihost = os.environ.get("TASK13_TPU_MULTIHOST") == "1"
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        keep_period=keep_period,
        create=False,
        async_options=ocp.AsyncOptions(timeout_secs=7200),
    )
    # TPU v6e workers have independent local disks.  An Orbax manager must
    # therefore be a *single-process* manager on each host: ``primary_host``
    # alone does not isolate the manager, and otherwise its finalize thread
    # still participates in the four-host global barrier while looking for a
    # directory that only exists on its local disk.  The caller transports the
    # four independently completed contributions to GCS and commits them with
    # the explicit Task13 manifest protocol.
    if tpu_multihost:
        options = dataclasses.replace(
            options,
            multiprocessing_options=ocp_options.MultiprocessingOptions(
                primary_host=None,
                barrier_sync_key_prefix="task13-local-orbax-",
            ),
            # Orbax >=0.11.24 scopes its asynchronous directory and signal
            # setup to the local process in this mode.  This is required when
            # every v6e worker writes the same logical step to a different
            # local filesystem before the explicit GCS shard transport.
            enable_per_process_directory_creation=True,
        )

    pytree_kwargs = {"array_metadata_validator": _LocalProcessArrayMetadataValidator()} if tpu_multihost else {}
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(**pytree_kwargs),
            "params": ocp.PyTreeCheckpointHandler(**pytree_kwargs),
        },
        options=options,
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        # The Task13 v6e-16 path uses a separate local checkpoint root on
        # every TPU VM.  Each local Orbax manager is therefore a primary for
        # its own contribution; restricting this callback to global process 0
        # leaves the remaining managers waiting for an absent asset commit.
        if os.environ.get("TASK13_TPU_MULTIHOST") == "1":
            args.callback(directory)
        elif jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
