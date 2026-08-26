# Task 13 TPU Feasibility Runbook

Version: v1.0  
Date: 2026-08-26  
Status: DRAFT — requires plan-author approval before any TPU environment, copy, job, or training action.

## Purpose and isolation

This is an isolated technical feasibility branch, not a Task 13 hardware-allocation change. The formal GPU plan remains unchanged: the complete Assembly block remains assigned to A100 80 GB when capacity is available, and the complete Hammer block remains assigned to RTX 5090. TPU outputs, losses, checkpoints, and evaluations are not scientific results and cannot count among the 26 formal models.

Only TPU-branch outputs may be written below:

~~~
/home/tanjunhao/Ego/dexjoco/outputs/task13_tpu_feasibility/v1
~~~

The following canonical mainline root is read-only:

~~~
/home/tanjunhao/Ego/dexjoco/outputs/task13_policy_matrix/v1
~~~

Any TPU-side inputs are derived read-only copies. The 5090 remains canonical owner of all Task 13 datasets, norm assets, and base checkpoints.

## Read-only audit record

Audited on lab-server-5090 on 2026-08-26:

- Repository: /home/tanjunhao/Ego/dexjoco at HEAD 8d23b0fab23b17a58c4b55f3942e17013aaf8267.
- The worktree is dirty: six tracked files are modified and many project files are untracked, including Task 13 material. No cleanup, checkout, reset, or write was performed.
- Sealed dataset/norm evidence exists in outputs/task13_policy_matrix/v1/manifests/dataset_and_norm_assets_seal.json and norm_stats_manifest.json. The latter is PASS and explicitly prohibits Task 13 norm-stat recomputation.
- Ten datasets and ten norm assets are sealed read-only. Assembly uses 44-D actions, 46-D state, and real ego/wrist_left/wrist_right cameras. Hammer uses 22-D actions, 23-D state, and real front/wrist cameras. Hammer's masked zero right-wrist compatibility input must remain.
- The base roots are checkpoints/pi05_base and checkpoints/pi05_base_action_dim_44. A path-and-size listing of both has SHA-256 5fd9667c3d4a664ba689a1ed4ac9b987947b0c751b1cc8ce083bdd9fe90f6eab. Phase 2 must make stronger per-file path/size/SHA-256 source and destination manifests.

Code review independently confirmed:

1. openpi/pyproject.toml pins JAX CUDA 0.5.3, Flax 0.10.2, and Orbax 0.11.13. TPU requires a separate environment and must not alter the GPU environment or lockfile.
2. train.py requires global batch 32 to divide exactly by jax.device_count().
3. sharding.py creates a mesh of (device_count/fsdp_devices, fsdp_devices). FSDP axis must divide the actual device count. FSDP 1 replicates parameters and does not combine HBM.
4. data_loader.py raises NotImplementedError when jax.process_count() exceeds 1. This branch permits only a single host and a single JAX process.
5. model.restore_params gives Orbax a current-device target sharding, so TPU-to-GPU restore is plausible but unproven until a real round trip.
6. train.py waits for checkpoint_manager.wait_until_finished(), which is mandatory before checkpoint hashing or transfer.

No TPU has been requested or allocated. Accordingly, this audit makes no claim about accelerator type, topology, HBM, quota, network, or capacity. The currently configured TPU SSH endpoints did not answer on port 22; this is not evidence of a TPU failure and must not be worked around before allocation.

## Non-negotiable rules

- Global batch remains 32. No gradient accumulation, batch reduction, precision reduction, camera removal, data regeneration, norm recomputation, or checkpoint format conversion.
- Keep pi0.5 LoRA, optimizer, schedule, precision, action semantics, datasets, and norm assets unchanged.
- A TPU FSDP topology is a separate technical experiment; it never revises the frozen GPU recipe.
- Do not touch, reserve, stop, restart, clean, or otherwise affect GPU/A100 processes.
- Do not upgrade the JAX/Flax/Orbax core stack or rewrite the multi-host loader without a new adjudication.

## Gates

| Gate | Evidence | Permission granted |
|---|---|---|
| G0 | Explicit approval of this runbook | Phase 0 read-only preflight |
| G1 | Single-host resource preflight plus isolated environment plan | Create environment and derived copies |
| G2 | TPU software probe and asset manifest PASS | Compile and run one training step |
| G3 | Hammer one-step PASS | Hammer 100-step smoke |
| G4 | Hammer 100-step PASS | Assembly one-step then 100-step smoke |
| G5 | TPU checkpoint plus GPU restore PASS | Throughput estimate and verdict |
| G6 | New plan-author/Claude decision | Any formal TPU allocation/training |

G0 through G5 do not authorize formal Task 13 TPU training.

## Phase 0: future allocated-resource preflight

After TPU allocation and only after G0, use read-only commands such as:

~~~
hostname; date -Is; uname -a
nproc; free -h; df -h / /home
python -c "import jax; print(jax.__version__); print(jax.process_count(), jax.process_index(), jax.device_count()); print(jax.devices())"
gcloud compute tpus tpu-vm describe <VM> --zone <ZONE> --format=json
~~~

Record accelerator type, slice topology, chips/devices, hosts/processes, per-chip HBM, host RAM, disk, zone, queue/reservation/quota, and a project-scoped route to the 5090. Confirm that any GPU used later for Phase 4 is idle without disturbing it.

PASS requires exactly one host, process_count=1, a device count in {1,2,4,8,16,32} so it divides batch 32, sufficient disk, identified hardware, usable quota, and a verified route to the canonical owner. Otherwise stop and report. Do not use a multi-host Pod, guess a model from an alias, or rewrite the data loader.

## Phase 1: isolated TPU environment

After G1, create an environment such as openpi-task13-tpu on the TPU VM only. Preserve the requested baseline versions JAX 0.5.3, Flax 0.10.2, Orbax 0.11.13 and record exact Python, NumPy, Torch, LeRobot, PyAV, JAX, jaxlib/libtpu, Flax, and Orbax versions and installation source in the TPU output root. Replace only the accelerator backend; do not edit openpi/pyproject.toml or the GPU lockfile.

Validate jax.devices, host-to-device transfer, an all-device collective, a disposable **local** Orbax save/restore followed by GCS upload/download restore, and Torch/LeRobot import without CUDA initialization. If the allocated TPU cannot run JAX 0.5.3 without an unadjudicated core-stack upgrade, stop with a compatibility report.

## Phase 2: derived input gate

After G1, copy only Hammer nominal_src, its matching sealed norm asset, the original 22-D base, and their source manifests into a temporary TPU-branch incoming directory. Build source and destination manifests containing each relative path, byte size, and SHA-256. They must match exactly; verify the source again after transfer. Then label the verified destination derived_read_only_copy and make it read-only.

Complete a full-sample LeRobot load and verify Hammer's 22-D action, 23-D state, real front/wrist inputs, expected norm SHA, and finite base parameter tree. Copy Assembly only after Hammer passes, and then use the canonical 44-D base plus all three real cameras.

## Phase 3: minimal training smoke

For actual device count N, register only FSDP axes that divide N. The fixed capacity sequence is FSDP 1, 2, 4, then 8; record non-divisors as NOT_APPLICABLE. Do not skip a legal axis based on loss or method results. Stop at the smallest axis that completes the gate; that selection is purely a TPU topology finding.

Use a TPU-only overlay configuration with the identical model, data, norm asset, global batch 32, optimizer, schedule, precision, camera semantics, seed 42, and 100 steps. Only the isolated output root and registered FSDP axis differ. The terminal step-99 Orbax checkpoint is required.

First run hammer_nail / nominal_src / seed42:

1. compile and complete one step;
2. require finite loss, gradient norm, parameter norm, correct norm asset, no dropped samples, and no camera/mask semantic error;
3. then complete 100 steps unchanged;
4. record compile time, post-compile step times, HBM/chip, host RAM, input wait, devices, FSDP axis, losses and norms;
5. wait for Orbax completion and produce a complete checkpoint-tree SHA-256 manifest.

Only after Hammer 100-step PASS, run bimanual_assembly / nominal_src / seed42 using the 44-D base and three real cameras. This is not a replacement for the pending A100 five-condition smoke.

### TPU spot checkpoint continuity policy

For any separately approved TPU technical run longer than the 100-step smoke, use a TPU-only save_interval of 5000. This does not alter the frozen GPU/A100 recipe or authorize formal TPU training. It creates six recovery points across a 30,000-step run and bounds a single spot-preemption loss to fewer than 5,000 steps.

During the 100-step smoke, measure steady-state step time and checkpoint save time. Before any longer TPU run, stop for plan-author review if 5,000 steps would exceed approximately 30 minutes of wall time, checkpoint writing materially stalls training, or the existing retention policy would discard a usable recovery point.

Orbax 0.11.13 cannot initialize its empty temporary checkpoint prefix directly on GCS: GCS object prefixes are not real directories, so Orbax fails before writing state. Therefore the approved TPU-only implementation writes each checkpoint locally, where Orbax atomically publishes the numbered step directory, then a sidecar copies that completed directory to the distinct GCS runs prefix. It compares total file bytes and writes `UPLOAD_COMPLETE` only after a successful copy. Recovery must use only a GCS step containing that marker. Never write under immutable `input_assets`; do not upgrade the core stack merely to alter this behavior.

This path was preflighted on 2026-08-26 with a synthetic two-integer Orbax checkpoint: local save, GCS upload, fresh download, and restore all passed (1,411 checkpoint bytes). The production 100-step and GPU restore gates remain unrun and require separate approval.

## Phase 4: TPU to GPU checkpoint round trip

For the first completed TPU smoke checkpoint, wait for asynchronous save completion, hash the whole tree, and copy it unchanged with its configuration/provenance/norm asset to a quarantine path on the 5090. Recheck that a chosen GPU is idle. Restore through the existing JAX policy-server path; do not convert formats.

Compare configuration, parameter-tree paths, shapes, dtypes, finiteness, and norm SHA. Run one frozen inference input under TPU-side and GPU-side restoration, requiring equal shapes and finite values; report max/mean difference, without demanding bitwise equality. A single clearly labelled non-formal GPU episode is optional interface evidence only and must not affect the mainline.

## Phase 5: report

Use measured values to estimate 30k-step, 13-model, and 26-model wall time, including compilation, checkpointing, input/video decode, quota/scheduling, and cost. Compare with the wait for A100 without changing the current allocation.

Verdict is PASS only when Hammer and Assembly each finish 100 steps, a checkpoint restores on GPU, and measured throughput/scheduling has practical value. CONDITIONAL covers a documented controlled decision needing adjudication. FAIL covers single-host/topology failure, unsupported stack, unavailable capacity, failed round trip, or no schedule value. Every verdict is only a recommendation; a separate decision is mandatory before formal TPU training.

## Hard stops

Immediately stop and report if a sealed input or scientific configuration would change; a GPU/A100 process would be affected; TPU is multi-host/multi-process; core-stack upgrade is needed; smoke produces NaN/Inf, data loss, wrong camera semantics, or non-finite state; GPU restore fails; or success requires lower batch/precision, fewer cameras, gradient accumulation, or a format conversion.

## Current state

This document is submitted for review only. No TPU has been requested, no dependency/environment has been created, and no TPU job, training smoke, GPU/A100 action, or mainline modification has occurred. The sealed computational input set (30,683,851,239 bytes) has been uploaded from the 5090 to `gs://euw4/user/tanjunhao/task13_tpu_feasibility/v1/input_assets/`; no experimental documents or repository copy were uploaded.

For the tutorial-compatible runtime identity, leave `--service-account` unspecified when creating a TPU VM. Cloud TPU then attaches the existing Compute Engine default service account `184047521632-compute@developer.gserviceaccount.com`. It is already a project Editor in `whyu01` and can access the existing bucket under the current project policy. The separate `tpu-bucket-writer` identity is not part of this plan.
