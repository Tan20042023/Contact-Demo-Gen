# Task 13 TPU Spot Prelaunch Checklist

Status: active preparation only. This checklist does not authorize a TPU VM, dependency install, TPU job, smoke, or formal Task 13 training.

## Scientific and ownership boundaries

- GPU/A100 remains the formal Task 13 mainline. Assembly stays allocated to A100 80 GB when capacity is available; Hammer stays allocated to RTX 5090.
- TPU work is a separately labelled technical feasibility branch. TPU loss, checkpoints, and evaluations do not count among the 26 formal models.
- Do not alter sealed data, norm stats, base checkpoints, model semantics, global batch 32, precision, camera meanings, seed protocol, or Task 13 conditions.
- Do not modify, clean, restart, reserve, or otherwise affect GPU/A100 workloads.

## Source code delivery

- GitHub repository: https://github.com/Tan20042023/Contact-Demo-Gen.git
- TPU branch: task13-tpu-feasibility-prep
- Current TPU preparation commit: 5d879a4 Prepare-default-TPU-runtime-preflight
- The branch is based on DexJoCo HEAD 8d23b0fab23b17a58c4b55f3942e17013aaf8267, not on the dirty 5090 worktree.
- The branch contains only:
  - isolated Task 13 TPU configs;
  - GCS-compatible checkpoint path handling through etils.epath;
  - TPU technical-run save interval 5000;
  - no copied GPU-mainline dirty changes.

Before using this branch on a TPU, review the branch diff and record its commit SHA in the TPU preflight.

## GCS asset layout

Input cache prefix:

~~~
gs://euw4/user/tanjunhao/task13_tpu_feasibility/v1/input_assets/
~~~

Required contents:

- checkpoints/pi05_base
- checkpoints/pi05_base_action_dim_44
- lerobot/bimanual_assembly/{nominal_src,repeat,visual,contact,combined}
- lerobot/hammer_nail/{nominal_src,repeat,visual,contact,combined}
- assets_full/bimanual_assembly/{nominal_src,repeat,visual,contact,combined}
- assets_full/hammer_nail/{nominal_src,repeat,visual,contact,combined}

These are derived copies. The canonical source remains the 5090. Do not upload documents, full repository copies, credentials, or experiment reports to this prefix.

Reserve a separate writable output prefix before TPU allocation:

~~~
gs://euw4/user/tanjunhao/task13_tpu_feasibility/v1/runs/
~~~

Never write checkpoint, training state, logs, or temporary test objects under input_assets.

## GCP access readiness

- Verified on 2026-08-26: project `whyu01` has Cloud TPU API enabled; the active human account is `tjh20042024@gmail.com`.
- The TPU request must omit `--service-account`, so the VM uses the existing Compute Engine default service account `184047521632-compute@developer.gserviceaccount.com`.
- That default identity is already a project Editor and can access the existing `euw4` bucket through the project's current bucket policy. This is the tutorial-compatible path and does not require any new IAM grant by the requester.
- `tpu-bucket-writer@whyu01.iam.gserviceaccount.com` and its prefix-scoped bindings are unused by this plan. Leave them unchanged unless the project owner later chooses a separately reviewed least-privilege migration.

- Keep the gcloud project explicit in all TPU commands (`--project=whyu01`); the current 5090 gcloud configuration has no default project set.
- The future TPU preflight must validate the default runtime identity's GCS read/list/write access and the local-checkpoint-to-GCS sync behavior before any formal run.

## Spot checkpoint policy

- TPU technical runs longer than 100 steps use save_interval 5000.
- Keep period remains 30000 unless a separate review changes it.
- The 100-step smoke still saves the terminal step-99 checkpoint.
- After the smoke, calculate 5000 times steady-state step time. If it exceeds about 30 minutes, checkpoint save time materially stalls training, or retention cannot keep a usable recovery checkpoint, stop for review before a longer run.
- Checkpoint state writes locally first. A TPU-only sidecar syncs each atomically completed numeric step to the GCS runs prefix, verifies byte totals, then writes its `UPLOAD_COMPLETE` marker. Resume only from marked steps so a preempted spot VM never uses a partial upload.

## Target TPU request

Preferred request: spot TPU v6e-4 in europe-west4-a. It is a single-VM four-chip configuration, compatible with global batch 32 and the pre-registered FSDP 2/4 tests.

Read-only verification on 2026-08-26 confirmed that both `europe-west4-a` and `us-east1-d` currently advertise `v6e-4` as an accelerator type in project `whyu01`. This confirms the shape is supported, not that spot capacity is currently available.

Fallback order:

1. v6e-4 in us-east1-d
2. v6e-8 in either v6e zone only if v6e-4 is insufficient
3. on-demand v4-8 only after a separate compatibility review

Do not request or run a complete 64-chip or 320-chip slice with the current loader. Multi-host jax.process_count is unsupported and a 64-device topology does not satisfy global batch 32 divisibility.

## Before requesting a TPU

- [ ] Verify every input asset upload completes and source/destination inventory is checked.
- [ ] Create or select a TPU VM service account in GCP project whyu01.
- [ ] Grant the service account read access to input_assets and scoped write access to runs; do not use the 5090 personal login as the long-term recovery mechanism.
- [ ] Confirm the requested TPU VM can use the service account.
- [ ] Prepare a local-only bootstrap script that clones the pinned branch, activates the service identity, creates the isolated environment, downloads only needed input assets, and runs preflight.
- [ ] Keep any service-account key out of Git, GCS, documents, and chat. Prefer an attached service account over a long-lived JSON key.
- [ ] Request at least 200 GB VM boot disk for local input cache, environment, compilation cache, and temporary checkpoint space.

## Immediately after allocation

- [ ] Use gcloud compute tpus tpu-vm ssh once before relying on bare SSH; record the new IP and update the SSH alias.
- [ ] Record accelerator type, slice topology, host count, JAX process count, device count, HBM, host RAM, free disk, runtime version, and quota/reservation state.
- [ ] Require one host, jax.process_count equal to 1, and a JAX device count dividing 32.
- [ ] Create an isolated TPU environment. Never alter the GPU openpi environment or its CUDA JAX installation.
- [ ] Verify JAX 0.5.3, Flax 0.10.2, Orbax 0.11.13, libtpu, Torch, LeRobot, and PyAV compatibility. Stop if a core stack upgrade is required.
- [ ] Verify GCS read/write from both gcloud and Python application credentials.
- [ ] Verify JAX devices, a small collective, host-to-device transfer, and disposable local Orbax save → GCS upload → fresh download/restore.
- [ ] Download inputs locally, compare every path/size/SHA-256 to the 5090 inventory, then mark the local copy read-only.

## Smoke and portability gates

- [ ] Run Hammer nominal_src seed42 compile plus one step at the first legal FSDP axis.
- [ ] Require finite loss, gradient norm, parameter norm, correct norm asset, no sample loss, and correct camera semantics.
- [ ] Complete Hammer 100 steps and wait for Orbax save completion.
- [ ] Hash the full step-99 checkpoint tree.
- [ ] Run Assembly nominal_src seed42 under the same rules, with 44-D base and all three real cameras.
- [ ] Restore the first TPU checkpoint through the existing GPU JAX policy-server path on a verified idle GPU.
- [ ] Compare parameter tree, shapes, dtypes, finiteness, norm SHA, and frozen inference output shapes.
- [ ] Report measured compile time, step time, HBM, host RAM, input wait, checkpoint time, spot-recovery behavior, and 30k-step ETA.

## Hard stops

Stop and report if the slice is multi-host, device count cannot divide batch 32, data loader rewrite is required, sealed input changes, a core stack upgrade is needed, any non-finite/data/camera error occurs, GCS checkpoint restore fails, GPU/A100 mainline would be affected, or success needs reduced batch/precision/cameras, gradient accumulation, or checkpoint format conversion.
