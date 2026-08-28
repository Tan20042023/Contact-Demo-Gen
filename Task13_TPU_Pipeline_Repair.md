# Task13 TPU pipeline repair record

**Status: training paused.** This document is the launch contract for the TPU
side branch after the 2026-08-28 checkpoint investigation. It supersedes any
informal instruction to repeatedly retry the 101-step smoke.

## What was demonstrated

- A v6e-16 has four independent worker filesystems, four JAX processes and 16
  global devices.
- Four-worker JAX initialization, the local sealed LeRobot inputs and a real
  Hammer `nominal_src` 100-step update loop completed (about 1.3 step/s after
  compilation).
- The source archive, input cache and direct worker bootstrap path work without
  any 5090 runtime dependency.

## What is *not* demonstrated

No all-worker checkpoint has saved, committed to GCS, been materialized on a
fresh slice and restored into a real subsequent update. Therefore neither P1
nor P2 is launchable. The current source release is a candidate, not a
qualified training release.

## Failures and their corrections

| Failure | Evidence | Permanent correction |
| --- | --- | --- |
| A per-worker local Orbax design was exercised in the 101-step smoke. | Workers blocked in Orbax save/finalize; no `COMMITTED.json` or `LATEST.json`. | Use one native shared-GCS Orbax root for all processes; a standalone checkpoint-contract gate is mandatory before any normal training. |
| Orbax 0.11.13 lacked the per-process-directory option; 0.11.24 calls a JAX 0.5.3-missing monitoring API during save. | 0.11.13 rejected the option; 0.11.24 failed with `jax.monitoring.record_scalar` absent. | Pin Orbax 0.11.23: it is the first release with the needed per-process signalling, declares `jax >= 0.5.0`, and does not make that monitoring call. Still qualify it with the gate before relying on it. |
| A one-worker diagnostic initialized JAX and retained the TPU runtime after timeout. | Later bootstrap reported the TPU already in use by that diagnostic PID. | Bootstrap must not enumerate TPU devices; only an all-worker preflight may initialize JAX. No ad-hoc JAX/Orbax probes on a single worker. |
| A retry targeted one worker after a distributed initialization failure. | That worker waited for the missing peers. | Bootstrap and every JAX-initializing check are all-four-worker operations; individual retries are limited to non-JAX file transfer/inspection. |
| Partial GCS prefixes were too easy to reuse. | R3 has diagnostic worker-0 data but no commit marker. | A new initial launch rejects every non-empty run prefix; resume requires `LATEST.json`. |
| A long smoke was treated as a checkpoint test. | Training passed but the recovery condition did not. | The contract test has a separate success artifact and must prove save, four manifests, commit, clean materialization, restore, and one post-restore update. |

## Repaired launch state machine

```text
READY + HEALTHY
  -> all-worker bootstrap (files, venv, inputs only; no TPU runtime)
  -> inspect four READY.json records
  -> all-worker preflight (the only normal place that initializes JAX)
  -> checkpoint-contract run (100-step smoke)
  -> clean all-worker resume + one update
  -> CHECKPOINT_CONTRACT_PASS.json
  -> formal technical run
```

Every transition fails closed. A failed contract run preserves its uniquely
named diagnostic prefix, records worker PIDs/log locations, and returns to the
analysis step; it is not automatically retried with a new checkpoint design.

## Exact acceptance criterion for the checkpoint contract

For one immutable source SHA and the same 4-host v6e-16 topology:

1. A 100-step smoke finishes one native Orbax step at the same shared GCS
   checkpoint root on all four processes.
2. The finalized step has non-empty GCS objects only after all four calls to
   `wait_until_finished()` return.
3. A single `COMMITTED.json` and `LATEST.json` identify the same post-update
   step, native checkpoint URI, `process_count=4` and source SHA.
4. A clean four-worker resume opens that same GCS root, restores, and completes
   one actual optimizer update.
5. `task13_tpu_v6e16_verify_contract.ps1` writes
   `CHECKPOINT_CONTRACT_PASS.json` only after it has
   checked all four conditions. It includes source SHA, topology, config, step,
   GCS prefix and UTC timestamp.

`task13_tpu_v6e16_launch.ps1` now has two purposes. The default
`checkpoint-contract` purpose accepts only `task13_tpu_smoke_*`; `formal`
accepts only `task13_tpu_technical_*` and requires a readable proof URI.

## Operational boundaries

- Never create, delete or reconfigure a TPU while diagnosing a checkpoint.
- Never use 5090 as a training, data, or checkpoint dependency. It may only be
  an authenticated control endpoint when the operator chooses it.
- Never run a one-worker command that imports JAX and enumerates devices.
- Never reuse partial output prefixes or treat a log line as recovery proof.
- Do not edit GPU/A100 code paths or artifacts for this TPU side branch.
