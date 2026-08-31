# Task13 TPU formal campaign Runbook

**Status:** formal topology selected; v6e-8 recovery qualification pending.

**Scope:** TPU side branch only. GPU/A100 remains the canonical main experiment.

**Campaign:** 26 predefined 30,000-step cells, global batch 32.

## 1. Architecture decision

The primary profile is **26 independent Spot v6e-8 slices in us-east1-d**.
Each v6e-8 is a single VM with eight chips. The pool requests at most 208 chips,
below the project limit of 320 Spot chips. Slots are asynchronous: the campaign
starts with the first healthy slice and never waits for all 26.

Why this is preferred:

- Task13 cells are independent and have no cross-cell collective.
- All 26 cells can be in the first logical wave.
- Batch 32 gives four samples/chip, a better utilization point than v6e-16's
  two samples/chip while retaining four-device FSDP groups.
- One VM per cell removes the v6e-16 four-VM recovery surface.
- A preemption affects one cell; its other 25 peers continue.
- 26 v6e-8 use 208 chips. There is no v6e-2. v6e-1 is intended primarily for
  testing and is not suitable for this model's four-device FSDP formal path.

The profile ranking is:

1. `v6e8x26`: primary, 208 chips, one logical wave, new one-process proof needed.
2. `v6e4x26`: availability fallback, 104 chips, approximately twice the cell
   wall time, new one-process/four-device proof needed.
3. `v6e16x13`: qualified recovery fallback, 208 chips, two logical waves,
   four VM processes per cell.

Do not mix profiles after the first formal cell starts. Process-count-dependent
data sharding and numerical order can differ even with the same global batch.
A profile switch is allowed only before formal launch, with a fresh campaign
state and a topology-specific recovery proof.

Official v6e shapes and VM types:
https://docs.cloud.google.com/tpu/docs/v6e.

## 2. Frozen experiment matrix

P2 runs first: two tasks × five conditions × seed 42 = 10 cells. P3 then runs
all non-nominal conditions for seeds 43 and 44 = 16 cells. P3 is predefined,
not selected based on P2 curves.

| Priority | Phase | Seeds | Tasks | Conditions | Cells |
| --- | --- | --- | --- | --- | ---: |
| 1 | P2 | 42 | Hammer, bimanual assembly | nominal, repeat, visual, contact, combined | 10 |
| 2 | P3 | 43, 44 | Hammer, bimanual assembly | repeat, visual, contact, combined | 16 |

Every cell freezes:

- 30,000 optimizer steps;
- global batch 32;
- the sealed data/base/norm/camera/action schema;
- four-device FSDP groups;
- source SHA and input bytes listed in the operations guide;
- v6e-8 checkpoint interval 1,000 steps;
- a unique experiment name and GCS attempt prefix.

The scheduler order is condition-major, Hammer then assembly for P2, followed
by seeds 43 and 44 with the four non-nominal conditions. Assignment timing has
no scientific meaning and may differ after preemption.

## 3. One-time v6e-8 recovery gate

Before initializing the formal campaign, there must be no formal output and the
following proof must exist:

```text
gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/
  checkpoint_contract_v6e8_60f7a53_v1/hammer_nail_nominal_src/
  provenance/CHECKPOINT_CONTRACT_PASS.json
```

Use exactly one Spot v6e-8. Bootstrap with `-ExpectedWorkers 1`. Launch the
Hammer smoke config to step 100 using `-ExpectedWorkers 1 -ExpectedDevices 8
-FsdpDevices 4`. Then perform a clean resume with
`--num-train-steps=101`, require the post-restore update witness, and run:

```powershell
.\task13_tpu_v6e16_verify_contract.ps1 `
  -ResumeRunId <resume-run-id> `
  -GcsRunUri gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/checkpoint_contract_v6e8_60f7a53_v1/hammer_nail_nominal_src `
  -SourceSha256 c1e6a96abc645b1d6abb66d4e64ad225946192c80a46e8a63cc97bd812af8c85 `
  -InitialStep 100 -ResumeStep 101 `
  -TpuName tanjunhao-tpu1 `
  -ExpectedWorkers 1 -ExpectedAcceleratorType v6e-8
```

This is a functional recovery gate, not a new scientific comparison. Do not
repeat GPU/FSDP equivalence tests.

## 4. Campaign initialization and persistence

The formal state file does not yet exist and must not be created until the gate
passes and the old v6e-16 queues have an explicit disposition. Initialize once:

```powershell
Set-Location G:\Ego\dexjoco
.\task13_tpu_formal_supervisor.ps1 -Initialize `
  -Campaign task13-tpu-formal-26-v1 -Profile v6e8x26
```

Inspect the 26 pending cells:

```powershell
.\task13_tpu_formal_supervisor.ps1 -Status
```

Install and start the persistent controller only after reviewing the state:

```powershell
.\task13_tpu_formal_supervisor.ps1 -InstallScheduledTask `
  -Profile v6e8x26 -ApproveAutoMutation
```

`-ApproveAutoMutation` is the standing approval boundary. Once active, the
supervisor may create, delete and recreate only `tanjunhao-tpu1..26` and their
matching `-qr` queues; bootstrap; launch/resume; and clean those queues after
completion. It cannot touch other GCP resources or the GPU/A100 mainline.

## 5. Per-slot asynchronous state machine

Each slot independently follows:

```text
ABSENT
  -> create exact Spot queued resource
  -> WAITING_FOR_RESOURCES / PROVISIONING
  -> READY + HEALTHY
  -> bootstrap immutable release and input cache
  -> topology/data/GCS preflight
  -> atomically claim next PENDING cell
  -> initial launch or committed resume
  -> monitor LATEST + worker process
  -> COMPLETE at committed step 30000
  -> claim another pending cell, or delete slot when campaign is drained
```

There is no barrier between slots. While pending cells exist, a preempted slot
retains its active cell and healthy slots claim other work first. When the
pending queue is empty, an idle healthy slot may atomically steal a cell only
from a slot recorded as waiting for replacement capacity. The old slot loses
ownership before any relaunch. Thus one chronically unavailable queue cannot
create a completion tail, and campaign completion does not depend on all 26
queues ever becoming ready.

The local JSON scheduler uses one process and a single-instance lock. Cell
claims and state writes are atomic. Task Scheduler restarts the process after
an unexpected exit or next login; GCS remains the recovery authority.

## 6. Preemption and checkpoint behavior

The controller treats queue `SUSPENDED`/`FAILED`, node terminal state, or
`READY` plus unhealthy health as loss of the allocation. It verifies the exact
node/queue identity, deletes that exact queued resource with its node, recreates
the same Spot request, waits independently, bootstraps and resumes.

Recovery decision:

- Valid `LATEST.json` below 30,000: resume the same attempt.
- Valid `LATEST.json` at 30,000: wait for processes to exit and mark complete.
- Empty prefix and no `LATEST`: initial launch.
- Non-empty prefix and no `LATEST`: preserve it as an abandoned diagnostic
  attempt and launch a new numbered attempt. Never delete or resume it.
- Mismatched source/config/experiment/process count: fail closed.

Orbax keeps at most one large checkpoint per cell. At completion the expected
large-checkpoint footprint is approximately 26 × 9.6 GB, about 250 GB, not
26 × 30 historical checkpoints. Small `COMMITTED.json` records remain for
audit. The bucket has no fixed capacity ceiling; charges follow stored bytes,
operations and any applicable transfer class.

## 7. Failure policy

The following are automatically recoverable:

- Spot preemption;
- service-suspended queue;
- unhealthy allocation;
- controller restart;
- training process loss when a valid committed checkpoint exists;
- process loss before the first checkpoint by starting a new attempt prefix.

The following block the campaign and delete its exact TPU queues to prevent
unbounded spend:

- three training crashes without checkpoint progress;
- five consecutive controller-path failures;
- proof/source/topology/provenance mismatch;
- persistent authentication, data, decoder, dependency or code failure.

After a block, inspect the state and log. Do not edit a running state file or
manually launch a second process into a supervisor-owned TPU.

## 8. Monitoring and stopping

Status and recent events:

```powershell
.\task13_tpu_formal_supervisor.ps1 -Status
Get-Content .\task13_tpu_formal_supervisor.log -Tail 100
```

The status reports campaign counts, every slot phase, active cell, latest TPU
observation and last error. GCS `LATEST.json` is authoritative for progress.

To stop intentionally:

```powershell
.\task13_tpu_formal_supervisor.ps1 -RequestStop
```

The live supervisor terminates Task13 training and deletes only the exact
campaign queues. After status becomes `STOPPED`, remove the Scheduled Task:

```powershell
.\task13_tpu_formal_supervisor.ps1 -UninstallScheduledTask
```

If the controller machine is offline, it cannot detect preemption or recreate
resources. Keep it powered on, logged in, online and gcloud-authenticated.

## 9. Completion

Campaign success requires all 26 cells to have a valid step-30,000 committed
checkpoint. The supervisor then deletes all exact campaign queues and changes
status to `COMPLETE`. Preserve:

- campaign state and event log;
- immutable source/input identities;
- per-cell final `LATEST.json`, `COMMITTED.json` and Orbax root;
- abandoned attempt prefixes until a separate cleanup inventory is approved.

TPU results remain labeled `tpu_sidebranch` and are not silently substituted
for GPU/A100 canonical conclusions.
