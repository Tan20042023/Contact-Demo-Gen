[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$RunId,
    [Parameter(Mandatory)] [string]$Config,
    [Parameter(Mandatory)] [string]$GcsRunUri,
    [Parameter(Mandatory)] [string]$SourceSha256,
    [string[]]$TrainArgs = @(),
    [switch]$Resume,
    [string]$TpuName = 'tanjunhao-tpu',
    [string]$Project = 'whyu01',
    [string]$Zone = 'us-east1-d',
    [string]$SshUser = 'tanjunhao',
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\google_compute_engine')
)

# Windows-native counterpart to task13_tpu_v6e16_launch.sh.  It deliberately
# uses the OpenSSH key validated by bootstrap instead of the local gcloud
# PuTTY integration, but does not create, alter, or tear down TPU resources.
$ErrorActionPreference = 'Stop'
$runUri = $GcsRunUri.TrimEnd('/')
if ($runUri -notmatch '^gs://') { throw "GcsRunUri must be a gs:// prefix: $GcsRunUri" }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "Compute Engine SSH key not found: $KeyPath" }

$node = gcloud compute tpus tpu-vm describe $TpuName "--project=$Project" "--zone=$Zone" --format=json | ConvertFrom-Json
if ($node.state -ne 'READY' -or $node.health -ne 'HEALTHY') {
    throw "TPU is not launchable: state=$($node.state) health=$($node.health)"
}
$ips = @($node.networkEndpoints | ForEach-Object { $_.accessConfig.externalIp } | Where-Object { $_ })
if ($ips.Count -ne 4) { throw "Expected four v6e-16 worker IPs; found $($ips.Count)" }

$latestExists = $false
& gcloud storage ls "$runUri/LATEST.json" *> $null
if ($LASTEXITCODE -eq 0) { $latestExists = $true }
if ($Resume -and -not $latestExists) { throw "Resume requested but LATEST.json is absent: $runUri" }
if (-not $Resume -and $latestExists) { throw "Committed output already exists; refuse initial launch: $runUri" }

$repoRel = "task13_v6e16/repo-$($SourceSha256.Substring(0, 12))/openpi"
$runRootRel = "task13_v6e16/runs/$RunId"
$logRel = "task13_v6e16/logs/$RunId"
# P0/P1 override values are numeric or simple CLI identifiers.  Deliberately
# reject shell syntax rather than trying to re-implement a general shell
# escaper in the Windows controller.
if ($TrainArgs | Where-Object { $_ -notmatch '^[A-Za-z0-9_.=/+-]+$' }) {
    throw 'TrainArgs may contain only letters, digits, dot, underscore, slash, equals, plus, or minus.'
}
$quotedArgs = $TrainArgs -join ' '
$resumeArg = if ($Resume) { ' --resume' } else { '' }

function Invoke-Workers([string]$Phase, [string]$RemoteCommand) {
    $tmp = Join-Path $env:TEMP "task13-tpu-$RunId-$Phase"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    # Pass a single base64 payload to the remote shell.  Start-Process's
    # Windows argument reconstruction otherwise corrupts the Python -c
    # semicolons in an all-worker preflight.
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($RemoteCommand))
    $payload = "echo '$encoded' | base64 -d | bash"
    $procs = foreach ($ip in $ips) {
        $safeIp = $ip.Replace('.', '_')
        $out = Join-Path $tmp "$safeIp.stdout.log"
        $err = Join-Path $tmp "$safeIp.stderr.log"
        Start-Process -FilePath ssh -WindowStyle Hidden -PassThru `
            -ArgumentList @('-i', $KeyPath, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=30', "$SshUser@$ip", $payload) `
            -RedirectStandardOutput $out -RedirectStandardError $err
    }
    $procs | Wait-Process
    foreach ($ip in $ips) {
        $safeIp = $ip.Replace('.', '_')
        $out = Join-Path $tmp "$safeIp.stdout.log"
        $err = Join-Path $tmp "$safeIp.stderr.log"
        $stdout = if (Test-Path $out) { Get-Content -Raw $out } else { '' }
        $stderr = if (Test-Path $err) { Get-Content -Raw $err } else { '' }
        Write-Host "[$Phase $ip] $stdout$stderr"
    }
    $failed = $procs | Where-Object { $_.ExitCode -ne 0 }
    if ($failed) { throw "$Phase failed on $($failed.Count) worker(s); logs=$tmp" }
}

$preflight = @"
set -euo pipefail
export PYTHONPATH="`$HOME/$repoRel/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TASK13_TPU_INPUT_ROOT="`$HOME/task13_v6e16/input_assets"
export TASK13_TPU_LOCAL_RUNS_ROOT="`$HOME/$runRootRel"
export TASK13_TPU_FSDP_DEVICES=4 TASK13_TPU_NUM_WORKERS=0
cd "`$HOME/$repoRel"
"`$HOME/task13_v6e16/venv/bin/python" -c "import jax; jax.distributed.initialize(); from openpi.training.config import get_config; from openpi.training.data_loader import create_torch_dataset; c=get_config('$Config'); d=c.data.create(c.assets_dirs,c.model); ds=create_torch_dataset(d,c.model.action_horizon,c.model); assert jax.process_count()==4 and jax.device_count()==16; print('LAUNCH_PREFLIGHT_PASS',jax.process_index(),len(ds))"
"@
Invoke-Workers 'preflight' $preflight

$train = @"
set -euo pipefail
export PYTHONPATH="`$HOME/$repoRel/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TASK13_TPU_MULTIHOST=1
export TASK13_TPU_INPUT_ROOT="`$HOME/task13_v6e16/input_assets"
export TASK13_TPU_LOCAL_RUNS_ROOT="`$HOME/$runRootRel"
export TASK13_TPU_FSDP_DEVICES=4 TASK13_TPU_NUM_WORKERS=0
export TASK13_TPU_GCS_RUN_URI='$runUri' TASK13_TPU_SOURCE_SHA256='$SourceSha256'
cd "`$HOME/$repoRel"
mkdir -p "`$HOME/$logRel"
nohup "`$HOME/task13_v6e16/venv/bin/python" scripts/train.py '$Config'$resumeArg $quotedArgs > "`$HOME/$logRel/train.log" 2>&1 < /dev/null &
echo TRAIN_PID=`$!
"@
Invoke-Workers 'launch' $train
Write-Host "LAUNCH_SUBMITTED run_id=$RunId uri=$runUri config=$Config resume=$($Resume.IsPresent)"
