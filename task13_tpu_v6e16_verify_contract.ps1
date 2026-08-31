[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$ResumeRunId,
    [Parameter(Mandatory)] [string]$GcsRunUri,
    [Parameter(Mandatory)] [string]$SourceSha256,
    [Parameter(Mandatory)] [int]$InitialStep,
    [Parameter(Mandatory)] [int]$ResumeStep,
    [string]$TpuName = 'tanjunhao-tpu',
    [string]$Project = 'whyu01',
    [string]$Zone = 'us-east1-d',
    [string]$SshUser = 'tanjunhao',
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\google_compute_engine'),
    [ValidateRange(1, 64)] [int]$ExpectedWorkers = 4,
    [string]$ExpectedAcceleratorType = 'v6e-16'
)

# This is the only producer of CHECKPOINT_CONTRACT_PASS.json.  Run it only
# after a separate --resume smoke launch has completed.  It is intentionally a
# verifier, never a launcher: it cannot create a TPU, install packages, or
# start a process.
$ErrorActionPreference = 'Stop'
$runUri = $GcsRunUri.TrimEnd('/')
if ($runUri -notmatch '^gs://') { throw "GcsRunUri must be a gs:// prefix: $GcsRunUri" }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "Compute Engine SSH key not found: $KeyPath" }

$node = gcloud compute tpus tpu-vm describe $TpuName "--project=$Project" "--zone=$Zone" --format=json | ConvertFrom-Json
if ($node.state -ne 'READY' -or $node.health -ne 'HEALTHY') {
    throw "TPU is not verifiable: state=$($node.state) health=$($node.health)"
}
$ips = @($node.networkEndpoints | ForEach-Object { $_.accessConfig.externalIp } | Where-Object { $_ })
if ($node.acceleratorType -ne $ExpectedAcceleratorType) { throw "Expected $ExpectedAcceleratorType; found $($node.acceleratorType)" }
if ($ips.Count -ne $ExpectedWorkers) { throw "Expected $ExpectedWorkers worker IPs; found $($ips.Count)" }

function Read-GcsJson([string]$Uri) {
    $content = & gcloud storage cat $Uri 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Required GCS object is unreadable: $Uri" }
    try { return ($content | Out-String | ConvertFrom-Json) } catch { throw "Invalid JSON at $Uri" }
}

$initialCommit = Read-GcsJson "$runUri/checkpoints/$InitialStep/COMMITTED.json"
$resumeCommit = Read-GcsJson "$runUri/checkpoints/$ResumeStep/COMMITTED.json"
$latest = Read-GcsJson "$runUri/LATEST.json"
if (
    $initialCommit.step -ne $InitialStep -or
    $resumeCommit.step -ne $ResumeStep -or
    $latest.step -ne $ResumeStep -or
    $initialCommit.process_count -ne $ExpectedWorkers -or
    $resumeCommit.process_count -ne $ExpectedWorkers -or
    $latest.process_count -ne $ExpectedWorkers -or
    $initialCommit.object_count -le 0 -or
    $resumeCommit.object_count -le 0
) {
    throw "The initial or resumed native-GCS commit metadata is incomplete for the $ExpectedWorkers-process topology."
}
if ($initialCommit.provenance.source_sha256 -ne $SourceSha256 -or $resumeCommit.provenance.source_sha256 -ne $SourceSha256) {
    throw 'Committed source SHA differs from the requested source SHA.'
}

# The native shared-GCS checkpoint is one Orbax root, not four worker-local
# checkpoint trees. Confirm that the latest finalized root contains metadata
# from every JAX process. (The original root may be rotated after resume.)
$objects = @(& gcloud storage ls --recursive "$($resumeCommit.checkpoint_uri)/**" 2>$null)
if ($LASTEXITCODE -ne 0 -or $objects.Count -eq 0) {
    throw "The resumed native-GCS checkpoint root is unreadable or empty: $($resumeCommit.checkpoint_uri)"
}
foreach ($worker in 0..($ExpectedWorkers - 1)) {
    if (-not ($objects | Where-Object { $_ -match "/array_metadatas/process_$worker(?:$|/)" })) {
        throw "No native Orbax array metadata witness for process $worker."
    }
}

foreach ($ip in $ips) {
    & ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "${SshUser}@${ip}" "grep -Fq 'TASK13_POST_RESTORE_UPDATE_PASS' `$HOME/task13_v6e16/logs/$ResumeRunId/train.log"
    if ($LASTEXITCODE -ne 0) { throw "No post-restore update witness in $ip train log for run $ResumeRunId." }
}

$proof = [ordered]@{
    schema = 'task13-tpu-native-gcs-checkpoint-contract-v3'
    status = 'PASS'
    source_sha256 = $SourceSha256
    process_count = $ExpectedWorkers
    accelerator_type = $ExpectedAcceleratorType
    tpu_name = $TpuName
    zone = $Zone
    initial_checkpoint_step = $InitialStep
    resumed_checkpoint_step = $ResumeStep
    run_uri = $runUri
    resume_run_id = $ResumeRunId
    verified_utc = (Get-Date).ToUniversalTime().ToString('o')
}
$tmp = Join-Path $env:TEMP "task13-checkpoint-contract-$ResumeRunId.json"
$proof | ConvertTo-Json | Set-Content -LiteralPath $tmp -Encoding utf8NoBOM
$proofUri = "$runUri/provenance/CHECKPOINT_CONTRACT_PASS.json"
& gcloud storage cp $tmp $proofUri
if ($LASTEXITCODE -ne 0) { throw "Could not publish checkpoint-contract proof: $proofUri" }
Write-Host "CHECKPOINT_CONTRACT_PASS proof_uri=$proofUri"
