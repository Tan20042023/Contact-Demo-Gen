[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$RunId,
    [Parameter(Mandatory)] [string]$GcsRunUri,
    [Parameter(Mandatory)] [string]$SourceSha256,
    [Parameter(Mandatory)] [int]$Step,
    [string]$TpuName = 'tanjunhao-tpu',
    [string]$Project = 'whyu01',
    [string]$Zone = 'us-east1-d',
    [string]$SshUser = 'tanjunhao',
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\google_compute_engine')
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
if ($ips.Count -ne 4) { throw "Expected four v6e-16 worker IPs; found $($ips.Count)" }

function Read-GcsJson([string]$Uri) {
    $content = & gcloud storage cat $Uri 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Required GCS object is unreadable: $Uri" }
    try { return ($content | Out-String | ConvertFrom-Json) } catch { throw "Invalid JSON at $Uri" }
}

$commit = Read-GcsJson "$runUri/checkpoints/$Step/COMMITTED.json"
$latest = Read-GcsJson "$runUri/LATEST.json"
if ($commit.step -ne $Step -or $latest.step -ne $Step -or $commit.process_count -ne 4 -or $latest.process_count -ne 4) {
    throw 'COMMITTED.json/LATEST.json do not describe the requested four-worker step.'
}
if ($commit.provenance.source_sha256 -ne $SourceSha256) {
    throw "Committed source SHA differs from requested source SHA."
}

foreach ($worker in 0..3) {
    $manifest = Read-GcsJson "$runUri/checkpoints/$Step/worker-$worker/manifest.json"
    if ($manifest.step -ne $Step -or $manifest.worker -ne $worker -or $manifest.process_count -ne 4 -or @($manifest.files).Count -eq 0) {
        throw "Invalid or empty manifest for worker $worker."
    }
}

foreach ($ip in $ips) {
    & ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "${SshUser}@${ip}" "grep -Fq 'TASK13_POST_RESTORE_UPDATE_PASS' `$HOME/task13_v6e16/logs/$RunId/train.log"
    if ($LASTEXITCODE -ne 0) { throw "No post-restore update witness in $ip train log for run $RunId." }
}

$proof = [ordered]@{
    schema = 'task13-v6e16-checkpoint-contract-v1'
    status = 'PASS'
    source_sha256 = $SourceSha256
    process_count = 4
    tpu_name = $TpuName
    zone = $Zone
    checkpoint_step = $Step
    run_uri = $runUri
    resume_run_id = $RunId
    verified_utc = (Get-Date).ToUniversalTime().ToString('o')
}
$tmp = Join-Path $env:TEMP "task13-checkpoint-contract-$RunId.json"
$proof | ConvertTo-Json | Set-Content -LiteralPath $tmp -Encoding utf8NoBOM
$proofUri = "$runUri/provenance/CHECKPOINT_CONTRACT_PASS.json"
& gcloud storage cp $tmp $proofUri
if ($LASTEXITCODE -ne 0) { throw "Could not publish checkpoint-contract proof: $proofUri" }
Write-Host "CHECKPOINT_CONTRACT_PASS proof_uri=$proofUri"
