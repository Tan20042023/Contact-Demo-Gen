[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$RunId,
    [Parameter(Mandatory)] [string]$CodeUri,
    [Parameter(Mandatory)] [string]$CodeSha256,
    [string]$InputUri = 'gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets',
    [Int64]$InputBytes = 30696986145,
    [string]$TpuName = 'tanjunhao-tpu',
    [string]$Project = 'whyu01',
    [string]$Zone = 'us-east1-d',
    [string]$SshUser = 'tanjunhao',
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\google_compute_engine'),
    [ValidateRange(1, 64)] [int]$ExpectedWorkers = 4,
    [switch]$SkipKnownHostsRefresh,
    [ValidateRange(30, 7200)] [int]$ReadyTimeoutSeconds = 3600
)

$ErrorActionPreference = 'Stop'
$workerScript = Join-Path $PSScriptRoot 'task13_tpu_v6e16_bootstrap_worker.sh'
if (-not (Test-Path -LiteralPath $workerScript)) { throw "Bootstrap worker script not found: $workerScript" }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "Compute Engine SSH key not found: $KeyPath" }

$node = gcloud compute tpus tpu-vm describe $TpuName "--project=$Project" "--zone=$Zone" --format=json | ConvertFrom-Json
if ($node.state -ne 'READY' -or $node.health -ne 'HEALTHY') {
    throw "TPU is not launchable: state=$($node.state) health=$($node.health)"
}
$ips = @($node.networkEndpoints | ForEach-Object { $_.accessConfig.externalIp } | Where-Object { $_ })
if ($ips.Count -ne $ExpectedWorkers) { throw "Expected $ExpectedWorkers worker IPs; found $($ips.Count)" }

# Spot recreations can reuse an external IP whose previous TPU VM had a
# different SSH host key.  The IP list above was obtained from the authoritative
# Cloud TPU node description, so remove only those stale local records before
# the first OpenSSH connection records the new keys with accept-new.  Do not
# disable host-key checking globally.
$knownHosts = Join-Path $env:USERPROFILE '.ssh\known_hosts'
if (-not $SkipKnownHostsRefresh) {
    foreach ($ip in $ips) {
        if (Test-Path -LiteralPath $knownHosts) {
            & ssh-keygen -R $ip -f $knownHosts *> $null
            & ssh-keygen -R "[$ip]:22" -f $knownHosts *> $null
        }
    }
}

# This gcloud call publishes the local Compute Engine public key for the actual
# TPU account.  Windows gcloud may use PuTTY and return after a host-key prompt;
# OpenSSH below performs the real noninteractive host connection and verification.
foreach ($worker in 0..($ExpectedWorkers - 1)) {
    & gcloud compute tpus tpu-vm ssh "${SshUser}@${TpuName}" "--project=$Project" "--zone=$Zone" "--worker=$worker" --command=true | Out-Host
}
foreach ($ip in $ips) {
    & ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "${SshUser}@${ip}" hostname | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "SSH validation failed for $ip" }
    & scp -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=yes $workerScript "${SshUser}@${ip}:task13_tpu_v6e16_bootstrap_worker.sh" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Bootstrap script copy failed for $ip" }
}

$remote = @"
set -euo pipefail
state="`$HOME/task13_v6e16/bootstrap/$RunId"
mkdir -p "`$state"
chmod 700 "`$HOME/task13_tpu_v6e16_bootstrap_worker.sh"
nohup env TASK13_TPU_RUN_ID='$RunId' TASK13_TPU_CODE_URI='$CodeUri' TASK13_TPU_CODE_SHA256='$CodeSha256' TASK13_TPU_INPUT_URI='$InputUri' TASK13_TPU_INPUT_BYTES='$InputBytes' "`$HOME/task13_tpu_v6e16_bootstrap_worker.sh" > "`$state/bootstrap.log" 2>&1 < /dev/null &
echo `$! > "`$state/bootstrap.pid"
echo BOOTSTRAP_STARTED host=`$(hostname) pid=`$(cat "`$state/bootstrap.pid")
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
foreach ($ip in $ips) {
    & ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=yes "${SshUser}@${ip}" "echo $encoded | base64 -d | bash" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Bootstrap launch failed for $ip" }
}

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $ready = 0
    foreach ($ip in $ips) {
        $readyJson = & ssh -i $KeyPath -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "${SshUser}@${ip}" "cat `$HOME/task13_v6e16/bootstrap/$RunId/READY.json" 2>$null
        if ($LASTEXITCODE -eq 0 -and $readyJson) {
            try {
                $record = ($readyJson | Out-String | ConvertFrom-Json)
                if (
                    $record.schema -eq 'task13-v6e16-ready-v3' -and
                    $record.source_sha256 -eq $CodeSha256 -and
                    $record.input_bytes -eq $InputBytes -and
                    $record.tpu_runtime_initialized -eq $false
                ) {
                    $ready++
                } else {
                    Write-Host "BOOTSTRAP_READY_REJECTED host=$ip (schema, source, input, or runtime-neutrality mismatch)"
                }
            } catch {
                Write-Host "BOOTSTRAP_READY_REJECTED host=$ip (invalid JSON)"
            }
        }
    }
    if ($ready -eq $ExpectedWorkers) { Write-Host "BOOTSTRAP_ALL_READY run_id=$RunId workers=$ExpectedWorkers"; exit 0 }
    Start-Sleep -Seconds 30
}
throw "Timed out waiting for $ExpectedWorkers READY.json records: run_id=$RunId"
