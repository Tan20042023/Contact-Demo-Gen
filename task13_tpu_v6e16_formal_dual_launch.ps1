[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$')]
    [string]$Campaign,

    [Parameter(Mandatory)]
    [ValidatePattern('^task13_tpu_technical_')]
    [string]$Config1,

    [Parameter(Mandatory)]
    [ValidatePattern('^task13_tpu_technical_')]
    [string]$Config2,

    [string]$Tpu1 = 'tanjunhao-tpu1',
    [string]$Tpu2 = 'tanjunhao-tpu2',
    [string[]]$TrainArgs1 = @(),
    [string[]]$TrainArgs2 = @(),
    [string]$Project = 'whyu01',
    [string]$Zone = 'us-east1-d',
    [string]$SshUser = 'tanjunhao',
    [string]$KeyPath = (Join-Path $env:USERPROFILE '.ssh\google_compute_engine'),
    [string]$InputUri = 'gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/input_assets',
    [Int64]$InputBytes = 30696986145,
    [string]$CodeUri = 'gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/bootstrap/60f7a53/source-layout-v2.tar.gz',
    [string]$SourceSha256 = 'c1e6a96abc645b1d6abb66d4e64ad225946192c80a46e8a63cc97bd812af8c85',
    [string]$ProofUri = 'gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs/checkpoint_contract_60f7a53_20260828a/hammer_nail_nominal_src/provenance/CHECKPOINT_CONTRACT_PASS.json',
    [string]$RunsRoot = 'gs://use1/user/tanjunhao/task13_tpu_sidebranch/v1/runs',
    [ValidateRange(30, 7200)] [int]$ReadyTimeoutSeconds = 3600,
    [switch]$ApproveFormalLaunch,
    [switch]$DryRun
)

# A control-plane orchestrator only. It never creates/deletes/recreates TPU
# resources, and delegates all runtime work to the reviewed bootstrap/launcher.
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $KeyPath)) { throw "Compute Engine SSH key not found: $KeyPath" }
if ($Tpu1 -eq $Tpu2) { throw 'Tpu1 and Tpu2 must name two different TPU VMs.' }
if ($SourceSha256 -notmatch '^[a-f0-9]{64}$') { throw 'SourceSha256 must be a lowercase SHA-256 digest.' }
if ($RunsRoot.TrimEnd('/') -notmatch '^gs://') { throw 'RunsRoot must be a gs:// prefix.' }
$allTrainArgs = @($TrainArgs1) + @($TrainArgs2)
if (@($allTrainArgs | Where-Object { $_ -notmatch '^[A-Za-z0-9_.=/+-]+$' }).Count -gt 0) {
    throw 'TrainArgs may contain only letters, digits, dot, underscore, slash, equals, plus, or minus.'
}

$bootstrapScript = Join-Path $PSScriptRoot 'task13_tpu_v6e16_bootstrap_all.ps1'
$launchScript = Join-Path $PSScriptRoot 'task13_tpu_v6e16_launch.ps1'
foreach ($script in @($bootstrapScript, $launchScript)) {
    if (-not (Test-Path -LiteralPath $script)) { throw "Required controller script not found: $script" }
}

$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runsRootClean = $RunsRoot.TrimEnd('/')
$cells = @(
    [ordered]@{
        slot = 1; tpu_name = $Tpu1; config = $Config1; train_args = @($TrainArgs1)
        run_id = "$Campaign-$Tpu1-$timestamp"
        run_uri = "$runsRootClean/$Campaign/$Tpu1/$Config1"
    },
    [ordered]@{
        slot = 2; tpu_name = $Tpu2; config = $Config2; train_args = @($TrainArgs2)
        run_id = "$Campaign-$Tpu2-$timestamp"
        run_uri = "$runsRootClean/$Campaign/$Tpu2/$Config2"
    }
)

function Get-HealthyTpu([System.Collections.IDictionary]$Cell) {
    $node = gcloud compute tpus tpu-vm describe $Cell.tpu_name "--project=$Project" "--zone=$Zone" --format=json | ConvertFrom-Json
    if ($node.state -ne 'READY' -or $node.health -ne 'HEALTHY') {
        throw "TPU $($Cell.tpu_name) is not launchable: state=$($node.state) health=$($node.health)"
    }
    if ($node.acceleratorType -ne 'v6e-16') {
        throw "TPU $($Cell.tpu_name) must be v6e-16; found $($node.acceleratorType)."
    }
    $ips = @($node.networkEndpoints | ForEach-Object { $_.accessConfig.externalIp } | Where-Object { $_ })
    if ($ips.Count -ne 4) { throw "TPU $($Cell.tpu_name) must expose four worker IPs; found $($ips.Count)." }
    $Cell.external_ips = $ips
}

function Assert-EmptyInitialPrefix([System.Collections.IDictionary]$Cell) {
    $objects = @(& gcloud storage ls --recursive "$($Cell.run_uri)/**" 2>$null)
    if ($LASTEXITCODE -eq 0 -and $objects.Count -gt 0) {
        throw "Formal initial launch refuses non-empty run prefix: $($Cell.run_uri)"
    }
}

# Validate common immutable inputs before touching either TPU. This is also
# intentionally executed by -DryRun, so it catches expired/mismatched proof.
& gcloud storage ls -l $CodeUri *> $null
if ($LASTEXITCODE -ne 0) { throw "Bootstrap source archive is unreadable: $CodeUri" }
$proofText = & gcloud storage cat $ProofUri 2>$null
if ($LASTEXITCODE -ne 0) { throw "Checkpoint contract proof is unreadable: $ProofUri" }
try { $proof = ($proofText | Out-String | ConvertFrom-Json) } catch { throw "Checkpoint contract proof is invalid JSON: $ProofUri" }
if ($proof.status -ne 'PASS' -or $proof.source_sha256 -ne $SourceSha256 -or $proof.process_count -ne 4) {
    throw 'Checkpoint contract proof does not PASS for the requested source SHA/four-worker topology.'
}

# Refresh host-key records once, serially. Bootstrap can then run in parallel
# without two jobs racing to rewrite the same known_hosts file.
$allIps = [System.Collections.Generic.List[string]]::new()
foreach ($cell in $cells) {
    Get-HealthyTpu $cell
    Assert-EmptyInitialPrefix $cell
    foreach ($ip in $cell.external_ips) { [void]$allIps.Add($ip) }
}
if ((@($allIps | Sort-Object -Unique)).Count -ne 8) { throw 'The two TPU VMs did not expose eight distinct worker IPs.' }
$knownHosts = Join-Path $env:USERPROFILE '.ssh\known_hosts'
if (-not $DryRun -and (Test-Path -LiteralPath $knownHosts)) {
    foreach ($ip in ($allIps | Sort-Object -Unique)) {
        & ssh-keygen -R $ip -f $knownHosts *> $null
        & ssh-keygen -R "[$ip]:22" -f $knownHosts *> $null
    }
}

$manifest = [ordered]@{
    schema = 'task13-v6e16-dual-formal-launch-v1'
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    dry_run = [bool]$DryRun
    project = $Project; zone = $Zone; source_uri = $CodeUri; source_sha256 = $SourceSha256
    checkpoint_contract_proof_uri = $ProofUri
    cells = $cells
}
$manifestDir = Join-Path $PSScriptRoot 'task13_tpu_launch_manifests'
New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
$manifestPath = Join-Path $manifestDir "$Campaign-$timestamp.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "DUAL_FORMAL_PREFLIGHT_PASS manifest=$manifestPath"
foreach ($cell in $cells) { Write-Host "CELL slot=$($cell.slot) tpu=$($cell.tpu_name) config=$($cell.config) uri=$($cell.run_uri)" }

if ($DryRun) {
    Write-Host 'DRY_RUN_PASS: no bootstrap or training was started.'
    exit 0
}
if (-not $ApproveFormalLaunch) {
    throw 'Formal launch requires -ApproveFormalLaunch after reviewing the printed two-cell manifest.'
}

$bootstrapJobs = foreach ($cell in $cells) {
    $bootstrapArgs = @{
        RunId = $cell.run_id; CodeUri = $CodeUri; CodeSha256 = $SourceSha256
        InputUri = $InputUri; InputBytes = $InputBytes; TpuName = $cell.tpu_name
        Project = $Project; Zone = $Zone; SshUser = $SshUser; KeyPath = $KeyPath
        SkipKnownHostsRefresh = $true; ReadyTimeoutSeconds = $ReadyTimeoutSeconds
    }
    Start-Job -Name "bootstrap-$($cell.tpu_name)" -ScriptBlock {
        param($ScriptPath, $Arguments)
        & $ScriptPath @Arguments
    } -ArgumentList $bootstrapScript, $bootstrapArgs
}
$bootstrapJobs | Wait-Job | Out-Null
$bootstrapFailures = @()
foreach ($job in $bootstrapJobs) {
    # gcloud's successful SSH-key propagation writes progress to stderr. Do
    # not let that informational stream abort the controller; the job state
    # and the bootstrap controller's four READY records remain authoritative.
    $jobOutput = @(Receive-Job $job -ErrorAction SilentlyContinue 2>&1)
    $jobOutput | Out-Host
    if ($job.State -ne 'Completed') { $bootstrapFailures += $job }
    Remove-Job $job -Force
}
if ($bootstrapFailures.Count -gt 0) {
    throw "Bootstrap failed on $($bootstrapFailures.Count) TPU(s); no formal training was launched."
}

# Each formal launcher repeats READY/HEALTHY, proof and empty-prefix gates just
# before dispatch. Independent cells launch in parallel to minimize Spot idle
# time; a failure is reported per cell and never triggers a recreate/recovery.
$launchJobs = foreach ($cell in $cells) {
    $launchArgs = @{
        RunId = $cell.run_id; Config = $cell.config; GcsRunUri = $cell.run_uri
        SourceSha256 = $SourceSha256; Purpose = 'formal'; CheckpointContractProofUri = $ProofUri
        TrainArgs = @($cell.train_args); TpuName = $cell.tpu_name; Project = $Project
        Zone = $Zone; SshUser = $SshUser; KeyPath = $KeyPath
    }
    Start-Job -Name "launch-$($cell.tpu_name)" -ScriptBlock {
        param($ScriptPath, $Arguments)
        & $ScriptPath @Arguments
    } -ArgumentList $launchScript, $launchArgs
}
$launchJobs | Wait-Job | Out-Null
$launchFailures = @()
foreach ($job in $launchJobs) {
    $jobOutput = @(Receive-Job $job -ErrorAction SilentlyContinue 2>&1)
    $jobOutput | Out-Host
    if ($job.State -ne 'Completed') { $launchFailures += $job }
    Remove-Job $job -Force
}
if ($launchFailures.Count -gt 0) {
    throw "Formal launch failed on $($launchFailures.Count) TPU(s); inspect the manifest and controller output. No recovery was attempted."
}
Write-Host "DUAL_FORMAL_LAUNCH_SUBMITTED manifest=$manifestPath"
