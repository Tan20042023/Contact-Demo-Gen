[CmdletBinding()]
param(
    [string]$QueueName = 'tanjunhao-tpu-qr',
    [string]$NodeId = 'tanjunhao-tpu',
    [string]$Project = 'whyu01',
    [string]$Zone = 'us-east1-d',
    [ValidateRange(15, 3600)]
    [int]$PollSeconds = 60,
    [ValidateRange(0, 10080)]
    [int]$MaxWaitMinutes = 0,
    [string]$StatePath = (Join-Path $PSScriptRoot 'task13_tpu_ready_state.json'),
    [string]$LogPath = (Join-Path $PSScriptRoot 'task13_tpu_ready_watcher.log'),
    [string]$RecreateScript = 'G:\Ego\recreate-tpu-after-preemption.ps1',
    [switch]$AutoRecreate,
    [switch]$Background,
    [switch]$Child,
    [string]$ScheduledTaskName = 'Task13-TPU-Ready-Watcher',
    [switch]$InstallScheduledTask
)

$ErrorActionPreference = 'Stop'

# Current Task13 policy: a Spot preemption is a stop condition.  Keep this
# guard even if an old shortcut or scheduled-task command line still supplies
# -AutoRecreate, so it cannot silently create or delete TPU resources.
if ($AutoRecreate) {
    throw 'Auto-recreate is disabled by the current Task13 TPU operating policy. Remove -AutoRecreate; a preemption must stop for operator review.'
}

function Write-WatcherLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $LogPath -Append
}

function Write-StateAtomically {
    param([System.Collections.IDictionary]$State)
    $parent = Split-Path -Parent $StatePath
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $temporary = "$StatePath.tmp"
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Get-GcloudJson {
    param([string[]]$Arguments)
    $output = & gcloud @Arguments '--format=json' 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    if (-not $output) { return $null }
    try { return ($output | ConvertFrom-Json) } catch { return $null }
}

function Start-BackgroundWatcher {
    $pwsh = (Get-Command pwsh -ErrorAction Stop).Source
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath,
        '-QueueName', $QueueName, '-NodeId', $NodeId, '-Project', $Project, '-Zone', $Zone,
        '-PollSeconds', $PollSeconds, '-MaxWaitMinutes', $MaxWaitMinutes,
        '-StatePath', $StatePath, '-LogPath', $LogPath, '-RecreateScript', $RecreateScript, '-Child'
    )
    if ($AutoRecreate) { $arguments += '-AutoRecreate' }
    $process = Start-Process -FilePath $pwsh -ArgumentList $arguments -WindowStyle Hidden -PassThru
    Write-Host "Watcher started in background (PID $($process.Id))."
    Write-Host "State: $StatePath"
    Write-Host "Log:   $LogPath"
}

function Install-WatcherScheduledTask {
    if (-not (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue)) {
        throw 'Windows Task Scheduler cmdlets are unavailable on this control machine.'
    }
    $pwsh = (Get-Command pwsh -ErrorAction Stop).Source
    $escapedScript = $PSCommandPath.Replace('"', '`"')
    $escapedState = $StatePath.Replace('"', '`"')
    $escapedLog = $LogPath.Replace('"', '`"')
    $escapedRecreate = $RecreateScript.Replace('"', '`"')
    $argument = "-NoProfile -ExecutionPolicy Bypass -File `"$escapedScript`" -QueueName `"$QueueName`" -NodeId `"$NodeId`" -Project `"$Project`" -Zone `"$Zone`" -PollSeconds $PollSeconds -MaxWaitMinutes $MaxWaitMinutes -StatePath `"$escapedState`" -LogPath `"$escapedLog`" -RecreateScript `"$escapedRecreate`" -Child"
    if ($AutoRecreate) { $argument += ' -AutoRecreate' }
    $action = New-ScheduledTaskAction -Execute $pwsh -Argument $argument
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 7) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $ScheduledTaskName -Action $action -Trigger $trigger -Settings $settings -Description 'Task13 Spot TPU ready/health watcher' -Force | Out-Null
    Start-ScheduledTask -TaskName $ScheduledTaskName
    Write-Host "Scheduled watcher installed and started: $ScheduledTaskName"
}

if ($InstallScheduledTask) {
    Install-WatcherScheduledTask
    exit 0
}

if ($Background -and -not $Child) {
    Start-BackgroundWatcher
    exit 0
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw 'gcloud was not found on PATH. Run this from the machine where the Google Cloud account is activated.'
}

$startedAt = Get-Date
$lastSignature = ''
$recreateIssuedFor = ''
$readyNotified = $false
Write-WatcherLog "watcher_started queue=$QueueName node=$NodeId project=$Project zone=$Zone poll_seconds=$PollSeconds auto_recreate=$AutoRecreate"

while ($true) {
    $now = Get-Date
    $queue = Get-GcloudJson @('compute', 'tpus', 'queued-resources', 'describe', $QueueName, "--project=$Project", "--zone=$Zone")
    $node = Get-GcloudJson @('compute', 'tpus', 'tpu-vm', 'describe', $NodeId, "--project=$Project", "--zone=$Zone")

    $queueState = if ($queue -and $queue.state) { [string]$queue.state.state } else { 'NOT_FOUND_OR_UNREADABLE' }
    $queueStateInitiator = if ($queue -and $queue.state -and $queue.state.stateInitiator) { [string]$queue.state.stateInitiator } else { $null }
    $nodeState = if ($node) { [string]$node.state } else { 'NOT_FOUND' }
    $health = if ($node -and $node.health) { [string]$node.health } else { 'NOT_FOUND' }
    $acceleratorType = if ($node -and $node.acceleratorType) { [string]$node.acceleratorType } else { $null }
    $runtimeVersion = if ($node -and $node.runtimeVersion) { [string]$node.runtimeVersion } else { $null }
    $ips = @()
    if ($node -and $node.networkEndpoints) {
        $ips = @($node.networkEndpoints | ForEach-Object { $_.accessConfig.externalIp } | Where-Object { $_ })
    }

    $ready = ($nodeState -eq 'READY' -and $health -eq 'HEALTHY')
    $state = [ordered]@{
        checked_at_utc = $now.ToUniversalTime().ToString('o')
        queue_name = $QueueName
        queue_state = $queueState
        queue_state_initiator = $queueStateInitiator
        node_id = $NodeId
        node_state = $nodeState
        health = $health
        ready = $ready
        project = $Project
        zone = $Zone
        accelerator_type = $acceleratorType
        runtime_version = $runtimeVersion
        external_ips = $ips
    }
    Write-StateAtomically $state

    $signature = "$queueState|$queueStateInitiator|$nodeState|$health|$($ips -join ',')"
    if ($signature -ne $lastSignature) {
        Write-WatcherLog "state_changed queue=$queueState initiator=$queueStateInitiator node=$nodeState health=$health ips=$($ips -join ',')"
        $lastSignature = $signature
    }

    if ($ready) {
        if (-not $readyNotified) {
            Write-WatcherLog "TPU_READY_AND_HEALTHY accelerator=$acceleratorType runtime=$runtimeVersion ips=$($ips -join ',')"
            Write-Host "TPU_READY_AND_HEALTHY: $NodeId ($acceleratorType, $Zone)"
            $readyNotified = $true
        }
    } else {
        $readyNotified = $false
    }

    $unhealthy = ($nodeState -eq 'READY' -and $health -like 'UNHEALTHY*')
    $serviceSuspended = ($queueState -eq 'SUSPENDED' -and $queueStateInitiator -eq 'SERVICE')
    $needsRecreate = $unhealthy -or $serviceSuspended
    if (-not $needsRecreate) {
        # A later preemption can have the same state signature as an earlier one.
        # Clear the one-shot guard once this recovery has demonstrably left its
        # terminal state, so the next service-initiated suspension is recoverable.
        $recreateIssuedFor = ''
    }
    if ($needsRecreate) {
        $reason = if ($serviceSuspended) { 'service_suspended_queue' } else { 'unhealthy_ready_node' }
        Write-WatcherLog "TPU_PREEMPTED_STOP reason=$reason; no_recreate_by_policy"
        Write-Host "TPU_PREEMPTED_STOP: $NodeId ($reason). No recreation was attempted."
        exit 3
    }

    if ($MaxWaitMinutes -gt 0 -and ((Get-Date) - $startedAt).TotalMinutes -ge $MaxWaitMinutes) {
        Write-WatcherLog "timeout_after_minutes=$MaxWaitMinutes"
        exit 2
    }
    Start-Sleep -Seconds $PollSeconds
}
