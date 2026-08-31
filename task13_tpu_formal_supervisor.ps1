[CmdletBinding(DefaultParameterSetName = 'Status')]
param(
    [Parameter(ParameterSetName = 'Initialize', Mandatory)] [switch]$Initialize,
    [Parameter(ParameterSetName = 'Run', Mandatory)] [switch]$Run,
    [Parameter(ParameterSetName = 'Background', Mandatory)] [switch]$Background,
    [Parameter(ParameterSetName = 'Install', Mandatory)] [switch]$InstallScheduledTask,
    [Parameter(ParameterSetName = 'Uninstall', Mandatory)] [switch]$UninstallScheduledTask,
    [Parameter(ParameterSetName = 'Stop', Mandatory)] [switch]$RequestStop,
    [Parameter(ParameterSetName = 'Status')] [switch]$Status,
    [Parameter(ParameterSetName = 'SelfTest', Mandatory)] [switch]$SelfTest,
    [string]$Campaign = 'task13-tpu-formal-26-v1',
    [ValidateSet('v6e8x26', 'v6e4x26', 'v6e16x13')] [string]$Profile = 'v6e8x26',
    [string]$StatePath = (Join-Path $PSScriptRoot 'task13_tpu_formal_campaign_state.json'),
    [string]$LogPath = (Join-Path $PSScriptRoot 'task13_tpu_formal_supervisor.log'),
    [ValidateRange(15, 3600)] [int]$PollSeconds = 60,
    [string]$ScheduledTaskName = 'Task13-TPU-Formal-Supervisor',
    [switch]$ApproveAutoMutation
)

$ErrorActionPreference = 'Stop'
$python = (Get-Command python -ErrorAction Stop).Source
$script = Join-Path $PSScriptRoot 'task13_tpu_formal_supervisor.py'
if (-not (Test-Path -LiteralPath $script)) { throw "Supervisor not found: $script" }

function Get-CommonArgs([string]$Command) {
    return @($script, $Command, '--campaign', $Campaign, '--profile', $Profile, '--state', $StatePath, '--log', $LogPath, '--poll-seconds', "$PollSeconds")
}

if ($Initialize) {
    & $python @(Get-CommonArgs 'init')
    exit $LASTEXITCODE
}
if ($SelfTest) {
    & $python @(Get-CommonArgs 'self-test')
    exit $LASTEXITCODE
}
if ($Status -or $PSCmdlet.ParameterSetName -eq 'Status') {
    & $python @(Get-CommonArgs 'status')
    exit $LASTEXITCODE
}
if ($RequestStop) {
    & $python @(Get-CommonArgs 'request-stop')
    exit $LASTEXITCODE
}
if ($UninstallScheduledTask) {
    $task = Get-ScheduledTask -TaskName $ScheduledTaskName -ErrorAction SilentlyContinue
    if ($task) {
        Stop-ScheduledTask -TaskName $ScheduledTaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $ScheduledTaskName -Confirm:$false
    }
    Write-Host "Scheduled supervisor removed: $ScheduledTaskName"
    exit 0
}

if (-not $ApproveAutoMutation) {
    throw 'This mode requires -ApproveAutoMutation because it may create/delete the two exact TPU queues and launch/resume formal training.'
}
$runArgs = @(Get-CommonArgs 'run') + '--approve-auto-mutation'

if ($Run) {
    & $python @runArgs
    exit $LASTEXITCODE
}
if ($Background) {
    $process = Start-Process -FilePath $python -ArgumentList $runArgs -WindowStyle Hidden -PassThru
    Write-Host "Supervisor started in background (PID $($process.Id))."
    Write-Host "State: $StatePath"
    Write-Host "Log:   $LogPath"
    exit 0
}
if ($InstallScheduledTask) {
    if (-not (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue)) {
        throw 'Windows Task Scheduler cmdlets are unavailable.'
    }
    $quotedArgs = $runArgs | ForEach-Object { '"' + $_.Replace('"', '`"') + '"' }
    $action = New-ScheduledTaskAction -Execute $python -Argument ($quotedArgs -join ' ')
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $ScheduledTaskName -Action $action -Trigger $trigger -Settings $settings -Description 'Task13 elastic Spot TPU formal campaign supervisor' -Force | Out-Null
    Start-ScheduledTask -TaskName $ScheduledTaskName
    Write-Host "Scheduled supervisor installed and started: $ScheduledTaskName"
    exit 0
}
