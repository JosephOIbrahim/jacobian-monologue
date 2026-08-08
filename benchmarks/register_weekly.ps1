# Registers a weekly automated benchmark pass (Sunday 03:00, matches the
# archive-server maintenance window). RUN THIS YOURSELF, ONCE -- registering
# a scheduled task is standing system configuration and stays a human call.
#
#   powershell -ExecutionPolicy Bypass -File benchmarks\register_weekly.ps1
#
# Remove later with:  Unregister-ScheduledTask -TaskName "JacobianMonologue-Bench"
$Root = Split-Path -Parent $PSScriptRoot
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Root\benchmarks\run.ps1`""
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3:00AM
Register-ScheduledTask -TaskName "JacobianMonologue-Bench" `
    -Action $Action -Trigger $Trigger `
    -Description "Weekly automated benchmark pass: measure, regression-check, sync docs, push on material change."
Write-Host "Registered: JacobianMonologue-Bench (Sundays 03:00)"
