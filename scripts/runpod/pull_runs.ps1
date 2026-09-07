# Pull training artifacts (runs/) home from the RunPod network volume over its S3 API,
# then delete them from the volume so it doesn't accumulate (the bucket *is* the volume).
# Works with NO pod running.
#
# Prereqs (one-time):
#   - RunPod -> Settings -> S3 API Keys  (access key + secret)
#   - aws configure --profile runpods3   (region eu-ro-1)
#
# Usage:  scripts\runpod\pull_runs.ps1              # sync runs/ home, then delete them from S3
#         scripts\runpod\pull_runs.ps1 -KeepRemote  # sync but leave the runs on the volume
#         scripts\runpod\pull_runs.ps1 -List        # just list what's on the volume (no pull/delete)
#
# NOTE: deleting removes the runs from the volume too, so a later cross-session `resume`
# from one of them would need it re-uploaded. Use -KeepRemote if you plan to extend a run.
param(
    [switch]$List,
    [switch]$KeepRemote
)

$ErrorActionPreference = "Stop"

# Locate aws.exe — it may not be on PATH (default per-user install lives under
# %LOCALAPPDATA%\Programs\Amazon\AWSCLIV2). Override with $env:AWS_CLI if needed.
function Resolve-Aws {
    if ($env:AWS_CLI -and (Test-Path $env:AWS_CLI)) { return $env:AWS_CLI }
    $cmd = Get-Command aws -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Amazon\AWSCLIV2\aws.exe",
        "$env:ProgramFiles\Amazon\AWSCLIV2\aws.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    throw "aws.exe not found. Install the AWS CLI or set `$env:AWS_CLI to its full path."
}
$Aws = Resolve-Aws

# --- Volume / endpoint config (edit if the network volume changes) ---
$Bucket   = "pjg2ftia6g"
$Region   = "eu-ro-1"
$Endpoint = "https://s3api-eu-ro-1.runpod.io"
$Profile  = if ($env:AWS_PROFILE) { $env:AWS_PROFILE } else { "runpods3" }
$RemoteRunsPrefix = "kAIsparov/runs"        # path of runs/ inside the volume

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$localRuns = Join-Path $repoRoot "runs"

$common = @(
    "--region", $Region,
    "--endpoint-url", $Endpoint,
    "--profile", $Profile
)

if ($List) {
    Write-Host ">> listing s3://$Bucket/$RemoteRunsPrefix/"
    & $Aws s3 ls "s3://$Bucket/$RemoteRunsPrefix/" @common
    exit $LASTEXITCODE
}

Write-Host ">> syncing s3://$Bucket/$RemoteRunsPrefix -> $localRuns"
& $Aws s3 sync "s3://$Bucket/$RemoteRunsPrefix" $localRuns @common
$syncExit = $LASTEXITCODE
if ($syncExit -ne 0) {
    throw "aws s3 sync failed (exit $syncExit) -- leaving the S3 copy untouched."
}

if ($KeepRemote) {
    Write-Host ">> done (runs left on the volume; -KeepRemote)."
    exit 0
}

# Sync succeeded -> the runs are safely local. Remove them from the volume.
Write-Host ">> sync OK. Deleting s3://$Bucket/$RemoteRunsPrefix/ from the volume ..."
& $Aws s3 rm "s3://$Bucket/$RemoteRunsPrefix" --recursive @common
if ($LASTEXITCODE -ne 0) {
    throw "Pulled OK, but deleting the S3 copy failed (exit $LASTEXITCODE). Re-run or clean it up manually."
}
Write-Host ">> done. Runs pulled to $localRuns and cleared from the volume."
