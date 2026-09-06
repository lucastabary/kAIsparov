# Pull training artifacts (runs/) home from the RunPod network volume over its S3 API.
# Works with NO pod running — the bucket is the volume itself.
#
# Prereqs (one-time):
#   - RunPod -> Settings -> S3 API Keys  (access key + secret)
#   - aws configure --profile runpod     (region eu-ro-1)
#
# Usage:  scripts\runpod\pull_runs.ps1            # sync runs/ into .\runs
#         scripts\runpod\pull_runs.ps1 -List      # just list what's on the volume
param(
    [switch]$List
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
$Bucket   = "9v22kl54a0"
$Region   = "eu-ro-1"
$Endpoint = "https://s3api-eu-ro-1.runpod.io"
$Profile  = "runpod"
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
Write-Host ">> done."
