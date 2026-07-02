# Run signal discovery on the bot's own broker data (H1 panels, 23 pairs).
#
# Prereqs (one-time, already done):
#   - panels exported:  py ..\export_discovery_panel.py
#   - NAT installed:    UV_PROJECT_ENVIRONMENT=.venv-nat uv sync --locked
#
# Usage (from the signal-discovery directory):
#   $env:NVIDIA_API_KEY = "<your key>"     # same key the bot uses as LLM_API_KEY
#   .\run-discovery.ps1
#   .\run-discovery.ps1 -Request "volatility breakout signals for forex"

param(
    [string]$Request = "momentum, mean-reversion and volatility signals for forex pairs"
)

if (-not $env:NVIDIA_API_KEY -and $env:LLM_API_KEY) {
    $env:NVIDIA_API_KEY = $env:LLM_API_KEY
}
if (-not $env:NVIDIA_API_KEY) {
    Write-Error "Set NVIDIA_API_KEY (or LLM_API_KEY) before running."
    exit 1
}

$env:SIGNAL_DATA_DIR = Join-Path $PSScriptRoot "src\signal_discovery_workflow\data\forex"

& (Join-Path $PSScriptRoot ".venv-nat\Scripts\nat.exe") run `
    --config_file (Join-Path $PSScriptRoot "configs\config-bot-forex.yml") `
    --input $Request

Write-Host ""
Write-Host "Accepted signals land in: src\signal_discovery_workflow\output\signal_*.json"
Write-Host "The bot picks them up automatically (DISCOVERED_SIGNALS_ENABLED=True, hourly rescan)."
