# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# Model access/login and video inference remain explicit steps in README.md.
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Executable, [string[]]$ArgumentList)
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable"
    }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Install Python 3.12 with the Windows Python launcher (py) first.'
    }
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        Invoke-Checked -Executable 'py' -ArgumentList @('-3.12', '-m', 'venv', '.venv')
    }
    Invoke-Checked -Executable $projectPython -ArgumentList @('-c', 'import sys; assert sys.version_info[:2] == (3, 12), "This setup requires Python 3.12"')
    Invoke-Checked -Executable $projectPython -ArgumentList @('-m', 'pip', 'install', 'torch==2.11.0', 'torchvision==0.26.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Invoke-Checked -Executable $projectPython -ArgumentList @('-m', 'pip', 'install', '-r', 'requirements-phase1.txt', '-c', 'requirements-phase1-lock.txt')
    Invoke-Checked -Executable $projectPython -ArgumentList @('-m', 'pip', 'check')
    Invoke-Checked -Executable $projectPython -ArgumentList @('phase1_run.py', '--check-env')
    Write-Host 'Environment ready. Follow README.md to log in to Hugging Face, prepare SAM3 and add a video.'
}
finally {
    Pop-Location
}
