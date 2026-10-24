$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskJava = (Get-ChildItem -LiteralPath (Join-Path $taskRoot 'tools/temurin21') -Filter java.exe -Recurse | Select-Object -First 1).FullName
if (-not $taskJava) { throw 'Portable Java runtime is missing.' }
$taskOutput = Join-Path $taskRoot 'data/source/candidates_5000/csv/patients.csv'
if (Test-Path -LiteralPath $taskOutput) { throw 'Candidate CSV already exists; do not overwrite the source snapshot.' }
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'logs') -Force | Out-Null
$taskArguments = @('-Xmx4g', '-jar', 'tools/synthea-with-dependencies.jar', '-p', '5000', '-s', '2026101601', '-cs', '2026101602', '-r', '20261016', '-e', '20261016', '-o', 'false', '-c', 'configs/synthea_candidates_5000.properties', 'Massachusetts')
& $taskJava @taskArguments *> (Join-Path $taskRoot 'logs/synthea_candidates_5000.log')
if ($LASTEXITCODE -ne 0) { throw "Synthea failed with exit code $LASTEXITCODE; inspect logs/synthea_candidates_5000.log." }
Write-Output 'Synthea generation completed.'
