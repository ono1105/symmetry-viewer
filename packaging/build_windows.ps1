# Build the packaged Windows version into dist\:
#
#   dist\symmetry-viewer\                    the program folder
#   dist\symmetry-viewer-windows-x64.zip     what gets handed out
#
# Runs on the GitHub Actions Windows runner (the release repository's
# .github/workflows/build.yml). Needs PowerShell 7, tar (built into Windows) and
# an internet connection. Like build_linux.sh it builds on a pinned
# python-build-standalone CPython, so both versions come from the same Python.
#
# Check the result with:
#   build\venv\Scripts\python.exe packaging\smoke_test.py dist\symmetry-viewer\symmetry-viewer.exe
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
Set-Location (Split-Path -Parent $PSScriptRoot)

$PbsTag = "20260901"
$PbsFile = "cpython-3.14.7+$PbsTag-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
$PbsSha256 = "ca3c33ca924dfcab3b74205a7a58a88b0255135c53f95497b26b5e60700fd66d"
$PbsUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsTag/$($PbsFile.Replace('+', '%2B'))"

$BuildDir = "build"
$PythonDir = Join-Path $BuildDir "python"
$VenvDir = Join-Path $BuildDir "venv"
$DistDir = "dist"
$AppDir = Join-Path $DistDir "symmetry-viewer"
$Archive = Join-Path $DistDir "symmetry-viewer-windows-x64.zip"

function Test-Tag([string] $Path) {
    return (Test-Path $Path) -and ((Get-Content $Path) -eq $PbsTag)
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
if (-not (Test-Tag (Join-Path $PythonDir ".pbs-tag"))) {
    $Download = Join-Path $BuildDir $PbsFile
    Invoke-WebRequest -Uri $PbsUrl -OutFile $Download
    $Actual = (Get-FileHash -Algorithm SHA256 $Download).Hash.ToLowerInvariant()
    if ($Actual -ne $PbsSha256) { throw "SHA256 mismatch for ${PbsFile}: $Actual" }
    if (Test-Path $PythonDir) { Remove-Item -Recurse -Force $PythonDir }
    tar -C $BuildDir -xzf $Download   # unpacks into build\python
    Remove-Item $Download
    Set-Content -Path (Join-Path $PythonDir ".pbs-tag") -Value $PbsTag
}

if (-not (Test-Tag (Join-Path $VenvDir ".pbs-tag"))) {
    if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
    & (Join-Path $PythonDir "python.exe") -m venv $VenvDir
    Set-Content -Path (Join-Path $VenvDir ".pbs-tag") -Value $PbsTag
}
$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -m pip install --quiet --upgrade pip
& $Python -m pip install --quiet -r packaging\requirements-build.txt

if (Test-Path $AppDir) { Remove-Item -Recurse -Force $AppDir }
if (Test-Path $Archive) { Remove-Item -Force $Archive }
& $Python -m PyInstaller --noconfirm --clean --distpath $DistDir --workpath (Join-Path $BuildDir "pyinstaller") packaging\symmetry_view.spec

Copy-Item packaging\README_windows.txt (Join-Path $AppDir "README.txt")
Copy-Item LICENSE (Join-Path $AppDir "LICENSE.txt")
tar -a -c -f $Archive -C $DistDir symmetry-viewer

"{0:N0} MB  {1}" -f ((Get-ChildItem -Recurse -File $AppDir | Measure-Object Length -Sum).Sum / 1MB), $AppDir
"{0:N0} MB  {1}" -f ((Get-Item $Archive).Length / 1MB), $Archive
"Built $Archive"
