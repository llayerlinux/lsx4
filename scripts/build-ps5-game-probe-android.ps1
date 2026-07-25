param(
    [string]$NdkRoot = "$env:LOCALAPPDATA\Android\Sdk\ndk\29.0.14206865",
    [int]$AndroidApi = 29,
    [string]$Output = "build\runtime-isolation-on\ps5_game_probe"
)

$ErrorActionPreference = "Stop"
$compiler = Join-Path $NdkRoot "toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe"
$sysroot = Join-Path $NdkRoot "toolchains\llvm\prebuilt\windows-x86_64\sysroot"
if (-not (Test-Path -LiteralPath $compiler)) {
    throw "NDK clang++ was not found at $compiler"
}

$outputPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\$Output"))
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

& $compiler `
    "--target=aarch64-none-linux-android$AndroidApi" `
    "--sysroot=$sysroot" `
    -std=c++23 -O2 -fPIE -pie `
    "-I$(Join-Path $PSScriptRoot '..\src')" `
    (Join-Path $PSScriptRoot "..\tools\ps5_game_probe.cpp") `
    -o $outputPath -ldl
if ($LASTEXITCODE -ne 0) {
    throw "PS5 game probe build failed with exit code $LASTEXITCODE"
}
Write-Host "Built $outputPath"
