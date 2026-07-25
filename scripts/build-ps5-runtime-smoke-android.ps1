param(
    [string]$NdkRoot = "$env:LOCALAPPDATA\Android\Sdk\ndk\29.0.14206865",
    [int]$AndroidApi = 29,
    [string]$Output = "build\runtime-isolation-on\ps5_runtime_smoke",
    [switch]$Sanitize
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

$compilerArguments = @(
    "--target=aarch64-none-linux-android$AndroidApi"
    "--sysroot=$sysroot"
    "-std=c++23"
    "-O2"
    "-Wall"
    "-Wextra"
    "-Werror"
    "-fPIE"
    "-pie"
)
if ($Sanitize) {
    $compilerArguments += @(
        "-fsanitize=address,undefined"
        "-fno-sanitize=vptr"
        "-fno-omit-frame-pointer"
        "-shared-libasan"
    )
}
$compilerArguments += @(
    "-I$(Join-Path $PSScriptRoot '..\src')"
    (Join-Path $PSScriptRoot "..\tests\ps5_runtime_smoke.cpp")
    "-o"
    $outputPath
    "-ldl"
)

& $compiler $compilerArguments
if ($LASTEXITCODE -ne 0) {
    throw "PS5 runtime smoke build failed with exit code $LASTEXITCODE"
}
Write-Host "Built $outputPath"
