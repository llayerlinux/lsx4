[CmdletBinding()]
param(
    [string]$AndroidNdk,
    [ValidateRange(21, 99)]
    [int]$AndroidApi = 29,
    [ValidateRange(1, 128)]
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount),
    [string]$MsysBash = "C:\msys64\usr\bin\bash.exe",
    [string]$WorkRoot,
    [string]$OutputRoot,
    [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$version = "7.1.1"
$archiveSha256 = "733984395E0DBBE5C046ABDA2DC49A5544E7E0E1E2366BBA849222AE9E3A03B1"
if (-not $WorkRoot) {
    $WorkRoot = Join-Path $repoRoot "build\lsx4-ffmpeg-android-pic-work"
}
$workRoot = [IO.Path]::GetFullPath($WorkRoot)
$archive = Join-Path $workRoot "ffmpeg-$version.tar.xz"
$sourceRoot = Join-Path $workRoot "ffmpeg-$version"
$buildRoot = Join-Path $workRoot "build-arm64-api$AndroidApi"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repoRoot "build\lsx4-ffmpeg-android-arm64-pic"
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
foreach ($configuredRoot in @($workRoot, $OutputRoot)) {
    $leaf = Split-Path -Leaf $configuredRoot
    if ($leaf -notlike "lsx4-ffmpeg-*") {
        throw "FFmpeg work/output directories must use an lsx4-ffmpeg-* leaf name: $configuredRoot"
    }
}

if (-not $AndroidNdk) {
    $AndroidNdk = $env:ANDROID_NDK_HOME
}
if (-not $AndroidNdk) {
    $AndroidNdk = $env:ANDROID_NDK_ROOT
}
if (-not $AndroidNdk) {
    $ndkParent = Join-Path $env:LOCALAPPDATA "Android\Sdk\ndk"
    $AndroidNdk = Get-ChildItem $ndkParent -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $AndroidNdk -or -not (Test-Path $AndroidNdk -PathType Container)) {
    throw "Android NDK was not found. Pass -AndroidNdk or set ANDROID_NDK_HOME."
}
if (-not (Test-Path $MsysBash -PathType Leaf)) {
    throw "MSYS2 bash was not found at $MsysBash"
}

$ndkBin = Join-Path $AndroidNdk "toolchains\llvm\prebuilt\windows-x86_64\bin"
$cc = Join-Path $ndkBin "aarch64-linux-android$AndroidApi-clang.cmd"
$cxx = Join-Path $ndkBin "aarch64-linux-android$AndroidApi-clang++.cmd"
foreach ($tool in @($cc, $cxx, (Join-Path $ndkBin "llvm-ar.exe"),
        (Join-Path $ndkBin "llvm-readelf.exe"))) {
    if (-not (Test-Path $tool -PathType Leaf)) {
        throw "Required NDK tool is missing: $tool"
    }
}

function Convert-ToMsysPath([string]$Path) {
    $fullPath = [IO.Path]::GetFullPath($Path)
    if ($fullPath.Contains("'")) {
        throw "Apostrophes in build paths are not supported: $fullPath"
    }
    $converted = & $MsysBash -lc "cygpath -u -- '$fullPath'"
    if ($LASTEXITCODE -ne 0) {
        throw "cygpath failed for $fullPath"
    }
    return $converted.Trim()
}

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try {
            return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace("-", "")
        } finally {
            $algorithm.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Assert-AndroidFfmpegArchive {
    $relocCheck = Join-Path $workRoot "reloc-check"
    if (Test-Path $relocCheck) {
        Remove-Item -LiteralPath $relocCheck -Recurse -Force
    }
    New-Item -ItemType Directory -Force $relocCheck | Out-Null
    Push-Location $relocCheck
    try {
        & (Join-Path $ndkBin "llvm-ar.exe") x `
            (Join-Path $OutputRoot "lib\libavutil.a") tx_float.o tx_float_neon.o
        if ($LASTEXITCODE -ne 0) {
            throw "Could not extract FFmpeg relocation-check objects"
        }
        $symbols = & (Join-Path $ndkBin "llvm-readelf.exe") --symbols "tx_float.o"
        if (-not ($symbols -match "GLOBAL\s+HIDDEN.+ff_tx_tab_32_float")) {
            throw "FFmpeg tx tables are not hidden; the archive is unsafe for an Android shared object"
        }
        $header = & (Join-Path $ndkBin "llvm-readelf.exe") -h "tx_float_neon.o"
        if (-not ($header -match "Machine:\s+AArch64")) {
            throw "FFmpeg object is not AArch64"
        }
    } finally {
        Pop-Location
    }
}

$requiredLibraries = @(
    "libavformat.a", "libavcodec.a", "libswscale.a", "libavutil.a",
    "libavfilter.a", "libswresample.a"
)
$installComplete = Test-Path (Join-Path $OutputRoot "include\libavcodec\avcodec.h")
foreach ($library in $requiredLibraries) {
    $installComplete = $installComplete -and (Test-Path (Join-Path $OutputRoot "lib\$library"))
}
if ($installComplete -and -not $ForceRebuild) {
    Assert-AndroidFfmpegArchive
    Write-Host "Reusing Android arm64 PIC FFmpeg at $OutputRoot"
    Write-Host "Verified: AArch64 ELF objects and hidden ff_tx_tab_* relocation targets."
    return
}

New-Item -ItemType Directory -Force $workRoot | Out-Null
if (-not (Test-Path $archive -PathType Leaf)) {
    & curl.exe -L --fail --retry 3 -o $archive "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz"
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg download failed"
    }
}
$actualSha256 = Get-Sha256 $archive
if ($actualSha256 -ne $archiveSha256) {
    throw "FFmpeg archive hash mismatch: expected $archiveSha256, got $actualSha256"
}
if (-not (Test-Path $sourceRoot -PathType Container)) {
    & tar.exe -xf $archive -C $workRoot
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg source extraction failed"
    }
}

foreach ($target in @($buildRoot, $OutputRoot)) {
    $resolvedTarget = [IO.Path]::GetFullPath($target)
    $allowedRoots = @(
        $repoRoot,
        [IO.Path]::GetFullPath([IO.Path]::GetTempPath()),
        $workRoot,
        $OutputRoot
    )
    $isAllowed = $false
    foreach ($allowedRoot in $allowedRoots) {
        if ($resolvedTarget.Equals($allowedRoot, [StringComparison]::OrdinalIgnoreCase) -or
            ($resolvedTarget.StartsWith($allowedRoot + [IO.Path]::DirectorySeparatorChar,
                [StringComparison]::OrdinalIgnoreCase))) {
            $isAllowed = $true
            break
        }
    }
    if (-not $isAllowed) {
        throw "Refusing to remove a build path outside the repository or temporary directory: $resolvedTarget"
    }
    if (Test-Path $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
    New-Item -ItemType Directory -Force $resolvedTarget | Out-Null
}

$sourceMsys = Convert-ToMsysPath $sourceRoot
$buildMsys = Convert-ToMsysPath $buildRoot
$outputMsys = Convert-ToMsysPath $OutputRoot
$ndkBinMsys = Convert-ToMsysPath $ndkBin
$configureCommand = @"
set -euo pipefail
export PATH="/mingw64/bin:${ndkBinMsys}:`$PATH"
cd "${buildMsys}"
"${sourceMsys}/configure" \
  --prefix="${outputMsys}" \
  --target-os=android --arch=aarch64 --enable-cross-compile \
  --cc=aarch64-linux-android${AndroidApi}-clang.cmd \
  --cxx=aarch64-linux-android${AndroidApi}-clang++.cmd \
  --host-cc=/mingw64/bin/gcc.exe \
  --ar=llvm-ar.exe --nm=llvm-nm.exe --ranlib=llvm-ranlib.exe --strip=llvm-strip.exe \
  --enable-pic --extra-cflags=-fPIC --extra-cflags=-fvisibility=hidden \
  --extra-cxxflags=-fPIC --extra-cxxflags=-fvisibility=hidden --extra-ldflags=-fPIC \
  --disable-autodetect --disable-doc --disable-programs --disable-network --disable-everything \
  --enable-avcodec --enable-avfilter --enable-avformat --enable-avutil \
  --enable-swresample --enable-swscale \
  --enable-decoder=aac --enable-decoder=aac_latm --enable-decoder=atrac3 \
  --enable-decoder=atrac3p --enable-decoder=atrac9 --enable-decoder=mp3 \
  --enable-decoder=pcm_s16le --enable-decoder=pcm_s8 --enable-decoder=h264 \
  --enable-decoder=mpeg4 --enable-decoder=mpeg2video --enable-decoder=mjpeg \
  --enable-decoder=mjpegb --enable-decoder=hevc \
  --enable-encoder=pcm_s16le --enable-encoder=ffv1 --enable-encoder=mpeg4 \
  --enable-encoder=ljpeg --enable-encoder=mjpeg --enable-muxer=avi \
  --enable-demuxer=hevc --enable-demuxer=h264 --enable-demuxer=m4v \
  --enable-demuxer=mp3 --enable-demuxer=mpegvideo --enable-demuxer=mpegps \
  --enable-demuxer=mjpeg --enable-demuxer=mov --enable-demuxer=avi \
  --enable-demuxer=aac --enable-demuxer=pmp --enable-demuxer=oma \
  --enable-demuxer=pcm_s16le --enable-demuxer=pcm_s8 --enable-demuxer=wav \
  --enable-parser=h264 --enable-parser=mpeg4video --enable-parser=mpegaudio \
  --enable-parser=mpegvideo --enable-parser=mjpeg --enable-parser=aac \
  --enable-parser=aac_latm --enable-protocol=file --enable-bsf=mjpeg2jpeg
"@
$buildCommand = @"
set -euo pipefail
export PATH="/mingw64/bin:${ndkBinMsys}:`$PATH"
cd "${buildMsys}"
make -j${Jobs}
make install
"@

$previousMsystem = $env:MSYSTEM
try {
    # The host table generators use MSYS2's MinGW64 compiler.  Keeping the
    # MinGW bin directory before the NDK bin directory prevents GCC from
    # accidentally picking Android's linker.
    $env:MSYSTEM = "MINGW64"
    & $MsysBash -lc $configureCommand
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg configure failed with exit code $LASTEXITCODE"
    }
    & $MsysBash -lc $buildCommand
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg build/install failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:MSYSTEM = $previousMsystem
}

Assert-AndroidFfmpegArchive
Write-Host "Built FFmpeg $version Android arm64 PIC at $OutputRoot"
Write-Host "Verified: AArch64 ELF objects and hidden ff_tx_tab_* relocation targets."
