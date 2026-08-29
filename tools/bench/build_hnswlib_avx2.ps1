[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Python,

    [string] $OutputDirectory = (
        Join-Path $PSScriptRoot '..\..\out\comparators\hnswlib-v0.8.0-msvc-avx2'
    ),

    [string] $VsDevCmd = (
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
    ),

    [string] $StockModule
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sourceCommit = '3f3429661187e4c24a490a0f148fc6bc89042b3d'
$sourceRepository = 'https://github.com/nmslib/hnswlib.git'
$sourceTag = 'v0.8.0'
$schemaVersion = 'ovvs.hnswlib-build.v1'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string] $Executable,
        [Parameter(Mandatory = $true)][string[]] $Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

function Get-FileRecord {
    param([Parameter(Mandatory = $true)][System.IO.FileInfo] $File)
    return [ordered]@{
        name = $File.Name
        bytes = $File.Length
        sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Get-SimdRecord {
    param(
        [Parameter(Mandatory = $true)][string] $Objdump,
        [Parameter(Mandatory = $true)][System.IO.FileInfo] $Module,
        [Parameter(Mandatory = $true)][string] $ArtifactDirectory,
        [Parameter(Mandatory = $true)][string] $Label
    )
    $disassemblyPath = Join-Path $ArtifactDirectory "$Label.disassembly.txt"
    & $Objdump -d --no-show-raw-insn $Module.FullName 2>&1 |
        Set-Content -LiteralPath $disassemblyPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "llvm-objdump failed for $($Module.FullName)"
    }
    $text = Get-Content -LiteralPath $disassemblyPath -Raw
    return [ordered]@{
        module = Get-FileRecord $Module
        disassembly = Get-FileRecord (Get-Item -LiteralPath $disassemblyPath)
        ymm_operands = ([regex]::Matches($text, '\bymm\d+\b')).Count
        vsubps = ([regex]::Matches($text, '\bvsubps\b')).Count
        vmulps = ([regex]::Matches($text, '\bvmulps\b')).Count
        vaddps = ([regex]::Matches($text, '\bvaddps\b')).Count
        vmovups = ([regex]::Matches($text, '\bvmovups\b')).Count
        legacy_subps = ([regex]::Matches($text, '(?<!v)\bsubps\b')).Count
        legacy_mulps = ([regex]::Matches($text, '(?<!v)\bmulps\b')).Count
    }
}

function Resolve-VsTool {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo] $VsDevCmdFile,
        [Parameter(Mandatory = $true)][string] $Name
    )
    $query = "call `"$($VsDevCmdFile.FullName)`" -arch=amd64 -host_arch=amd64 >nul && where.exe $Name"
    $matches = @(
        & $env:ComSpec /d /s /c $query |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
    if ($LASTEXITCODE -ne 0 -or $matches.Count -eq 0) {
        throw "could not resolve required Visual Studio tool: $Name"
    }
    return Get-Item -LiteralPath $matches[0]
}

$pythonFile = Get-Item -LiteralPath $Python
$vsDevCmdFile = Get-Item -LiteralPath $VsDevCmd
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $outputPath) {
    throw "refusing to overwrite an existing comparator directory: $outputPath"
}

$null = New-Item -ItemType Directory -Path $outputPath
$sourcePath = Join-Path $outputPath 'source'
$venvPath = Join-Path $outputPath '.venv-build'
$dependencyWheelhouse = Join-Path $outputPath 'dependency-wheelhouse'
$wheelPath = Join-Path $outputPath 'wheel'
$modulePath = Join-Path $outputPath 'module'
$artifactsPath = Join-Path $outputPath 'verification'
foreach ($directory in @($dependencyWheelhouse, $wheelPath, $modulePath, $artifactsPath)) {
    $null = New-Item -ItemType Directory -Path $directory
}

Invoke-Checked git @(
    'clone', '--depth', '1', '--branch', $sourceTag, $sourceRepository, $sourcePath
)
$resolvedCommit = (& git -C $sourcePath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $resolvedCommit -ne $sourceCommit) {
    throw "source commit mismatch: expected $sourceCommit, found $resolvedCommit"
}
$sourceTree = (& git -C $sourcePath rev-parse 'HEAD^{tree}').Trim()
$sourceEpoch = (& git -C $sourcePath show -s --format=%ct HEAD).Trim()

Invoke-Checked $pythonFile.FullName @('-m', 'venv', $venvPath)
$buildPython = Get-Item -LiteralPath (Join-Path $venvPath 'Scripts\python.exe')
$dependencyPins = @(
    'pip==26.1.1',
    'build==1.5.0',
    'setuptools==80.9.0',
    'wheel==0.45.1',
    'pybind11==2.13.6',
    'numpy==2.2.6',
    'packaging==25.0',
    'pyproject-hooks==1.2.0',
    'colorama==0.4.6'
)
$downloadArguments = @(
    '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps', '--dest', $dependencyWheelhouse
) + $dependencyPins
$installArguments = @(
    '-m', 'pip', 'install', '--no-index', '--no-deps', '--find-links', $dependencyWheelhouse
) + $dependencyPins
Invoke-Checked $buildPython.FullName $downloadArguments
Invoke-Checked $buildPython.FullName $installArguments

$buildLog = Join-Path $artifactsPath 'build.log'
$buildCommand = (
    "call `"$($vsDevCmdFile.FullName)`" -arch=amd64 -host_arch=amd64 >nul " +
    "&& set `"VSLANG=1033`" && set `"DISTUTILS_USE_SDK=1`" && set `"MSSdk=1`" " +
    "&& set `"PYTHONNOUSERSITE=1`" && set `"PIP_NO_INDEX=1`" " +
    "&& set `"PYTHONHASHSEED=0`" && set `"SOURCE_DATE_EPOCH=$sourceEpoch`" " +
    "&& set `"PYTHONPATH=`" && set `"CFLAGS=`" && set `"CXXFLAGS=`" " +
    "&& set `"_CL_=`" && set `"LDFLAGS=`" " +
    "&& set `"CL=/O2 /openmp /arch:AVX2`" && set `"LINK=/Brepro`" " +
    "&& `"$($buildPython.FullName)`" -m build --wheel --no-isolation " +
    "--outdir `"$wheelPath`" `"$sourcePath`""
)
& $env:ComSpec /d /s /c $buildCommand 2>&1 |
    Tee-Object -FilePath $buildLog
if ($LASTEXITCODE -ne 0) {
    throw "hnswlib build failed with exit code $LASTEXITCODE"
}

$wheels = @(Get-ChildItem -LiteralPath $wheelPath -Filter '*.whl' -File)
if ($wheels.Count -ne 1) {
    throw "expected one hnswlib wheel, found $($wheels.Count)"
}
Invoke-Checked $buildPython.FullName @(
    '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', $modulePath,
    $wheels[0].FullName
)
$modules = @(Get-ChildItem -LiteralPath $modulePath -Filter 'hnswlib*.pyd' -File)
if ($modules.Count -ne 1) {
    throw "expected one hnswlib extension, found $($modules.Count)"
}
$module = $modules[0]

$cl = Resolve-VsTool $vsDevCmdFile 'cl.exe'
$dumpbin = Resolve-VsTool $vsDevCmdFile 'dumpbin.exe'
$objdump = Resolve-VsTool $vsDevCmdFile 'llvm-objdump.exe'

$headersPath = Join-Path $artifactsPath 'candidate.headers.txt'
$dependentsPath = Join-Path $artifactsPath 'candidate.dependents.txt'
& $dumpbin.FullName /headers $module.FullName 2>&1 |
    Set-Content -LiteralPath $headersPath -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'dumpbin /headers failed' }
& $dumpbin.FullName /dependents $module.FullName 2>&1 |
    Set-Content -LiteralPath $dependentsPath -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'dumpbin /dependents failed' }
$headersText = Get-Content -LiteralPath $headersPath -Raw
$dependentsText = Get-Content -LiteralPath $dependentsPath -Raw
if ($headersText -notmatch '8664 machine \(x64\)') {
    throw 'candidate is not an x64 PE image'
}

$candidateSimd = Get-SimdRecord $objdump.FullName $module $artifactsPath 'candidate'
if (
    $candidateSimd.ymm_operands -le 0 -or
    $candidateSimd.vsubps -le 0 -or
    $candidateSimd.vmulps -le 0 -or
    $candidateSimd.vaddps -le 0 -or
    $candidateSimd.vmovups -le 0
) {
    throw 'candidate does not contain the required AVX/YMM float-distance evidence'
}

$stockSimd = $null
if ($StockModule) {
    $stockFile = Get-Item -LiteralPath $StockModule
    $stockSimd = Get-SimdRecord $objdump.FullName $stockFile $artifactsPath 'stock'
}

$smokeCode = @'
import importlib.util
import json
import sys
import numpy as np

module_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("hnswlib", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError("module spec unavailable")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data = np.arange(16 * 128, dtype=np.float32).reshape(16, 128)
index = module.Index(space="l2", dim=128)
index.init_index(max_elements=16, ef_construction=20, M=8, random_seed=7)
index.add_items(data, np.arange(16), num_threads=2)
index.set_ef(10)
ids, distances = index.knn_query(data[:2], k=3, num_threads=2)
print(json.dumps({
    "status": "success",
    "ids": ids.tolist(),
    "finite_distances": bool(np.isfinite(distances).all()),
}, sort_keys=True))
'@
$smokeOutput = & $buildPython.FullName -c $smokeCode $module.FullName
if ($LASTEXITCODE -ne 0) { throw 'candidate import/add/query smoke failed' }
$importSmoke = $smokeOutput | ConvertFrom-Json
if ($importSmoke.status -ne 'success' -or $importSmoke.finite_distances -ne $true) {
    throw 'candidate import/add/query smoke emitted an invalid result'
}

$pythonVersion = (& $pythonFile.FullName --version 2>&1).Trim()
$pythonCacheTag = (& $pythonFile.FullName -c 'import sys; print(sys.implementation.cache_tag)').Trim()
$compilerVersion = (& $env:ComSpec /d /s /c (
    "call `"$($vsDevCmdFile.FullName)`" -arch=amd64 -host_arch=amd64 >nul && cl 2>&1"
) | Select-String -Pattern 'Compiler Version' | Select-Object -First 1).Line.Trim()
$dependencyWheels = @(
    Get-ChildItem -LiteralPath $dependencyWheelhouse -Filter '*.whl' -File |
        Sort-Object Name |
        ForEach-Object { Get-FileRecord $_ }
)
& git -C $sourcePath diff --quiet
if ($LASTEXITCODE -ne 0) {
    throw 'tracked hnswlib source files changed during the comparator build'
}
$sourceTrackedClean = $true

$manifest = [ordered]@{
    schema_version = $schemaVersion
    source = [ordered]@{
        repository = $sourceRepository
        tag = $sourceTag
        version = '0.8.0'
        commit = $resolvedCommit
        tree = $sourceTree
        tracked_files_clean_after_build = $sourceTrackedClean
    }
    toolchain = [ordered]@{
        python_path = $pythonFile.FullName
        python_version = $pythonVersion
        python_cache_tag = $pythonCacheTag
        compiler = $compilerVersion
        compiler_path = $cl.FullName
        dumpbin_path = $dumpbin.FullName
        llvm_objdump_path = $objdump.FullName
        compile_options = @('/O2', '/openmp', '/EHsc', '/arch:AVX2')
        injected_environment = [ordered]@{
            CL = '/O2 /openmp /arch:AVX2'
            LINK = '/Brepro'
            SOURCE_DATE_EPOCH = $sourceEpoch
            PYTHONHASHSEED = '0'
            PYTHONNOUSERSITE = '1'
            PIP_NO_INDEX = '1'
            PYTHONPATH = ''
            CFLAGS = ''
            CXXFLAGS = ''
            _CL_ = ''
            LDFLAGS = ''
        }
    }
    dependencies = [ordered]@{
        pins = $dependencyPins
        wheels = $dependencyWheels
    }
    build = [ordered]@{
        command = $buildCommand
        log = Get-FileRecord (Get-Item -LiteralPath $buildLog)
        wheel = Get-FileRecord $wheels[0]
    }
    binary = [ordered]@{
        path = $module.FullName
        bytes = $module.Length
        sha256 = (Get-FileHash -LiteralPath $module.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    verification = [ordered]@{
        scope = 'static binary evidence; not runtime active-cycle attribution'
        expected_hot_path = 'hnswlib v0.8.0 AVX/YMM float distance kernels on an AVX2-capable x64 CPU'
        pe_machine = 'x64'
        threading_runtime = 'python binding ParallelFor uses std::thread; upstream /openmp is retained but VCOMP140.DLL is not expected without an OpenMP pragma'
        vcomp140_imported = [bool]($dependentsText -match 'VCOMP140\.DLL')
        import_add_query_smoke = $importSmoke
        candidate = $candidateSimd
        stock_negative_control = $stockSimd
        headers = Get-FileRecord (Get-Item -LiteralPath $headersPath)
        dependents = Get-FileRecord (Get-Item -LiteralPath $dependentsPath)
    }
}
$manifestPath = Join-Path $outputPath 'hnswlib-build.json'
$manifest | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Output "module=$($module.FullName)"
Write-Output "provenance=$manifestPath"
Write-Output "sha256=$($manifest.binary.sha256)"
