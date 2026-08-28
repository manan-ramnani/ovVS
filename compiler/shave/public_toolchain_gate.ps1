#requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CompilerRoot,

    [Parameter(Mandatory = $true)]
    [string]$OpenVinoRoot,

    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$Configuration = 'RelWithDebInfo',

    [string]$BaselineRunner,

    [string]$MoviToolsRoot,
    [string]$EvidenceDir,
    [string]$OpenVinoBuildLog,
    [string]$CompilerBuildLog,

    [string]$CustomKernelName = 'ovvs_gate_identity',
    [string]$CustomKernelSource,
    [string]$CustomKernelDescriptor,
    [string]$CustomKernelElf,
    [string]$CustomKernelRegistration,
    [string]$ExpectedCustomElfSha256,
    [string]$CustomVerificationRunner,
    [string[]]$CustomVerificationArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum GateExitCode {
    Pass = 0
    BaselineBlocked = 10
    PublicToolchainUnsupported = 20
    ReadyForCustomVerification = 21
    CustomVerificationFailed = 30
    InvalidInvocation = 40
}

$ExpectedCompilerCommit = '6761af885b8ff54ddf0da5bf8ad44e30746b2f62'
$BaselineFilter = 'MatMulWithDivide/MatMulWithDivideTestCommon.NPU3720_TestKindSubgraph/*'
$BuiltInSoftwareFilter = '*ActivationLayerTest_SW_FP*NPU3720*Sigmoid*'

function Get-AbsolutePath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-PathInside([string]$Candidate, [string]$Parent) {
    if ([string]::IsNullOrWhiteSpace($Candidate) -or [string]::IsNullOrWhiteSpace($Parent)) {
        return $false
    }
    $candidateFull = (Get-AbsolutePath $Candidate).TrimEnd('\', '/')
    $parentFull = (Get-AbsolutePath $Parent).TrimEnd('\', '/')
    return $candidateFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $candidateFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Get-FileEvidence([string]$Path) {
    $exists = -not [string]::IsNullOrWhiteSpace($Path) -and
        (Test-Path -LiteralPath $Path -PathType Leaf)
    $result = [ordered]@{
        path = $Path
        exists = $exists
        length = $null
        sha256 = $null
        fileVersion = $null
        productVersion = $null
        creationUtc = $null
        lastWriteUtc = $null
    }
    if (-not $exists) {
        return $result
    }

    $item = Get-Item -LiteralPath $Path
    $result.length = $item.Length
    $result.sha256 = Get-Sha256 $Path
    $result.fileVersion = $item.VersionInfo.FileVersion
    $result.productVersion = $item.VersionInfo.ProductVersion
    $result.creationUtc = $item.CreationTimeUtc.ToString('o')
    $result.lastWriteUtc = $item.LastWriteTimeUtc.ToString('o')
    return $result
}

function Test-FileBuiltDuringFreshLog([string]$Artifact, [string]$Log) {
    if (-not (Test-Path -LiteralPath $Artifact -PathType Leaf) -or
        -not (Test-Path -LiteralPath $Log -PathType Leaf)) {
        return $false
    }
    $artifactItem = Get-Item -LiteralPath $Artifact
    $logItem = Get-Item -LiteralPath $Log
    $now = [DateTime]::UtcNow
    return $logItem.CreationTimeUtc -ge $now.AddHours(-24) -and
        $artifactItem.LastWriteTimeUtc -ge $logItem.CreationTimeUtc.AddSeconds(-2) -and
        $artifactItem.LastWriteTimeUtc -le $logItem.LastWriteTimeUtc.AddSeconds(2)
}

function Test-SuccessfulLoadAttribution([string]$Output, [string]$LibraryPath) {
    if ([string]::IsNullOrWhiteSpace($Output) -or [string]::IsNullOrWhiteSpace($LibraryPath)) {
        return $false
    }
    $escapedPath = [regex]::Escape($LibraryPath)
    $success = $Output -match "(?im)^(?:(?=.*\b(?:loaded|using)\b)(?=.*$escapedPath)).*$"
    $pathFailure = $Output -match "(?im)^(?:(?=.*\b(?:not\s+loaded|unable|fallback|fail(?:ed|ure)?|error|cannot)\b)(?=.*$escapedPath)).*$"
    $fallbackSelection = $Output -match '(?im)^.*\b(?:using|select(?:ed)?)\s+(?:the\s+)?(?:driver|production|installed)\s+compiler\b.*$'
    return $success -and -not $pathFailure -and -not $fallbackSelection
}

function Get-CMakeCacheValue([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $pattern = '^' + [regex]::Escape($Name) + ':[^=]+=(.*)$'
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match $pattern) {
            return $Matches[1].Trim()
        }
    }
    return $null
}

function Test-CMakeTrue([string]$Value) {
    return $Value -match '^(?i:1|ON|TRUE|YES|Y)$'
}

function Test-CMakeFalse([string]$Value) {
    return $Value -match '^(?i:0|OFF|FALSE|NO|N|IGNORE|NOTFOUND|.*-NOTFOUND)$'
}

function Test-PathEqual([string]$Left, [string]$Right) {
    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) {
        return $false
    }
    $leftFull = (Get-AbsolutePath $Left).TrimEnd('\', '/')
    $rightFull = (Get-AbsolutePath $Right).TrimEnd('\', '/')
    return $leftFull.Equals($rightFull, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-GitCommit([string]$Root) {
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        return $null
    }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $git) {
        return $null
    }
    $commit = & $git.Source -C $Root rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return ($commit | Select-Object -First 1).Trim()
}

function Get-GitState([string]$Root) {
    $result = [ordered]@{
        commit = Get-GitCommit $Root
        clean = $false
        status = @()
    }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $git -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
        return $result
    }
    $status = @(& $git.Source -C $Root status --porcelain=v1 --untracked-files=all 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return $result
    }
    $result.status = @($status)
    $result.clean = $status.Count -eq 0
    return $result
}

function Get-Elf32Evidence([string]$Path) {
    $result = [ordered]@{
        valid = $false
        class = $null
        data = $null
        type = $null
        machine = $null
        entry = $null
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $result
    }
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 28) {
            return $result
        }
        $header = [byte[]]::new(28)
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length) {
            return $result
        }
    } finally {
        $stream.Dispose()
    }
    if ($header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
        $header[2] -ne 0x4c -or $header[3] -ne 0x46) {
        return $result
    }
    $result.class = [int]$header[4]
    $result.data = [int]$header[5]
    if ($result.data -ne 1) {
        return $result
    }
    $result.type = [BitConverter]::ToUInt16($header, 16)
    $result.machine = [BitConverter]::ToUInt16($header, 18)
    $result.entry = ('0x{0:X8}' -f [BitConverter]::ToUInt32($header, 24))
    $result.valid = $result.class -eq 1 -and $result.type -eq 2 -and $result.machine -eq 2
    return $result
}

function Add-Blocker([string]$Stage, [string]$Code, [string]$Detail, [string]$Path = $null) {
    $script:gate.blockers.Add([ordered]@{
        stage = $Stage
        code = $Code
        detail = $Detail
        path = $Path
    })
}

function Write-GateEvidence {
    $script:gate.finishedUtc = [DateTime]::UtcNow.ToString('o')
    $jsonPath = Join-Path $script:evidenceRoot 'gate.json'
    $script:gate | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding utf8NoBOM

    $summaryPath = Join-Path $script:evidenceRoot 'gate-summary.md'
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# ovVS public NPU compiler gate')
    $lines.Add('')
    $lines.Add("- Status: ``$($script:gate.status)``")
    $lines.Add("- Exit code: ``$($script:gate.exitCode)``")
    $lines.Add("- Compiler commit: ``$($script:gate.source.compilerCommit)``")
    $lines.Add("- Required OpenVINO commit: ``$($script:gate.source.requiredOpenVinoCommit)``")
    $lines.Add("- Actual OpenVINO commit: ``$($script:gate.source.openVinoCommit)``")
    $lines.Add("- Baseline passed: ``$($script:gate.stages.unchangedGraph.passed)``")
    $lines.Add("- Built-in software-layer check passed: ``$($script:gate.stages.builtInSoftwareLayer.passed)``")
    $lines.Add("- Custom kernel executed: ``$($script:gate.stages.customKernel.executed)``")
    $lines.Add('')
    $lines.Add('## Blockers')
    $lines.Add('')
    if ($script:gate.blockers.Count -eq 0) {
        $lines.Add('- None.')
    } else {
        foreach ($blocker in $script:gate.blockers) {
            $suffix = if ($blocker.path) { " (``$($blocker.path)``)" } else { '' }
            $lines.Add("- ``$($blocker.code)``: $($blocker.detail)$suffix")
        }
    }
    $lines.Add('')
    $lines.Add('Built-in Intel ActShave binaries and host-linked ovVS C functions never count as a custom kernel.')
    $lines | Set-Content -LiteralPath $summaryPath -Encoding utf8NoBOM
}

function Complete-Gate([string]$Status, [GateExitCode]$ExitCode) {
    $script:gate.status = $Status
    $script:gate.exitCode = [int]$ExitCode
    Write-GateEvidence
    Write-Host "OVVS_NPU_COMPILER_GATE_RESULT=$Status"
    Write-Host "OVVS_NPU_COMPILER_GATE_EXIT=$([int]$ExitCode)"
    Write-Host "OVVS_NPU_COMPILER_GATE_EVIDENCE=$script:evidenceRoot"
    exit ([int]$ExitCode)
}

function Invoke-Captured([string]$Name, [string]$Executable, [string[]]$Arguments) {
    $logPath = Join-Path $script:evidenceRoot ($Name + '.log')
    $result = [ordered]@{
        executable = $Executable
        arguments = @($Arguments)
        exitCode = $null
        log = $logPath
        output = ''
    }

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        $result.output = "Executable not found: $Executable"
        $result.output | Set-Content -LiteralPath $logPath -Encoding utf8NoBOM
        return $result
    }

    $captured = @()
    try {
        $captured = @(& $Executable @Arguments 2>&1)
        $result.exitCode = $LASTEXITCODE
    } catch {
        $captured = @($_ | Out-String)
        $result.exitCode = -1
    }
    $result.output = ($captured | Out-String)
    $result.output | Set-Content -LiteralPath $logPath -Encoding utf8NoBOM
    return $result
}

function Test-GTestPass([object]$Run) {
    if ($null -eq $Run.exitCode -or $Run.exitCode -ne 0) {
        return $false
    }
    return $Run.output -match '(?m)^\[\s*PASSED\s*\]\s+[1-9][0-9]*\s+test'
}

function Restore-Environment([hashtable]$Saved) {
    foreach ($entry in $Saved.GetEnumerator()) {
        if ($null -eq $entry.Value) {
            Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
        } else {
            Set-Item -LiteralPath "Env:$($entry.Key)" -Value $entry.Value
        }
    }
}

function Invoke-PublicToolchainInventory {
    $kernelRoot = Join-Path $compilerRootFull 'sw_runtime_kernels\kernels'
    $kernelSourceRoot = Join-Path $kernelRoot 'src'
    $kernelDescriptorRoot = Join-Path $kernelRoot 'descrip'
    $prebuiltRoot = Join-Path $kernelRoot 'prebuild\act_shave_bin'
    $movitoolsManifestUnderscore = Join-Path $compilerRootFull 'artifacts\vpuip_2\revisions.json'
    $movitoolsManifestDocumented = Join-Path $compilerRootFull 'artifacts\vpuip2\revisions.json'
    $script:gate.stages.prerequisites = [ordered]@{
        sourceRootExists = Test-Path -LiteralPath $kernelSourceRoot -PathType Container
        descriptorRootExists = Test-Path -LiteralPath $kernelDescriptorRoot -PathType Container
        trackedSourceCount = 0
        trackedDescriptorCount = 0
        prebuiltElfCount = @(Get-ChildItem -LiteralPath $prebuiltRoot -Filter '*.elf' -File -ErrorAction SilentlyContinue).Count
        moviToolsManifestUnderscoreExists = Test-Path -LiteralPath $movitoolsManifestUnderscore -PathType Leaf
        moviToolsManifestDocumentedExists = Test-Path -LiteralPath $movitoolsManifestDocumented -PathType Leaf
        moviToolsRoot = $null
        moviCompile = $null
        moviLink = $null
        runtimeLibraries = @()
        builtInSigmoidElf = $null
    }

    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCommand) {
        $trackedKernelFiles = @(& $gitCommand.Source -C $compilerRootFull ls-tree -r --name-only HEAD -- sw_runtime_kernels/kernels 2>$null)
        $script:gate.stages.prerequisites.trackedSourceCount = @($trackedKernelFiles | Where-Object { $_ -like 'sw_runtime_kernels/kernels/src/*' }).Count
        $script:gate.stages.prerequisites.trackedDescriptorCount = @($trackedKernelFiles | Where-Object { $_ -like 'sw_runtime_kernels/kernels/descrip/*' }).Count
    }

    if ($script:gate.stages.prerequisites.trackedSourceCount -eq 0 -or
        $script:gate.stages.prerequisites.trackedDescriptorCount -eq 0) {
        Add-Blocker 'custom_dependency' 'public_kernel_sources_unavailable' "The public tree has $($script:gate.stages.prerequisites.prebuiltElfCount) prebuilt Intel ELFs but no tracked kernel sources/descriptors. With ENABLE_SHAVE_BINARIES_BUILD=ON, CMake removes every prebuilt lacking a descriptor, so the public tree cannot reproduce or safely update that set." $kernelRoot
    }

    if ([string]::IsNullOrWhiteSpace($MoviToolsRoot)) {
        $MoviToolsRoot = $env:IE_NPU_FORCE_MV_TOOLS_PATH
    }
    if ([string]::IsNullOrWhiteSpace($MoviToolsRoot)) {
        if (-not $script:gate.stages.prerequisites.moviToolsManifestUnderscoreExists -and
            -not $script:gate.stages.prerequisites.moviToolsManifestDocumentedExists) {
            Add-Blocker 'custom_dependency' 'movitools_manifest_unavailable' 'The public clone has neither manifest path referenced by the README/CMake downloader.' $movitoolsManifestUnderscore
        } else {
            Add-Blocker 'custom_dependency' 'movitools_not_materialized' 'The manifest exists, but no isolated MoviTools root was supplied.'
        }
    } else {
        $moviRootFull = Get-AbsolutePath $MoviToolsRoot
        $script:gate.stages.prerequisites.moviToolsRoot = $moviRootFull
        $moviCompile = Join-Path $moviRootFull 'win64\bin\moviCompile.exe'
        $moviLink = Join-Path $moviRootFull 'win64\bin\moviLLD.exe'
        $runtimeLibraries = @(
            (Join-Path $moviRootFull 'common\moviCompile\lib\37xxxx\mlibm.a'),
            (Join-Path $moviRootFull 'common\moviCompile\lib\37xxxx\mlibc_lite.a'),
            (Join-Path $moviRootFull 'common\moviCompile\lib\37xxxx\mlibcrt.a')
        )
        $script:gate.stages.prerequisites.moviCompile = Get-FileEvidence $moviCompile
        $script:gate.stages.prerequisites.moviLink = Get-FileEvidence $moviLink
        $script:gate.stages.prerequisites.runtimeLibraries = @(
            $runtimeLibraries | ForEach-Object { Get-FileEvidence $_ }
        )
        foreach ($requiredTool in @($moviCompile, $moviLink) + $runtimeLibraries) {
            if (-not (Test-Path -LiteralPath $requiredTool -PathType Leaf)) {
                Add-Blocker 'custom_dependency' 'movitools_file_missing' 'A required 3720xx compiler/link/runtime file is absent.' $requiredTool
            }
        }
    }

    $builtInSigmoid = Join-Path $prebuiltRoot 'activation_sigmoid.3720xx.elf'
    if (Test-Path -LiteralPath $builtInSigmoid -PathType Leaf) {
        $script:gate.stages.prerequisites.builtInSigmoidElf = [ordered]@{
            file = Get-FileEvidence $builtInSigmoid
            elf = Get-Elf32Evidence $builtInSigmoid
            classification = 'intel_prebuilt_not_custom'
        }
    }

    $customInputs = [ordered]@{
        source = Get-AbsolutePath $CustomKernelSource
        descriptor = Get-AbsolutePath $CustomKernelDescriptor
        elf = Get-AbsolutePath $CustomKernelElf
        registration = Get-AbsolutePath $CustomKernelRegistration
    }
    $script:gate.stages.customKernel.inputs = $customInputs

    foreach ($pair in $customInputs.GetEnumerator()) {
        if ([string]::IsNullOrWhiteSpace($pair.Value) -or -not (Test-Path -LiteralPath $pair.Value -PathType Leaf)) {
            Add-Blocker 'custom_input' "custom_$($pair.Key)_missing" "The exact custom $($pair.Key) artifact is required; an Intel prebuilt or host function cannot substitute." $pair.Value
        }
    }

    if ($customInputs.source -and (Test-Path -LiteralPath $customInputs.source -PathType Leaf)) {
        if (-not (Test-PathInside $customInputs.source $kernelSourceRoot)) {
            Add-Blocker 'custom_input' 'custom_source_not_packaged' 'The source must be in the compiler kernel source tree used by the descriptor.' $customInputs.source
        }
        if ([System.IO.Path]::GetFileNameWithoutExtension($customInputs.source) -ne $CustomKernelName) {
            Add-Blocker 'custom_input' 'custom_source_name_mismatch' 'The source basename must equal CustomKernelName.' $customInputs.source
        }
    }

    if ($customInputs.elf -and (Test-Path -LiteralPath $customInputs.elf -PathType Leaf)) {
        $actualName = [System.IO.Path]::GetFileName($customInputs.elf)
        $expectedName = "$CustomKernelName.3720xx.elf"
        $expectedPackagedElf = Join-Path $prebuiltRoot $expectedName
        $actualHash = Get-Sha256 $customInputs.elf
        $elfEvidence = Get-Elf32Evidence $customInputs.elf
        $script:gate.stages.customKernel.elfSha256 = $actualHash
        $script:gate.stages.customKernel.elf = $elfEvidence
        if ($actualName -ne $expectedName) {
            Add-Blocker 'custom_input' 'custom_elf_name_mismatch' "Expected $expectedName; found $actualName." $customInputs.elf
        }
        if (-not (Test-PathEqual $customInputs.elf $expectedPackagedElf)) {
            Add-Blocker 'custom_input' 'custom_elf_not_packaged' 'The exact ELF must be in prebuild/act_shave_bin for CMake resource embedding.' $customInputs.elf
        }
        if (-not $elfEvidence.valid -or $elfEvidence.entry -ne '0x1D000000') {
            Add-Blocker 'custom_input' 'custom_elf_format_invalid' 'Expected a little-endian ELF32 SPARC executable with entry 0x1D000000 from the public 3720xx linker script.' $customInputs.elf
        }
        if ([string]::IsNullOrWhiteSpace($ExpectedCustomElfSha256) -or
            $actualHash -ne $ExpectedCustomElfSha256.ToUpperInvariant()) {
            Add-Blocker 'custom_input' 'custom_elf_provenance_unverified' 'Pass the SHA-256 emitted by the isolated MoviTools build. Renaming a prebuilt ELF is not accepted.' $customInputs.elf
        }
        $stockHashes = @(Get-ChildItem -LiteralPath $prebuiltRoot -Filter '*.elf' -File -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-PathEqual $_.FullName $customInputs.elf) } |
            ForEach-Object { Get-Sha256 $_.FullName })
        if ($stockHashes -contains $actualHash) {
            Add-Blocker 'custom_input' 'custom_elf_matches_stock_prebuilt' 'The claimed custom ELF is byte-identical to an Intel prebuilt kernel.' $customInputs.elf
        }
    }

    if ($customInputs.descriptor -and (Test-Path -LiteralPath $customInputs.descriptor -PathType Leaf)) {
        if (-not (Test-PathInside $customInputs.descriptor $kernelDescriptorRoot)) {
            Add-Blocker 'custom_input' 'custom_descriptor_not_packaged' 'The descriptor must be in sw_runtime_kernels/kernels/descrip.' $customInputs.descriptor
        }
        $descriptorText = Get-Content -LiteralPath $customInputs.descriptor -Raw
        $sourceName = if ($customInputs.source) { [System.IO.Path]::GetFileName($customInputs.source) } else { "$CustomKernelName.c" }
        if ($descriptorText -notmatch ('(?m)set\s*\(\s*kernel_src\s+"?' + [regex]::Escape($sourceName) + '"?\s*\)') -or
            $descriptorText -notmatch '(?m)set\s*\(\s*kernel_cpunum\s+"?3720"?\s*\)') {
            Add-Blocker 'custom_input' 'custom_descriptor_mismatch' 'The descriptor must name the exact source and target kernel_cpunum 3720.' $customInputs.descriptor
        }
    }
    if ($customInputs.registration -and (Test-Path -LiteralPath $customInputs.registration -PathType Leaf)) {
        if (-not (Test-PathInside $customInputs.registration $compilerRootFull)) {
            Add-Blocker 'custom_input' 'custom_registration_outside_compiler' 'Registration must be part of the pinned compiler fork.' $customInputs.registration
        }
        $registrationText = Get-Content -LiteralPath $customInputs.registration -Raw
        if ($registrationText -notmatch [regex]::Escape($CustomKernelName)) {
            Add-Blocker 'custom_input' 'custom_registration_mismatch' 'The compiler lowering/kernel-info registration does not name the custom entry point.' $customInputs.registration
        }
    }
}

$compilerRootFull = Get-AbsolutePath $CompilerRoot
$openVinoRootFull = Get-AbsolutePath $OpenVinoRoot
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
if ([string]::IsNullOrWhiteSpace($EvidenceDir)) {
    $EvidenceDir = Join-Path ([System.IO.Path]::GetTempPath()) "ovvs-npu-compiler-gate\$stamp"
}
$script:evidenceRoot = Get-AbsolutePath $EvidenceDir
if ((Test-PathInside $script:evidenceRoot $compilerRootFull) -or
    (Test-PathInside $script:evidenceRoot $openVinoRootFull)) {
    [Console]::Error.WriteLine("Evidence must be outside both source trees: $script:evidenceRoot")
    Write-Host 'OVVS_NPU_COMPILER_GATE_RESULT=INVALID_INVOCATION'
    Write-Host "OVVS_NPU_COMPILER_GATE_EXIT=$([int][GateExitCode]::InvalidInvocation)"
    exit ([int][GateExitCode]::InvalidInvocation)
}
if (Test-Path -LiteralPath $script:evidenceRoot -PathType Container) {
    $existingEvidence = @(Get-ChildItem -LiteralPath $script:evidenceRoot -Force -ErrorAction Stop)
    if ($existingEvidence.Count -gt 0) {
        [Console]::Error.WriteLine("Evidence directory must be new or empty: $script:evidenceRoot")
        Write-Host 'OVVS_NPU_COMPILER_GATE_RESULT=INVALID_INVOCATION'
        Write-Host "OVVS_NPU_COMPILER_GATE_EXIT=$([int][GateExitCode]::InvalidInvocation)"
        exit ([int][GateExitCode]::InvalidInvocation)
    }
}
New-Item -ItemType Directory -Path $script:evidenceRoot -Force | Out-Null

$script:gate = [ordered]@{
    schemaVersion = 1
    startedUtc = [DateTime]::UtcNow.ToString('o')
    finishedUtc = $null
    status = 'RUNNING'
    exitCode = $null
    isolation = [ordered]@{
        evidenceDir = $script:evidenceRoot
        scriptDownloadsPerformed = $false
        scriptModifiedProductionRuntime = $false
        externalRunnerAttempted = $false
        externalRunnerEffectsAudited = $false
    }
    invocation = [ordered]@{
        platform = 'NPU3720'
        baselineFilter = $BaselineFilter
        builtInSoftwareFilter = $BuiltInSoftwareFilter
        customKernelName = $CustomKernelName
        expectedCustomElfSha256 = $ExpectedCustomElfSha256
    }
    source = [ordered]@{
        compilerRoot = $compilerRootFull
        compilerCommit = $null
        compilerGit = $null
        expectedCompilerCommit = $ExpectedCompilerCommit
        openVinoRoot = $openVinoRootFull
        openVinoCommit = $null
        openVinoGit = $null
        requiredOpenVinoCommit = $null
        configuration = $Configuration
        compilerBuildDir = Join-Path $compilerRootFull "build-x86_64\$Configuration"
        openVinoBuildDir = Join-Path $openVinoRootFull "build-x86_64\$Configuration"
    }
    stages = [ordered]@{
        unchangedGraph = [ordered]@{
            passed = $false
            loadedCompilerPathAttributed = $false
            buildProof = [ordered]@{}
            runner = $null
            list = $null
            run = $null
        }
        builtInSoftwareLayer = [ordered]@{
            passed = $false
            compilerInPluginExplicit = $false
            profilingAttributed = $false
            list = $null
            run = $null
            classification = 'intel_prebuilt_actshave_not_custom'
        }
        prerequisites = [ordered]@{}
        customKernel = [ordered]@{
            name = $CustomKernelName
            executed = $false
            oracleMatch = $false
            deviceAttributed = $false
            inputs = $null
            elfSha256 = $null
            elf = $null
            verificationRunner = $null
            verificationAccepted = $false
            run = $null
        }
    }
    blockers = [System.Collections.Generic.List[object]]::new()
    notes = @(
        'add_extension and ov::Op::evaluate are not NPU kernel registration',
        'host-linked compiler/shave C is an oracle only',
        'an unrelated or renamed Intel prebuilt ELF is not a custom ovVS kernel',
        'a loose SHAVE ELF is not a Level Zero native graph'
    )
}

# Inventory the exact source pair before running anything. The unchanged graph still
# remains the first executable gate.
$script:gate.source.compilerGit = Get-GitState $compilerRootFull
$script:gate.source.openVinoGit = Get-GitState $openVinoRootFull
$script:gate.source.compilerCommit = $script:gate.source.compilerGit.commit
$script:gate.source.openVinoCommit = $script:gate.source.openVinoGit.commit
$compatibilityFile = Join-Path $compilerRootFull 'validation\openvino_config.json'
if (Test-Path -LiteralPath $compatibilityFile -PathType Leaf) {
    try {
        $compatibility = Get-Content -LiteralPath $compatibilityFile -Raw | ConvertFrom-Json
        $script:gate.source.requiredOpenVinoCommit = [string]$compatibility.openvinotoolkit
    } catch {
        Add-Blocker 'unchanged_graph' 'invalid_openvino_compatibility_file' $_.Exception.Message $compatibilityFile
    }
} else {
    Add-Blocker 'unchanged_graph' 'missing_openvino_compatibility_file' 'The compiler compatibility pin is absent.' $compatibilityFile
}

if ($script:gate.source.compilerCommit -ne $ExpectedCompilerCommit) {
    Add-Blocker 'unchanged_graph' 'compiler_commit_mismatch' "Expected $ExpectedCompilerCommit; found $($script:gate.source.compilerCommit)." $compilerRootFull
} elseif (-not $script:gate.source.compilerGit.clean) {
    Add-Blocker 'unchanged_graph' 'compiler_source_dirty' 'The unchanged baseline requires a clean compiler tree. Preserve custom work in a separate commit/build after this evidence.' $compilerRootFull
}
if ([string]::IsNullOrWhiteSpace($script:gate.source.openVinoCommit)) {
    Add-Blocker 'unchanged_graph' 'openvino_source_missing' 'A source checkout and developer build are required; a production wheel/runtime is not a Compiler-In-Plugin build.' $openVinoRootFull
} elseif ($script:gate.source.requiredOpenVinoCommit -and
    $script:gate.source.openVinoCommit -ne $script:gate.source.requiredOpenVinoCommit) {
    Add-Blocker 'unchanged_graph' 'openvino_commit_mismatch' "Expected $($script:gate.source.requiredOpenVinoCommit); found $($script:gate.source.openVinoCommit)." $openVinoRootFull
} elseif (-not $script:gate.source.openVinoGit.clean) {
    Add-Blocker 'unchanged_graph' 'openvino_source_dirty' 'The unchanged baseline requires a clean pinned OpenVINO tree.' $openVinoRootFull
}

$compilerCache = Join-Path $script:gate.source.compilerBuildDir 'CMakeCache.txt'
$openVinoCache = Join-Path $script:gate.source.openVinoBuildDir 'CMakeCache.txt'
$expectedOpenVinoDeveloperPackage = $script:gate.source.openVinoBuildDir
$openVinoBuildLogFull = Get-AbsolutePath $OpenVinoBuildLog
$compilerBuildLogFull = Get-AbsolutePath $CompilerBuildLog
foreach ($buildLogPath in @($openVinoBuildLogFull, $compilerBuildLogFull)) {
    if ($buildLogPath -and ((Test-PathInside $buildLogPath $compilerRootFull) -or
            (Test-PathInside $buildLogPath $openVinoRootFull))) {
        Add-Blocker 'unchanged_graph' 'build_log_inside_source_tree' 'Retained build logs must be outside both source trees.' $buildLogPath
    }
}
$compilerBuildProof = [ordered]@{
    cmakeCache = Get-FileEvidence $compilerCache
    cmakeHomeDirectory = Get-CMakeCacheValue $compilerCache 'CMAKE_HOME_DIRECTORY'
    cmakeBuildType = Get-CMakeCacheValue $compilerCache 'CMAKE_BUILD_TYPE'
    developerBuild = Get-CMakeCacheValue $compilerCache 'ENABLE_DEVELOPER_BUILD'
    shaveBinariesBuild = Get-CMakeCacheValue $compilerCache 'ENABLE_SHAVE_BINARIES_BUILD'
    openVinoDeveloperPackage = Get-CMakeCacheValue $compilerCache 'OpenVINODeveloperPackage_DIR'
    buildLog = Get-FileEvidence $compilerBuildLogFull
}
$openVinoBuildProof = [ordered]@{
    cmakeCache = Get-FileEvidence $openVinoCache
    cmakeHomeDirectory = Get-CMakeCacheValue $openVinoCache 'CMAKE_HOME_DIRECTORY'
    cmakeBuildType = Get-CMakeCacheValue $openVinoCache 'CMAKE_BUILD_TYPE'
    intelNpuEnabled = Get-CMakeCacheValue $openVinoCache 'ENABLE_INTEL_NPU'
    inTreeNpuCompilerEnabled = Get-CMakeCacheValue $openVinoCache 'ENABLE_INTEL_NPU_COMPILER'
    buildLog = Get-FileEvidence $openVinoBuildLogFull
}
$script:gate.stages.unchangedGraph.buildProof = [ordered]@{
    compiler = $compilerBuildProof
    openVino = $openVinoBuildProof
    compilerLibraries = @()
    npuPluginLibraries = @()
}

if (-not $compilerBuildProof.cmakeCache.exists) {
    Add-Blocker 'unchanged_graph' 'compiler_cmake_cache_missing' 'Configure the pinned npu_compiler developer preset before running the gate.' $compilerCache
} else {
    if (-not (Test-PathEqual $compilerBuildProof.cmakeHomeDirectory $compilerRootFull)) {
        Add-Blocker 'unchanged_graph' 'compiler_cache_source_mismatch' "CMAKE_HOME_DIRECTORY is $($compilerBuildProof.cmakeHomeDirectory)." $compilerCache
    }
    if ($compilerBuildProof.cmakeBuildType -ne $Configuration) {
        Add-Blocker 'unchanged_graph' 'compiler_cache_configuration_mismatch' "Expected $Configuration; found $($compilerBuildProof.cmakeBuildType)." $compilerCache
    }
    if (-not (Test-CMakeTrue $compilerBuildProof.developerBuild)) {
        Add-Blocker 'unchanged_graph' 'compiler_not_developer_build' 'ENABLE_DEVELOPER_BUILD must be true so the local Compiler-In-Plugin is available.' $compilerCache
    }
    if (-not (Test-CMakeFalse $compilerBuildProof.shaveBinariesBuild)) {
        Add-Blocker 'unchanged_graph' 'baseline_shave_build_not_disabled' 'The clean unchanged baseline requires ENABLE_SHAVE_BINARIES_BUILD=OFF.' $compilerCache
    }
    if (-not (Test-PathEqual $compilerBuildProof.openVinoDeveloperPackage $expectedOpenVinoDeveloperPackage)) {
        Add-Blocker 'unchanged_graph' 'compiler_openvino_build_mismatch' "OpenVINODeveloperPackage_DIR is $($compilerBuildProof.openVinoDeveloperPackage)." $compilerCache
    }
}
if (-not $compilerBuildProof.buildLog.exists) {
    Add-Blocker 'unchanged_graph' 'compiler_build_log_missing' 'Retain the clean verbose compiler build log, including the pinned git commit, outside the source tree.' $compilerBuildLogFull
} else {
    $compilerBuildLogText = Get-Content -LiteralPath $compilerBuildLogFull -Raw
    if ($compilerBuildLogText -notmatch [regex]::Escape($ExpectedCompilerCommit) -or
        $compilerBuildLogText -notmatch 'openvino_intel_npu_compiler') {
        Add-Blocker 'unchanged_graph' 'compiler_build_log_unbound' 'The build log must contain the pinned compiler commit and core compiler target.' $compilerBuildLogFull
    }
}

if (-not $openVinoBuildProof.cmakeCache.exists) {
    Add-Blocker 'unchanged_graph' 'openvino_cmake_cache_missing' 'Configure the pinned isolated OpenVINO developer build before running the gate.' $openVinoCache
} else {
    if (-not (Test-PathEqual $openVinoBuildProof.cmakeHomeDirectory $openVinoRootFull)) {
        Add-Blocker 'unchanged_graph' 'openvino_cache_source_mismatch' "CMAKE_HOME_DIRECTORY is $($openVinoBuildProof.cmakeHomeDirectory)." $openVinoCache
    }
    if ($openVinoBuildProof.cmakeBuildType -ne $Configuration) {
        Add-Blocker 'unchanged_graph' 'openvino_cache_configuration_mismatch' "Expected $Configuration; found $($openVinoBuildProof.cmakeBuildType)." $openVinoCache
    }
    if (-not (Test-CMakeTrue $openVinoBuildProof.intelNpuEnabled)) {
        Add-Blocker 'unchanged_graph' 'openvino_npu_plugin_disabled' 'ENABLE_INTEL_NPU must be true.' $openVinoCache
    }
    if (-not (Test-CMakeFalse $openVinoBuildProof.inTreeNpuCompilerEnabled)) {
        Add-Blocker 'unchanged_graph' 'openvino_in_tree_compiler_enabled' 'ENABLE_INTEL_NPU_COMPILER must be false; the pinned external compiler supplies Compiler-In-Plugin.' $openVinoCache
    }
}
if (-not $openVinoBuildProof.buildLog.exists) {
    Add-Blocker 'unchanged_graph' 'openvino_build_log_missing' 'Retain the clean verbose OpenVINO build log, including the pinned git commit, outside the source tree.' $openVinoBuildLogFull
} else {
    $openVinoBuildLogText = Get-Content -LiteralPath $openVinoBuildLogFull -Raw
    if ([string]::IsNullOrWhiteSpace($script:gate.source.requiredOpenVinoCommit) -or
        $openVinoBuildLogText -notmatch [regex]::Escape($script:gate.source.requiredOpenVinoCommit) -or
        $openVinoBuildLogText -notmatch 'openvino_intel_npu_plugin') {
        Add-Blocker 'unchanged_graph' 'openvino_build_log_unbound' 'The build log must contain the pinned OpenVINO commit and NPU plugin target.' $openVinoBuildLogFull
    }
}

if ([string]::IsNullOrWhiteSpace($BaselineRunner)) {
    $BaselineRunner = Join-Path $openVinoRootFull "bin\intel64\$Configuration\npuFuncTests.exe"
}
$baselineRunnerFull = Get-AbsolutePath $BaselineRunner
$script:gate.stages.unchangedGraph.runner = Get-FileEvidence $baselineRunnerFull
$expectedRunnerDir = Join-Path $openVinoRootFull "bin\intel64\$Configuration"
if (-not (Test-PathInside $baselineRunnerFull $expectedRunnerDir)) {
    Add-Blocker 'unchanged_graph' 'baseline_runner_outside_build' 'npuFuncTests must come from the pinned isolated OpenVINO output directory.' $baselineRunnerFull
}
if (-not (Test-Path -LiteralPath $baselineRunnerFull -PathType Leaf)) {
    Add-Blocker 'unchanged_graph' 'baseline_runner_missing' 'Build the pinned OpenVINO plus Compiler-In-Plugin before running this gate.' $baselineRunnerFull
}

$compilerLibraries = @(Get-ChildItem -LiteralPath (Split-Path -Parent $baselineRunnerFull) -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(lib)?openvino_intel_npu_compiler(d)?\.(dll|so|dylib)(\..*)?$' })
$npuPluginLibraries = @(Get-ChildItem -LiteralPath (Split-Path -Parent $baselineRunnerFull) -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(lib)?openvino_intel_npu_plugin(d)?\.(dll|so|dylib)(\..*)?$' })
$script:gate.stages.unchangedGraph.buildProof.compilerLibraries = @(
    $compilerLibraries | ForEach-Object { Get-FileEvidence $_.FullName }
)
$script:gate.stages.unchangedGraph.buildProof.npuPluginLibraries = @(
    $npuPluginLibraries | ForEach-Object { Get-FileEvidence $_.FullName }
)
if ($compilerLibraries.Count -ne 1) {
    Add-Blocker 'unchanged_graph' 'compiler_in_plugin_library_ambiguous' "Expected exactly one locally built core compiler library beside npuFuncTests; found $($compilerLibraries.Count). Loader-only or production DLLs do not count." (Split-Path -Parent $baselineRunnerFull)
} elseif ($compilerBuildProof.buildLog.exists -and
    -not (Test-FileBuiltDuringFreshLog $compilerLibraries[0].FullName $compilerBuildLogFull)) {
    Add-Blocker 'unchanged_graph' 'compiler_binary_not_fresh' 'The core compiler timestamp is not inside a build-log window created within the last 24 hours. A no-op or copied stale build does not pass.' $compilerLibraries[0].FullName
}
if ($npuPluginLibraries.Count -ne 1) {
    Add-Blocker 'unchanged_graph' 'openvino_npu_plugin_library_ambiguous' "Expected exactly one locally built NPU plugin library beside npuFuncTests; found $($npuPluginLibraries.Count)." (Split-Path -Parent $baselineRunnerFull)
} elseif ($openVinoBuildProof.buildLog.exists -and
    -not (Test-FileBuiltDuringFreshLog $npuPluginLibraries[0].FullName $openVinoBuildLogFull)) {
    Add-Blocker 'unchanged_graph' 'openvino_npu_plugin_not_fresh' 'The NPU plugin timestamp is not inside a build-log window created within the last 24 hours.' $npuPluginLibraries[0].FullName
}

# Inventory is read-only and deliberately precedes the first executable gate so
# a BASELINE_BLOCKED report still retains the later public-toolchain boundary.
Invoke-PublicToolchainInventory

if (($script:gate.blockers | Where-Object stage -eq 'unchanged_graph').Count -gt 0) {
    Complete-Gate 'BASELINE_BLOCKED' ([GateExitCode]::BaselineBlocked)
}

$savedEnvironment = @{
    IE_NPU_TESTS_PLATFORM = $env:IE_NPU_TESTS_PLATFORM
    IE_NPU_TESTS_RUN_INFER = $env:IE_NPU_TESTS_RUN_INFER
    OV_NPU_LOG_LEVEL = $env:OV_NPU_LOG_LEVEL
}
try {
    $env:IE_NPU_TESTS_PLATFORM = 'NPU3720'
    $env:IE_NPU_TESTS_RUN_INFER = '1'
    $env:OV_NPU_LOG_LEVEL = 'LOG_TRACE'

    $baselineList = Invoke-Captured 'unchanged-graph-list' $baselineRunnerFull @(
        "--gtest_filter=$BaselineFilter",
        '--gtest_list_tests'
    )
    $script:gate.stages.unchangedGraph.list = $baselineList
    if ($baselineList.exitCode -ne 0 -or $baselineList.output -notmatch 'NPU3720') {
        Add-Blocker 'unchanged_graph' 'baseline_test_not_listed' "No NPU3720 test matched $BaselineFilter." $baselineRunnerFull
        Complete-Gate 'BASELINE_BLOCKED' ([GateExitCode]::BaselineBlocked)
    }

    $baselineRun = Invoke-Captured 'unchanged-graph-run' $baselineRunnerFull @(
        "--gtest_filter=$BaselineFilter"
    )
    $script:gate.stages.unchangedGraph.run = $baselineRun
    $script:gate.stages.unchangedGraph.passed = Test-GTestPass $baselineRun
    if (-not $script:gate.stages.unchangedGraph.passed) {
        Add-Blocker 'unchanged_graph' 'baseline_run_failed' 'The unchanged pinned Compiler-In-Plugin graph did not complete with at least one passing NPU test.' $baselineRun.log
        Complete-Gate 'BASELINE_BLOCKED' ([GateExitCode]::BaselineBlocked)
    }
    $compilerLibraryPath = $compilerLibraries[0].FullName
    $script:gate.stages.unchangedGraph.loadedCompilerPathAttributed =
        Test-SuccessfulLoadAttribution $baselineRun.output $compilerLibraryPath
    if (-not $script:gate.stages.unchangedGraph.loadedCompilerPathAttributed) {
        Add-Blocker 'unchanged_graph' 'loaded_compiler_path_unattributed' 'The NPU log did not identify the exact locally hashed core compiler library. A passing test alone cannot exclude a stale or production compiler.' $baselineRun.log
        Complete-Gate 'BASELINE_BLOCKED' ([GateExitCode]::BaselineBlocked)
    }

    # This proves only that an Intel-prebuilt software layer can be packaged and run.
    # It is deliberately not accepted as a custom ovVS kernel.
    $builtInList = Invoke-Captured 'builtin-sw-list' $baselineRunnerFull @(
        "--gtest_filter=$BuiltInSoftwareFilter",
        '--gtest_list_tests'
    )
    $script:gate.stages.builtInSoftwareLayer.list = $builtInList
    if ($builtInList.exitCode -eq 0 -and $builtInList.output -match 'NPU3720') {
        $builtInRun = Invoke-Captured 'builtin-sw-run' $baselineRunnerFull @(
            "--gtest_filter=$BuiltInSoftwareFilter"
        )
        $script:gate.stages.builtInSoftwareLayer.run = $builtInRun
        $script:gate.stages.builtInSoftwareLayer.passed = Test-GTestPass $builtInRun
        if (-not $script:gate.stages.builtInSoftwareLayer.passed) {
            Add-Blocker 'built_in_software_layer' 'builtin_sw_run_failed' 'The pinned built-in Sigmoid ActShave graph did not pass on NPU.' $builtInRun.log
        } else {
            Add-Blocker 'built_in_software_layer' 'builtin_sw_cip_shave_unattributed' 'This pinned NPU3720 Activation test selects ReferenceSW but does not explicitly set NPU_COMPILER_TYPE=PLUGIN or enable profiling. Treat it only as stock SW-graph evidence, not a Compiler-In-Plugin/Shave proof.' $builtInRun.log
        }
    } else {
        Add-Blocker 'built_in_software_layer' 'builtin_sw_test_not_listed' "No NPU3720 built-in software-layer test matched $BuiltInSoftwareFilter." $builtInList.log
    }
} finally {
    Restore-Environment $savedEnvironment
}

if (($script:gate.blockers | Where-Object stage -eq 'custom_dependency').Count -gt 0) {
    Complete-Gate 'UNSUPPORTED_PUBLIC_TOOLCHAIN' ([GateExitCode]::PublicToolchainUnsupported)
}

if (($script:gate.blockers | Where-Object stage -eq 'built_in_software_layer').Count -gt 0) {
    Complete-Gate 'CUSTOM_VERIFICATION_FAILED' ([GateExitCode]::CustomVerificationFailed)
}

if (($script:gate.blockers | Where-Object stage -eq 'custom_input').Count -gt 0) {
    Complete-Gate 'CUSTOM_INPUTS_REQUIRED' ([GateExitCode]::ReadyForCustomVerification)
}

if ([string]::IsNullOrWhiteSpace($CustomVerificationRunner)) {
    Add-Blocker 'custom_verification' 'custom_verification_runner_missing' 'Prerequisites are present, but no oracle-equivalent on-device verification was supplied.'
    Complete-Gate 'READY_FOR_CUSTOM_VERIFICATION' ([GateExitCode]::ReadyForCustomVerification)
}

$customVerificationRunnerFull = Get-AbsolutePath $CustomVerificationRunner
$script:gate.stages.customKernel.verificationRunner = Get-FileEvidence $customVerificationRunnerFull
$script:gate.isolation.externalRunnerAttempted = $true
$customRun = Invoke-Captured 'custom-kernel-run' $customVerificationRunnerFull $CustomVerificationArgs
$script:gate.stages.customKernel.run = $customRun
$requiredMarkers = @(
    "OVVS_CUSTOM_KERNEL=$CustomKernelName",
    'ORACLE_MATCH=1',
    'DEVICE=NPU',
    "PERF_NODE=$CustomKernelName;EXEC_TYPE=Shave"
)
$missingMarkers = @($requiredMarkers | Where-Object { $customRun.output -notmatch [regex]::Escape($_) })
$script:gate.stages.customKernel.executed = $customRun.exitCode -eq 0 -and $missingMarkers.Count -eq 0
$script:gate.stages.customKernel.oracleMatch = $customRun.output -match 'ORACLE_MATCH=1'
$script:gate.stages.customKernel.deviceAttributed = $customRun.output -match 'DEVICE=NPU' -and
    $customRun.output -match [regex]::Escape("PERF_NODE=$CustomKernelName;EXEC_TYPE=Shave")
if (-not $script:gate.stages.customKernel.executed) {
    Add-Blocker 'custom_verification' 'custom_kernel_not_proven' "Missing markers: $($missingMarkers -join ', ')." $customRun.log
    Complete-Gate 'CUSTOM_VERIFICATION_FAILED' ([GateExitCode]::CustomVerificationFailed)
}

Add-Blocker 'custom_verification' 'self_attested_verifier_not_accepted' 'The caller-supplied runner and markers are retained as evidence but cannot authorize PASS. Add a checked-in bounded fixture that independently parses the CPU oracle, compiler/ELF hashes, and raw per-node NPU profiling.' $customRun.log
Complete-Gate 'CUSTOM_VERIFICATION_UNPROVEN' ([GateExitCode]::CustomVerificationFailed)
