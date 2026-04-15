param(
    [string]$ConfigPath = "",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $repoRoot "config\local\export_t1_only_with_age.json"
}

<#
Subject-level IXI age resolution policy:
1. Resolve one canonical metadata record per IXI_ID.
2. Prefer AGE from Table$ when it is uniquely available.
3. If AGE is absent, reconstruct age from DOB + STUDY_DATE.
4. If STUDY_DATE is absent in Table$, use Study Date$.
5. Exclude IXI_IDs with conflicting AGE/DOB/STUDY_DATE evidence.
#>

function Ensure-Dir {
    param([string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Convert-ToNullableString {
    param([object]$Value)
    if ($null -eq $Value) {
        return $null
    }

    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }

    return $text.Trim()
}

function Convert-ToNullableDouble {
    param([object]$Value)
    $text = Convert-ToNullableString $Value
    if ($null -eq $text) {
        return $null
    }

    try {
        return [math]::Round([double]$text, 2)
    }
    catch {
        return $null
    }
}

function Convert-ToCanonicalDate {
    param([object]$Value)
    $text = Convert-ToNullableString $Value
    if ($null -eq $text) {
        return $null
    }

    try {
        return ([datetime]$text).ToString("yyyy-MM-dd")
    }
    catch {
        return $null
    }
}

function Compute-AgeFromDates {
    param(
        [string]$Dob,
        [string]$StudyDate
    )

    if ([string]::IsNullOrWhiteSpace($Dob) -or [string]::IsNullOrWhiteSpace($StudyDate)) {
        return $null
    }

    return [math]::Round((([datetime]$StudyDate - [datetime]$Dob).TotalDays / 365.2425), 2)
}

function Join-UniqueValues {
    param([object[]]$Values)

    $clean = New-Object System.Collections.Generic.List[object]
    foreach ($value in @($Values | Where-Object { $null -ne $_ -and -not [string]::IsNullOrWhiteSpace([string]$_) } | Sort-Object -Unique)) {
        $clean.Add($value)
    }
    if ($null -eq $clean -or $clean.Count -eq 0) {
        return ""
    }

    return (($clean | ForEach-Object { [string]$_ }) -join "|")
}

function Resolve-SingleValue {
    param([object[]]$Values)

    $clean = New-Object System.Collections.Generic.List[object]
    foreach ($value in @($Values | Where-Object { $null -ne $_ -and -not [string]::IsNullOrWhiteSpace([string]$_) } | Sort-Object -Unique)) {
        $clean.Add($value)
    }
    if ($null -eq $clean -or $clean.Count -eq 0) {
        return ""
    }
    if ($clean.Count -eq 1) {
        return [string]$clean[0]
    }

    return ($clean -join "|")
}

function Load-ExcelTable {
    param(
        [string]$WorkbookPath,
        [string]$SheetName
    )

    $connString = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$WorkbookPath;Extended Properties='Excel 8.0;HDR=YES;IMEX=1';"
    $conn = New-Object System.Data.OleDb.OleDbConnection($connString)
    $conn.Open()

    try {
        $cmd = $conn.CreateCommand()
        $cmd.CommandText = "SELECT * FROM [$SheetName]"
        $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($cmd)
        $table = New-Object System.Data.DataTable
        [void]$adapter.Fill($table)
        return [pscustomobject]@{ Table = $table }
    }
    finally {
        $conn.Close()
    }
}

function Get-RowValueIfColumnExists {
    param(
        [object]$Row,
        [string]$ColumnName
    )

    if ([string]::IsNullOrWhiteSpace($ColumnName)) {
        return $null
    }

    if ($null -eq $Row -or $null -eq $Row.Table -or -not $Row.Table.Columns.Contains($ColumnName)) {
        return $null
    }

    return $Row[$ColumnName]
}

function Get-RequiredConfigValue {
    param(
        [object]$Config,
        [string]$Key
    )

    $value = Convert-ToNullableString $Config.$Key
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing required config key '$Key' in $ConfigPath"
    }

    return $value
}

function Get-ObjectPropertyValue {
    param(
        [object]$Object,
        [string]$PropertyName
    )

    if ($null -eq $Object -or [string]::IsNullOrWhiteSpace($PropertyName)) {
        return $null
    }

    $property = $Object.PSObject.Properties[$PropertyName]
    if ($null -eq $property) {
        return $null
    }

    return $property.Value
}

function Normalize-SimonPhenotype {
    param([string]$CsvPath)

    $raw = Get-Content -LiteralPath $CsvPath -Raw
    $raw = [regex]::Replace($raw, '^[^,]*Session,', 'Session,')
    return $raw | ConvertFrom-Csv
}

function Resolve-IXIRowPerSubject {
    param(
        [int]$IXIId,
        [object[]]$SourceRows,
        [hashtable]$StudyDateMap,
        [string]$SexColumn
    )

    if ($null -eq $SourceRows -or $SourceRows.Count -eq 0) {
        return [pscustomobject]@{
            IXI_ID                = $IXIId
            resolution_category   = "no_table_row"
            resolved_age          = $null
            age_source            = ""
            normalized_dob        = ""
            normalized_study_date = ""
            duplicate_count       = 0
            conflict_note         = "No row in Table$ for this IXI_ID"
            sex_id                = ""
            ethnic_id             = ""
            marital_id            = ""
            occupation_id         = ""
            qualification_id      = ""
            iq                    = ""
            visit_date            = ""
            date_of_birth         = ""
        }
    }

    $normalizedRows = @(foreach ($row in $SourceRows) {
        $studyDate = Convert-ToCanonicalDate $row["STUDY_DATE"]
        if ([string]::IsNullOrWhiteSpace($studyDate) -and $StudyDateMap.ContainsKey($IXIId)) {
            $studyDate = $StudyDateMap[$IXIId]
        }

        [pscustomobject]@{
            age              = Convert-ToNullableDouble (Get-RowValueIfColumnExists -Row $row -ColumnName "AGE")
            dob              = Convert-ToCanonicalDate (Get-RowValueIfColumnExists -Row $row -ColumnName "DOB")
            study_date       = $studyDate
            inferred_age     = Compute-AgeFromDates -Dob (Convert-ToCanonicalDate (Get-RowValueIfColumnExists -Row $row -ColumnName "DOB")) -StudyDate $studyDate
            sex_id           = Convert-ToNullableString (Get-RowValueIfColumnExists -Row $row -ColumnName $SexColumn)
            ethnic_id        = Convert-ToNullableString (Get-RowValueIfColumnExists -Row $row -ColumnName "ETHNIC_ID")
            marital_id       = Convert-ToNullableString (Get-RowValueIfColumnExists -Row $row -ColumnName "MARITAL_ID")
            occupation_id    = Convert-ToNullableString (Get-RowValueIfColumnExists -Row $row -ColumnName "OCCUPATION_ID")
            qualification_id = Convert-ToNullableString (Get-RowValueIfColumnExists -Row $row -ColumnName "QUALIFICATION_ID")
            iq               = Convert-ToNullableString (Get-RowValueIfColumnExists -Row $row -ColumnName "IQ")
        }
    })

    $ageValues = @($normalizedRows | Select-Object -ExpandProperty age | Where-Object { $null -ne $_ } | Sort-Object -Unique)
    $dobValues = @($normalizedRows | Select-Object -ExpandProperty dob | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
    $studyValues = @($normalizedRows | Select-Object -ExpandProperty study_date | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
    $inferredAgeValues = @($normalizedRows | Select-Object -ExpandProperty inferred_age | Where-Object { $null -ne $_ } | Sort-Object -Unique)

    $resolutionCategory = "missing_age"
    $resolvedAge = $null
    $ageSource = ""
    $conflictNote = ""

    if ($ageValues.Count -gt 1) {
        $resolutionCategory = "conflicting_metadata"
        $conflictNote = "Multiple distinct AGE values: $(Join-UniqueValues $ageValues)"
    }
    elseif ($inferredAgeValues.Count -gt 1) {
        $resolutionCategory = "conflicting_metadata"
        $conflictNote = "Multiple DOB/STUDY_DATE combinations imply different ages: $(Join-UniqueValues $inferredAgeValues)"
    }
    elseif ($ageValues.Count -eq 1) {
        $resolutionCategory = "direct_age"
        $resolvedAge = [double]$ageValues[0]
        $ageSource = "IXI.xls:AGE"
    }
    elseif ($inferredAgeValues.Count -eq 1) {
        $resolutionCategory = "reconstructed_age"
        $resolvedAge = [double]$inferredAgeValues[0]
        $ageSource = "IXI.xls:DOB+StudyDate"
    }
    else {
        $resolutionCategory = "missing_age"
        $conflictNote = "No usable AGE and no unambiguous DOB+STUDY_DATE pair"
    }

    return [pscustomobject]@{
        IXI_ID                = $IXIId
        resolution_category   = $resolutionCategory
        resolved_age          = $resolvedAge
        age_source            = $ageSource
        normalized_dob        = Resolve-SingleValue $dobValues
        normalized_study_date = Resolve-SingleValue $studyValues
        duplicate_count       = $normalizedRows.Count
        conflict_note         = $conflictNote
        sex_id                = Resolve-SingleValue ($normalizedRows | Select-Object -ExpandProperty sex_id)
        ethnic_id             = Resolve-SingleValue ($normalizedRows | Select-Object -ExpandProperty ethnic_id)
        marital_id            = Resolve-SingleValue ($normalizedRows | Select-Object -ExpandProperty marital_id)
        occupation_id         = Resolve-SingleValue ($normalizedRows | Select-Object -ExpandProperty occupation_id)
        qualification_id      = Resolve-SingleValue ($normalizedRows | Select-Object -ExpandProperty qualification_id)
        iq                    = Resolve-SingleValue ($normalizedRows | Select-Object -ExpandProperty iq)
        visit_date            = Resolve-SingleValue $studyValues
        date_of_birth         = Resolve-SingleValue $dobValues
    }
}

function Build-IXIAgeMap {
    param([object[]]$CanonicalRows)

    $map = @{}
    foreach ($row in $CanonicalRows) {
        if ($row.resolution_category -in @("direct_age", "reconstructed_age")) {
            $map[[int]$row.IXI_ID] = $row
        }
    }
    return $map
}

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "Missing config file: $ConfigPath. Copy config\export_t1_only_with_age.example.json into config\local\export_t1_only_with_age.json and fill in local paths."
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

$ixiRoot = Get-RequiredConfigValue -Config $config -Key "ixi_root"
$ixiDataRoot = Join-Path $ixiRoot "ixi"
$ixiXls = Get-RequiredConfigValue -Config $config -Key "ixi_workbook"
$simonRoot = Get-RequiredConfigValue -Config $config -Key "simon_bids_root"
$simonPheno = Get-RequiredConfigValue -Config $config -Key "simon_pheno_csv"
$outRoot = Get-RequiredConfigValue -Config $config -Key "output_root"

if (-not (Test-Path -LiteralPath $ixiXls)) {
    throw "Missing IXI demographics file: $ixiXls"
}
if (-not (Test-Path -LiteralPath $simonPheno)) {
    throw "Missing SIMON phenotype file: $simonPheno"
}

if (Test-Path -LiteralPath $outRoot) {
    if (-not $Force) {
        throw "Output folder already exists: $outRoot. Re-run with -Force to regenerate it."
    }
    Remove-Item -LiteralPath $outRoot -Recurse -Force
}

$ixiOutImages = Join-Path $outRoot "ixi\images"
$ixiOutMeta = Join-Path $outRoot "ixi\metadata"
$simonOutImages = Join-Path $outRoot "simon\images"
$simonOutMeta = Join-Path $outRoot "simon\metadata"
$simonOutSidecars = Join-Path $simonOutMeta "t1_sidecars"

Ensure-Dir $ixiOutImages
Ensure-Dir $ixiOutMeta
Ensure-Dir $simonOutImages
Ensure-Dir $simonOutMeta
Ensure-Dir $simonOutSidecars

# IXI: read workbook and resolve one canonical subject row per IXI_ID.
$ixiMain = (Load-ExcelTable -WorkbookPath $ixiXls -SheetName "Table$").Table
$ixiStudyDates = (Load-ExcelTable -WorkbookPath $ixiXls -SheetName "Study Date$").Table
$ixiSexCol = (($ixiMain.Columns | ForEach-Object ColumnName) | Where-Object { $_ -like "SEX_ID*" } | Select-Object -First 1)
if (-not $ixiSexCol) {
    throw "Could not find the IXI sex column in Table$."
}

$studyDateMap = @{}
foreach ($row in $ixiStudyDates.Rows) {
    $id = Convert-ToNullableDouble $row["ixi_id"]
    $date = Convert-ToCanonicalDate $row["study_date"]
    if ($null -ne $id -and -not [string]::IsNullOrWhiteSpace($date)) {
        $studyDateMap[[int]$id] = $date
    }
}

$ixiT1Files = Get-ChildItem -LiteralPath $ixiDataRoot -Recurse -File |
    Where-Object { $_.FullName -match '\\T1\\NIfTI\\.*\.nii\.gz$' } |
    Sort-Object Name

$ixiSubjectIds = $ixiT1Files |
    ForEach-Object {
        if ($_.Name -match '^IXI(\d+)-') {
            [int]$Matches[1]
        }
    } |
    Sort-Object -Unique

$ixiRowGroups = @{}
foreach ($group in ($ixiMain.Rows | Group-Object { [int]$_['IXI_ID'] })) {
    $ixiRowGroups[[int]$group.Name] = @($group.Group)
}

$ixiSubjectResolution = foreach ($ixiId in $ixiSubjectIds) {
    $rows = if ($ixiRowGroups.ContainsKey($ixiId)) { $ixiRowGroups[$ixiId] } else { @() }
    Resolve-IXIRowPerSubject -IXIId $ixiId -SourceRows $rows -StudyDateMap $studyDateMap -SexColumn $ixiSexCol
}

$ixiAgeMap = Build-IXIAgeMap -CanonicalRows $ixiSubjectResolution

$ixiManifest = New-Object System.Collections.Generic.List[object]
$ixiExcluded = New-Object System.Collections.Generic.List[object]

foreach ($file in $ixiT1Files) {
    if ($file.Name -notmatch '^IXI(?<id>\d+)-(?<site>[^-]+)-(?<scan>\d+)-T1\.nii\.gz$') {
        throw "Unexpected IXI filename format: $($file.Name)"
    }

    $ixiId = [int]$Matches["id"]
    $site = $Matches["site"]
    $scanId = $Matches["scan"]
    $subjectRow = $ixiSubjectResolution | Where-Object { [int]$_.IXI_ID -eq $ixiId } | Select-Object -First 1

    if (-not $ixiAgeMap.ContainsKey($ixiId)) {
        $ixiExcluded.Add([pscustomobject]@{
            IXI_ID = $ixiId
            filename = $file.Name
            exclusion_reason = if ($null -ne $subjectRow) { $subjectRow.resolution_category } else { "no_resolution_row" }
        })
        continue
    }

    $resolved = $ixiAgeMap[$ixiId]
    $destFile = Join-Path $ixiOutImages $file.Name
    Copy-Item -LiteralPath $file.FullName -Destination $destFile -Force

    $ixiManifest.Add([pscustomobject]@{
        dataset = "IXI"
        subject_id = $ixiId
        site = $site
        scan_id = $scanId
        t1_filename = $file.Name
        t1_path = $destFile
        age = $resolved.resolved_age
        sex_id = $resolved.sex_id
        ethnic_id = $resolved.ethnic_id
        marital_id = $resolved.marital_id
        occupation_id = $resolved.occupation_id
        qualification_id = $resolved.qualification_id
        iq = $resolved.iq
        visit_date = $resolved.visit_date
        date_of_birth = $resolved.date_of_birth
        age_source = $resolved.age_source
    })
}

Copy-Item -LiteralPath $ixiXls -Destination (Join-Path $ixiOutMeta "IXI.xls") -Force
$ixiSubjectResolution |
    Sort-Object IXI_ID |
    Export-Csv -LiteralPath (Join-Path $ixiOutMeta "ixi_subject_resolution.csv") -NoTypeInformation -Encoding UTF8
$ixiExcluded |
    Sort-Object IXI_ID |
    Export-Csv -LiteralPath (Join-Path $ixiOutMeta "excluded_t1_missing_or_ambiguous_age.csv") -NoTypeInformation -Encoding UTF8
$ixiManifest |
    Sort-Object subject_id |
    Export-Csv -LiteralPath (Join-Path $outRoot "ixi\manifest.csv") -NoTypeInformation -Encoding UTF8

# SIMON: keep the existing session-level age join.
$simonPhenoRows = Normalize-SimonPhenotype -CsvPath $simonPheno
$duplicateSimonSessions = @($simonPhenoRows | Group-Object { [int]$_.Session } | Where-Object { $_.Count -gt 1 })
if ($duplicateSimonSessions.Count -gt 0) {
    throw "SIMON phenotype CSV contains duplicate Session values: $((($duplicateSimonSessions | Select-Object -ExpandProperty Name) -join ','))"
}

$simonPhenoMap = @{}
foreach ($row in $simonPhenoRows) {
    if ([string]::IsNullOrWhiteSpace([string]$row.Age)) {
        throw "SIMON phenotype row has missing Age for Session=$($row.Session)"
    }
    $simonPhenoMap[[int]$row.Session] = $row
}

$simonFiles = Get-ChildItem -LiteralPath $simonRoot -Recurse -File |
    Where-Object { $_.Name -match '_T1w\.nii\.gz$' } |
    Sort-Object Name

$simonManifest = New-Object System.Collections.Generic.List[object]

foreach ($file in $simonFiles) {
    if ($file.Name -notmatch '^(?<subject>sub-[^_]+)_ses-(?<session>\d{3})(?:_acq-(?<acq>[^_]+))?_run-(?<run>\d+)_T1w\.nii\.gz$') {
        throw "Unexpected SIMON filename format: $($file.Name)"
    }

    $subjectId = $Matches["subject"]
    $sessionId = [int]$Matches["session"]
    $acquisitionLabel = $Matches["acq"]
    $run = [int]$Matches["run"]

    if (-not $simonPhenoMap.ContainsKey($sessionId)) {
        throw "No SIMON phenotype row found for Session=$sessionId"
    }

    $jsonSource = [regex]::Replace($file.FullName, '\.nii\.gz$', '.json')
    if (-not (Test-Path -LiteralPath $jsonSource)) {
        throw "Missing SIMON T1 sidecar JSON for $($file.Name)"
    }

    $json = Get-Content -LiteralPath $jsonSource -Raw | ConvertFrom-Json
    $pheno = $simonPhenoMap[$sessionId]
    $destNii = Join-Path $simonOutImages $file.Name
    $destJson = Join-Path $simonOutSidecars ([System.IO.Path]::GetFileName($jsonSource))

    Copy-Item -LiteralPath $file.FullName -Destination $destNii -Force
    Copy-Item -LiteralPath $jsonSource -Destination $destJson -Force

    $simonManifest.Add([pscustomobject]@{
        dataset = "SIMON"
        subject_id = $subjectId
        session_id = $sessionId
        run = $run
        acquisition_label = $acquisitionLabel
        acquisition_date = Convert-ToCanonicalDate (Get-ObjectPropertyValue -Object $pheno -PropertyName "Acquisition_date")
        age = [math]::Round([double](Get-ObjectPropertyValue -Object $pheno -PropertyName "Age"), 2)
        handedness = [string](Get-ObjectPropertyValue -Object $pheno -PropertyName "Handedness")
        institution_name = [string](Get-ObjectPropertyValue -Object $pheno -PropertyName "institution_name")
        manufacturer = [string](Get-ObjectPropertyValue -Object $pheno -PropertyName "manufacturer")
        man_model_name = [string](Get-ObjectPropertyValue -Object $pheno -PropertyName "man_model_name")
        json_manufacturer = [string](Get-ObjectPropertyValue -Object $json -PropertyName "Manufacturer")
        json_model = [string](Get-ObjectPropertyValue -Object $json -PropertyName "ManufacturersModelName")
        json_institution = [string](Get-ObjectPropertyValue -Object $json -PropertyName "InstitutionName")
        json_field_strength = [string](Get-ObjectPropertyValue -Object $json -PropertyName "MagneticFieldStrength")
        t1_filename = $file.Name
        t1_path = $destNii
        json_filename = [System.IO.Path]::GetFileName($jsonSource)
        json_path = $destJson
        age_source = "SIMON_pheno (4).csv"
    })
}

Copy-Item -LiteralPath (Join-Path $simonRoot "dataset_description.json") -Destination (Join-Path $simonOutMeta "dataset_description.json") -Force
Copy-Item -LiteralPath (Join-Path $simonRoot "README") -Destination (Join-Path $simonOutMeta "README") -Force
Copy-Item -LiteralPath $simonPheno -Destination (Join-Path $simonOutMeta "SIMON_pheno (4).csv") -Force
$simonManifest |
    Sort-Object session_id, run, acquisition_label |
    Export-Csv -LiteralPath (Join-Path $outRoot "simon\manifest.csv") -NoTypeInformation -Encoding UTF8

# Validation summary
$ixiManifestRows = Import-Csv -LiteralPath (Join-Path $outRoot "ixi\manifest.csv")
$ixiResolutionRows = Import-Csv -LiteralPath (Join-Path $ixiOutMeta "ixi_subject_resolution.csv")
$ixiExcludedRows = Import-Csv -LiteralPath (Join-Path $ixiOutMeta "excluded_t1_missing_or_ambiguous_age.csv")
$simonManifestRows = Import-Csv -LiteralPath (Join-Path $outRoot "simon\manifest.csv")

if (@($ixiManifestRows | Where-Object { [string]::IsNullOrWhiteSpace($_.age) }).Count -gt 0) {
    throw "IXI manifest contains missing ages."
}
if (@($simonManifestRows | Where-Object { [string]::IsNullOrWhiteSpace($_.age) }).Count -gt 0) {
    throw "SIMON manifest contains missing ages."
}

Write-Output "Created: $outRoot"
Write-Output "IXI T1 files inspected: $(@($ixiT1Files).Count)"
Write-Output "IXI subject resolution rows: $(@($ixiResolutionRows).Count)"
Write-Output "IXI manifest rows: $(@($ixiManifestRows).Count)"
Write-Output "IXI excluded T1 rows: $(@($ixiExcludedRows).Count)"
Write-Output "SIMON manifest rows: $(@($simonManifestRows).Count)"
