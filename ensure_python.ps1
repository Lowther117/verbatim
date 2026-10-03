<#
  Finds a real Python 3.11+ with tkinter, or installs one. (3.11 is the oldest
  version the speech engine publishes wheels for.)

  "Real" excludes the WindowsApps execution-alias stub, which exists purely to
  open the Microsoft Store and reports a version of nothing. Writes the chosen
  interpreter's path to standard output and nothing else, so the batch file can
  capture it.
#>
$ErrorActionPreference = "Stop"

function Test-RealPython([string]$exe) {
  if (-not $exe) { return $false }
  if (-not (Test-Path $exe)) { return $false }
  if ($exe -like "*\WindowsApps\*") { return $false }
  try {
    $probe = & $exe -c "import sys,tkinter;print(sys.version_info[0],sys.version_info[1],tkinter.TkVersion)" 2>$null
  } catch { return $false }
  if (-not $probe) { return $false }
  $bits = $probe -split '\s+'
  if ([int]$bits[0] -ne 3) { return $false }
  if ([int]$bits[1] -lt 11) { return $false }
  if ([double]$bits[2] -lt 8.6) { return $false }
  return $true
}

function Find-Python {
  if ($env:VERBATIM_BUILD_PYTHON -and (Test-RealPython $env:VERBATIM_BUILD_PYTHON)) {
    return $env:VERBATIM_BUILD_PYTHON
  }
  # py launcher first: it knows about every installed version
  $py = (Get-Command py -ErrorAction SilentlyContinue)
  if ($py) {
    foreach ($v in @("-3.13","-3.12","-3.14","-3.11","-3")) {
      try {
        $p = & py $v -c "import sys;print(sys.executable)" 2>$null
        if (Test-RealPython $p) { return $p }
      } catch { }
    }
  }
  foreach ($c in (Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
    if (Test-RealPython $c.Source) { return $c.Source }
  }
  foreach ($root in @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles\Python*",
                      "${env:ProgramFiles(x86)}\Python*", "C:\Python*")) {
    foreach ($d in (Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending)) {
      $p = Join-Path $d.FullName "python.exe"
      if (Test-RealPython $p) { return $p }
    }
  }
  return $null
}

$found = Find-Python
if ($found) { Write-Output $found; exit 0 }

Write-Host "No suitable Python found. Installing one..." -ForegroundColor Yellow
$installed = $false
if (Get-Command winget -ErrorAction SilentlyContinue) {
  try {
    winget install --id Python.Python.3.12 --scope user --silent `
      --accept-package-agreements --accept-source-agreements | Out-Host
    # winget does not throw on failure; check its exit code and whether a
    # usable interpreter actually appeared before trusting it
    if ($LASTEXITCODE -eq 0) {
      $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                  [System.Environment]::GetEnvironmentVariable("Path","User")
      if (Find-Python) { $installed = $true }
    }
    if (-not $installed) { Write-Host "winget did not produce a usable Python; using the python.org installer." }
  } catch { Write-Host "winget failed: $_" }
}
if (-not $installed) {
  $url = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
  $tmp = Join-Path $env:TEMP "python-3.12.7-amd64.exe"
  Write-Host "Downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
  Write-Host "Running the installer (per-user, includes tcl/tk)..."
  Start-Process -FilePath $tmp -ArgumentList `
    "/quiet","InstallAllUsers=0","PrependPath=1","Include_tcltk=1","Include_pip=1" -Wait
}
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")
$found = Find-Python
if ($found) { Write-Output $found; exit 0 }
Write-Error "Python was installed but could not be found afterwards. Open a new terminal and run this again."
exit 1
