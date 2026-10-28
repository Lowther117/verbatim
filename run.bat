@echo off
setlocal EnableExtensions
REM Run Verbatim from source on Windows, setting up a virtual environment
REM the first time. Use build-exe.bat if you want a standalone program instead.
cd /d "%~dp0"
set "VENV=%~dp0.venv"
if not exist "%VENV%\Scripts\python.exe" goto :setup
REM  The environment can exist with its packages missing (a first run whose
REM  pip install failed); pythonw would then fail silently on every launch.
"%VENV%\Scripts\python.exe" -c "import sherpa_onnx, numpy, imageio_ffmpeg, certifi" >nul 2>&1 && goto :run
echo Finishing setup - installing the packages.
goto :install

:setup
echo First run - setting Python up. This happens once.
REM  %PY% is set and used in separate statements on purpose: inside one
REM  parenthesised block cmd expands variables before the block runs.
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ensure_python.ps1"`) do set "PY=%%P"
if not defined PY goto :nopython
if not exist "%PY%" goto :nopython
"%PY%" -m venv "%VENV%"
if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
:install
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >nul
"%VENV%\Scripts\python.exe" -m pip install --only-binary :all: -r "%~dp0requirements.txt"
if errorlevel 1 ( pause & exit /b 1 )

:run
REM  miniaudio (gapless read-along audio) is optional and may have no wheel yet
REM  for a brand-new Python, so it is tried once rather than on every launch.
if exist "%VENV%\.miniaudio-tried" goto :launch
"%VENV%\Scripts\python.exe" -m pip install --only-binary :all: miniaudio >nul 2>&1
type nul > "%VENV%\.miniaudio-tried"
:launch
if "%~1"=="" goto :window
REM  with arguments (transcribe, fetch, models, selftest) stay in this console
"%VENV%\Scripts\python.exe" "%~dp0verbatim.py" %*
exit /b %errorlevel%

:window
start "" "%VENV%\Scripts\pythonw.exe" "%~dp0verbatim.py"
exit /b 0

:nopython
echo Could not find or install Python.
pause
exit /b 1
