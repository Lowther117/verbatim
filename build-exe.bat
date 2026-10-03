@echo off
setlocal EnableExtensions
REM ---------------------------------------------------------------------------
REM  Verbatim - Windows build. Double-click this file.
REM  Installs Python if needed, installs every dependency, downloads the speech
REM  models, and produces the folder dist\Verbatim with Verbatim.exe inside it.
REM
REM  It is a folder rather than one .exe on purpose: about 700 MB of speech
REM  models are baked in, and a single-file exe would unpack all of that to a
REM  temporary folder every time it started.
REM
REM  Which speech model is baked in:  set VERBATIM_BAKE=parakeet-v2  (default)
REM  Any id from "verbatim models" works; "none" bakes only the small helper
REM  models and the app downloads a speech model on first use instead.
REM
REM  NVIDIA graphics card (experimental, not tested by the author):
REM      set VERBATIM_GPU=cuda
REM  swaps in the CUDA build of the speech engine. The PC also needs the
REM  CUDA 12 and cuDNN 9 runtimes installed. Afterwards run the speed test in
REM  the app (Models menu): it uses the graphics card only if it is quicker.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
set "LOG=%~dp0build-win-log.txt"
set "VENV=%~dp0.venv-build"
if not defined VERBATIM_BAKE set "VERBATIM_BAKE=parakeet-v2"
set "BAKE_IDS=support %VERBATIM_BAKE%"
if /i "%VERBATIM_BAKE%"=="none" set "BAKE_IDS=support"
echo Verbatim build started %DATE% %TIME% > "%LOG%"

echo.
echo ===============================================
echo   Building Verbatim for Windows
echo   Full log: build-win-log.txt
echo ===============================================
echo.

echo [1/7] Finding or installing Python...
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ensure_python.ps1" 2^>^> "%LOG%"`) do set "PY=%%P"
if not defined PY goto :fail
if not exist "%PY%" goto :fail
echo       using %PY%
echo Using Python: %PY% >> "%LOG%"

echo [2/7] Creating a clean build environment...
if exist "%VENV%" rmdir /s /q "%VENV%"
"%PY%" -m venv "%VENV%" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
set "VPY=%VENV%\Scripts\python.exe"

echo [3/7] Installing dependencies (wheels only, no compiling)...
"%VPY%" -m pip install --upgrade pip wheel >> "%LOG%" 2>&1
"%VPY%" -m pip install --only-binary :all: -r "%~dp0requirements.txt" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
"%VPY%" -m pip install --only-binary :all: pyinstaller >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
if /i not "%VERBATIM_GPU%"=="cuda" goto :nogpu
echo       Swapping in the NVIDIA CUDA build of the speech engine ^(experimental^)...
"%VPY%" -m pip install --only-binary :all: --force-reinstall "sherpa-onnx==1.13.8+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
:nogpu

echo [4/7] Checking the code before packaging...
REM  The working directory is already this folder, so '.' is the source tree.
REM  %~dp0 must NOT go inside the Python string: it always ends in a backslash,
REM  which would escape the closing quote and make it an unterminated literal.
"%VPY%" -c "import sys; sys.path.insert(0,'.'); import sherpa_onnx, imageio_ffmpeg, numpy; import verbatimlib.ui, verbatimlib.engine, verbatimlib.export, verbatimlib.selftest; print('imports ok')" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

echo [5/7] Downloading the speech models to bake in (%BAKE_IDS%)...
echo       About 500 MB to download the first time; kept in build-models for later builds.
REM  --stage copies exactly the models named here into a fresh build-bake folder,
REM  so a model fetched for an earlier build is not baked into this one as well.
"%VPY%" "%~dp0verbatim.py" fetch --into "%~dp0build-models" --stage "%~dp0build-bake" %BAKE_IDS%
if errorlevel 1 goto :fail

echo [6/7] Packaging (this takes a few minutes)...
if exist "%~dp0build" rmdir /s /q "%~dp0build"
if exist "%~dp0dist" rmdir /s /q "%~dp0dist"
"%VPY%" -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --name "Verbatim" ^
  --collect-all sherpa_onnx ^
  --collect-all imageio_ffmpeg ^
  --collect-all tkinterdnd2 ^
  --collect-all certifi ^
  --collect-submodules verbatimlib ^
  --add-data "verbatimlib\data;verbatimlib\data" ^
  --add-data "build-bake;models" ^
  --hidden-import tkinter ^
  --hidden-import tkinter.ttk ^
  --hidden-import tkinter.filedialog ^
  --hidden-import tkinter.messagebox ^
  --exclude-module pytest ^
  "%~dp0verbatim_app.py" >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
if not exist "%~dp0dist\Verbatim\Verbatim.exe" goto :fail

echo [7/7] Running the self-test on the built program...
echo       It transcribes a short two-speaker sample for real and checks every
echo       export format. About a minute; the window stays hidden while it works.
if exist "%~dp0dist\Verbatim\verbatim-selftest.txt" del /q "%~dp0dist\Verbatim\verbatim-selftest.txt"
pushd "%~dp0dist\Verbatim"
REM  A windowed build has no console, so a plain invocation returns at once and
REM  the report would be read before it exists. start /wait blocks until the
REM  self-test has actually finished.
start "Verbatim self-test" /wait "%~dp0dist\Verbatim\Verbatim.exe" selftest
popd
if exist "%~dp0dist\Verbatim\verbatim-selftest.txt" (
  type "%~dp0dist\Verbatim\verbatim-selftest.txt"
  findstr /C:"PROBLEMS FOUND" "%~dp0dist\Verbatim\verbatim-selftest.txt" >nul && goto :selftestfail
) else (
  echo       ^(no self-test report was written - check it by hand^)
)

echo.
echo ===============================================
echo   DONE.  dist\Verbatim\Verbatim.exe is ready.
echo   Move the whole dist\Verbatim folder wherever
echo   you like and run Verbatim.exe from inside it.
echo ===============================================
echo.
pause
exit /b 0

:selftestfail
echo.
echo ===============================================
echo   BUILT, BUT THE SELF-TEST REPORTED PROBLEMS.
echo   See dist\Verbatim\verbatim-selftest.txt above.
echo ===============================================
echo.
pause
exit /b 2

:fail
echo.
echo ===============================================
echo   BUILD FAILED.  Last 40 lines of the log:
echo ===============================================
powershell -NoProfile -Command "Get-Content -Tail 40 '%LOG%'"
echo.
echo Full log: %LOG%
echo.
pause
exit /b 1
