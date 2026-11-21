@echo off
setlocal
title UStack
cd /d "%~dp0"
rem UStack launcher for Windows: double-click to run UStack from these sources.
rem The first start makes a Python environment in %LOCALAPPDATA%\UStack\venv and installs requirements.txt.
set "VENV=%LOCALAPPDATA%\UStack\venv"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" (
  echo Setting up UStack's Python environment in %VENV% ...
  python -m venv "%VENV%"
  if errorlevel 1 goto :nopython
  "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 goto :error
)
"%PY%" -m ustack %*
if errorlevel 1 goto :error
goto :eof
:nopython
echo Python 3.10 or newer was not found on PATH. Install it from https://www.python.org and try again.
pause
exit /b 1
:error
echo UStack stopped with an error (see the messages above).
pause
exit /b 1
