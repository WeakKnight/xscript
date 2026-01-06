@echo off
echo ========================================
echo Running XScript VM Test Suite
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Running Entity GPU Tests...
python -m pytest tests/test_entity_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [2/5] Running Spawn GPU Tests...
python -m pytest tests/test_spawn_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [3/5] Running Dispatch GPU Tests...
python -m pytest tests/test_dispatch_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [4/5] Running GC GPU Tests...
python -m pytest tests/test_gc_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [5/5] Running VM GPU Tests...
python -m pytest tests/test_vm_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo ========================================
echo All tests passed!
echo ========================================
goto :end

:error
echo.
echo ========================================
echo Tests failed with error code %ERRORLEVEL%
echo ========================================
exit /b %ERRORLEVEL%

:end
