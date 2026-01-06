@echo off
echo ========================================
echo Running XScript Full Test Suite
echo ========================================
echo.

cd /d "%~dp0"

echo [1/8] Running Compiler Tests...
python -m pytest tests/test_compiler.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [2/8] Running API Tests...
python -m pytest tests/test_api.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [3/8] Running Integration Tests...
python -m pytest tests/test_integration.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [4/8] Running Entity GPU Tests...
python -m pytest tests/test_entity_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [5/8] Running Spawn GPU Tests...
python -m pytest tests/test_spawn_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [6/8] Running Dispatch GPU Tests...
python -m pytest tests/test_dispatch_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [7/8] Running GC GPU Tests...
python -m pytest tests/test_gc_gpu.py -v
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [8/8] Running VM GPU Tests...
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
