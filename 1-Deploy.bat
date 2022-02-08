@echo off

git submodule update --init --recursive

mkdir _Compiler
pushd _Compiler
cmake .. -A x64
popd

if %ERRORLEVEL% neq 0 call :ErrorOccured

exit /b 0

:ErrorOccured

pause
exit /b 1
