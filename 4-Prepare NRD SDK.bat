@echo off

set NRD_DIR=External\NRD
set NRI_DIR=External\NRIFramework\External\NRI

set "use_pause=y"
set "copy_shaders="
set "no_copy_shaders="
set "copy_integration="
set "no_copy_integration="

:PARSE
if "%~1"=="" goto :MAIN

if /i "%~1"=="-h"                 goto :HELP
if /i "%~1"=="--help"             goto :HELP
  
if /i "%~1"=="--no-pause"         set "use_pause="

if /i "%~1"=="--copy-shaders"     set "copy_shaders=y"
if /i "%~1"=="--no-copy-shaders"  set "no_copy_shaders=y"

if /i "%~1"=="--integration"      set "copy_integration=y"
if /i "%~1"=="--no-integration"   set "no_copy_integration=y"

shift
goto :PARSE

:MAIN

rd /q /s "_NRD_SDK"

mkdir "_NRD_SDK\Include"
mkdir "_NRD_SDK\Integration"
mkdir "_NRD_SDK\Lib\Debug"
mkdir "_NRD_SDK\Lib\Release"

cd "_NRD_SDK"

copy "..\%NRD_DIR%\Integration\*" "Integration"
copy "..\%NRD_DIR%\Include\*" "Include"
copy "..\_Build\Debug\NRD.dll" "Lib\Debug"
copy "..\_Build\Debug\NRD.lib" "Lib\Debug"
copy "..\_Build\Debug\NRD.pdb" "Lib\Debug"
copy "..\_Build\Release\NRD.dll" "Lib\Release"
copy "..\_Build\Release\NRD.lib" "Lib\Release"
copy "..\%NRD_DIR%\LICENSE.txt" "."
copy "..\%NRD_DIR%\README.md" "."

echo.
if defined copy_shaders goto :SHADERS
if defined no_copy_shaders goto :NRI
set /P M=Do you need the shader source code for a white-box integration? [y/n]
if /I "%M%" neq "y" goto :NRI

:SHADERS
mkdir "Shaders"

xcopy "..\%NRD_DIR%\Shaders\" "Shaders" /s
copy "..\%NRD_DIR%\External\MathLib\*.hlsli" "Shaders\Source"


echo.
if defined copy_integration goto :NRI
if defined no_copy_integration goto :END
set /P M=Do you need NRI required for NRDIntegration? [y/n]
if /I "%M%" neq "y" goto :END

:NRI
cd ..

rd /q /s "_NRI_SDK"

mkdir "_NRI_SDK\Include\Extensions"
mkdir "_NRI_SDK\Lib\Debug"
mkdir "_NRI_SDK\Lib\Release"

cd "_NRI_SDK"

copy "..\%NRI_DIR%\Include\*" "Include"
copy "..\%NRI_DIR%\Include\Extensions\*" "Include\Extensions"
copy "..\_Build\Debug\NRI.dll" "Lib\Debug"
copy "..\_Build\Debug\NRI.lib" "Lib\Debug"
copy "..\_Build\Debug\NRI.pdb" "Lib\Debug"
copy "..\_Build\Release\NRI.dll" "Lib\Release"
copy "..\_Build\Release\NRI.lib" "Lib\Release"
copy "..\%NRI_DIR%\LICENSE.txt" "."

cd ..

:END

cd ..
if defined use_pause pause
exit /b %errorlevel%

:HELP
echo. -h, --help          show help message
echo. --no-pause          skip pause in the end of script
echo. -s, --copy-shaders  copy shadres for a white-box integration
echo. -i, --integration   copy NRDIntegration
exit

