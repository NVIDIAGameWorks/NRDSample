@echo off

set NRD_DIR=External\NRD

rd /q /s "_NRD_SDK"

mkdir "_NRD_SDK"
cd "_NRD_SDK"

copy "..\%NRD_DIR%\LICENSE.txt" "."
copy "..\%NRD_DIR%\README.md" "."

mkdir "Integration"
copy "..\%NRD_DIR%\Integration\*" "Integration"

mkdir "Include"
copy "..\%NRD_DIR%\Include\*" "Include"

mkdir "Lib"
mkdir "Lib\Debug"
copy "..\_Build\Debug\NRD.dll" "Lib\Debug"
copy "..\_Build\Debug\NRD.lib" "Lib\Debug"
copy "..\_Build\Debug\NRD.pdb" "Lib\Debug"
mkdir "Lib\Release"
copy "..\_Build\Release\NRD.dll" "Lib\Release"
copy "..\_Build\Release\NRD.lib" "Lib\Release"

echo.
set /P M=Do you need the shader source code for a white-box integration? [y/n]
if /I "%M%" neq "y" goto NRI

mkdir "Shaders"
copy "..\%NRD_DIR%\Source\Shaders\*" "Shaders"
copy "..\%NRD_DIR%\Source\Shaders\Include\*" "Shaders"
copy "..\%NRD_DIR%\External\MathLib\*.hlsli" "Shaders"
copy "..\%NRD_DIR%\Include\*.hlsli" "Shaders"

:NRI

echo.
set /P M=Do you need NRI required for NRDIntegration? [y/n]
if /I "%M%" neq "y" goto END

set NRI_DIR=External\NRIFramework\External\NRI

mkdir "_NRI_SDK"
cd "_NRI_SDK"

copy "..\..\%NRI_DIR%\LICENSE.txt" "."

mkdir "Include"
copy "..\..\%NRI_DIR%\Include\*" "Include"

mkdir "Include\Extensions"
copy "..\..\%NRI_DIR%\Include\Extensions\*" "Include\Extensions"

mkdir "Lib"
mkdir "Lib\Debug"
copy "..\..\_Build\Debug\NRI.dll" "Lib\Debug"
copy "..\..\_Build\Debug\NRI.lib" "Lib\Debug"
copy "..\..\_Build\Debug\NRI.pdb" "Lib\Debug"
mkdir "Lib\Release"
copy "..\..\_Build\Release\NRI.dll" "Lib\Release"
copy "..\..\_Build\Release\NRI.lib" "Lib\Release"

:END

exit /b 0