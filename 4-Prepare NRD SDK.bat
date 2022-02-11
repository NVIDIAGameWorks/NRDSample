@echo off

set NRD_DIR=External\NRD
set NRI_DIR=External\NRIFramework\External\NRI

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
set /P M=Do you need the shader source code for a white-box integration? [y/n]
if /I "%M%" neq "y" goto NRI

mkdir "Shaders"

copy "..\%NRD_DIR%\Source\Shaders\*" "Shaders"
copy "..\%NRD_DIR%\Source\Shaders\Include\*" "Shaders"
copy "..\%NRD_DIR%\External\MathLib\*.hlsli" "Shaders"
copy "..\%NRD_DIR%\Include\*.hlsli" "Shaders"

:NRI

cd ..

echo.
set /P M=Do you need NRI required for NRDIntegration? [y/n]
if /I "%M%" neq "y" goto END

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

