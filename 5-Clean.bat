@echo off

set "keep_packman="
set "no_keep_packman="

:PARSE
if "%~1"=="" goto :MAIN

if /i "%~1"=="-h"                goto :HELP
if /i "%~1"=="--help"            goto :HELP

if /i "%~1"=="--keep-packman"    set "keep_packman=y"
if /i "%~1"=="--no-keep-packman" set "no_keep_packman=y"

shift
goto :PARSE

:MAIN
if defined keep_packman goto :KEEP_PACKMAN
if defined no_keep_packman goto :DELETE_PACKMAN
set /P M=Do you want to delete PACKMAN repository? [y/n]
if /I "%M%" neq "y" goto KEEP_PACKMAN

:DELETE_PACKMAN
if exist "%PM_PACKAGES_ROOT%" rd /q /s "%PM_PACKAGES_ROOT%"

:KEEP_PACKMAN
if exist "_Build" rd /q /s "_Build"
if exist "_Compiler" rd /q /s "_Compiler"
if exist "_Data" rd /q /s "_Data"
if exist "_Shaders" rd /q /s "_Shaders"
if exist "_NRD_SDK" rd /q /s "_NRD_SDK"
if exist "_NRI_SDK" rd /q /s "_NRI_SDK"
if exist "External/DXC" rd /q /s "External/DXC"
if exist "External/NRIFramework/External/Assimp" rd /q /s "External/NRIFramework/External/Assimp"
if exist "External/NRIFramework/External/Detex" rd /q /s "External/NRIFramework/External/Detex"
if exist "External/NRIFramework/External/Glfw" rd /q /s "External/NRIFramework/External/Glfw"
if exist "External/NRIFramework/External/ImGui" rd /q /s "External/NRIFramework/External/ImGui"
if exist "External/NRIFramework/External/NRI/External/AGS" rd /q /s "External/NRIFramework/External/NRI/External/AGS"
if exist "External/NRIFramework/External/NRI/External/NVAPI" rd /q /s "External/NRIFramework/External/NRI/External/NVAPI"

if exist "build" rd /q /s "build"

exit

:HELP
echo. -h, --help          show help message
echo. --keep-packman      keep Packman files
echo. --no-keep-packman   don't keep Packman files
exit
