@echo off

set /P M=Do you want to delete PACKMAN repository? [y/n]
if /I "%M%" neq "y" goto KEEP_PACKMAN

:DELETE_PACKMAN
rd /q /s "%PM_PACKAGES_ROOT%"

:KEEP_PACKMAN
rd /q /s "_Build"
rd /q /s "_Compiler"
rd /q /s "_Data"
rd /q /s "_Shaders"
rd /q /s "_NRD_SDK"
rd /q /s "_NRI_SDK"
rd /q /s "External/DXC"
rd /q /s "External/NRIFramework/External/Assimp"
rd /q /s "External/NRIFramework/External/Detex"
rd /q /s "External/NRIFramework/External/Glfw"
rd /q /s "External/NRIFramework/External/ImGui"
rd /q /s "External/NRIFramework/External/NRI/External/AGS"
rd /q /s "External/NRIFramework/External/NRI/External/NVAPI"

rd /q /s "build"
