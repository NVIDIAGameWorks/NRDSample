@echo off

set DIR_BIN=%cd%\_Build\Release

if not exist "%DIR_BIN%" (
    set DIR_BIN=%cd%\_Build\Debug
)

if not exist "%DIR_BIN%" (
    echo The project is not compiled!
    pause
    exit /b
)
echo Running NRD sample from '%DIR_BIN%'...
echo.

set API=D3D12
echo 1 - D3D12
echo 2 - VULKAN
:CHOOSE_API
    set /P M=Choose API [1-2]:
    if %M%==1 (
        set API=D3D12
        goto RESOLUTION
    )
    if %M%==2 (
        set API=VULKAN
        goto RESOLUTION
    )
    goto CHOOSE_API

:RESOLUTION
set WIDTH=1920
set HEIGHT=1080
echo.
echo 1 - 1080p
echo 2 - 1440p
echo 3 - 2160p
echo 4 - 1080p (ultra wide)
echo 5 - 1440p (ultra wide)
:CHOOSE_RESOLUTION
    set /P M=Choose resolution [1-5]:
    if %M%==1 (
        set WIDTH=1920
        set HEIGHT=1080
        goto VSYNC
    )
    if %M%==2 (
        set WIDTH=2560
        set HEIGHT=1440
        goto VSYNC
    )
    if %M%==3 (
        set WIDTH=3840
        set HEIGHT=2160
        goto VSYNC
    )
    if %M%==4 (
        set WIDTH=2560
        set HEIGHT=1080
        goto VSYNC
    )
    if %M%==5 (
        set WIDTH=3440
        set HEIGHT=1440
        goto VSYNC
    )
    goto CHOOSE_RESOLUTION

:VSYNC
set VSYNC=0
echo.
echo 1 - off
echo 2 - on
:CHOOSE_VSYNC
    set /P M=Vsync [1-2]:
    if %M%==1 (
        set VSYNC=0
        goto DLSS
    )
    if %M%==2 (
        set VSYNC=1
        goto DLSS
    )
    goto CHOOSE_VSYNC

:DLSS
set DLSS=0
echo.
echo 1 - off
echo 2 - ultra performance
echo 3 - performance
echo 4 - balanced
echo 5 - quality
echo 6 - ultra quality
:CHOOSE_DLSS
    set /P M=DLSS mode [1-6]:
    if %M%==1 (
        set DLSS=-1
        goto SCENE
    )
    if %M%==2 (
        set DLSS=0
        goto SCENE
    )
    if %M%==3 (
        set DLSS=1
        goto SCENE
    )
    if %M%==4 (
        set DLSS=2
        goto SCENE
    )
    if %M%==5 (
        set DLSS=3
        goto SCENE
    )
    if %M%==6 (
        set DLSS=4
        goto SCENE
    )
    goto CHOOSE_DLSS

:SCENE
set SCENE=Bistro/BistroInterior.fbx
echo.
echo 1 - Bistro (interior)
echo 2 - Bistro (exterior)
echo 3 - Shader balls
echo 4 - Zero day
:CHOOSE_SCENE
    set /P M=Choose scene [1-4]:
    if %M%==1 (
        set SCENE=Bistro\BistroInterior.fbx
        goto RUN
    )
    if %M%==2 (
        set SCENE=Bistro\BistroExterior.fbx
        goto RUN
    )
    if %M%==3 (
        set SCENE=ShaderBalls\ShaderBalls.obj
        goto RUN
    )
    if %M%==4 (
        set SCENE=ZeroDay\MEASURE_SEVEN/MEASURE_SEVEN.fbx
        goto RUN
    )
    goto CHOOSE_SCENE

:RUN
start "NRD sample" "%DIR_BIN%\NRDSample.exe" --width=%WIDTH% --height=%HEIGHT% --api=%API% --testMode --swapInterval=%VSYNC% --scene=%SCENE% --dlssQuality=%DLSS%

exit /b
