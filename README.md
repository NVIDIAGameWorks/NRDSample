# NRD Sample

All-in-one repository including all relevant pieces to see [*NRD (NVIDIA Real-time Denoisers)*](https://github.com/NVIDIAGameWorks/RayTracingDenoiser) in action. The sample is cross-platform, it's based on [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIAGameWorks/NRI) to bring cross-GraphicsAPI support.

## Build instructions

- Install [*Cmake*](https://cmake.org/download/) 3.15+
- Install on
    - Windows: latest *WindowsSDK*, *VulkanSDK*
    - Linux (x86-64): *VulkanSDK*, *libx11-dev*, *libxrandr-dev*, *libwayland-dev*
    - Linux (aarch64): find a precompiled binary for [*DXC*](https://github.com/microsoft/DirectXShaderCompiler), *libx11-dev*, *libxrandr-dev*, *libwayland-dev*
- Build (variant 1) - using *Git* and *CMake* explicitly
    - Clone project and init submodules
    - Generate and build project using *CMake*
- Build (variant 2) - by running scripts:
    - Run `1-Deploy`
    - Run `2-Build`

### CMake options

- `USE_MINIMAL_DATA=ON` - download minimal resource package (90MB)
- `DISABLE_SHADER_COMPILATION=ON` - disable compilation of shaders (shaders can be built on other platform)
- `DXC_CUSTOM_PATH=custom/path/to/dxc` - custom path to *DXC* (will be used if VulkanSDK is not found)
- `USE_DXC_FROM_PACKMAN_ON_AARCH64=OFF` - use default path for *DXC*

## How to run

- Run `3-Run NRD sample` script and answer the cmdline questions to set the runtime parameters
- If [Smart Command Line Arguments extension for Visual Studio](https://marketplace.visualstudio.com/items?itemName=MBulli.SmartCommandlineArguments) is installed, all command line arguments will be loaded into corresponding window
- The executables can be found in `_Build`. The executable loads resources from `_Data`, therefore please run the samples with working directory set to the project root folder (needed pieces of the command line can be found in `3-Run NRD sample` script)

## Minimum Requirements

Any Ray Tracing compatible GPU:
- RTX 3000 series
- RTX 2000 series
- GTX 1660 (Ti, S)
- GTX 1000 series (GPUs with at least 6GB of memory)
- AMD RX 6000 series

## Usage

- Press MOUSE_RIGHT to move...
- W/S/A/D - move camera
- MOUSE_SCROLL - accelerate / decelerate
- F1 - hide UI toggle
- F2 - switch to the next denoiser (*REBLUR* => *RELAX* => ..., *SIGMA* is the only denoiser for shadows)
- SPACE - pause animation toggle

Notes:
- RELAX doesn't support AO / SO denoising. If RELAX is the current denoiser, ambient term will be automatically ignored, bypassing settings in the UI In such cases the default behavior can be returned by pressing the `Default settings` button or choosing a new test, if `--testMode` is set in the command line.
