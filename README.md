# NRD Sample

## Build instructions
### Windows
- Install **WindowsSDK** and **VulkanSDK**
- Clone project and init submodules
- Generate and build project using **cmake**

Or by running scripts only:
- Run `1-Deploy.bat`
- Run `2-Build.bat`

### Linux (x86-64)
- Install **VulkanSDK**, **libx11-dev**, **libxrandr-dev**
- Clone project and init submodules
- Generate and build project using **cmake**

### Linux (aarch64)
- Install **libx11-dev**, **libxrandr-dev**
- Clone project and init submodules
- Generate and build project using **cmake**

### CMake options
- `-DUSE_MINIMAL_DATA=ON` - download minimal resource package (90MB)
- `-DDISABLE_SHADER_COMPILATION=ON` - disable compilation of shaders (shaders can be built on other platform)
- `-DDXC_CUSTOM_PATH=my/path/to/dxc` - custom path to **dxc**
- `-DUSE_DXC_FROM_PACKMAN_ON_AARCH64=OFF` - use default path for **dxc**

## How to run
- Run `3-Run NRD sample.bat` script and answer the cmdline questions to set the runtime parameters
- The executables can be found in `_Build`. The executable loads resources from `_Data`, therefore please run the samples with working directory set to the project root folder (needed pieces of the command line can be found in `3-Run NRD sample.bat` script)

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
