# NRD Sample

All-in-one repository including all relevant pieces to see [*NRD (NVIDIA Real-time Denoisers)*](https://github.com/NVIDIAGameWorks/RayTracingDenoiser) in action. The sample is cross-platform, it's based on [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIAGameWorks/NRI) to bring cross-GraphicsAPI support.

*NRD sample* is a land for high performance path tracing for games. Some features to highlight:
- minimalistic path tracer utilizing *Trace Ray Inline*
- quarter, half (checkerboard) and full resolution tracing
- full resolution tracing with probabilistic diffuse / specular selection at the primary hit
- NRD denoising (including occlusion-only and spherical harmonics modes)
- overhead-free multi-bounce propagation (even in case of a single bounce) based on reusing the previously denoised frame
- reference accumulation
- many RPP and bounces
- reflections on transparent surfaces
- physically based ambient estimation using RT
- mip level calculation
- curvature estimation

## Build instructions

- Install [*Cmake*](https://cmake.org/download/) 3.15+
- Install on
    - Windows: latest *WindowsSDK* (22000+), *VulkanSDK* (1.3.216+)
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

## Requirements

Any ray tracing compatible GPU.

## Usage

- Right mouse button + W/S/A/D - move camera
- Mouse scroll - accelerate / decelerate
- F1 - toggle "gDebug" (can be useful for debugging and experiments)
- F2 - go to next test (only if *TESTS* section is unfolded)
- F3 - toggle emission
- Tab - UI toggle
- Space - animation toggle
- PgUp/PgDown - switch between denoisers

By default *NRD* is used in common mode. But it can also be used in occlusion-only (including directional) and SH (spherical harmonics) modes in the sample. To change the behavior `NRD_MODE` macro needs to be changed from `NORMAL` to `OCCLUSION`, `SH` or `DIRECTIONAL_OCCLUSION` in two places: `NRDSample.cpp` and `Shared.hlsli`.

Notes:
- RELAX doesn't support AO / SO denoising. If RELAX is the current denoiser, ambient term will be flat, but energy correct.
