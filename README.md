# NRD SAMPLE

All-in-one repository including all relevant pieces to see [*NRD (NVIDIA Real-time Denoisers)*](https://github.com/NVIDIAGameWorks/RayTracingDenoiser) in action. The sample is cross-platform, it's based on [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIAGameWorks/NRI) to bring cross-GraphicsAPI support.

*NRD sample* is a land for high performance path tracing for games. Some features to highlight:
- minimalistic path tracer utilizing *Trace Ray Inline*
- HALF resolution (checkerboard), FULL resolution and FULL resolution tracing with PROBABILISTIC diffuse / specular selection at the primary hit
- NRD denoising, including occlusion-only and spherical harmonics / gaussian modes
- overhead-free multi-bounce propagation (even in case of a single bounce) based on reusing the previously denoised frame
- SHARC radiance cache
- reference accumulation
- several rays per pixel and bounces
- realistic glass with multi-bounce reflections and refractions
- mip level calculation
- curvature estimation

### A NOTE ABOUT THE TRACER

The path tracer in the sample has been designed to respect performance. Instead of using a commonly used solution, which in general looks like:
```c++
// Resources
ByteAddressBuffer g_BindlessBuffers[];
Texture2D g_BindlessTextures[];
StructuredBuffer<InstanceData> g_InstanceData;
StructuredBuffer<GeometryData> g_GeometryData;
StructuredBuffer<MaterialData> g_MaterialData;

// Geometry fetching
uint instanceIndex = rayQuery.InstanceIndex();
uint geometryIndex = rayQuery.GeometryIndex();
uint primitiveIndex = rayQuery.PrimitiveIndex();

InstanceData instanceData = g_InstanceData[ instanceIndex ];
GeometryData geometryData = g_GeometryData[ instanceData.geometryBaseIndex + geometryIndex ];

ByteAddressBuffer indexBuffer = g_BindlessBuffers[ NonUniformResourceIndex( geometryData.indexBufferIndex ) ];
ByteAddressBuffer vertexBuffer = g_BindlessBuffers[ NonUniformResourceIndex( geometryData.vertexBufferIndex ) ];

uint3 indices = indexBuffer.Load3( geometryData.indexOffset + primitiveIndex * INDEX_STRIDE );

float3 p0 = DecodePosition( vertexBuffer.Load3( geometryData.vertexOffset + indices[0] * VERTEX_STRIDE ) );
float3 p1 = DecodePosition( vertexBuffer.Load3( geometryData.vertexOffset + indices[1] * VERTEX_STRIDE ) );
float3 p2 = DecodePosition( vertexBuffer.Load3( geometryData.vertexOffset + indices[2] * VERTEX_STRIDE ) );
float3 p = Interpolate( p0, p1, p2, barycentrics );

float3 n0 = DecodeNormal( vertexBuffer.Load3( geometryData.vertexOffset + offset1 + indices[0] * VERTEX_STRIDE ) );
float3 n1 = DecodeNormal( vertexBuffer.Load3( geometryData.vertexOffset + offset1 + indices[1] * VERTEX_STRIDE ) );
float3 n2 = DecodeNormal( vertexBuffer.Load3( geometryData.vertexOffset + offset1 + indices[2] * VERTEX_STRIDE ) );
float3 n = Interpolate( n0, n1, n2, barycentrics );
n = Rotate( instanceData.transform );

float2 uv0 = DecodeUv( vertexBuffer.Load2( geometryData.vertexOffset + offset2 + indices[0] * VERTEX_STRIDE ) );
float2 uv1 = DecodeUv( vertexBuffer.Load2( geometryData.vertexOffset + offset2 + indices[1] * VERTEX_STRIDE ) );
float2 uv2 = DecodeUv( vertexBuffer.Load2( geometryData.vertexOffset + offset2 + indices[2] * VERTEX_STRIDE ) );
float2 uv = Interpolate( uv0, uv1, uv2, barycentrics );

// Material fetching
MaterialData materialData = g_MaterialData[ geometryData.materialIndex ];

Texture2D texture1 = g_BindlessTextures[ NonUniformResourceIndex( materialData.textureIndex1 ) ];
float4 data1 = texture1.SampleLevel( ... );

Texture2D texture2 = g_BindlessTextures[ NonUniformResourceIndex( materialData.textureIndex2 ) ];
float4 data2 = texture2.SampleLevel( ... );

Texture2D texture3 = g_BindlessTextures[ NonUniformResourceIndex( materialData.textureIndex3 ) ];
float4 data1 = texture3.SampleLevel( ... );
```
To get a vertex data we need:
- fetch the vertex data per element through 4 indirections
- vertex position is interpolated despite that it's already in BVH
- in our case to fetch all vertex data we need to do 14 HLSL fetches

The path tracer uses the following scheme:
```c++
// Resources
StructuredBuffer<InstanceData>, g_InstanceData;
StructuredBuffer<PrimitiveData> g_PrimitiveData;
Texture2D g_BindlessTextures[];

// Geometry fetching
uint instanceIndex = rayQuery.InstanceIndex();
uint geometryIndex = rayQuery.GeometryIndex();
uint primitiveIndex = rayQuery.PrimitiveIndex();

InstanceData instanceData = g_InstanceData[ instanceIndex + geometryIndex ];
PrimitiveData primitiveData = g_PrimitiveData[ primitiveIndex ];

float3x3 mObjectToWorld = (float3x3)rayQuery.ObjectToWorld3x4();
if( instanceData.isStatic )
    mObjectToWorld = (float3x3)instanceData.mWorldToWorldPrev;

float3 p = rayOrigin + rayDirection * rayQuery.RayT();

float3 n0 = DecodeNormal( primitiveData.n0 );
float3 n1 = DecodeNormal( primitiveData.n1 );
float3 n2 = DecodeNormal( primitiveData.n2 );
float3 n = Interpolate( n0, n1, n2, barycentrics );
n = Rotate( mObjectToWorld );

float2 uv0 = DecodeUv( primitiveData.uv0 );
float2 uv1 = DecodeUv( primitiveData.uv1 );
float2 uv2 = DecodeUv( primitiveData.uv2 );
float2 uv = Interpolate( uv0, uv1, uv2, barycentrics );

// Material fetching
Texture2D texture1 = g_BindlessTextures[ NonUniformResourceIndex( instanceData.textureBaseIndex ) ];
float4 data1 = texture1.SampleLevel( ... );

Texture2D texture2 = g_BindlessTextures[ NonUniformResourceIndex( instanceData.textureBaseIndex + 1 ) ];
float4 data2 = texture2.SampleLevel( ... );

Texture2D texture3 = g_BindlessTextures[ NonUniformResourceIndex( instanceData.textureBaseIndex + 2 ) ];
float4 data1 = texture3.SampleLevel( ... );
```
To get a vertex data we need:
- fetch the primitive data explicitly without indirections
- interpolate vertex elements using primitive data
- in our case to fetch all vertex data we need to do 2 HLSL fetches

This approach simplifies and accelerates ray tracing, but adds difficulties to BVH management. Deleting a BLAS adds a contiguous region of free elements in `g_PrimitiveData`, which needs to be tracked and potentially re-used in the future when a suitable object appears. If estimated geometry sizes are known, this memory-fragmentation-free approach is more than applicable.

# BUILD INSTRUCTIONS

- Install [*Cmake*](https://cmake.org/download/) 3.15+
- Install on
    - Windows: latest *WindowsSDK* and *VulkanSDK*
    - Linux (x86-64): latest *VulkanSDK*, *libx11-dev*, *libxrandr-dev*, *libwayland-dev*
    - Linux (aarch64): find a precompiled binary for [*DXC*](https://github.com/microsoft/DirectXShaderCompiler), *libx11-dev*, *libxrandr-dev*, *libwayland-dev*
- Build (variant 1) - using *Git* and *CMake* explicitly
    - Clone project and init submodules
    - Generate and build project using *CMake*
- Build (variant 2) - by running scripts:
    - Run `1-Deploy`
    - Run `2-Build`

### CMAKE OPTIONS

- `USE_MINIMAL_DATA=ON` - download minimal resource package (90MB)
- `DISABLE_SHADER_COMPILATION=ON` - disable compilation of shaders (shaders can be built on other platform)
- `DXC_CUSTOM_PATH=custom/path/to/dxc` - custom path to *DXC* (will be used if VulkanSDK is not found)
- `USE_DXC_FROM_PACKMAN_ON_AARCH64=OFF` - use default path for *DXC*

# HOW TO RUN

- Run `3-Run NRD sample` script and answer the cmdline questions to set the runtime parameters
- If [Smart Command Line Arguments extension for Visual Studio](https://marketplace.visualstudio.com/items?itemName=MBulli.SmartCommandlineArguments) is installed, all command line arguments will be loaded into corresponding window
- The executables can be found in `_Bin`. The executable loads resources from `_Data`, therefore please run the samples with working directory set to the project root folder (needed pieces of the command line can be found in `3-Run NRD sample` script)

### REQUIREMENTS

Any ray tracing compatible GPU.

### USAGE

- Right mouse button + W/S/A/D - move camera
- Mouse scroll - accelerate / decelerate
- F1 - toggle "gDebug" (can be useful for debugging and experiments)
- F2 - go to next test (only if *TESTS* section is unfolded)
- F3 - toggle emission
- Tab - UI toggle
- Space - animation toggle
- PgUp/PgDown - switch between denoisers

By default *NRD* is used in common mode. But it can also be used in occlusion-only (including directional) and SH (spherical harmonics) modes in the sample. To change the behavior `NRD_MODE` macro needs to be changed from `NORMAL` to `OCCLUSION`, `SH` or `DIRECTIONAL_OCCLUSION` in `Shared.hlsli`.

Notes:
- RELAX doesn't support AO / SO denoising. If RELAX is the current denoiser, ambient term will be flat, but energy correct.
