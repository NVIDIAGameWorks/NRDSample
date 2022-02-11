/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SampleBase.h"
#include "Extensions/NRIRayTracing.h"

#include "NRD.h"
#include "NRDIntegration.hpp"

#include "Extensions/NRIWrapperD3D11.h"
#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIWrapperVK.h"
#include "DLSS/DLSSIntegration.hpp"

#define NRD_COMBINED 1
#define NRD_OCCLUSION_ONLY 0

constexpr bool CAMERA_RELATIVE = true;
constexpr bool CAMERA_LEFT_HANDED = true;

constexpr uint32_t TEXTURES_PER_MATERIAL = 4;
constexpr uint32_t FG_TEX_SIZE = 256;
constexpr uint32_t ANIMATED_INSTANCE_MAX_NUM = 512;
constexpr float NEAR_Z = 0.001f; // m
constexpr float ACCUMULATION_PERIOD_IN_SECONDS = 0.5f;
constexpr auto BOTTOM_LEVEL_BUILD_FLAGS = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;
constexpr auto TOP_LEVEL_BUILD_FLAGS = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;

#define UI_YELLOW ImVec4(1.0f, 0.9f, 0.0f, 1.0f)

// Choose one of:

// Variant 1 - baseline (NRD_NORMAL_ENCODING = NRD_NORMAL_ENCODING_UNORM8 required)
// #define NORMAL_FORMAT nri::Format::RGBA8_UNORM

// Variant 2 - more precise normals (NRD_NORMAL_ENCODING = NRD_NORMAL_ENCODING_OCT10 required)
#define NORMAL_FORMAT nri::Format::R10_G10_B10_A2_UNORM

// Variant 3 - maximize precision of curvature coputations (NRD_NORMAL_ENCODING = NRD_NORMAL_ENCODING_UNORM16 required)
// #define NORMAL_FORMAT nri::Format::RGBA16_UNORM

// See HLSL
#define FLAG_FIRST_BIT                  20
#define INSTANCE_ID_MASK                ( ( 1 << FLAG_FIRST_BIT ) - 1 )
#define FLAG_OPAQUE_OR_ALPHA_OPAQUE     0x01
#define FLAG_TRANSPARENT                0x02
#define FLAG_EMISSION                   0x04
#define FLAG_FORCED_EMISSION            0x08

enum Denoiser : int32_t
{
    REBLUR,
    RELAX,

    DENOISER_MAX_NUM
};

enum Resolution : int32_t
{
    RESOLUTION_FULL,
    RESOLUTION_HALF,
    RESOLUTION_QUARTER
};

enum class Buffer : uint32_t
{
    GlobalConstants,
    InstanceDataStaging,
    WorldTlasDataStaging,
    LightTlasDataStaging,

    PrimitiveData,
    InstanceData,
    WorldScratch,
    LightScratch,

    UploadHeapBufferNum = 4
};

enum class Texture : uint32_t
{
    IntegrateBRDF,
    Ambient,
    ViewZ,
    Motion,
    Normal_Roughness,
    PrimaryMip,
    Downsampled_ViewZ,
    Downsampled_Motion,
    Downsampled_Normal_Roughness,
    BaseColor_Metalness,
    DirectLighting,
    DirectEmission,
    TransparentLighting,
    Shadow,
    Diff,
    DiffDirectionPdf,
    Spec,
    SpecDirectionPdf,
    Unfiltered_ShadowData,
    Unfiltered_Diff,
    Unfiltered_Spec,
    Unfiltered_Shadow_Translucency,
    ComposedLighting_ViewZ,
    TaaHistory,
    TaaHistoryPrev,
    DlssOutput,
    Final,
    MaterialTextures
};

enum class Pipeline : uint32_t
{
    IntegrateBRDF,
    AmbientRays,
    PrimaryRays,
    DirectLighting,
    IndirectRays,
    Composition,
    Temporal,
    Upsample,
    PreDlss,
    AfterDlss
};

enum class Descriptor : uint32_t
{
    World_AccelerationStructure,
    Light_AccelerationStructure,

    PrimitiveData_Buffer,
    InstanceData_Buffer,

    IntegrateBRDF_Texture,
    IntegrateBRDF_StorageTexture,
    Ambient_Texture,
    Ambient_StorageTexture,
    ViewZ_Texture,
    ViewZ_StorageTexture,
    Motion_Texture,
    Motion_StorageTexture,
    Normal_Roughness_Texture,
    Normal_Roughness_StorageTexture,
    PrimaryMip_Texture,
    PrimaryMip_StorageTexture,
    Downsampled_ViewZ_Texture,
    Downsampled_ViewZ_StorageTexture,
    Downsampled_Motion_Texture,
    Downsampled_Motion_StorageTexture,
    Downsampled_Normal_Roughness_Texture,
    Downsampled_Normal_Roughness_StorageTexture,
    BaseColor_Metalness_Texture,
    BaseColor_Metalness_StorageTexture,
    DirectLighting_Texture,
    DirectLighting_StorageTexture,
    DirectEmission_Texture,
    DirectEmission_StorageTexture,
    TransparentLighting_Texture,
    TransparentLighting_StorageTexture,
    Shadow_Texture,
    Shadow_StorageTexture,
    Diff_Texture,
    Diff_StorageTexture,
    DiffDirectionPdf_Texture,
    DiffDirectionPdf_StorageTexture,
    Spec_Texture,
    Spec_StorageTexture,
    SpecDirectionPdf_Texture,
    SpecDirectionPdf_StorageTexture,
    Unfiltered_ShadowData_Texture,
    Unfiltered_ShadowData_StorageTexture,
    Unfiltered_Diff_Texture,
    Unfiltered_Diff_StorageTexture,
    Unfiltered_Spec_Texture,
    Unfiltered_Spec_StorageTexture,
    Unfiltered_Shadow_Translucency_Texture,
    Unfiltered_Shadow_Translucency_StorageTexture,
    ComposedLighting_ViewZ_Texture,
    ComposedLighting_ViewZ_StorageTexture,
    TaaHistory_Texture,
    TaaHistory_StorageTexture,
    TaaHistoryPrev_Texture,
    TaaHistoryPrev_StorageTexture,
    DlssOutput_Texture,
    DlssOutput_StorageTexture,
    Final_Texture,
    Final_StorageTexture,
    MaterialTextures
};

enum class DescriptorSet : uint32_t
{
    IntegrateBRDF0,
    AmbientRays1,
    PrimaryRays1,
    DirectLighting1,
    IndirectRays1,
    Composition1,
    Temporal1a,
    Temporal1b,
    Upsample1a,
    Upsample1b,
    PreDlss1,
    AfterDlss1,
    RayTracing2
};

struct NRIInterface
    : public nri::CoreInterface
    , public nri::SwapChainInterface
    , public nri::RayTracingInterface
    , public nri::HelperInterface
{};

struct Frame
{
    nri::DeviceSemaphore* deviceSemaphore;
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
    nri::Descriptor* globalConstantBufferDescriptor;
    nri::DescriptorSet* globalConstantBufferDescriptorSet;
    uint64_t globalConstantBufferOffset;
};

struct GlobalConstantBufferData
{
    float4x4 gWorldToView;
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToClipPrev;
    float4x4 gWorldToClip;
    float4 gDiffHitDistParams;
    float4 gSpecHitDistParams;
    float4 gCameraFrustum;
    float4 gSunDirection_gExposure;
    float4 gCameraOrigin_gMipBias;
    float4 gTrimmingParams_gEmissionIntensity;
    float4 gViewDirection_gIsOrtho;
    float2 gOutputSize;
    float2 gInvOutputSize;
    float2 gScreenSize;
    float2 gInvScreenSize;
    float2 gRectSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gNearZ;
    float gAmbientAccumSpeed;
    float gAmbient;
    float gAmbientInComposition;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gIndirectDiffuse;
    float gIndirectSpecular;
    float gSunAngularRadius;
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gTransparent;
    float gReference;
    uint32_t gDenoiserType;
    uint32_t gDisableShadowsAndEnableImportanceSampling;
    uint32_t gOnScreen;
    uint32_t gFrameIndex;
    uint32_t gForcedMaterial;
    uint32_t gUseNormalMap;
    uint32_t gWorldSpaceMotion;
    uint32_t gTracingMode;
    uint32_t gSampleNum;
    uint32_t gBounceNum;
    uint32_t gOcclusionOnly;
};

struct NrdSettings
{
    float       blurRadius                         = 30.0f;
    float       adaptiveRadiusScale                = 5.0f;
    float       stabilizationStrength              = 1.0f;
    float       normalWeightStrictness             = 1.0f;
    float       disocclusionThreshold              = 1.0f;
    float       residualNoiseLevel                 = 3.0f;

    int32_t     maxAccumulatedFrameNum             = 31;
    int32_t     maxFastAccumulatedFrameNum         = 7;
    int32_t     prePassMode                        = (int32_t)nrd::PrePassMode::SIMPLE;

    bool        referenceAccumulation              = false;
    bool        antilagIntensity                   = true;
    bool        antilagHitDistance                 = true;
    bool        enableAntiFirefly                  = false;
};

struct Settings
{
    NrdSettings nrdSettings                        = {};

    double      motionStartTime                    = 0.0;

    float       maxFps                             = 60.0f;
    float       camFov                             = 90.0f;
    float       sunAzimuth                         = -147.0f;
    float       sunElevation                       = 45.0f;
    float       sunAngularDiameter                 = 0.533f;
    float       exposure                           = 80.0f;
    float       roughnessOverride                  = 0.0f;
    float       metalnessOverride                  = 0.0f;
    float       emissionIntensity                  = 1.0f;
    float       debug                              = 0.0f;
    float       meterToUnitsMultiplier             = 1.0f;
    float       emulateMotionSpeed                 = 1.0f;
    float       animatedObjectScale                = 1.0f;
    float       separator                          = 0.0f;
    float       animationProgress                  = 0.0f;
    float       animationSpeed                     = 0.0f;
    float       diffHitDistScale                   = 3.0f;
    float       specHitDistScale                   = 3.0f;

    int32_t     onScreen                           = 0;
    int32_t     forcedMaterial                     = 0;
    int32_t     animatedObjectNum                  = 5;
    int32_t     activeAnimation                    = 0;
    int32_t     motionMode                         = 0;
    int32_t     denoiser                           = REBLUR;
    int32_t     rpp                                = 1;
    int32_t     bounceNum                          = 1;
    int32_t     tracingMode                        = RESOLUTION_HALF;

    bool        limitFps                           = false;
    bool        ambient                            = true;
    bool        reference                          = false;
    bool        indirectDiffuse                    = true;
    bool        indirectSpecular                   = true;
    bool        normalMap                          = true;
    bool        TAA                                = true;
    bool        animatedObjects                    = false;
    bool        animateCamera                      = false;
    bool        animateSun                         = false;
    bool        nineBrothers                       = false;
    bool        blink                              = false;
    bool        pauseAnimation                     = true;
    bool        emission                           = false;
    bool        isMotionVectorInWorldSpace         = true;
    bool        linearMotion                       = true;
    bool        emissiveObjects                    = false;
    bool        importanceSampling                 = true;
    bool        specularLobeTrimming               = true;
    bool        ortho                              = false;
    bool        adaptiveAccumulation               = false;
};

struct DescriptorDesc
{
    const char* debugName;
    void* resource;
    nri::Format format;
    nri::TextureUsageBits textureUsage;
    nri::BufferUsageBits bufferUsage;
    bool isArray;
};

struct TextureState
{
    Texture texture;
    nri::AccessBits nextAccess;
    nri::TextureLayout nextLayout;
};

struct AnimationParameters
{
    float3 rotationAxis;
    float3 elipseAxis;
    float durationSec = 5.0f;
    float progressedSec = 0.0f;
    float inverseRotation = 1.0f;
    float inverseDirection = 1.0f;
    float angleRad = 0.0f;
};

struct AnimatedInstance
{
    double3 position = double3::Zero();
    double3 basePosition = double3::Zero();
    AnimationParameters animation;
    uint32_t instanceID;

    float4x4 Animate(float elapsedSeconds, float scale)
    {
        float weight = (animation.progressedSec + elapsedSeconds) / animation.durationSec;
        weight = weight * 2.0f - 1.0f;
        weight = Pi(weight);

        float3 localPosition;
        localPosition.x = Cos(weight * animation.inverseDirection);
        localPosition.y = Sin(weight * animation.inverseDirection);
        localPosition.z = localPosition.y;

        position = basePosition + ToDouble( localPosition * animation.elipseAxis * scale );

        animation.angleRad = weight * animation.inverseRotation;
        animation.progressedSec += elapsedSeconds;
        animation.progressedSec = (animation.progressedSec >= animation.durationSec) ? 0.0f : animation.progressedSec;

        float4x4 transform;
        transform.SetupByRotation(animation.angleRad, animation.rotationAxis);
        transform.AddScale(scale);

        return transform;
    }
};

struct PrimitiveData
{
    uint32_t uv0;
    uint32_t uv1;
    uint32_t uv2;
    uint32_t n0oct;

    uint32_t n1oct;
    uint32_t n2oct;
    uint32_t t0oct;
    uint32_t t1oct;

    uint32_t t2oct;
    uint32_t b0s_b1s;
    uint32_t b2s_worldToUvUnits;
    uint32_t padding;
};

struct InstanceData
{
    uint32_t basePrimitiveIndex;
    uint32_t baseTextureIndex;
    uint32_t averageBaseColor;
    uint32_t unused;

    float4 mWorldToWorldPrev0;
    float4 mWorldToWorldPrev1;
    float4 mWorldToWorldPrev2;
};

class Sample : public SampleBase
{
public:
    Sample() :
        m_Reblur(BUFFERED_FRAME_MAX_NUM),
        m_Relax(BUFFERED_FRAME_MAX_NUM),
        m_Sigma(BUFFERED_FRAME_MAX_NUM),
        m_Reference(BUFFERED_FRAME_MAX_NUM)
    {}

    ~Sample();

    bool Initialize(nri::GraphicsAPI graphicsAPI);
    void PrepareFrame(uint32_t frameIndex);
    void RenderFrame(uint32_t frameIndex);

    inline nri::Texture*& Get(Texture index)
    { return m_Textures[(uint32_t)index]; }

    inline nri::TextureTransitionBarrierDesc& GetState(Texture index)
    { return m_TextureStates[(uint32_t)index]; }

    inline nri::Format GetFormat(Texture index)
    { return m_TextureFormats[(uint32_t)index]; }

    inline nri::Buffer*& Get(Buffer index)
    { return m_Buffers[(uint32_t)index]; }

    inline nri::Pipeline*& Get(Pipeline index)
    { return m_Pipelines[(uint32_t)index]; }

    inline nri::PipelineLayout*& GetPipelineLayout(Pipeline index)
    { return m_PipelineLayouts[(uint32_t)index]; }

    inline nri::Descriptor*& Get(Descriptor index)
    { return m_Descriptors[(uint32_t)index]; }

    inline nri::DescriptorSet*& Get(DescriptorSet index)
    { return m_DescriptorSets[(uint32_t)index]; }

private:
    void CreateCommandBuffers();
    void CreateSwapChain(nri::Format& swapChainFormat);
    void CreateResources(nri::Format swapChainFormat);
    void CreatePipelines();
    void CreateDescriptorSets();
    void CreateBottomLevelAccelerationStructures();
    void CreateTopLevelAccelerationStructure();
    void UpdateConstantBuffer(uint32_t frameIndex, float globalResetFactor);
    void UploadStaticData();
    void LoadScene();
    void SetupAnimatedObjects();
    void CreateUploadBuffer(uint64_t size, nri::Buffer*& buffer, nri::Memory*& memory);
    void CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory);
    void BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::GeometryObject* objects, const uint32_t objectNum);
    void BuildTopLevelAccelerationStructure(nri::CommandBuffer& commandBuffer, uint32_t bufferedFrameIndex);
    void CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint16_t width, uint16_t height, uint16_t mipNum, uint16_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state);
    void CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage, nri::Format format = nri::Format::UNKNOWN);
    void CreateDescriptors(const std::vector<DescriptorDesc>& descriptorDescs);
    uint32_t BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, nri::TextureTransitionBarrierDesc* transitions, uint32_t transitionMaxNum);

    inline float3 GetSunDirection() const
    {
        float3 sunDirection;
        sunDirection.x = Cos( DegToRad(m_Settings.sunAzimuth) ) * Cos( DegToRad(m_Settings.sunElevation) );
        sunDirection.y = Sin( DegToRad(m_Settings.sunAzimuth) ) * Cos( DegToRad(m_Settings.sunElevation) );
        sunDirection.z = Sin( DegToRad(m_Settings.sunElevation) );

        return sunDirection;
    }

    inline float3 GetTrimmingParams() const
    {
        // See NRDSettings.h - it's a good start
        return (m_Settings.specularLobeTrimming && !m_Settings.reference) ? float3(0.85f, 0.04f, 0.11f) : float3(1.0f, 0.0f, 0.0001f);
    }

    inline void GetAntilagSettings(nrd::AntilagIntensitySettings& antilagIntensitySettings, nrd::AntilagHitDistanceSettings& antilagHitDistanceSettings) const
    {
        float scale = 0.25f + 0.75f / (1.0f + (m_Settings.rpp - 1) * 0.25f);

        antilagIntensitySettings.thresholdMin = 0.03f * scale;
        antilagIntensitySettings.thresholdMax = 0.20f * scale;
        antilagIntensitySettings.sigmaScale = 1.0f * scale;
        antilagIntensitySettings.sensitivityToDarkness = 0.01f; // IMPORTANT: tuned for the sky model in the sample
        antilagIntensitySettings.enable = m_Settings.nrdSettings.antilagIntensity;

        antilagHitDistanceSettings.thresholdMin = 0.015f * scale;
        antilagHitDistanceSettings.thresholdMax = 0.15f * scale;
        antilagHitDistanceSettings.sigmaScale = 1.0f * scale;
        antilagHitDistanceSettings.sensitivityToDarkness = 0.1f;
        antilagHitDistanceSettings.enable = m_Settings.nrdSettings.antilagHitDistance;
    }

private:
    NrdIntegration m_Reblur;
    NrdIntegration m_Relax;
    NrdIntegration m_Sigma;
    NrdIntegration m_Reference;

    DlssIntegration m_DLSS;

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::CommandQueue* m_CommandQueue = nullptr;
    nri::QueueSemaphore* m_BackBufferAcquireSemaphore = nullptr;
    nri::QueueSemaphore* m_BackBufferReleaseSemaphore = nullptr;
    nri::AccelerationStructure* m_WorldTlas = nullptr;
    nri::AccelerationStructure* m_LightTlas = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::TextureTransitionBarrierDesc> m_TextureStates;
    std::vector<nri::Format> m_TextureFormats;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Memory*> m_MemoryAllocations;
    std::vector<nri::Descriptor*> m_Descriptors;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::PipelineLayout*> m_PipelineLayouts;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::AccelerationStructure*> m_BLASs;
    std::vector<BackBuffer> m_SwapChainBuffers;
    std::vector<AnimatedInstance> m_AnimatedInstances;
    std::array<float, 256> m_FrameTimes = {};
    Timer m_Timer;
    float3 m_PrevLocalPos = {};
    float2 m_RectSizePrev = {};
    uint2 m_OutputResolution = {};
    uint2 m_ScreenResolution = {};
    utils::Scene m_Scene;
    nrd::RelaxDiffuseSpecularSettings m_RelaxSettings = {}; // TODO: after code stabilization move to Settings and adjust unit tests
    Settings m_Settings = {};
    Settings m_PrevSettings = {};
    Settings m_DefaultSettings = {};
    const nri::DeviceDesc* m_DeviceDesc = nullptr;
    uint64_t m_ConstantBufferSize = 0;
    uint32_t m_DefaultInstancesOffset = 0;
    uint32_t m_LastSelectedTest = uint32_t(-1);
    uint32_t m_TestNum = uint32_t(-1);
    float m_AmbientAccumFrameNum = 0.0f;
    float m_ResolutionScale = 1.0f;
    float m_MinResolutionScale = 50.0f;
    float m_Sharpness = 0.0f;
    bool m_HasTransparentObjects = false;
    bool m_ShowUi = true;
    bool m_ForceHistoryReset = false;
};

Sample::~Sample()
{
    NRI.WaitForIdle(*m_CommandQueue);

    m_DLSS.Shutdown();

    m_Reblur.Destroy();
    m_Relax.Destroy();
    m_Sigma.Destroy();
    m_Reference.Destroy();

    for (Frame& frame : m_Frames)
    {
        NRI.DestroyCommandBuffer(*frame.commandBuffer);
        NRI.DestroyDeviceSemaphore(*frame.deviceSemaphore);
        NRI.DestroyCommandAllocator(*frame.commandAllocator);
        NRI.DestroyDescriptor(*frame.globalConstantBufferDescriptor);
    }

    for (BackBuffer& backBuffer : m_SwapChainBuffers)
    {
        NRI.DestroyDescriptor(*backBuffer.colorAttachment);
        NRI.DestroyFrameBuffer(*backBuffer.frameBufferUI);
    }

    for (uint32_t i = 0; i < m_Textures.size(); i++)
        NRI.DestroyTexture(*m_Textures[i]);

    for (uint32_t i = 0; i < m_Buffers.size(); i++)
        NRI.DestroyBuffer(*m_Buffers[i]);

    for (uint32_t i = 0; i < m_Descriptors.size(); i++)
        NRI.DestroyDescriptor(*m_Descriptors[i]);

    for (uint32_t i = 0; i < m_Pipelines.size(); i++)
        NRI.DestroyPipeline(*m_Pipelines[i]);

    for (uint32_t i = 0; i < m_PipelineLayouts.size(); i++)
        NRI.DestroyPipelineLayout(*m_PipelineLayouts[i]);

    for (uint32_t i = 0; i < m_BLASs.size(); i++)
        NRI.DestroyAccelerationStructure(*m_BLASs[i]);

    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyAccelerationStructure(*m_WorldTlas);
    NRI.DestroyAccelerationStructure(*m_LightTlas);
    NRI.DestroyQueueSemaphore(*m_BackBufferAcquireSemaphore);
    NRI.DestroyQueueSemaphore(*m_BackBufferReleaseSemaphore);
    NRI.DestroySwapChain(*m_SwapChain);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
        NRI.FreeMemory(*m_MemoryAllocations[i]);

    DestroyUserInterface();

    nri::DestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI)
{
    nri::PhysicalDeviceGroup physicalDeviceGroup = {};
    if (!helper::FindPhysicalDeviceGroup(physicalDeviceGroup))
        return false;

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.spirvBindingOffsets = SPIRV_BINDING_OFFSETS;
    deviceCreationDesc.physicalDeviceGroup = &physicalDeviceGroup;
    DlssIntegration::SetupDeviceExtensions(deviceCreationDesc);
    NRI_ABORT_ON_FAILURE( nri::CreateDevice(deviceCreationDesc, m_Device) );

    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) );

    NRI_ABORT_ON_FAILURE( NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, m_CommandQueue));
    NRI_ABORT_ON_FAILURE( NRI.CreateQueueSemaphore(*m_Device, m_BackBufferAcquireSemaphore));
    NRI_ABORT_ON_FAILURE( NRI.CreateQueueSemaphore(*m_Device, m_BackBufferReleaseSemaphore));

    m_DeviceDesc = &NRI.GetDeviceDesc(*m_Device);
    m_ConstantBufferSize = helper::GetAlignedSize(sizeof(GlobalConstantBufferData), m_DeviceDesc->constantBufferOffsetAlignment);
    m_OutputResolution = uint2(GetWindowWidth(), GetWindowHeight());
    m_ScreenResolution = m_OutputResolution;

    if (m_DlssQuality != uint32_t(-1))
    {
        if (m_DLSS.InitializeLibrary(*m_Device, ""))
        {
            DlssSettings dlssSettings = {};
            if (m_DLSS.GetOptimalSettings({m_OutputResolution.x, m_OutputResolution.y}, (DlssQuality)m_DlssQuality, dlssSettings))
            {
                DlssInitDesc dlssInitDesc = {};
                dlssInitDesc.outputResolution = {m_OutputResolution.x, m_OutputResolution.y};
                dlssInitDesc.quality = (DlssQuality)m_DlssQuality;
                dlssInitDesc.isContentHDR = true;

                m_DLSS.Initialize(m_CommandQueue, dlssInitDesc);

                float sx = float(dlssSettings.minRenderResolution.Width) / float(dlssSettings.renderResolution.Width);
                float sy = float(dlssSettings.minRenderResolution.Height) / float(dlssSettings.renderResolution.Height);
                float minResolutionScale = sy > sx ? sy : sx;

                m_ScreenResolution = {dlssSettings.renderResolution.Width, dlssSettings.renderResolution.Height};
                m_MinResolutionScale = Floor(minResolutionScale * 100.0f + 0.99f);
                m_Sharpness = dlssSettings.sharpness;

                printf("DLSS: render resolution (%u, %u)\n", m_ScreenResolution.x, m_ScreenResolution.y);
            }
            else
            {
                m_DLSS.Shutdown();

                printf("DLSS: unsupported mode!\n");
            }
        }
    }

    nri::Format swapChainFormat = nri::Format::UNKNOWN;
    LoadScene();
    CreateCommandBuffers();
    CreateSwapChain(swapChainFormat);
    CreatePipelines();
    CreateBottomLevelAccelerationStructures();
    CreateTopLevelAccelerationStructure();
    CreateResources(swapChainFormat);
    CreateDescriptorSets();
    UploadStaticData();
    SetupAnimatedObjects();

    // REBLUR
    {
        const nrd::MethodDesc methodDescs[] =
        {
            #if( NRD_OCCLUSION_ONLY == 1 )
                #if( NRD_COMBINED == 1 )
                    { nrd::Method::REBLUR_DIFFUSE_SPECULAR_OCCLUSION, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                #else
                    { nrd::Method::REBLUR_DIFFUSE_OCCLUSION, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                    { nrd::Method::REBLUR_SPECULAR_OCCLUSION, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                #endif
            #else
                #if( NRD_COMBINED == 1 )
                    { nrd::Method::REBLUR_DIFFUSE_SPECULAR, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                #else
                    { nrd::Method::REBLUR_DIFFUSE, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                    { nrd::Method::REBLUR_SPECULAR, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                #endif
            #endif
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodNum = helper::GetCountOf(methodDescs);
        NRI_ABORT_ON_FALSE( m_Reblur.Initialize(*m_Device, NRI, NRI, denoiserCreationDesc) );
    }

    // RELAX
    {
        const nrd::MethodDesc methodDescs[] =
        {
            #if( NRD_COMBINED == 1 )
                { nrd::Method::RELAX_DIFFUSE_SPECULAR, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
            #else
                { nrd::Method::RELAX_DIFFUSE, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
                { nrd::Method::RELAX_SPECULAR, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
            #endif
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Relax.Initialize(*m_Device, NRI, NRI, denoiserCreationDesc) );
    }

    // SIGMA
    {
        const nrd::MethodDesc methodDescs[] =
        {
            { nrd::Method::SIGMA_SHADOW_TRANSLUCENCY, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodNum = helper::GetCountOf(methodDescs);
        NRI_ABORT_ON_FALSE( m_Sigma.Initialize(*m_Device, NRI, NRI, denoiserCreationDesc) );
    }

    // REFERENCE
    {
        const nrd::MethodDesc methodDescs[] =
        {
            { nrd::Method::REFERENCE, (uint16_t)m_ScreenResolution.x, (uint16_t)m_ScreenResolution.y },
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodNum = helper::GetCountOf(methodDescs);
        NRI_ABORT_ON_FALSE( m_Reference.Initialize(*m_Device, NRI, NRI, denoiserCreationDesc) );
    }

    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
    m_Scene.UnloadResources();

    m_DefaultSettings = m_Settings;

    return CreateUserInterface(*m_Device, NRI, NRI, m_OutputResolution.x, m_OutputResolution.y, swapChainFormat);
}

void Sample::SetupAnimatedObjects()
{
    const float3 maxSize = Abs(m_Scene.aabb.vMax) + Abs(m_Scene.aabb.vMin);

    Rand::Seed(106937, &m_FastRandState);

    for (uint32_t i = 0; i < ANIMATED_INSTANCE_MAX_NUM; i++)
    {
        uint32_t instanceIndex = i % m_DefaultInstancesOffset;
        float3 tmpPosition = Rand::uf3(&m_FastRandState) * maxSize - Abs(m_Scene.aabb.vMin);

        AnimatedInstance tmpAnimatedInstance = {};
        tmpAnimatedInstance.instanceID = helper::GetCountOf(m_Scene.instances);
        tmpAnimatedInstance.position = ToDouble( tmpPosition );
        tmpAnimatedInstance.basePosition = tmpAnimatedInstance.position;
        tmpAnimatedInstance.animation.durationSec = Rand::uf1(&m_FastRandState) * 10.0f + 5.0f;
        tmpAnimatedInstance.animation.progressedSec = tmpAnimatedInstance.animation.durationSec * Rand::uf1(&m_FastRandState);
        tmpAnimatedInstance.animation.rotationAxis = Normalize( Rand::sf3(&m_FastRandState) + 1e-6f );
        tmpAnimatedInstance.animation.elipseAxis = Rand::sf3(&m_FastRandState) * 5.0f;
        tmpAnimatedInstance.animation.inverseDirection = Sign( Rand::sf1(&m_FastRandState) );
        tmpAnimatedInstance.animation.inverseRotation = Sign( Rand::sf1(&m_FastRandState) );
        m_AnimatedInstances.push_back(tmpAnimatedInstance);

        const utils::Instance& tmpInstance = m_Scene.instances[instanceIndex];
        m_Scene.instances.push_back(tmpInstance);
    }
}

void Sample::PrepareFrame(uint32_t frameIndex)
{
    m_PrevSettings = m_Settings;
    m_Camera.SavePreviousState();

    PrepareUserInterface();

    if (IsKeyToggled(Key::Space))
        m_Settings.pauseAnimation = !m_Settings.pauseAnimation;
    if (IsKeyToggled(Key::F1))
        m_ShowUi = !m_ShowUi;
    if (IsKeyToggled(Key::F2))
        m_Settings.denoiser = (m_Settings.denoiser + 1) % DENOISER_MAX_NUM;
    if (IsKeyToggled(Key::F3))
        m_Settings.debug = Step(0.5f, 1.0f - m_Settings.debug);

    float avgFrameTime = m_Timer.GetVerySmoothedElapsedTime();

    float prevResolutionSCale = m_ResolutionScale;
    m_ResolutionScale *= 100.0f;

    if (!IsKeyPressed(Key::LAlt) && m_ShowUi)
    {
        ImGui::SetNextWindowPos(ImVec2(5.0f, 5.0f), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f));
        ImGui::Begin("Settings (F1 - hide)", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
        {
            char avg[64];
            snprintf(avg, sizeof(avg), "%.1f FPS (%.2f ms)", 1000.0f / avgFrameTime, avgFrameTime);

            ImVec4 colorFps = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
            if (avgFrameTime > 1000.0f / 60.0f)
                colorFps = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
            if (avgFrameTime > 1000.0f / 30.0f)
                colorFps = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);

            float lo = avgFrameTime * 0.5f;
            float hi = avgFrameTime * 1.5f;

            const uint32_t N = helper::GetCountOf(m_FrameTimes);
            uint32_t head = frameIndex % N;
            m_FrameTimes[head] = m_Timer.GetElapsedTime();
            ImGui::PushStyleColor(ImGuiCol_Text, colorFps);
            ImGui::PlotLines("Performance", m_FrameTimes.data(), N, head, avg, lo, hi, ImVec2(0, 80.0f));
            ImGui::PopStyleColor();

            if (IsButtonPressed(Button::Right))
            {
                ImGui::Text("Move - W/S/A/D");
                ImGui::Text("Accelerate - MOUSE SCROLL");
            }
            else
            {
                ImGui::PushID("CAMERA");
                {
                    #if( NRD_OCCLUSION_ONLY == 1 )
                        static const char* onScreenModes[] =
                        {
                            "Ambient occlusion",
                            "Specular occlusion",
                        };
                    #else
                        static const char* onScreenModes[] =
                        {
                            "Final",
                            "Denoised diffuse",
                            "Denoised specular",
                            "Ambient occlusion",
                            "Specular occlusion",
                            "Shadow",
                            "Base color",
                            "Normal",
                            "Roughness",
                            "Metalness",
                            "World units",
                            "Mesh",
                            "Mip level (primary)",
                            "Mip level (specular)",
                        };
                    #endif

                    static const char* motionMode[] =
                    {
                        "Left / Right",
                        "Up / Down",
                        "Forward / Backward",
                        "Mixed"
                    };

                    ImGui::Text("CAMERA (press RIGHT MOUSE BOTTON for free-fly mode)");
                    ImGui::Separator();
                    ImGui::SliderFloat("Field of view (deg)", &m_Settings.camFov, 1.0f, 160.0f, "%.1f");
                    ImGui::SliderFloat("Exposure", &m_Settings.exposure, 0.0f, 1000.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat("Resolution scale (%)", &m_ResolutionScale, m_MinResolutionScale, 100.0f, "%.1f");
                    ImGui::Combo("On screen", &m_Settings.onScreen, onScreenModes, helper::GetCountOf(onScreenModes));
                    if (!m_DLSS.IsInitialized())
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.nrdSettings.referenceAccumulation && m_Settings.TAA) ? UI_YELLOW : ImGui::GetStyleColorVec4(ImGuiCol_Text));
                            ImGui::Checkbox("TAA", &m_Settings.TAA);
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                    }
                    else
                        ImGui::SliderFloat("Sharpness", &m_Sharpness, -1.0f, 1.0f, "%.2f");
                    ImGui::Checkbox("3D MVs", &m_Settings.isMotionVectorInWorldSpace);
                    ImGui::SameLine();
                    ImGui::Checkbox("Ortho", &m_Settings.ortho);
                    ImGui::SameLine();
                    ImGui::Checkbox("FPS cap", &m_Settings.limitFps);
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.motionStartTime > 0.0 ? UI_YELLOW : ImGui::GetStyleColorVec4(ImGuiCol_Text));
                    bool isPressed = ImGui::Button("Emulate motion");
                    ImGui::PopStyleColor();
                    if (isPressed)
                        m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? 0.0 : -1.0;
                    if (m_Settings.motionStartTime > 0.0)
                    {
                        ImGui::SliderFloat("Slower / Faster", &m_Settings.emulateMotionSpeed, -10.0f, 10.0f);
                        ImGui::SetNextItemWidth(160.0f);
                        ImGui::Combo("Mode", &m_Settings.motionMode, motionMode, helper::GetCountOf(motionMode));
                        ImGui::SameLine();
                        ImGui::Checkbox("Linear", &m_Settings.linearMotion);
                    }
                    if (m_Settings.limitFps)
                        ImGui::SliderFloat("Min / Max FPS", &m_Settings.maxFps, 30.0f, 120.0f, "%.0f");
                }
                ImGui::PopID();
                ImGui::NewLine();
                ImGui::PushID("MATERIALS");
                {
                    static const char* forcedMaterial[] =
                    {
                        "None",
                        "Gypsum",
                        "Cobalt",
                    };

                    ImGui::Text("MATERIALS");
                    ImGui::Separator();
                    ImGui::SliderFloat2("Roughness / Metalness", &m_Settings.roughnessOverride, 0.0f, 1.0f, "%.3f");
                    ImGui::SetNextItemWidth(90.0f);
                    ImGui::Combo("Material", &m_Settings.forcedMaterial, forcedMaterial, helper::GetCountOf(forcedMaterial));
                    ImGui::SameLine();
                    ImGui::Checkbox("Emission", &m_Settings.emission);
                    if (m_Settings.emission)
                        ImGui::SliderFloat("Emission intensity", &m_Settings.emissionIntensity, 0.0f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::PopID();

                if (m_Settings.onScreen == 10)
                    ImGui::SliderFloat("Units in 1 meter", &m_Settings.meterToUnitsMultiplier, 0.001f, 100.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
                else
                {
                    ImGui::NewLine();
                    ImGui::PushID("WORLD");
                    {
                        ImGui::Text("WORLD");
                        ImGui::Separator();
                        ImGui::SliderFloat2("Sun position (deg)", &m_Settings.sunAzimuth, -180.0f, 180.0f, "%.1f");
                        ImGui::SliderFloat("Sun angular size (deg)", &m_Settings.sunAngularDiameter, 0.0f, 3.0f, "%.1f");
                        ImGui::Checkbox("Animate sun", &m_Settings.animateSun);
                        ImGui::SameLine();
                        ImGui::Checkbox("Animate objects", &m_Settings.animatedObjects);
                        if (!m_Scene.animations.empty() && m_Scene.animations[m_Settings.activeAnimation].cameraNode.animationNodeID != -1)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("Animate camera", &m_Settings.animateCamera);
                        }

                        if (m_Settings.animatedObjects)
                        {
                            if (!m_Settings.nineBrothers)
                                ImGui::SliderInt("Object number", &m_Settings.animatedObjectNum, 1, (int32_t)ANIMATED_INSTANCE_MAX_NUM);
                            ImGui::SliderFloat("Object scale", &m_Settings.animatedObjectScale, 0.1f, 2.0f);
                            ImGui::Checkbox("\"9 brothers\"", &m_Settings.nineBrothers);
                            ImGui::SameLine();
                            ImGui::Checkbox("Blink", &m_Settings.blink);
                            ImGui::SameLine();
                            ImGui::Checkbox("Emissive", &m_Settings.emissiveObjects);
                        }

                        if (m_Settings.animateSun || m_Settings.animatedObjects || !m_Scene.animations.empty())
                        {
                            if (m_Settings.animatedObjects)
                                ImGui::SameLine();
                            ImGui::Checkbox("Pause (SPACE)", &m_Settings.pauseAnimation);
                            ImGui::SliderFloat("Slower / Faster", &m_Settings.animationSpeed, -10.0f, 10.0f);
                        }

                        if (!m_Scene.animations.empty())
                        {
                            if (m_Scene.animations[m_Settings.activeAnimation].durationMs != 0.0f)
                            {
                                char animationLabel[128];
                                snprintf(animationLabel, sizeof(animationLabel), "Animation %.1f sec (%%)", 0.001f * m_Scene.animations[m_Settings.activeAnimation].durationMs / (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed)));
                                ImGui::SliderFloat(animationLabel, &m_Settings.animationProgress, 0.0f, 99.999f);

                                if (m_Scene.animations.size() > 1)
                                {
                                    char items[1024] = {'\0'};
                                    size_t offset = 0;
                                    char* iterator = items;
                                    for (auto animation : m_Scene.animations)
                                    {
                                        const size_t size = std::min(sizeof(items), animation.animationName.length() + 1);
                                        memcpy(iterator + offset, animation.animationName.c_str(), size);
                                        offset += animation.animationName.length() + 1;
                                    }
                                    ImGui::Combo("Animated scene", &m_Settings.activeAnimation, items, helper::GetCountOf(m_Scene.animations));
                                }
                            }
                        }

                        m_Settings.sunElevation = Clamp(m_Settings.sunElevation, -90.0f, 90.0f);
                    }
                    ImGui::PopID();
                    ImGui::NewLine();
                    ImGui::PushID("INDIRECT RAYS");
                    {
                        const float sceneRadiusInMeters = m_Scene.aabb.GetRadius() / m_Settings.meterToUnitsMultiplier;

                        static const char* resolution[] =
                        {
                            "Full",
                            "Half",
                            "Quarter",
                        };

                        ImGui::Text("INDIRECT RAYS");
                        ImGui::Separator();
                        ImGui::SliderInt2("Samples / Bounces", &m_Settings.rpp, 1, 8);
                        ImGui::SliderFloat2("AO / SO range (m)", &m_Settings.diffHitDistScale, 0.01f, sceneRadiusInMeters, "%.2f");
                        ImGui::SetNextItemWidth(100.0f);
                        ImGui::Combo("Resolution", &m_Settings.tracingMode, resolution, helper::GetCountOf(resolution));
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.nrdSettings.referenceAccumulation && m_Settings.specularLobeTrimming) ? UI_YELLOW : ImGui::GetStyleColorVec4(ImGuiCol_Text));
                            ImGui::Checkbox("Specular lobe trimming", &m_Settings.specularLobeTrimming);
                        ImGui::PopStyleColor();
                        ImGui::Checkbox("Diffuse", &m_Settings.indirectDiffuse);
                        ImGui::SameLine();
                        ImGui::Checkbox("Specular", &m_Settings.indirectSpecular);
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.ambient && m_Settings.denoiser == RELAX) ? UI_YELLOW : ImGui::GetStyleColorVec4(ImGuiCol_Text));
                            ImGui::Checkbox("Ambient", &m_Settings.ambient);
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::Checkbox("Normal map", &m_Settings.normalMap);

                        const float3& sunDirection = GetSunDirection();
                        bool cmp = sunDirection.z < 0.0f && m_Settings.importanceSampling;
                        if (cmp)
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
                        ImGui::Checkbox("Importance sampling", &m_Settings.importanceSampling);
                        if (cmp)
                            ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::Checkbox("Reference", &m_Settings.reference);
                    }
                    ImGui::PopID();
                    ImGui::NewLine();
                    ImGui::PushID("DENOISER");
                    {
                        const nrd::LibraryDesc& nrdLibraryDesc = nrd::GetLibraryDesc();

                        ImGui::Text("NRD v%u.%u.%u - %s / SIGMA (F2 - change)", nrdLibraryDesc.versionMajor, nrdLibraryDesc.versionMinor, nrdLibraryDesc.versionBuild, m_Settings.denoiser == REBLUR ? "REBLUR" : "RELAX");
                        ImGui::Separator();
                        ImGui::SliderFloat("Disocclusion (%)", &m_Settings.nrdSettings.disocclusionThreshold, 0.25f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                        if (m_Settings.denoiser == REBLUR)
                            ImGui::SliderInt("History length (frames)", &m_Settings.nrdSettings.maxAccumulatedFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);
                        else
                            ImGui::SliderInt2("History length (frames)", &m_Settings.nrdSettings.maxAccumulatedFrameNum, 0, nrd::REBLUR_MAX_HISTORY_FRAME_NUM);

                        ImGui::Checkbox("Anti-firefly", &m_Settings.nrdSettings.enableAntiFirefly);
                        ImGui::SameLine();
                        ImGui::Checkbox("Adaptive accum", &m_Settings.adaptiveAccumulation);
                        if (m_Settings.denoiser == REBLUR)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("Reference accum", &m_Settings.nrdSettings.referenceAccumulation);
                        }

                        static const char* prePassMode[] =
                        {
                            "Off",
                            "Simple",
                            "Advanced",
                        };

                        if (m_Settings.denoiser == REBLUR)
                        {
                            ImGui::Text("SPATIAL FILTERING:");
                            ImGui::Combo("Pre-pass", &m_Settings.nrdSettings.prePassMode, prePassMode, helper::GetCountOf(prePassMode));
                            ImGui::SliderFloat("Blur radius (px)", &m_Settings.nrdSettings.blurRadius, 0.0f, 60.0f, "%.1f");
                            ImGui::SliderFloat("Adaptive radius scale", &m_Settings.nrdSettings.adaptiveRadiusScale, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderFloat("Normal weight strictness", &m_Settings.nrdSettings.normalWeightStrictness, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Stabilization strength", &m_Settings.nrdSettings.stabilizationStrength, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Residual noise level", &m_Settings.nrdSettings.residualNoiseLevel, 1.0f, 10.0f, "%.2f");

                            nrd::AntilagIntensitySettings antilagIntensitySettings = {};
                            nrd::AntilagHitDistanceSettings antilagHitDistanceSettings = {};
                            GetAntilagSettings(antilagIntensitySettings, antilagHitDistanceSettings);

                            char s[128];
                            snprintf(s, sizeof(s), "[%.1f; %.1f]", antilagIntensitySettings.thresholdMin * 100.0f, antilagIntensitySettings.thresholdMax * 100.0f);
                            ImGui::Text("ANTI-LAG:");
                            ImGui::Checkbox("Intensity", &m_Settings.nrdSettings.antilagIntensity);
                            ImGui::SameLine();
                            ImGui::Text(s);

                            snprintf(s, sizeof(s), "[%.1f; %.1f]", antilagHitDistanceSettings.thresholdMin * 100.0f, antilagHitDistanceSettings.thresholdMax * 100.0f);
                            ImGui::SameLine();
                            ImGui::Checkbox("Hit distance", &m_Settings.nrdSettings.antilagHitDistance);
                            ImGui::SameLine();
                            ImGui::Text(s);
                        }
                        else if (m_Settings.denoiser == RELAX)
                        {
                            ImGui::Text("REPROJECTION:");
                            ImGui::SliderFloat("Spec variance boost", &m_RelaxSettings.specularVarianceBoost, 0.0f, 8.0f, "%.2f");
                            ImGui::SliderFloat("Clamping sigma scale", &m_RelaxSettings.historyClampingColorBoxSigmaScale, 0.0f, 10.0f, "%.1f");

                            ImGui::Text("SPATIAL FILTERING:");
                            ImGui::Combo("Pre-pass", &m_Settings.nrdSettings.prePassMode, prePassMode, helper::GetCountOf(prePassMode));
                            ImGui::SliderInt("A-trous iterations", (int32_t*)&m_RelaxSettings.atrousIterationNum, 2, 8);
                            ImGui::SliderFloat2("Spec / Diff lum weight", &m_RelaxSettings.specularPhiLuminance, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderFloat2("Normal / Lum relaxation", &m_RelaxSettings.normalEdgeStoppingRelaxation, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Roughness relaxation", &m_RelaxSettings.roughnessEdgeStoppingRelaxation, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Depth weight", &m_RelaxSettings.depthThreshold, 0.0f, 1.0f, "%.3f");
                            ImGui::SliderFloat("Normal weight", &m_RelaxSettings.phiNormal, 1.0f, 256.0f, "%.0f");

                            ImGui::Text("DISOCCLUSION FIX:");
                            ImGui::SliderFloat("Edge-stop normal power", &m_RelaxSettings.disocclusionFixEdgeStoppingNormalPower, 0.0f, 128.0f, "%.1f");
                            ImGui::SliderFloat("Max kernel radius", &m_RelaxSettings.disocclusionFixMaxRadius, 0.0f, 100.0f, "%.1f");
                            ImGui::SliderInt("Frames to fix", (int32_t*)&m_RelaxSettings.disocclusionFixNumFramesToFix, 0, 10);

                            ImGui::Text("SPATIAL VARIANCE ESTIMATION:");
                            ImGui::SliderInt("History threshold", (int32_t*)&m_RelaxSettings.spatialVarianceEstimationHistoryThreshold, 0, 10);
                        }

                        m_ForceHistoryReset = ImGui::Button("Reset history");
                        ImGui::SameLine();

                        if (ImGui::Button("Change denoiser"))
                            m_Settings.denoiser = (m_Settings.denoiser + 1) % DENOISER_MAX_NUM;
                    }
                    ImGui::PopID();
                    ImGui::NewLine();
                    ImGui::Separator();
                    ImGui::SliderFloat("Debug (F3 - toggle)", &m_Settings.debug, 0.0f, 1.0f, "%.6f");
                    ImGui::SliderFloat("Input / Denoised", &m_Settings.separator, 0.0f, 1.0f, "%.2f");

                    if (ImGui::Button("Reload shaders"))
                    {
                        CreatePipelines();
                        printf("Ready!\n");
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Default settings"))
                    {
                        m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                        m_Settings = m_DefaultSettings;
                        m_ForceHistoryReset = true;
                    }

                    // Tests
                    if (m_TestMode)
                    {
                        ImGui::NewLine();
                        ImGui::Text("TESTS (F4 - next)");
                        ImGui::Separator();

                        char s[64];
                        std::string sceneName = std::string( utils::GetFileName(m_SceneFile) );
                        size_t dotPos = sceneName.find_last_of(".");
                        if (dotPos != std::string::npos)
                            sceneName.replace(dotPos, 4, ".bin");
                        const std::string path = utils::GetFullPath(sceneName, utils::DataFolder::TESTS);
                        const uint32_t testByteSize = sizeof(m_Settings) + Camera::GetStateSize();

                        // Get number of tests
                        if (m_TestNum == uint32_t(-1))
                        {
                            FILE* fp = fopen(path.c_str(), "rb");
                            if (fp)
                            {
                                // Use this code to convert tests to reflect new Settings and Camera layouts
                                #if 0
                                    typedef Settings SettingsOld; // adjust if needed
                                    typedef Camera CameraOld; // adjust if needed

                                    const uint32_t oldItemSize = sizeof(SettingsOld) + CameraOld::GetStateSize();

                                    fseek(fp, 0, SEEK_END);
                                    m_TestNum = ftell(fp) / oldItemSize;
                                    fseek(fp, 0, SEEK_SET);

                                    FILE* fpNew;
                                    fopen_s(&fpNew, (path + ".new").c_str(), "wb");

                                    for (uint32_t i = 0; i < m_TestNum && fpNew; i++)
                                    {
                                        SettingsOld settingsOld;
                                        fread_s(&settingsOld, sizeof(SettingsOld), 1, sizeof(SettingsOld), fp);

                                        CameraOld cameraOld;
                                        fread_s(cameraOld.GetState(), CameraOld::GetStateSize(), 1, CameraOld::GetStateSize(), fp);

                                        // Convert Old to New here
                                        m_Settings = settingsOld;
                                        m_Camera.state = cameraOld.state;

                                        // ...

                                        fwrite(&m_Settings, 1, sizeof(m_Settings), fpNew);
                                        fwrite(m_Camera.GetState(), 1, Camera::GetStateSize(), fpNew);
                                    }

                                    fclose(fp);
                                    fclose(fpNew);

                                    __debugbreak();
                                #endif

                                fseek(fp, 0, SEEK_END);
                                m_TestNum = ftell(fp) / testByteSize;
                                fclose(fp);
                            }
                            else
                                m_TestNum = 0;
                        }

                        // Adjust current test index
                        bool isTestChanged = false;
                        if (IsKeyToggled(Key::F4) && m_TestNum)
                        {
                            m_LastSelectedTest++;
                            isTestChanged = true;
                        }

                        if (m_LastSelectedTest == uint32_t(-1) || !m_TestNum)
                            m_LastSelectedTest = uint32_t(-1);
                        else
                            m_LastSelectedTest %= m_TestNum;

                        // Main buttons
                        uint32_t i = 0;
                        for (; i < m_TestNum; i++)
                        {
                            snprintf(s, sizeof(s), "%u", i + 1);

                            if (i % 14 != 0)
                                ImGui::SameLine();

                            bool cmp = i == m_LastSelectedTest;
                            if (cmp)
                                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));

                            if (ImGui::Button(s, ImVec2(25.0f, 0.0f)) || isTestChanged)
                            {
                                uint32_t test = isTestChanged ? m_LastSelectedTest : i;
                                FILE* fp = fopen(path.c_str(), "rb");

                                if (fp && fseek(fp, test * testByteSize, SEEK_SET) == 0)
                                {
                                    size_t elemNum = fread(&m_Settings, sizeof(m_Settings), 1, fp);
                                    if (elemNum == 1)
                                        elemNum = fread(m_Camera.GetState(), Camera::GetStateSize(), 1, fp);

                                    m_LastSelectedTest = test;

                                    // File read error
                                    if (elemNum != 1)
                                    {
                                        m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                                        m_Settings = m_DefaultSettings;
                                    }

                                    // Reset some settings to defaults to avoid a potential confusion
                                    m_Settings.debug = 0.0f;
                                    m_Settings.denoiser = REBLUR;
                                    m_ForceHistoryReset = true;
                                }

                                if (fp)
                                    fclose(fp);

                                isTestChanged = false;
                            }

                            if (cmp)
                                ImGui::PopStyleColor();
                        }

                        if (i % 14 != 0)
                            ImGui::SameLine();

                        // "Add" button
                        if (ImGui::Button("Add"))
                        {
                            FILE* fp = fopen(path.c_str(), "ab");

                            if (fp)
                            {
                                m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? -1.0 : 0.0;

                                fwrite(&m_Settings, sizeof(m_Settings), 1, fp);
                                fwrite(m_Camera.GetState(), Camera::GetStateSize(), 1, fp);
                                fclose(fp);

                                m_TestNum = uint32_t(-1);
                            }
                        }

                        if ((i + 1) % 14 != 0)
                            ImGui::SameLine();

                        // "Del" button
                        snprintf(s, sizeof(s), "Del %u", m_LastSelectedTest + 1);
                        if (m_TestNum != uint32_t(-1) && m_LastSelectedTest != uint32_t(-1) && ImGui::Button(s))
                        {
                            std::vector<uint8_t> data;
                            utils::LoadFile(path, data);

                            FILE* fp = fopen(path.c_str(), "wb");

                            if (fp)
                            {
                                for (i = 0; i < m_TestNum; i++)
                                {
                                    if (i != m_LastSelectedTest)
                                        fwrite(&data[i * testByteSize], 1, testByteSize, fp);
                                }

                                fclose(fp);

                                m_TestNum = uint32_t(-1);
                            }
                        }
                    }
                }
            }
        }
        ImGui::End();
    }

    // Update camera
    cBoxf cameraLimits = m_Scene.aabb;
    cameraLimits.Scale(2.0f);

    CameraDesc desc = {};
    desc.limits = cameraLimits;
    desc.aspectRatio = float( GetWindowWidth() ) / float( GetWindowHeight() );
    desc.horizontalFov = RadToDeg( Atan( Tan( DegToRad( m_Settings.camFov ) * 0.5f ) *  desc.aspectRatio * 9.0f / 16.0f ) * 2.0f ); // recalculate to ultra-wide if needed
    desc.nearZ = NEAR_Z * m_Settings.meterToUnitsMultiplier;
    desc.farZ = 10000.0f * m_Settings.meterToUnitsMultiplier;
    desc.isCustomMatrixSet = m_Settings.animateCamera;
    desc.isLeftHanded = CAMERA_LEFT_HANDED;
    desc.orthoRange = m_Settings.ortho ? Tan( DegToRad( m_Settings.camFov ) * 0.5f ) * 3.0f * m_Settings.meterToUnitsMultiplier : 0.0f;
    GetCameraDescFromInputDevices(desc);

    const float animationSpeed = m_Settings.pauseAnimation ? 0.0f : (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
    const float scale = m_Settings.animatedObjectScale * m_Settings.meterToUnitsMultiplier / 2.0f;
    const float objectAnimationDelta = animationSpeed * m_Timer.GetElapsedTime() * 0.001f;

    if (m_Settings.motionStartTime > 0.0)
    {
        float time = float(m_Timer.GetTimeStamp() - m_Settings.motionStartTime);
        float amplitude = 40.0f * m_Camera.state.motionScale;
        float period = 0.0003f * time * (m_Settings.emulateMotionSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.emulateMotionSpeed)) : (1.0f + m_Settings.emulateMotionSpeed));

        float3 localPos = m_Camera.state.mWorldToView.GetRow0().To3d();
        if (m_Settings.motionMode == 1)
            localPos = m_Camera.state.mWorldToView.GetRow1().To3d();
        else if (m_Settings.motionMode == 2)
            localPos = m_Camera.state.mWorldToView.GetRow2().To3d();
        else if (m_Settings.motionMode == 3)
        {
            float3 rows[3] = { m_Camera.state.mWorldToView.GetRow0().To3d(), m_Camera.state.mWorldToView.GetRow1().To3d(), m_Camera.state.mWorldToView.GetRow2().To3d() };
            float f = Sin( Pi(period * 3.0f) );
            localPos = Normalize( f < 0.0f ? Lerp( rows[1], rows[0], float3( Abs(f) ) ) : Lerp( rows[1], rows[2], float3(f) ) );
        }
        localPos *= amplitude * (m_Settings.linearMotion ? WaveTriangle(period) - 0.5f : Sin( Pi(period) ) * 0.5f);

        desc.dUser = localPos - m_PrevLocalPos;
        m_PrevLocalPos = localPos;
    }
    else if (m_Settings.motionStartTime == -1.0)
    {
        m_Settings.motionStartTime = m_Timer.GetTimeStamp();
        m_PrevLocalPos = float3::Zero();
    }

    m_Scene.Animate(animationSpeed, m_Timer.GetElapsedTime(), m_Settings.animationProgress, m_Settings.activeAnimation, m_Settings.animateCamera ? &desc.customMatrix : nullptr);
    m_Camera.Update(desc, frameIndex);

    if (m_Settings.nineBrothers)
    {
        m_Settings.animatedObjectNum = 9;

        const float3& vRight = m_Camera.state.mViewToWorld.GetCol0().xmm;
        const float3& vTop = m_Camera.state.mViewToWorld.GetCol1().xmm;
        const float3& vForward = m_Camera.state.mViewToWorld.GetCol2().xmm;

        float3 basePos = ToFloat(m_Camera.state.globalPosition);

        for (int32_t i = -1; i <= 1; i++ )
        {
            for (int32_t j = -1; j <= 1; j++ )
            {
                const uint32_t index = (i + 1) * 3 + (j + 1);

                float x = float(i) * scale * 4.0f;
                float y = float(j) * scale * 4.0f;
                float z = 10.0f * scale * (CAMERA_LEFT_HANDED ? 1.0f : -1.0f);

                float3 pos = basePos + vRight * x + vTop * y + vForward * z;

                utils::Instance& instance = m_Scene.instances[ m_AnimatedInstances[index].instanceID ];
                instance.position = ToDouble( pos );
                instance.rotation = m_Camera.state.mViewToWorld;
                instance.rotation.SetTranslation( float3::Zero() );
                instance.rotation.AddScale(scale);
            }
        }
    }
    else if (m_Settings.animatedObjects)
    {
        for (int32_t i = 0; i < m_Settings.animatedObjectNum; i++)
        {
            float4x4 transform = m_AnimatedInstances[i].Animate(objectAnimationDelta, scale);

            utils::Instance& instance = m_Scene.instances[ m_AnimatedInstances[i].instanceID ];
            instance.rotation = transform;
            instance.position = m_AnimatedInstances[i].position;
        }
    }

    m_ResolutionScale *= 0.01f;

    if (prevResolutionSCale != m_ResolutionScale || m_PrevSettings.tracingMode != m_Settings.tracingMode)
    {
        printf
        (
            "\nResolution change:\n"
            "Output        : %ux%u\n"
            "Primary rays  : %ux%u\n"
            "Indirect rays : %ux%u\n",
            m_OutputResolution.x, m_OutputResolution.y,
            uint32_t(m_ScreenResolution.x * m_ResolutionScale + 0.5f), uint32_t(m_ScreenResolution.y * m_ResolutionScale + 0.5f),
            uint32_t(m_ScreenResolution.x * m_ResolutionScale * (m_Settings.tracingMode > 0 ? 0.5f : 1.0f) + 0.5f), uint32_t(m_ScreenResolution.y * m_ResolutionScale * (m_Settings.tracingMode > 1 ? 0.5f : 1.0f) + 0.5f)
        );
    }
}

void Sample::CreateSwapChain(nri::Format& swapChainFormat)
{
    nri::SwapChainDesc swapChainDesc = {};
    swapChainDesc.windowSystemType = GetWindowSystemType();
    swapChainDesc.window = GetWindow();
    swapChainDesc.commandQueue = m_CommandQueue;
    swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
    swapChainDesc.verticalSyncInterval = m_SwapInterval;
    swapChainDesc.width = (uint16_t)m_OutputResolution.x;
    swapChainDesc.height = (uint16_t)m_OutputResolution.y;
    swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;

    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

    uint32_t swapChainTextureNum = 0;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum, swapChainFormat);

    nri::ClearValueDesc clearColor = {};
    nri::FrameBufferDesc frameBufferDesc = {};
    frameBufferDesc.colorAttachmentNum = 1;
    frameBufferDesc.colorClearValues = &clearColor;

    for (uint32_t i = 0; i < swapChainTextureNum; i++)
    {
        m_SwapChainBuffers.emplace_back();
        BackBuffer& backBuffer = m_SwapChainBuffers.back();

        backBuffer = {};
        backBuffer.texture = swapChainTextures[i];

        nri::Texture2DViewDesc textureViewDesc = {backBuffer.texture, nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, backBuffer.colorAttachment));

        frameBufferDesc.colorAttachments = &backBuffer.colorAttachment;
        NRI_ABORT_ON_FAILURE(NRI.CreateFrameBuffer(*m_Device, frameBufferDesc, backBuffer.frameBufferUI));
    }
}

void Sample::CreateCommandBuffers()
{
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE(NRI.CreateDeviceSemaphore(*m_Device, true, frame.deviceSemaphore));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, frame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer));
    }
}

void Sample::CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint16_t width, uint16_t height, uint16_t mipNum, uint16_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state)
{
    nri::Texture* texture = nullptr;
    const nri::CTextureDesc textureDesc = nri::CTextureDesc::Texture2D(format, width, height, mipNum, arraySize, usage);
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, texture));
    m_Textures.push_back(texture);

    if (state != nri::AccessBits::UNKNOWN)
    {
        nri::TextureTransitionBarrierDesc transition = nri::TextureTransition(texture, state, state == nri::AccessBits::SHADER_RESOURCE ? nri::TextureLayout::SHADER_RESOURCE : nri::TextureLayout::GENERAL);
        m_TextureStates.push_back(transition);
        m_TextureFormats.push_back(format);
    }

    descriptorDescs.push_back( {debugName, texture, format, usage, nri::BufferUsageBits::NONE, arraySize > 1} );
}

void Sample::CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage, nri::Format format)
{
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = elements * stride;
    bufferDesc.structureStride = (format == nri::Format::UNKNOWN && stride != 1) ? stride : 0;
    bufferDesc.usageMask = usage;

    nri::Buffer* buffer = nullptr;
    NRI_ABORT_ON_FAILURE( NRI.CreateBuffer(*m_Device, bufferDesc, buffer) );
    m_Buffers.push_back(buffer);

    descriptorDescs.push_back( {debugName, buffer, format, nri::TextureUsageBits::NONE, usage} );
}

inline nri::Format ConvertFormatToTextureStorageCompatible(nri::Format format)
{
    switch (format)
    {
        case nri::Format::D16_UNORM:                return nri::Format::R16_UNORM;
        case nri::Format::D24_UNORM_S8_UINT:        return nri::Format::R24_UNORM_X8;
        case nri::Format::D32_SFLOAT:               return nri::Format::R32_SFLOAT;
        case nri::Format::D32_SFLOAT_S8_UINT_X24:   return nri::Format::R32_SFLOAT_X8_X24;
        case nri::Format::RGBA8_SRGB:               return nri::Format::RGBA8_UNORM;
        case nri::Format::BGRA8_SRGB:               return nri::Format::BGRA8_UNORM;
    }

    return format;
}

void Sample::CreateDescriptors(const std::vector<DescriptorDesc>& descriptorDescs)
{
    nri::Descriptor* descriptor = nullptr;
    for (const DescriptorDesc& desc : descriptorDescs)
    {
        if (desc.textureUsage == nri::TextureUsageBits::NONE)
        {
            if (desc.bufferUsage == nri::BufferUsageBits::CONSTANT_BUFFER)
            {
                for (uint32_t i = 0; i < BUFFERED_FRAME_MAX_NUM; i++)
                {
                    nri::BufferViewDesc bufferDesc = {};
                    bufferDesc.buffer = Get(Buffer::GlobalConstants);
                    bufferDesc.viewType = nri::BufferViewType::CONSTANT;
                    bufferDesc.offset = i * m_ConstantBufferSize;
                    bufferDesc.size = m_ConstantBufferSize;

                    NRI_ABORT_ON_FAILURE( NRI.CreateBufferView(bufferDesc, m_Frames[i].globalConstantBufferDescriptor) );
                    m_Frames[i].globalConstantBufferOffset = bufferDesc.offset;
                }
            }
            else if (desc.bufferUsage & nri::BufferUsageBits::SHADER_RESOURCE)
            {
                const nri::BufferViewDesc viewDesc = {(nri::Buffer*)desc.resource, nri::BufferViewType::SHADER_RESOURCE, desc.format};
                NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(viewDesc, descriptor));
                m_Descriptors.push_back(descriptor);
            }

            NRI.SetBufferDebugName(*(nri::Buffer*)desc.resource, desc.debugName);
        }
        else
        {
            nri::Texture2DViewDesc viewDesc = {(nri::Texture*)desc.resource, desc.isArray ? nri::Texture2DViewType::SHADER_RESOURCE_2D_ARRAY : nri::Texture2DViewType::SHADER_RESOURCE_2D, desc.format};
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(viewDesc, descriptor));
            m_Descriptors.push_back(descriptor);

            if (desc.textureUsage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE)
            {
                viewDesc.format = ConvertFormatToTextureStorageCompatible(desc.format);
                viewDesc.viewType = desc.isArray ? nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D_ARRAY : nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D;
                NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(viewDesc, descriptor));
                m_Descriptors.push_back(descriptor);
            }

            NRI.SetTextureDebugName(*(nri::Texture*)desc.resource, desc.debugName);
        }
    }
}

void Sample::CreateResources(nri::Format swapChainFormat)
{
    std::vector<DescriptorDesc> descriptorDescs;

    const uint16_t w = (uint16_t)m_ScreenResolution.x;
    const uint16_t h = (uint16_t)m_ScreenResolution.y;
    const uint64_t instanceDataSize = (m_Scene.instances.size() + ANIMATED_INSTANCE_MAX_NUM) * sizeof(InstanceData);
    const uint64_t worldScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*m_WorldTlas);
    const uint64_t lightScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*m_LightTlas);

    // nri::MemoryLocation::HOST_UPLOAD
    CreateBuffer(descriptorDescs, "Buffer::GlobalConstants", m_ConstantBufferSize * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::CONSTANT_BUFFER);
    CreateBuffer(descriptorDescs, "Buffer::InstanceDataStaging", instanceDataSize * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::NONE);
    CreateBuffer(descriptorDescs, "Buffer::WorldTlasDataStaging", (m_Scene.instances.size() + ANIMATED_INSTANCE_MAX_NUM) * sizeof(nri::GeometryObjectInstance) * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER);
    CreateBuffer(descriptorDescs, "Buffer::LightTlasDataStaging", (m_Scene.instances.size() + ANIMATED_INSTANCE_MAX_NUM) * sizeof(nri::GeometryObjectInstance) * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER);

    // nri::MemoryLocation::DEVICE
    CreateBuffer(descriptorDescs, "Buffer::PrimitiveData", m_Scene.primitives.size(), sizeof(PrimitiveData), nri::BufferUsageBits::SHADER_RESOURCE, nri::Format::UNKNOWN);
    CreateBuffer(descriptorDescs, "Buffer::InstanceData", instanceDataSize / sizeof(InstanceData), sizeof(InstanceData), nri::BufferUsageBits::SHADER_RESOURCE, nri::Format::UNKNOWN);
    CreateBuffer(descriptorDescs, "Buffer::WorldScratch", worldScratchBufferSize, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::LightScratch", lightScratchBufferSize, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);

#if( NRD_OCCLUSION_ONLY == 1 )
    nri::Format dataFormat = nri::Format::R16_SFLOAT;
#else
    nri::Format dataFormat = nri::Format::RGBA16_SFLOAT;
#endif

    nri::Format outputFormat = m_DLSS.IsInitialized() ? nri::Format::RGBA16_SFLOAT : swapChainFormat;

    CreateTexture(descriptorDescs, "Texture::IntegrateBRDF", nri::Format::RG16_SFLOAT, FG_TEX_SIZE, FG_TEX_SIZE, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::Ambient", nri::Format::RGBA16_SFLOAT, 2, 2, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ViewZ", nri::Format::R32_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Motion", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Normal_Roughness", NORMAL_FORMAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::PrimaryMip", nri::Format::R8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Downsampled_ViewZ", nri::Format::R32_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Downsampled_Motion", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Downsampled_Normal_Roughness", NORMAL_FORMAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::BaseColor_Metalness", nri::Format::RGBA8_SRGB, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectLighting", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectEmission", nri::Format::R11_G11_B10_UFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::TransparentLighting", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Shadow", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DiffDirectionPdf", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::SpecDirectionPdf", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_ShadowData", nri::Format::RG16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Shadow_Translucency", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::ComposedLighting_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::TaaHistory", nri::Format::R10_G10_B10_A2_UNORM, m_OutputResolution.x, m_OutputResolution.y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::TaaHistoryPrev", nri::Format::R10_G10_B10_A2_UNORM, m_OutputResolution.x, m_OutputResolution.y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::DlssOutput", nri::Format::R11_G11_B10_UFLOAT, m_OutputResolution.x, m_OutputResolution.y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::Final", swapChainFormat, (uint16_t)m_OutputResolution.x, (uint16_t)m_OutputResolution.y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::COPY_SOURCE);

    // Material textures
    for (const utils::Texture* textureData : m_Scene.textures)
        CreateTexture(descriptorDescs, "", textureData->GetFormat(), textureData->GetWidth(), textureData->GetHeight(), textureData->GetMipNum(), textureData->GetArraySize(), nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);

    constexpr uint32_t offset = uint32_t(Buffer::UploadHeapBufferNum);

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    resourceGroupDesc.bufferNum = offset;
    resourceGroupDesc.buffers = m_Buffers.data();

    size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE( NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = helper::GetCountOf(m_Buffers) - offset;
    resourceGroupDesc.buffers = m_Buffers.data() + offset;
    resourceGroupDesc.textureNum = helper::GetCountOf(m_Textures);
    resourceGroupDesc.textures = m_Textures.data();

    baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE( NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    CreateDescriptors(descriptorDescs);
}

void Sample::CreatePipelines()
{
    if (!m_Pipelines.empty())
    {
        NRI.WaitForIdle(*m_CommandQueue);

        for (uint32_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(*m_Pipelines[i]);
        m_Pipelines.clear();

        m_Reblur.CreatePipelines();
        m_Relax.CreatePipelines();
        m_Sigma.CreatePipelines();
        m_Reference.CreatePipelines();
    }

    utils::ShaderCodeStorage shaderCodeStorage;
    nri::PipelineLayout* pipelineLayout = nullptr;
    nri::Pipeline* pipeline = nullptr;

    nri::SamplerDesc samplerDescs[3] = {};
    {
        samplerDescs[0].addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDescs[0].minification = nri::Filter::LINEAR;
        samplerDescs[0].magnification = nri::Filter::LINEAR;
        samplerDescs[0].mip = nri::Filter::LINEAR;
        samplerDescs[0].mipMax = 16.0f;

        samplerDescs[1].addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDescs[1].minification = nri::Filter::NEAREST;
        samplerDescs[1].magnification = nri::Filter::NEAREST;
        samplerDescs[1].mip = nri::Filter::NEAREST;
        samplerDescs[1].mipMax = 16.0f;

        samplerDescs[2].addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
        samplerDescs[2].minification = nri::Filter::LINEAR;
        samplerDescs[2].magnification = nri::Filter::LINEAR;
    }

    const nri::StaticSamplerDesc staticSamplersDesc[] =
    {
        { samplerDescs[0], 1, nri::ShaderStage::ALL },
        { samplerDescs[1], 2, nri::ShaderStage::ALL },
        { samplerDescs[2], 3, nri::ShaderStage::ALL },
    };

    const nri::DescriptorRangeDesc descriptorRanges0[] =
    {
        { 0, 1, nri::DescriptorType::CONSTANT_BUFFER, nri::ShaderStage::ALL },
    };

    // Ray tracing resources
    const uint32_t textureNum = helper::GetCountOf(m_Scene.materials) * TEXTURES_PER_MATERIAL;
    nri::DescriptorRangeDesc descriptorRanges2[] =
    {
        { 0, 2, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::ShaderStage::ALL },
        { 2, 2, nri::DescriptorType::STRUCTURED_BUFFER, nri::ShaderStage::ALL },
        { 4, textureNum, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL, nri::VARIABLE_DESCRIPTOR_NUM, nri::DESCRIPTOR_ARRAY },
    };

    { // Pipeline::IntegrateBRDF
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "IntegrateBRDF.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::AmbientRays
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL },
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
            { descriptorRanges2, helper::GetCountOf(descriptorRanges2) }
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "AmbientRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::PrimaryRays
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 3, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 3, 10, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL },
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
            { descriptorRanges2, helper::GetCountOf(descriptorRanges2) }
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "PrimaryRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::DirectLighting
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 2, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 2, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "DirectLighting.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::IndirectRays
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 7, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 7, 7, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
            { descriptorRanges2, helper::GetCountOf(descriptorRanges2) }
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "IndirectRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Composition
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 9, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 9, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "Composition.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Temporal
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 4, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 4, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "Temporal.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Upsample
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 1, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 1, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "Upsample.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::PreDlss
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 3, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 3, 3, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "PreDlss.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::AfterDlss
        const nri::DescriptorRangeDesc descriptorRanges1[] =
        {
            { 0, 1, nri::DescriptorType::TEXTURE, nri::ShaderStage::ALL },
            { 1, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::ALL }
        };

        const nri::DescriptorSetDesc descriptorSetDesc[] =
        {
            { descriptorRanges0, helper::GetCountOf(descriptorRanges0), staticSamplersDesc, helper::GetCountOf(staticSamplersDesc) },
            { descriptorRanges1, helper::GetCountOf(descriptorRanges1) },
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
        pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, pipelineLayout));
        m_PipelineLayouts.push_back(pipelineLayout);

        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.computeShader = utils::LoadShader(m_DeviceDesc->graphicsAPI, "AfterDlss.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }
}

void Sample::CreateDescriptorSets()
{
    nri::DescriptorSet* descriptorSet = nullptr;

    nri::DescriptorPoolDesc descriptorPoolDesc = {};
    descriptorPoolDesc.descriptorSetMaxNum = 128;
    descriptorPoolDesc.staticSamplerMaxNum = 3 * BUFFERED_FRAME_MAX_NUM;
    descriptorPoolDesc.storageTextureMaxNum = 128;
    descriptorPoolDesc.textureMaxNum = 128 + uint32_t(m_Scene.materials.size()) * TEXTURES_PER_MATERIAL;
    descriptorPoolDesc.accelerationStructureMaxNum = 16;
    descriptorPoolDesc.bufferMaxNum = 16;
    descriptorPoolDesc.structuredBufferMaxNum = 16;
    descriptorPoolDesc.constantBufferMaxNum = 1 * BUFFERED_FRAME_MAX_NUM;
    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));

    // Constant buffer
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::PrimaryRays), 0, &frame.globalConstantBufferDescriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { &frame.globalConstantBufferDescriptor, 1 },
        };

        NRI.UpdateDescriptorRanges(*frame.globalConstantBufferDescriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::IntegrateBRDF0
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::IntegrateBRDF), 0, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::IntegrateBRDF_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::AmbientRays1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::AmbientRays), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Ambient_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::PrimaryRays1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::PrimaryRays), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get( Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::ScramblingRanking1spp) ),
            Get( Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::SobolSequence) ),
            Get(Descriptor::Ambient_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Motion_StorageTexture),
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
            Get(Descriptor::BaseColor_Metalness_StorageTexture),
            Get(Descriptor::PrimaryMip_StorageTexture),
            Get(Descriptor::DirectLighting_StorageTexture),
            Get(Descriptor::DirectEmission_StorageTexture),
            Get(Descriptor::TransparentLighting_StorageTexture),
            Get(Descriptor::Unfiltered_ShadowData_StorageTexture),
            Get(Descriptor::Unfiltered_Shadow_Translucency_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::DirectLighting1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::DirectLighting), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::DirectEmission_Texture),
            Get(Descriptor::Shadow_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::DirectLighting_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::IndirectRays1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::IndirectRays), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::ViewZ_Texture),
            Get(Descriptor::Normal_Roughness_Texture),
            Get(Descriptor::BaseColor_Metalness_Texture),
            Get(Descriptor::PrimaryMip_Texture),
            Get(Descriptor::ComposedLighting_ViewZ_Texture),
            Get(Descriptor::Ambient_Texture),
            Get(Descriptor::Motion_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Unfiltered_Diff_StorageTexture),
            Get(Descriptor::Unfiltered_Spec_StorageTexture),
            Get(Descriptor::DiffDirectionPdf_StorageTexture),
            Get(Descriptor::SpecDirectionPdf_StorageTexture),
            Get(Descriptor::Downsampled_ViewZ_StorageTexture),
            Get(Descriptor::Downsampled_Motion_StorageTexture),
            Get(Descriptor::Downsampled_Normal_Roughness_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Composition1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::Composition), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::ViewZ_Texture),
            Get(Descriptor::Downsampled_ViewZ_Texture),
            Get(Descriptor::Normal_Roughness_Texture),
            Get(Descriptor::BaseColor_Metalness_Texture),
            Get(Descriptor::DirectLighting_Texture),
            Get(Descriptor::Ambient_Texture),
            Get(Descriptor::IntegrateBRDF_Texture),
            Get(Descriptor::Diff_Texture),
            Get(Descriptor::Spec_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::ComposedLighting_ViewZ_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Temporal1a
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::Temporal), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Motion_Texture),
            Get(Descriptor::ComposedLighting_ViewZ_Texture),
            Get(Descriptor::TransparentLighting_Texture),
            Get(Descriptor::TaaHistoryPrev_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::TaaHistory_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Temporal1b
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::Temporal), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Motion_Texture),
            Get(Descriptor::ComposedLighting_ViewZ_Texture),
            Get(Descriptor::TransparentLighting_Texture),
            Get(Descriptor::TaaHistory_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::TaaHistoryPrev_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Upsample1a
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::Upsample), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistory_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Upsample1b
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::Upsample), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistoryPrev_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::PreDlss1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::PreDlss), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Motion_Texture),
            Get(Descriptor::TransparentLighting_Texture),
            Get(Descriptor::ComposedLighting_ViewZ_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Unfiltered_ShadowData_StorageTexture),
            Get(Descriptor::Unfiltered_Diff_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::AfterDlss1
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::AfterDlss), 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::DlssOutput_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::RayTracing2
        std::vector<nri::Descriptor*> rtTextures(m_Scene.materials.size() * TEXTURES_PER_MATERIAL);
        for (size_t i = 0; i < m_Scene.materials.size(); i++)
        {
            const size_t index = i * TEXTURES_PER_MATERIAL;
            const utils::Material& material = m_Scene.materials[i];

            rtTextures[index] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.diffuseMapIndex) );
            rtTextures[index + 1] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.specularMapIndex) );
            rtTextures[index + 2] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.normalMapIndex) );
            rtTextures[index + 3] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.emissiveMapIndex) );
        }

        const nri::Descriptor* rtBuffers[] =
        {
            Get(Descriptor::InstanceData_Buffer),
            Get(Descriptor::PrimitiveData_Buffer)
        };

        const nri::Descriptor* rtAccelerationStructures[] =
        {
            Get(Descriptor::World_AccelerationStructure),
            Get(Descriptor::Light_AccelerationStructure)
        };

        const nri::DescriptorRangeUpdateDesc rtDescriptorRangeUpdateDesc[] =
        {
            { rtAccelerationStructures, helper::GetCountOf(rtAccelerationStructures) },
            { rtBuffers, helper::GetCountOf(rtBuffers) },
            { rtTextures.data(), helper::GetCountOf(rtTextures) }
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *GetPipelineLayout(Pipeline::AmbientRays), 2, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, helper::GetCountOf(rtTextures)));
        m_DescriptorSets.push_back(descriptorSet);

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(rtDescriptorRangeUpdateDesc), rtDescriptorRangeUpdateDesc);
    }
}

void Sample::UploadStaticData()
{
    // PrimitiveData
    std::vector<PrimitiveData> primitiveData( m_Scene.primitives.size() );
    uint32_t n = 0;
    for (const utils::Mesh& mesh : m_Scene.meshes)
    {
        uint32_t triangleNum = mesh.indexNum / 3;
        for (uint32_t j = 0; j < triangleNum; j++)
        {
            uint32_t primitiveIndex = mesh.indexOffset / 3 + j;
            const utils::Primitive& primitive = m_Scene.primitives[primitiveIndex];

            const utils::UnpackedVertex& v0 = m_Scene.unpackedVertices[ mesh.vertexOffset + m_Scene.indices[primitiveIndex * 3] ];
            const utils::UnpackedVertex& v1 = m_Scene.unpackedVertices[ mesh.vertexOffset + m_Scene.indices[primitiveIndex * 3 + 1] ];
            const utils::UnpackedVertex& v2 = m_Scene.unpackedVertices[ mesh.vertexOffset + m_Scene.indices[primitiveIndex * 3 + 2] ];

            float2 n0 = Packed::EncodeUnitVector( float3(v0.normal), true );
            float2 n1 = Packed::EncodeUnitVector( float3(v1.normal), true );
            float2 n2 = Packed::EncodeUnitVector( float3(v2.normal), true );

            float2 t0 = Packed::EncodeUnitVector( float4(v0.tangent).xmm, true );
            float2 t1 = Packed::EncodeUnitVector( float4(v1.tangent).xmm, true );
            float2 t2 = Packed::EncodeUnitVector( float4(v2.tangent).xmm, true );

            PrimitiveData& data = primitiveData[n++];
            data.uv0 = Packed::sf2_to_h2(v0.uv[0], v0.uv[1]);
            data.uv1 = Packed::sf2_to_h2(v1.uv[0], v1.uv[1]);
            data.uv2 = Packed::sf2_to_h2(v2.uv[0], v2.uv[1]);

            data.n0oct = Packed::sf2_to_h2(n0.x, n0.y);
            data.n1oct = Packed::sf2_to_h2(n1.x, n1.y);
            data.n2oct = Packed::sf2_to_h2(n2.x, n2.y);

            data.t0oct = Packed::sf2_to_h2(t0.x, t0.y);
            data.t1oct = Packed::sf2_to_h2(t1.x, t1.y);
            data.t2oct = Packed::sf2_to_h2(t2.x, t2.y);

            data.b0s_b1s = Packed::sf2_to_h2(v0.tangent[3], v1.tangent[3]);
            data.b2s_worldToUvUnits = Packed::sf2_to_h2(v2.tangent[3], primitive.worldToUvUnits);
            data.padding = 0;
        }
    }

    // MaterialTextures
    uint32_t subresourceNum = 0;
    for (const utils::Texture* texture : m_Scene.textures)
        subresourceNum += texture->GetArraySize() * texture->GetMipNum();

    std::vector<nri::TextureUploadDesc> textureData( m_Scene.textures.size() );
    std::vector<nri::TextureSubresourceUploadDesc> subresources( subresourceNum );
    uint32_t subresourceOffset = 0;

    nri::TextureUploadDesc* textureDataDesc = textureData.data();
    for (const utils::Texture* texture : m_Scene.textures)
    {
        for (uint32_t layer = 0; layer < texture->GetArraySize(); layer++)
            for (uint32_t mip = 0; mip < texture->GetMipNum(); mip++)
                texture->GetSubresource(subresources[subresourceOffset + layer * texture->GetMipNum() + mip], mip, layer);

        textureDataDesc->subresources = &subresources[subresourceOffset];
        textureDataDesc->mipNum = texture->GetMipNum();
        textureDataDesc->arraySize = texture->GetArraySize();
        textureDataDesc->texture = Get( (Texture)((uint32_t)Texture::MaterialTextures + textureDataDesc - textureData.data()) );
        textureDataDesc->nextLayout = nri::TextureLayout::SHADER_RESOURCE;
        textureDataDesc->nextAccess = nri::AccessBits::SHADER_RESOURCE;
        textureDataDesc++;

        subresourceOffset += texture->GetArraySize() * texture->GetMipNum();
    }

    for (const nri::TextureTransitionBarrierDesc& state : m_TextureStates)
    {
        nri::TextureUploadDesc desc = {};
        desc.nextAccess = state.nextAccess;
        desc.nextLayout = state.nextLayout;
        desc.texture = (nri::Texture*)state.texture;

        textureData.push_back(desc);
    }

    // Buffer data
    nri::BufferUploadDesc dataDescArray[] =
    {
        { primitiveData.data(), helper::GetByteSizeOf(primitiveData), Get(Buffer::PrimitiveData), 0, nri::AccessBits::SHADER_RESOURCE },
    };

    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_CommandQueue, textureData.data(), helper::GetCountOf(textureData), dataDescArray, helper::GetCountOf(dataDescArray)));
}

void Sample::CreateBottomLevelAccelerationStructures()
{
    for (const utils::Mesh& mesh : m_Scene.meshes)
    {
        const uint64_t vertexDataSize = mesh.vertexNum * sizeof(utils::Vertex);
        const uint64_t indexDataSize = mesh.indexNum * sizeof(utils::Index);

        nri::Buffer* tempBuffer = nullptr;
        nri::Memory* tempMemory = nullptr;
        CreateUploadBuffer(vertexDataSize + indexDataSize, tempBuffer, tempMemory);

        uint8_t* data = (uint8_t*)NRI.MapBuffer(*tempBuffer, 0, nri::WHOLE_SIZE);
        memcpy(data, &m_Scene.vertices[mesh.vertexOffset], (size_t)vertexDataSize);
        memcpy(data + vertexDataSize, &m_Scene.indices[mesh.indexOffset], (size_t)indexDataSize);
        NRI.UnmapBuffer(*tempBuffer);

        nri::GeometryObject geometryObject = {};
        geometryObject.type = nri::GeometryType::TRIANGLES;
        geometryObject.flags = nri::BottomLevelGeometryBits::NONE;
        geometryObject.triangles.vertexBuffer = tempBuffer;
        geometryObject.triangles.vertexOffset = 0;
        geometryObject.triangles.vertexNum = mesh.vertexNum;
        geometryObject.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
        geometryObject.triangles.vertexStride = sizeof(utils::Vertex);
        geometryObject.triangles.indexBuffer = tempBuffer;
        geometryObject.triangles.indexOffset = vertexDataSize;
        geometryObject.triangles.indexNum = mesh.indexNum;
        geometryObject.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;

        nri::AccelerationStructureDesc blasDesc = {};
        blasDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
        blasDesc.flags = BOTTOM_LEVEL_BUILD_FLAGS;
        blasDesc.instanceOrGeometryObjectNum = 1;
        blasDesc.geometryObjects = &geometryObject;

        nri::AccelerationStructure* blas = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, blasDesc, blas));
        m_BLASs.push_back(blas);

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*blas, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = { memory, blas };
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        BuildBottomLevelAccelerationStructure(*blas, &geometryObject, 1);

        NRI.DestroyBuffer(*tempBuffer);
        NRI.FreeMemory(*tempMemory);
    }
}

void Sample::CreateTopLevelAccelerationStructure()
{
    nri::AccelerationStructureDesc tlasDesc = {};
    tlasDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
    tlasDesc.flags = TOP_LEVEL_BUILD_FLAGS;
    tlasDesc.instanceOrGeometryObjectNum = helper::GetCountOf(m_Scene.instances) + ANIMATED_INSTANCE_MAX_NUM;

    // Descriptor::World_AccelerationStructure
    {
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, tlasDesc, m_WorldTlas));

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*m_WorldTlas, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = { memory, m_WorldTlas };
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        nri::Descriptor* descriptor;
        NRI.CreateAccelerationStructureDescriptor(*m_WorldTlas, 0, descriptor);
        m_Descriptors.push_back(descriptor);
    }

    // Descriptor::Light_AccelerationStructure
    {
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, tlasDesc, m_LightTlas));

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*m_LightTlas, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = { memory, m_LightTlas };
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        nri::Descriptor* descriptor;
        NRI.CreateAccelerationStructureDescriptor(*m_LightTlas, 0, descriptor);
        m_Descriptors.push_back(descriptor);
    }
}

void Sample::CreateUploadBuffer(uint64_t size, nri::Buffer*& buffer, nri::Memory*& memory)
{
    const nri::BufferDesc bufferDesc = { size, 0, nri::BufferUsageBits::RAY_TRACING_BUFFER };
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryInfo(*buffer, nri::MemoryLocation::HOST_UPLOAD, memoryDesc);

    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = { memory, buffer };
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory)
{
    const uint64_t scratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(accelerationStructure);

    const nri::BufferDesc bufferDesc = { scratchBufferSize, 0, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE };
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryInfo(*buffer, nri::MemoryLocation::DEVICE, memoryDesc);

    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = { memory, buffer };
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::GeometryObject* objects, const uint32_t objectNum)
{
    nri::Buffer* scratchBuffer = nullptr;
    nri::Memory* scratchBufferMemory = nullptr;
    CreateScratchBuffer(accelerationStructure, scratchBuffer, scratchBufferMemory);

    nri::CommandAllocator* commandAllocator = nullptr;
    NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, commandAllocator);

    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    NRI.BeginCommandBuffer(*commandBuffer, nullptr, 0);
    {
        NRI.CmdBuildBottomLevelAccelerationStructure(*commandBuffer, objectNum, objects, BOTTOM_LEVEL_BUILD_FLAGS, accelerationStructure, *scratchBuffer, 0);
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBuffers = &commandBuffer;
    workSubmissionDesc.commandBufferNum = 1;
    NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, nullptr);

    NRI.WaitForIdle(*m_CommandQueue);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);
    NRI.DestroyBuffer(*scratchBuffer);
    NRI.FreeMemory(*scratchBufferMemory);
}

void Sample::BuildTopLevelAccelerationStructure(nri::CommandBuffer& commandBuffer, uint32_t bufferedFrameIndex)
{
    bool isAnimatedObjects = m_Settings.animatedObjects;
    if (m_Settings.blink)
    {
        double period = 0.0003 * m_Timer.GetTimeStamp() * (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
        isAnimatedObjects &= WaveTriangle(period) > 0.5;
    }

    const uint64_t tlasCount = m_Scene.instances.size() - m_DefaultInstancesOffset;
    const uint64_t tlasDataSize = tlasCount * sizeof(nri::GeometryObjectInstance);
    const uint64_t tlasDataOffset = tlasDataSize * bufferedFrameIndex;
    const uint64_t instanceDataSize = tlasCount * sizeof(InstanceData);
    const uint64_t instanceDataOffset = instanceDataSize * bufferedFrameIndex;
    const uint64_t instanceCount = m_Scene.instances.size() - (m_AnimatedInstances.size() - m_Settings.animatedObjectNum * isAnimatedObjects);
    const uint64_t staticInstanceCount = m_Scene.instances.size() - m_AnimatedInstances.size();

    auto instanceData = (InstanceData*)NRI.MapBuffer(*Get(Buffer::InstanceDataStaging), instanceDataOffset, instanceDataSize);
    auto worldTlasData = (nri::GeometryObjectInstance*)NRI.MapBuffer(*Get(Buffer::WorldTlasDataStaging), tlasDataOffset, tlasDataSize);
    auto lightTlasData = (nri::GeometryObjectInstance*)NRI.MapBuffer(*Get(Buffer::LightTlasDataStaging), tlasDataOffset, tlasDataSize);

    Rand::Seed(105361, &m_FastRandState);

    uint32_t worldInstanceNum = 0;
    uint32_t lightInstanceNum = 0;
    m_HasTransparentObjects = false;
    for (size_t i = m_DefaultInstancesOffset; i < instanceCount; i++)
    {
        utils::Instance& instance = m_Scene.instances[i];
        const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];

        if (material.IsOff()) // TODO: not an elegant way to skip "bad objects" (alpha channel is set to 0)
            continue;

        assert( worldInstanceNum <= INSTANCE_ID_MASK );

        float4x4 mObjectToWorld = instance.rotation;
        mObjectToWorld.AddTranslation( m_Camera.GetRelative( instance.position ) );

        float4x4 mObjectToWorldPrev = instance.rotationPrev;
        mObjectToWorldPrev.AddTranslation( m_Camera.GetRelative( instance.positionPrev ) );

        // Use fp64 to avoid imprecision problems on close up views (InvertOrtho can't be used due to scaling factors)
        double4x4 mWorldToObjectd = ToDouble( mObjectToWorld );
        mWorldToObjectd.Invert();
        float4x4 mWorldToObject = ToFloat( mWorldToObjectd );

        float4x4 mWorldToWorldPrev = mObjectToWorldPrev * mWorldToObject;
        mWorldToWorldPrev.Transpose3x4();

        instance.positionPrev = instance.position;
        instance.rotationPrev = instance.rotation;

        mObjectToWorld.Transpose3x4();

        uint32_t flags = 0;
        if (material.IsEmissive()) // TODO: importance sampling can be significantly accelerated if ALL emissives will be placed into a single BLAS, which will be the only one in a special TLAS!
            flags = m_Settings.emission ? FLAG_EMISSION : FLAG_OPAQUE_OR_ALPHA_OPAQUE;
        else if (m_Settings.emissiveObjects && i > staticInstanceCount && Rand::uf1(&m_FastRandState) > 0.66f)
            flags = m_Settings.emission ? FLAG_FORCED_EMISSION : FLAG_OPAQUE_OR_ALPHA_OPAQUE;
        else if (material.IsTransparent())
        {
            flags = FLAG_TRANSPARENT;
            m_HasTransparentObjects = true;
        }
        else
            flags = FLAG_OPAQUE_OR_ALPHA_OPAQUE;

        uint32_t basePrimitiveIndex = mesh.indexOffset / 3;
        uint32_t instanceIdAndFlags = worldInstanceNum | (flags << FLAG_FIRST_BIT);

        uint32_t packedMaterial = Packed::uf4_to_uint<7, 7, 7, 0>(material.avgBaseColor);
        packedMaterial |= Packed::uf4_to_uint<11, 10, 6, 5>( float4(0.0f, 0.0f, material.avgSpecularColor.y, material.avgSpecularColor.z) );

        instanceData->basePrimitiveIndex = basePrimitiveIndex;
        instanceData->baseTextureIndex = instance.materialIndex * TEXTURES_PER_MATERIAL;
        instanceData->averageBaseColor = packedMaterial;
        instanceData->mWorldToWorldPrev0 = mWorldToWorldPrev.col0;
        instanceData->mWorldToWorldPrev1 = mWorldToWorldPrev.col1;
        instanceData->mWorldToWorldPrev2 = mWorldToWorldPrev.col2;
        instanceData++;

        nri::GeometryObjectInstance tlasInstance = {};
        memcpy(tlasInstance.transform, mObjectToWorld.a16, sizeof(tlasInstance.transform));
        tlasInstance.instanceId = instanceIdAndFlags;
        tlasInstance.mask = flags;
        tlasInstance.shaderBindingTableLocalOffset = 0;
        tlasInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE | ((material.IsOpaque() || material.IsTransparent()) ? nri::TopLevelInstanceBits::FORCE_OPAQUE : nri::TopLevelInstanceBits::NONE);
        tlasInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*m_BLASs[instance.meshIndex], 0);

        if (flags & (FLAG_EMISSION | FLAG_FORCED_EMISSION))
        {
            *lightTlasData++ = tlasInstance;
            lightInstanceNum++;
        }

        *worldTlasData++ = tlasInstance;
        worldInstanceNum++;
    }

    NRI.UnmapBuffer(*Get(Buffer::InstanceDataStaging));
    NRI.UnmapBuffer(*Get(Buffer::WorldTlasDataStaging));
    NRI.UnmapBuffer(*Get(Buffer::LightTlasDataStaging));

    const nri::BufferTransitionBarrierDesc transitions[] =
    {
        { Get(Buffer::InstanceData), nri::AccessBits::SHADER_RESOURCE,  nri::AccessBits::COPY_DESTINATION },
    };

    nri::TransitionBarrierDesc transitionBarriers = {};
    transitionBarriers.buffers = transitions;
    transitionBarriers.bufferNum = helper::GetCountOf(transitions);
    NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

    NRI.CmdCopyBuffer(commandBuffer, *Get(Buffer::InstanceData), 0, 0, *Get(Buffer::InstanceDataStaging), 0, instanceDataOffset, instanceDataSize);
    NRI.CmdBuildTopLevelAccelerationStructure(commandBuffer, worldInstanceNum, *Get(Buffer::WorldTlasDataStaging), tlasDataOffset, TOP_LEVEL_BUILD_FLAGS, *m_WorldTlas, *Get(Buffer::WorldScratch), 0);
    NRI.CmdBuildTopLevelAccelerationStructure(commandBuffer, lightInstanceNum, *Get(Buffer::LightTlasDataStaging), tlasDataOffset, TOP_LEVEL_BUILD_FLAGS, *m_LightTlas, *Get(Buffer::LightScratch), 0);
}

void Sample::UpdateConstantBuffer(uint32_t frameIndex, float globalResetFactor)
{
    // Sun animation
    if (m_Settings.animateSun)
    {
        const float animationSpeed = m_Settings.pauseAnimation ? 0.0f : (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
        float period = float( animationSpeed * 0.0001 * m_Timer.GetTimeStamp() );
        m_Settings.sunElevation = WaveTriangle(period) * 30.0f;
    }

    // Ambient accumulation
    const float maxSeconds = 0.5f;
    float maxAccumFrameNum = maxSeconds * 1000.0f / m_Timer.GetSmoothedElapsedTime();
    m_AmbientAccumFrameNum = (m_AmbientAccumFrameNum + 1.0f) * globalResetFactor;
    m_AmbientAccumFrameNum = Min(m_AmbientAccumFrameNum, maxAccumFrameNum);

    const float3& sunDirection = GetSunDirection();
    float emissionIntensity = m_Settings.emissionIntensity * float(m_Settings.emission);

    uint32_t rectW = uint32_t(m_ScreenResolution.x * m_ResolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_ScreenResolution.y * m_ResolutionScale + 0.5f);

    float2 outputSize = float2( float(m_OutputResolution.x), float(m_OutputResolution.y) );
    float2 screenSize = float2( float(m_ScreenResolution.x), float(m_ScreenResolution.y) );
    float2 rectSize = float2( float(rectW), float(rectH) );
    float2 jitter = (m_Settings.TAA ? m_Camera.state.viewportJitter : 0.0f) / rectSize;
    float baseMipBias = -0.5f + log2f(m_ResolutionScale);

    float3 viewDir = float3(m_Camera.state.mViewToWorld.GetCol2().xmm) * (CAMERA_LEFT_HANDED ? -1.0f : 1.0f);

    nrd::HitDistanceParameters diffHitDistanceParameters = {};
    diffHitDistanceParameters.A = m_Settings.diffHitDistScale * m_Settings.meterToUnitsMultiplier;

    nrd::HitDistanceParameters specHitDistanceParameters = {};
    specHitDistanceParameters.A = m_Settings.specHitDistScale * m_Settings.meterToUnitsMultiplier;

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const uint64_t rangeOffset = m_Frames[bufferedFrameIndex].globalConstantBufferOffset;
    nri::Buffer* globalConstants = Get(Buffer::GlobalConstants);
    auto data = (GlobalConstantBufferData*)NRI.MapBuffer(*globalConstants, rangeOffset, sizeof(GlobalConstantBufferData));
    {
        data->gWorldToView = m_Camera.state.mWorldToView;
        data->gViewToWorld = m_Camera.state.mViewToWorld;
        data->gViewToClip = m_Camera.state.mViewToClip;
        data->gWorldToClipPrev = m_Camera.statePrev.mWorldToClip;
        data->gWorldToClip = m_Camera.state.mWorldToClip;
        data->gDiffHitDistParams = float4( diffHitDistanceParameters.A, diffHitDistanceParameters.B, diffHitDistanceParameters.C, diffHitDistanceParameters.D );
        data->gSpecHitDistParams = float4( specHitDistanceParameters.A, specHitDistanceParameters.B, specHitDistanceParameters.C, specHitDistanceParameters.D );
        data->gCameraFrustum = m_Camera.state.frustum;
        data->gSunDirection_gExposure = sunDirection;
        data->gSunDirection_gExposure.w = m_Settings.exposure;
        data->gCameraOrigin_gMipBias = m_Camera.state.position;
        data->gCameraOrigin_gMipBias.w = m_DLSS.IsInitialized() ? (baseMipBias + log2f(float(m_ScreenResolution.x) / float(m_OutputResolution.x))) : (m_Settings.TAA ? baseMipBias : 0.0f);
        data->gTrimmingParams_gEmissionIntensity = GetTrimmingParams();
        data->gTrimmingParams_gEmissionIntensity.w = emissionIntensity;
        data->gViewDirection_gIsOrtho = float4( viewDir.x, viewDir.y, viewDir.z, m_Camera.m_IsOrtho );
        data->gOutputSize = outputSize;
        data->gInvOutputSize = float2(1.0f, 1.0f) / outputSize;
        data->gScreenSize = screenSize;
        data->gInvScreenSize = float2(1.0f, 1.0f) / screenSize;
        data->gRectSize = rectSize;
        data->gInvRectSize = float2(1.0f, 1.0f) / rectSize;
        data->gRectSizePrev = m_RectSizePrev;
        data->gJitter = jitter;
        data->gNearZ = (CAMERA_LEFT_HANDED ? 1.0f : -1.0f) * NEAR_Z * m_Settings.meterToUnitsMultiplier;
        data->gAmbientAccumSpeed = 1.0f / (1.0f + m_AmbientAccumFrameNum);
        data->gAmbient = m_Settings.ambient ? 1.0f : 0.0f;
        data->gAmbientInComposition = (m_Settings.ambient && m_Settings.denoiser != RELAX) ? 1.0f : 0.0f;
        data->gSeparator = m_Settings.separator;
        data->gRoughnessOverride = m_Settings.roughnessOverride;
        data->gMetalnessOverride = m_Settings.metalnessOverride;
        data->gUnitToMetersMultiplier = 1.0f / m_Settings.meterToUnitsMultiplier;
        data->gIndirectDiffuse = m_Settings.indirectDiffuse ? 1.0f : 0.0f;
        data->gIndirectSpecular = m_Settings.indirectSpecular ? 1.0f : 0.0f;
        data->gSunAngularRadius = DegToRad( m_Settings.sunAngularDiameter * 0.5f );
        data->gTanSunAngularRadius = Tan( DegToRad( m_Settings.sunAngularDiameter * 0.5f ) );
        data->gTanPixelAngularRadius = Tan( 0.5f * DegToRad(m_Settings.camFov) / m_OutputResolution.x );
        data->gDebug = m_Settings.debug;
        data->gTransparent = m_HasTransparentObjects ? 1.0f : 0.0f;
        data->gReference = m_Settings.reference ? 1.0f : 0.0f;
        data->gDenoiserType = (uint32_t)m_Settings.denoiser;
        data->gDisableShadowsAndEnableImportanceSampling = (sunDirection.z < 0.0f && m_Settings.importanceSampling) ? 1 : 0;
        data->gOnScreen = m_Settings.onScreen + NRD_OCCLUSION_ONLY * 3; // preserve original mapping
        data->gFrameIndex = frameIndex;
        data->gForcedMaterial = m_Settings.forcedMaterial;
        data->gUseNormalMap = m_Settings.normalMap ? 1 : 0;
        data->gWorldSpaceMotion = m_Settings.isMotionVectorInWorldSpace ? 1 : 0;
        data->gTracingMode = m_Settings.reference ? 0 : m_Settings.tracingMode;
        data->gSampleNum = m_Settings.rpp;
        data->gBounceNum = m_Settings.bounceNum;
        data->gOcclusionOnly = NRD_OCCLUSION_ONLY;
    }
    NRI.UnmapBuffer(*globalConstants);

    m_RectSizePrev = rectSize;
}

void Sample::LoadScene()
{
    std::string sceneFile = utils::GetFullPath("Cubes/Cubes.obj", utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE( utils::LoadScene(sceneFile, m_Scene, false) );
    m_DefaultInstancesOffset = helper::GetCountOf(m_Scene.meshes);

    sceneFile = utils::GetFullPath(m_SceneFile, utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE( utils::LoadScene(sceneFile, m_Scene, false) );

    if (m_SceneFile.find("BistroInterior") != std::string::npos)
    {
        m_Settings.exposure = 80.0f;
        m_Settings.emissionIntensity = 1.0f;
        m_Settings.emission = true;
        m_Settings.animatedObjectScale = 0.5f;
        m_Settings.sunElevation = 7.0f;
    }
    else if (m_SceneFile.find("BistroExterior") != std::string::npos)
    {
        m_Settings.exposure = 50.0f;
        m_Settings.emissionIntensity = 1.0f;
        m_Settings.emission = true;
    }
    else if (m_SceneFile.find("ShaderBalls") != std::string::npos)
        m_Settings.exposure = 1.7f;
    else if (m_SceneFile.find("ZeroDay") != std::string::npos)
    {
        m_Settings.exposure = 25.0f;
        m_Settings.emissionIntensity = 2.3f;
        m_Settings.emission = true;
        m_Settings.roughnessOverride = 0.07f;
        m_Settings.metalnessOverride = 0.25f;
        m_Settings.camFov = 75.0f;
        m_Settings.animationSpeed = -0.6f;
        m_Settings.sunElevation = -90.0f;
        m_Settings.sunAngularDiameter = 0.0f;
    }
}

uint32_t Sample::BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, nri::TextureTransitionBarrierDesc* transitions, uint32_t transitionMaxNum)
{
    uint32_t n = 0;

    for (uint32_t i = 0; i < stateNum; i++)
    {
        const TextureState& state = states[i];
        nri::TextureTransitionBarrierDesc& transition = GetState(state.texture);

        bool isStateChanged = transition.nextAccess != state.nextAccess || transition.nextLayout != state.nextLayout;
        bool isStorageBarrier = transition.nextAccess == nri::AccessBits::SHADER_RESOURCE_STORAGE && state.nextAccess == nri::AccessBits::SHADER_RESOURCE_STORAGE;
        if (isStateChanged || isStorageBarrier)
        {
            assert( n < transitionMaxNum );
            transitions[n++] = nri::TextureTransition(transition, state.nextAccess, state.nextLayout);
        }
    }

    return n;
}

void Sample::RenderFrame(uint32_t frameIndex)
{
    std::array<nri::TextureTransitionBarrierDesc, 32> optimizedTransitions = {};

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];
    const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain, *m_BackBufferAcquireSemaphore);
    const BackBuffer* backBuffer = &m_SwapChainBuffers[backBufferIndex];
    const bool isEven = !(frameIndex & 0x1);
    nri::TransitionBarrierDesc transitionBarriers = {};
    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;

    NRI.WaitForSemaphore(*m_CommandQueue, *frame.deviceSemaphore);
    NRI.ResetCommandAllocator(*frame.commandAllocator);

    // Global history reset
    float sunCurr = Smoothstep( -0.9f, 0.05f, Sin( DegToRad(m_Settings.sunElevation) ) );
    float sunPrev = Smoothstep( -0.9f, 0.05f, Sin( DegToRad(m_PrevSettings.sunElevation) ) );
    float resetHistoryFactor = 1.0f - Smoothstep( 0.0f, 0.2f, Abs(sunCurr - sunPrev) );

    if (m_PrevSettings.denoiser != m_Settings.denoiser)
        resetHistoryFactor = 0.0f;
    if (m_PrevSettings.ortho != m_Settings.ortho)
        resetHistoryFactor = 0.0f;
    if (m_PrevSettings.nrdSettings.referenceAccumulation != m_Settings.nrdSettings.referenceAccumulation)
        resetHistoryFactor = 0.0f;
    if (m_PrevSettings.onScreen != m_Settings.onScreen)
        resetHistoryFactor = 0.0f;
    if (m_PrevSettings.reference != m_Settings.reference)
        resetHistoryFactor = 0.0f;
    if (m_ForceHistoryReset || frameIndex == 0)
        resetHistoryFactor = 0.0f;

    // Sizes
    uint32_t rectW = uint32_t(m_ScreenResolution.x * m_ResolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_ScreenResolution.y * m_ResolutionScale + 0.5f);
    uint32_t rectGridW = (rectW + 15) / 16;
    uint32_t rectGridH = (rectH + 15) / 16;
    uint32_t outputGridW = (m_OutputResolution.x + 15) / 16;
    uint32_t outputGridH = (m_OutputResolution.y + 15) / 16;
    uint32_t screenGridW = (m_ScreenResolution.x + 15) / 16;
    uint32_t screenGridH = (m_ScreenResolution.y + 15) / 16;

    // NRD settings
    if (m_Settings.adaptiveAccumulation)
    {
        float maxAccumFrameNum = ACCUMULATION_PERIOD_IN_SECONDS * 1000.0f / m_Timer.GetSmoothedElapsedTime();
        m_Settings.nrdSettings.maxAccumulatedFrameNum = int32_t(maxAccumFrameNum + 0.5f);
        m_Settings.nrdSettings.maxAccumulatedFrameNum = Min(m_Settings.nrdSettings.maxAccumulatedFrameNum, int32_t(nrd::REBLUR_MAX_HISTORY_FRAME_NUM));
    }

    uint32_t maxAccumulatedFrameNum = uint32_t(m_Settings.nrdSettings.maxAccumulatedFrameNum * resetHistoryFactor + 0.5f);
    uint32_t maxFastAccumulatedFrameNum = uint32_t(m_Settings.nrdSettings.maxFastAccumulatedFrameNum * resetHistoryFactor + 0.5f);
    float2 jitter = m_Settings.TAA ? m_Camera.state.viewportJitter : 0.0f;
    int32_t tracingMode = m_Settings.reference ? 0 : m_Settings.tracingMode;
    float resolutionScaleQuarter = m_ResolutionScale * (tracingMode == RESOLUTION_QUARTER ? 0.5f : 1.0f);

    nrd::CommonSettings commonSettings = {};
    memcpy(commonSettings.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));
    memcpy(commonSettings.viewToClipMatrixPrev, &m_Camera.statePrev.mViewToClip, sizeof(m_Camera.statePrev.mViewToClip));
    memcpy(commonSettings.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
    memcpy(commonSettings.worldToViewMatrixPrev, &m_Camera.statePrev.mWorldToView, sizeof(m_Camera.statePrev.mWorldToView));
    commonSettings.motionVectorScale[0] = m_Settings.isMotionVectorInWorldSpace ? 1.0f : 1.0f / float(rectW);
    commonSettings.motionVectorScale[1] = m_Settings.isMotionVectorInWorldSpace ? 1.0f : 1.0f / float(rectH);
    commonSettings.cameraJitter[0] = jitter.x;
    commonSettings.cameraJitter[1] = jitter.y;
    commonSettings.resolutionScale[0] = m_ResolutionScale;
    commonSettings.resolutionScale[1] = m_ResolutionScale;
    commonSettings.denoisingRange = 4.0f * m_Scene.aabb.GetRadius();
    commonSettings.disocclusionThreshold = m_Settings.nrdSettings.disocclusionThreshold * 0.01f;
    commonSettings.splitScreen = m_Settings.reference ? 1.0f : m_Settings.separator;
    commonSettings.debug = m_Settings.debug;
    commonSettings.frameIndex = frameIndex;
    commonSettings.accumulationMode = resetHistoryFactor == 0.0f ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.isMotionVectorInWorldSpace = m_Settings.isMotionVectorInWorldSpace;

    // NRD user pool
    NrdUserPool userPool =
    {{
        // IN_MV
        {&GetState(Texture::Motion), GetFormat(Texture::Motion)},

        // IN_NORMAL_ROUGHNESS
        {&GetState(Texture::Normal_Roughness), GetFormat(Texture::Normal_Roughness)},

        // IN_VIEWZ
        {&GetState(Texture::ViewZ), GetFormat(Texture::ViewZ)},

        // IN_DIFF_RADIANCE_HITDIST
        {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)},

        // IN_SPEC_RADIANCE_HITDIST
        {&GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec)},

        // IN_DIFF_HITDIST
        {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)}, // needed for NRD_OCCLUSION_ONLY

        // IN_SPEC_HITDIST
        {&GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec)}, // needed for NRD_OCCLUSION_ONLY

        // IN_DIFF_DIRECTION_HITDIST
        {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)}, // not used, needed only for DIFFUSE_DIRECTIONAL_OCCLUSION denoiser validation

        // IN_DIFF_DIRECTION_PDF
        {&GetState(Texture::DiffDirectionPdf), GetFormat(Texture::DiffDirectionPdf)},

        // IN_SPEC_DIRECTION_PDF
        {&GetState(Texture::SpecDirectionPdf), GetFormat(Texture::SpecDirectionPdf)},

        // IN_DIFF_CONFIDENCE
        {nullptr, nri::Format::UNKNOWN}, // yes, unnecessary inputs / outputs can be set to 0

        // IN_SPEC_CONFIDENCE
        {nullptr, nri::Format::UNKNOWN}, // yes, unnecessary inputs / outputs can be set to 0

        // IN_SHADOWDATA
        {&GetState(Texture::Unfiltered_ShadowData), GetFormat(Texture::Unfiltered_ShadowData)},

        // IN_SHADOW_TRANSLUCENCY
        {&GetState(Texture::Unfiltered_Shadow_Translucency), GetFormat(Texture::Unfiltered_Shadow_Translucency)},

        // OUT_SHADOW_TRANSLUCENCY
        {&GetState(Texture::Shadow), GetFormat(Texture::Shadow)},

        // OUT_DIFF_RADIANCE_HITDIST
        {&GetState(Texture::Diff), GetFormat(Texture::Diff)},

        // OUT_SPEC_RADIANCE_HITDIST
        {&GetState(Texture::Spec), GetFormat(Texture::Spec)},

        // OUT_DIFF_HITDIST
        {&GetState(Texture::Diff), GetFormat(Texture::Diff)}, // needed for NRD_OCCLUSION_ONLY

        // OUT_SPEC_HITDIST
        {&GetState(Texture::Spec), GetFormat(Texture::Spec)}, // needed for NRD_OCCLUSION_ONLY

        // OUT_DIFF_DIRECTION_HITDIST
        {&GetState(Texture::Diff), GetFormat(Texture::Diff)}, // not used, needed only for DIFFUSE_DIRECTIONAL_OCCLUSION denoiser validation
    }};

    UpdateConstantBuffer(frameIndex, resetHistoryFactor);

    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool, 0);
    {
        // Preintegrate F (for specular) and G (for diffuse) terms (only once)
        if (frameIndex == 0)
        {
            NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::IntegrateBRDF));
            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::IntegrateBRDF));
            NRI.CmdSetDescriptorSets(commandBuffer, 0, 1, &Get(DescriptorSet::IntegrateBRDF0), nullptr);

            const uint32_t gridWidth = (FG_TEX_SIZE + 15) / 16;
            const uint32_t gridHeight = (FG_TEX_SIZE + 15) / 16;
            NRI.CmdDispatch(commandBuffer, gridWidth, gridHeight, 1);

            const nri::TextureTransitionBarrierDesc transitions[] =
            {
                nri::TextureTransition(GetState(Texture::IntegrateBRDF), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE),
            };
            transitionBarriers.textures = transitions;
            transitionBarriers.textureNum = helper::GetCountOf(transitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
        }

        { // TLAS
            helper::Annotation annotation(NRI, commandBuffer, "TLAS");

            BuildTopLevelAccelerationStructure(commandBuffer, bufferedFrameIndex);
        }

        { // Ambient rays
            helper::Annotation annotation(NRI, commandBuffer, "Ambient rays");

            const nri::BufferTransitionBarrierDesc bufferTransitions[] =
            {
                {Get(Buffer::InstanceData), nri::AccessBits::COPY_DESTINATION,  nri::AccessBits::SHADER_RESOURCE},
            };

            const TextureState transitions[] =
            {
                // Output
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
            transitionBarriers.buffers = bufferTransitions;
            transitionBarriers.bufferNum = helper::GetCountOf(bufferTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
            transitionBarriers.bufferNum = 0;

            NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::AmbientRays));
            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::AmbientRays));

            const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::AmbientRays1), Get(DescriptorSet::RayTracing2) };
            NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

            NRI.CmdDispatch(commandBuffer, 2, 2, 1);
        }

        { // Primary rays
            helper::Annotation annotation(NRI, commandBuffer, "Primary rays");

            const TextureState transitions[] =
            {
                // Input
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::Motion, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Shadow_Translucency, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::PrimaryRays));
            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::PrimaryRays));

            const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::PrimaryRays1), Get(DescriptorSet::RayTracing2) };
            NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        { // Shadow denoising
            helper::Annotation annotation(NRI, commandBuffer, "Shadow denoising");

            nrd::SigmaShadowSettings shadowSettings = {};

            m_Sigma.SetMethodSettings(nrd::Method::SIGMA_SHADOW_TRANSLUCENCY, &shadowSettings);
            m_Sigma.Denoise(frameIndex, commandBuffer, commonSettings, userPool);

            // NRD integration layer binds its own descriptor pool, we need to re-bind ours back
            NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
        }

        { // Direct lighting
            helper::Annotation annotation(NRI, commandBuffer, "Direct lighting");

            const TextureState transitions[] =
            {
                // Input
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Shadow, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::TransparentLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::DirectLighting));
            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DirectLighting));

            const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::DirectLighting1) };
            NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        { // Indirect rays
            helper::Annotation annotation(NRI, commandBuffer, "Indirect rays");

            const TextureState transitions[] =
            {
                // Input
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::ComposedLighting_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Motion, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::Unfiltered_Diff, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Spec, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DiffDirectionPdf, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::SpecDirectionPdf, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Downsampled_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Downsampled_Motion, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Downsampled_Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::IndirectRays));
            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::IndirectRays));

            const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::IndirectRays1), Get(DescriptorSet::RayTracing2) };
            NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

            uint32_t rectWmod = uint32_t(m_ScreenResolution.x * resolutionScaleQuarter + 0.5f);
            uint32_t rectHmod = uint32_t(m_ScreenResolution.y * resolutionScaleQuarter + 0.5f);
            uint32_t rectGridWmod = (rectWmod + 15) / 16;
            uint32_t rectGridHmod = (rectHmod + 15) / 16;

            NRI.CmdDispatch(commandBuffer, rectGridWmod, rectGridHmod, 1);
        }

        { // Diffuse & specular indirect lighting denoising
            helper::Annotation annotation(NRI, commandBuffer, "Indirect lighting denoising");

            if (tracingMode == RESOLUTION_QUARTER)
            {
                userPool[(size_t)nrd::ResourceType::IN_MV] = {&GetState(Texture::Downsampled_Motion), GetFormat(Texture::Downsampled_Motion)};
                userPool[(size_t)nrd::ResourceType::IN_NORMAL_ROUGHNESS] = {&GetState(Texture::Downsampled_Normal_Roughness), GetFormat(Texture::Downsampled_Normal_Roughness)};
                userPool[(size_t)nrd::ResourceType::IN_VIEWZ] = {&GetState(Texture::Downsampled_ViewZ), GetFormat(Texture::Downsampled_ViewZ)};

                commonSettings.resolutionScale[0] = resolutionScaleQuarter;
                commonSettings.resolutionScale[1] = resolutionScaleQuarter;
            }

            if (m_Settings.denoiser == REBLUR)
            {
                const float3 trimmingParams = GetTrimmingParams();

                nrd::AntilagIntensitySettings antilagIntensitySettings = {};
                nrd::AntilagHitDistanceSettings antilagHitDistanceSettings = {};
                GetAntilagSettings(antilagIntensitySettings, antilagHitDistanceSettings);

                nrd::HitDistanceParameters diffHitDistanceParameters = {};
                diffHitDistanceParameters.A = m_Settings.diffHitDistScale * m_Settings.meterToUnitsMultiplier;

                nrd::HitDistanceParameters specHitDistanceParameters = {};
                specHitDistanceParameters.A = m_Settings.specHitDistScale * m_Settings.meterToUnitsMultiplier;

                nrd::ReblurDiffuseSpecularSettings reblurSettings = {};

                reblurSettings.diffuse.hitDistanceParameters = diffHitDistanceParameters;
                reblurSettings.diffuse.antilagIntensitySettings = antilagIntensitySettings;
                reblurSettings.diffuse.antilagHitDistanceSettings = antilagHitDistanceSettings;
                reblurSettings.diffuse.materialMask = 1;
                reblurSettings.diffuse.maxAccumulatedFrameNum = maxAccumulatedFrameNum;
                reblurSettings.diffuse.blurRadius = m_Settings.nrdSettings.blurRadius;
                reblurSettings.diffuse.maxAdaptiveRadiusScale = m_Settings.nrdSettings.adaptiveRadiusScale;
                reblurSettings.diffuse.normalWeightStrictness = m_Settings.nrdSettings.normalWeightStrictness;
                reblurSettings.diffuse.stabilizationStrength = m_Settings.nrdSettings.stabilizationStrength;
                reblurSettings.diffuse.residualNoiseLevel = m_Settings.nrdSettings.residualNoiseLevel * 0.01f;
                reblurSettings.diffuse.checkerboardMode = tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                reblurSettings.diffuse.prePassMode = (nrd::PrePassMode)m_Settings.nrdSettings.prePassMode;
                reblurSettings.diffuse.enableAntiFirefly = m_Settings.nrdSettings.enableAntiFirefly;
                reblurSettings.diffuse.enableReferenceAccumulation = m_Settings.nrdSettings.referenceAccumulation;

                reblurSettings.specular.hitDistanceParameters = specHitDistanceParameters;
                reblurSettings.specular.lobeTrimmingParameters = { trimmingParams.x, trimmingParams.y, trimmingParams.z };
                reblurSettings.specular.antilagIntensitySettings = antilagIntensitySettings;
                reblurSettings.specular.antilagHitDistanceSettings = antilagHitDistanceSettings;
                reblurSettings.specular.materialMask = 0;
                reblurSettings.specular.maxAccumulatedFrameNum = reblurSettings.diffuse.maxAccumulatedFrameNum;
                reblurSettings.specular.blurRadius = reblurSettings.diffuse.blurRadius;
                reblurSettings.specular.maxAdaptiveRadiusScale = reblurSettings.diffuse.maxAdaptiveRadiusScale;
                reblurSettings.specular.normalWeightStrictness = reblurSettings.diffuse.normalWeightStrictness;
                reblurSettings.specular.stabilizationStrength = reblurSettings.diffuse.stabilizationStrength;
                reblurSettings.specular.residualNoiseLevel = reblurSettings.diffuse.residualNoiseLevel;
                reblurSettings.specular.checkerboardMode = tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::BLACK : nrd::CheckerboardMode::OFF;
                reblurSettings.specular.prePassMode = reblurSettings.diffuse.prePassMode;
                reblurSettings.specular.enableAntiFirefly = reblurSettings.diffuse.enableAntiFirefly;
                reblurSettings.specular.enableReferenceAccumulation = reblurSettings.diffuse.enableReferenceAccumulation;

                #if( NRD_OCCLUSION_ONLY == 0 )
                    #if( NRD_COMBINED == 1 )
                        m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR, &reblurSettings);
                    #else
                        m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE, &reblurSettings.diffuse);
                        m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR, &reblurSettings.specular);
                    #endif
                #else
                    #if( NRD_COMBINED == 1 )
                        m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR_OCCLUSION, &reblurSettings);
                    #else
                        m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_OCCLUSION, &reblurSettings.diffuse);
                        m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR_OCCLUSION, &reblurSettings.specular);
                    #endif
                #endif

                m_Reblur.Denoise(frameIndex, commandBuffer, commonSettings, userPool);
            }
            else if (m_Settings.denoiser == RELAX)
            {
                m_RelaxSettings.diffuseMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_RelaxSettings.specularMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_RelaxSettings.specularMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_RelaxSettings.checkerboardMode = tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                m_RelaxSettings.enableAntiFirefly = m_Settings.nrdSettings.enableAntiFirefly;
                m_RelaxSettings.diffusePrepassBlurRadius = m_Settings.nrdSettings.prePassMode ? 50.0f : 0.0f;
                m_RelaxSettings.specularPrepassBlurRadius = m_Settings.nrdSettings.prePassMode ? 50.0f : 0.0f;

                #if( NRD_COMBINED == 1 )
                    m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SPECULAR, &m_RelaxSettings);
                #else
                    nrd::RelaxDiffuseSettings diffuseSettings = {};
                    diffuseSettings.prepassBlurRadius                            = m_RelaxSettings.diffusePrepassBlurRadius;
                    diffuseSettings.diffuseMaxAccumulatedFrameNum                = m_RelaxSettings.diffuseMaxAccumulatedFrameNum;
                    diffuseSettings.diffuseMaxFastAccumulatedFrameNum            = m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum;
                    diffuseSettings.disocclusionFixEdgeStoppingNormalPower       = m_RelaxSettings.disocclusionFixEdgeStoppingNormalPower;
                    diffuseSettings.disocclusionFixMaxRadius                     = m_RelaxSettings.disocclusionFixMaxRadius;
                    diffuseSettings.disocclusionFixNumFramesToFix                = m_RelaxSettings.disocclusionFixNumFramesToFix;
                    diffuseSettings.historyClampingColorBoxSigmaScale            = m_RelaxSettings.historyClampingColorBoxSigmaScale;
                    diffuseSettings.spatialVarianceEstimationHistoryThreshold    = m_RelaxSettings.spatialVarianceEstimationHistoryThreshold;
                    diffuseSettings.atrousIterationNum                           = m_RelaxSettings.atrousIterationNum;
                    diffuseSettings.diffusePhiLuminance                          = m_RelaxSettings.diffusePhiLuminance;
                    diffuseSettings.minLuminanceWeight                           = m_RelaxSettings.minLuminanceWeight;
                    diffuseSettings.phiNormal                                    = m_RelaxSettings.phiNormal;
                    diffuseSettings.depthThreshold                               = m_RelaxSettings.depthThreshold;
                    diffuseSettings.checkerboardMode                             = tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                    diffuseSettings.enableAntiFirefly                            = m_RelaxSettings.enableAntiFirefly;

                    nrd::RelaxSpecularSettings specularSettings = {};
                    specularSettings.prepassBlurRadius                           = m_RelaxSettings.specularPrepassBlurRadius;
                    specularSettings.specularMaxAccumulatedFrameNum              = m_RelaxSettings.specularMaxAccumulatedFrameNum;
                    specularSettings.specularMaxFastAccumulatedFrameNum          = m_RelaxSettings.specularMaxFastAccumulatedFrameNum;
                    specularSettings.specularVarianceBoost                       = m_RelaxSettings.specularVarianceBoost;
                    specularSettings.disocclusionFixEdgeStoppingNormalPower      = m_RelaxSettings.disocclusionFixEdgeStoppingNormalPower;
                    specularSettings.disocclusionFixMaxRadius                    = m_RelaxSettings.disocclusionFixMaxRadius;
                    specularSettings.disocclusionFixNumFramesToFix               = m_RelaxSettings.disocclusionFixNumFramesToFix;
                    specularSettings.historyClampingColorBoxSigmaScale           = m_RelaxSettings.historyClampingColorBoxSigmaScale;
                    specularSettings.spatialVarianceEstimationHistoryThreshold   = m_RelaxSettings.spatialVarianceEstimationHistoryThreshold;
                    specularSettings.atrousIterationNum                          = m_RelaxSettings.atrousIterationNum;
                    specularSettings.specularPhiLuminance                        = m_RelaxSettings.specularPhiLuminance;
                    specularSettings.minLuminanceWeight                          = m_RelaxSettings.minLuminanceWeight;
                    specularSettings.phiNormal                                   = m_RelaxSettings.phiNormal;
                    specularSettings.depthThreshold                              = m_RelaxSettings.depthThreshold;
                    specularSettings.specularLobeAngleFraction                   = m_RelaxSettings.specularLobeAngleFraction;
                    specularSettings.specularLobeAngleSlack                      = m_RelaxSettings.specularLobeAngleSlack;
                    specularSettings.roughnessEdgeStoppingRelaxation             = m_RelaxSettings.roughnessEdgeStoppingRelaxation;
                    specularSettings.normalEdgeStoppingRelaxation                = m_RelaxSettings.normalEdgeStoppingRelaxation;
                    specularSettings.luminanceEdgeStoppingRelaxation             = m_RelaxSettings.luminanceEdgeStoppingRelaxation;
                    specularSettings.checkerboardMode                            = tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::BLACK : nrd::CheckerboardMode::OFF;
                    specularSettings.enableSpecularVirtualHistoryClamping        = m_RelaxSettings.enableSpecularVirtualHistoryClamping;
                    specularSettings.enableRoughnessEdgeStopping                 = m_RelaxSettings.enableRoughnessEdgeStopping;
                    specularSettings.enableAntiFirefly                           = m_RelaxSettings.enableAntiFirefly;

                    m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE, &diffuseSettings);
                    m_Relax.SetMethodSettings(nrd::Method::RELAX_SPECULAR, &specularSettings);
                #endif

                m_Relax.Denoise(frameIndex, commandBuffer, commonSettings, userPool);
            }

            // NRD integration layer binds its own descriptor pool, we need to re-bind ours back
            NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
        }

        { // Composition
            helper::Annotation annotation(NRI, commandBuffer, "Composition");

            const TextureState transitions[] =
            {
                // Input
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Downsampled_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Diff, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Spec, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::ComposedLighting_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::Composition));
            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Composition));

            const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::Composition1) };
            NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        if (m_Settings.reference)
        { // Reference
            helper::Annotation annotation(NRI, commandBuffer, "Reference denoising");

            nrd::ReferenceSettings referenceSettings = {};

            userPool[(size_t)nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST] = {&GetState(Texture::ComposedLighting_ViewZ), GetFormat(Texture::ComposedLighting_ViewZ)};
            userPool[(size_t)nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST] = {&GetState(Texture::ComposedLighting_ViewZ), GetFormat(Texture::ComposedLighting_ViewZ)};

            commonSettings.splitScreen = m_Settings.separator;

            m_Reference.SetMethodSettings(nrd::Method::REFERENCE, &referenceSettings);
            m_Reference.Denoise(frameIndex, commandBuffer, commonSettings, userPool);

            // NRD integration layer binds its own descriptor pool, we need to re-bind ours back
            NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
        }

        if (m_DLSS.IsInitialized())
        {
            { // Pre
                helper::Annotation annotation(NRI, commandBuffer, "PreDlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::Motion, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::TransparentLighting, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::ComposedLighting_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                    {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                    {Texture::Unfiltered_Diff, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::PreDlss));
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::PreDlss));

                const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::PreDlss1) };
                NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

                NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            }

            { // DLSS
                helper::Annotation annotation(NRI, commandBuffer, "Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Unfiltered_Diff, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                DlssDispatchDesc dlssDesc = {};
                dlssDesc.texInput = Get(Texture::Unfiltered_Diff);
                dlssDesc.texMv = Get(Texture::Unfiltered_ShadowData);
                dlssDesc.texDepth = Get(Texture::ViewZ);
                dlssDesc.texOutput = Get(Texture::DlssOutput);

                dlssDesc.descriptorInput = Get(Descriptor::Unfiltered_Diff_Texture);
                dlssDesc.descriptorMv = Get(Descriptor::Unfiltered_ShadowData_Texture);
                dlssDesc.descriptorDepth = Get(Descriptor::ViewZ_Texture);
                dlssDesc.descriptorOutput = Get(Descriptor::DlssOutput_StorageTexture);

                dlssDesc.sharpness = m_Sharpness;
                dlssDesc.renderOrScaledResolution = {rectW, rectH};
                dlssDesc.motionVectorScale[0] = 1.0f;
                dlssDesc.motionVectorScale[1] = 1.0f;
                dlssDesc.jitter[0] = -m_Camera.state.viewportJitter.x;
                dlssDesc.jitter[1] = -m_Camera.state.viewportJitter.y;
                dlssDesc.physicalDeviceIndex = 0;
                dlssDesc.reset = resetHistoryFactor == 0.0f;

                m_DLSS.Evaluate(&commandBuffer, dlssDesc);

                NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
            }

            { // After
                helper::Annotation annotation(NRI, commandBuffer, "AfterDlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::AfterDlss));
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::AfterDlss));

                const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(DescriptorSet::AfterDlss1) };
                NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

                NRI.CmdDispatch(commandBuffer, outputGridW, outputGridH, 1);
            }
        }
        else
        {
            const Texture taaSrc = isEven ? Texture::TaaHistoryPrev : Texture::TaaHistory;
            const Texture taaDst = isEven ? Texture::TaaHistory : Texture::TaaHistoryPrev;

            { // Temporal
                helper::Annotation annotation(NRI, commandBuffer, "Temporal");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::Motion, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::ComposedLighting_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::TransparentLighting, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {taaSrc, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {taaDst, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::Temporal));
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Temporal));

                const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(isEven ? DescriptorSet::Temporal1a : DescriptorSet::Temporal1b) };
                NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

                NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            }

            { // Upsample or copy
                helper::Annotation annotation(NRI, commandBuffer, "Upsample");

                const TextureState transitions[] =
                {
                    // Input
                    {taaDst, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions.data(), helper::GetCountOf(optimizedTransitions));
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipelineLayout(commandBuffer, *GetPipelineLayout(Pipeline::Upsample));
                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Upsample));

                const nri::DescriptorSet* descriptorSets[] = { frame.globalConstantBufferDescriptorSet, Get(isEven ? DescriptorSet::Upsample1a : DescriptorSet::Upsample1b) };
                NRI.CmdSetDescriptorSets(commandBuffer, 0, helper::GetCountOf(descriptorSets), descriptorSets, nullptr);

                NRI.CmdDispatch(commandBuffer, screenGridW, screenGridH, 1);
            }
        }

        { // Copy to back-buffer
            const nri::TextureTransitionBarrierDesc copyTransitions[] =
            {
                nri::TextureTransition(GetState(Texture::Final), nri::AccessBits::COPY_SOURCE, nri::TextureLayout::GENERAL),
                nri::TextureTransition(backBuffer->texture, nri::AccessBits::UNKNOWN, nri::AccessBits::COPY_DESTINATION, nri::TextureLayout::UNKNOWN, nri::TextureLayout::GENERAL),
            };
            transitionBarriers.textures = copyTransitions;
            transitionBarriers.textureNum = helper::GetCountOf(copyTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdCopyTexture(commandBuffer, *backBuffer->texture, 0, nullptr, *Get(Texture::Final), 0, nullptr);
        }

        { // UI
            const nri::TextureTransitionBarrierDesc beforeTransitions = nri::TextureTransition(backBuffer->texture, nri::AccessBits::COPY_DESTINATION, nri::AccessBits::COLOR_ATTACHMENT, nri::TextureLayout::GENERAL, nri::TextureLayout::COLOR_ATTACHMENT);
            transitionBarriers.textures = &beforeTransitions;
            transitionBarriers.textureNum = 1;
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdBeginRenderPass(commandBuffer, *backBuffer->frameBufferUI, nri::RenderPassBeginFlag::SKIP_FRAME_BUFFER_CLEAR);
            RenderUserInterface(commandBuffer);
            NRI.CmdEndRenderPass(commandBuffer);

            const nri::TextureTransitionBarrierDesc afterTransitions = nri::TextureTransition(backBuffer->texture, nri::AccessBits::COLOR_ATTACHMENT, nri::AccessBits::UNKNOWN, nri::TextureLayout::COLOR_ATTACHMENT, nri::TextureLayout::PRESENT);
            transitionBarriers.textures = &afterTransitions;
            transitionBarriers.textureNum = 1;
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
        }
    }
    NRI.EndCommandBuffer(commandBuffer);

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.wait = &m_BackBufferAcquireSemaphore;
    workSubmissionDesc.waitNum = 1;
    workSubmissionDesc.commandBuffers = &frame.commandBuffer;
    workSubmissionDesc.commandBufferNum = 1;
    workSubmissionDesc.signal = &m_BackBufferReleaseSemaphore;
    workSubmissionDesc.signalNum = 1;
    NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, frame.deviceSemaphore);

    NRI.SwapChainPresent(*m_SwapChain, *m_BackBufferReleaseSemaphore);

    m_Timer.UpdateElapsedTimeSinceLastSave();

    float msLimit = 1000.0f / m_Settings.maxFps;
    while( m_Timer.GetElapsedTime() < msLimit && m_Settings.limitFps)
        m_Timer.UpdateElapsedTimeSinceLastSave();

    m_Timer.SaveCurrentTime();
}

SAMPLE_MAIN(Sample, 0);
