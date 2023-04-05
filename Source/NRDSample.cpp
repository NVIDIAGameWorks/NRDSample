/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NRIFramework.h"
#include "Extensions/NRIRayTracing.h"

// NRD and NRI-based integration
#include "NRD.h"
#include "NRDIntegration.hpp"

// DLSS and NRI-based integration
#include "Extensions/NRIWrapperVK.h"
#include "DLSS/DLSSIntegration.hpp"

// NIS
#include "NGX/NVIDIAImageScaling/NIS/NIS_Config.h"

//=================================================================================
// Settings
//=================================================================================

// Fused or separate denoising selection
//      0 - DIFFUSE and SPECULAR
//      1 - DIFFUSE_SPECULAR
#define NRD_COMBINED                                1

// IMPORTANT: adjust same macro in "Shared.hlsli"
//      NORMAL                - common (non specialized) denoisers
//      OCCLUSION             - OCCLUSION (ambient or specular occlusion only) denoisers
//      SH                    - SH (spherical harmonics or spherical gaussian) denoisers
//      DIRECTIONAL_OCCLUSION - DIRECTIONAL_OCCLUSION (ambient occlusion in SH mode) denoisers
#define NRD_MODE                                    NORMAL

constexpr uint32_t MAX_ANIMATED_INSTANCE_NUM        = 512;
constexpr auto BLAS_BUILD_BITS                      = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;
constexpr auto TLAS_BUILD_BITS                      = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;
constexpr float ACCUMULATION_TIME                   = 0.5f; // seconds
constexpr float NEAR_Z                              = 0.001f; // m
constexpr float GLASS_THICKNESS                     = 0.002f; // m
constexpr bool CAMERA_RELATIVE                      = true;
constexpr bool CAMERA_LEFT_HANDED                   = true;
constexpr bool ALLOW_BLAS_MERGING                   = true;

//=================================================================================
// Important tests, sensitive to regressions or just testing base functionality
//=================================================================================

const std::vector<uint32_t> interior_checkMeTests =
{{
    1, 3, 6, 8, 9, 10, 12, 13, 14, 23, 27, 28, 29, 31, 32, 35, 43, 44, 47, 53,
    59, 60, 62, 67, 75, 76, 79, 81, 95, 96, 107, 109, 111, 110, 114, 120, 124,
    126, 127, 132, 133, 134, 139, 140, 142, 145, 148, 150, 155, 156, 157, 160,
    161, 162, 164, 168, 169, 171, 172, 173, 174
}};

//=================================================================================
// Tests, where IQ improvement would be "nice to have"
//=================================================================================

const std::vector<uint32_t> REBLUR_interior_improveMeTests =
{{
    108, 153, 158
}};

const std::vector<uint32_t> RELAX_interior_improveMeTests =
{{
    114, 144, 148, 156, 159
}};

//=================================================================================

constexpr int32_t MAX_HISTORY_FRAME_NUM             = (int32_t)std::min(60u, std::min(nrd::REBLUR_MAX_HISTORY_FRAME_NUM, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
constexpr uint32_t TEXTURES_PER_MATERIAL            = 4;
constexpr uint32_t MAX_TEXTURE_TRANSITION_NUM       = 32;

// UI
#define UI_YELLOW                                   ImVec4(1.0f, 0.9f, 0.0f, 1.0f)
#define UI_GREEN                                    ImVec4(0.5f, 0.9f, 0.0f, 1.0f)
#define UI_RED                                      ImVec4(1.0f, 0.1f, 0.0f, 1.0f)
#define UI_HEADER                                   ImVec4(0.7f, 1.0f, 0.7f, 1.0f)
#define UI_HEADER_BACKGROUND                        ImVec4(0.7f * 0.3f, 1.0f * 0.3f, 0.7f * 0.3f, 1.0f)
#define UI_DEFAULT                                  ImGui::GetStyleColorVec4(ImGuiCol_Text)

// NRD variant
#define NORMAL                                      0
#define OCCLUSION                                   1
#define SH                                          2
#define DIRECTIONAL_OCCLUSION                       3

// See HLSL
#define FLAG_FIRST_BIT                              20
#define INSTANCE_ID_MASK                            ( ( 1 << FLAG_FIRST_BIT ) - 1 )
#define FLAG_DEFAULT                                0x01 // set only if nothing is set
#define FLAG_TRANSPARENT                            0x02 // transparent
#define FLAG_FORCED_EMISSION                        0x04 // animated emissive cube

enum Denoiser : int32_t
{
    REBLUR,
    RELAX,
    REFERENCE,

    DENOISER_MAX_NUM
};

enum Resolution : int32_t
{
    RESOLUTION_FULL,
    RESOLUTION_FULL_PROBABILISTIC,
    RESOLUTION_HALF
};

enum MvType : int32_t
{
    MV_2D,
    MV_25D,
    MV_3D,
};

enum class AccelerationStructure : uint32_t
{
    TLAS_World,
    TLAS_Emissive,

    BLAS_StaticOpaque,
    BLAS_StaticTransparent,
    BLAS_StaticEmissive,

    BLAS_Other // all other BLAS start from here
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
    Ambient,
    ViewZ,
    Mv,
    Normal_Roughness,
    PrimaryMipAndCurvature,
    BaseColor_Metalness,
    DirectLighting,
    DirectEmission,
    Shadow,
    Diff,
    Spec,
    Unfiltered_ShadowData,
    Unfiltered_Diff,
    Unfiltered_Spec,
    Unfiltered_Shadow_Translucency,
    Validation,
    Composed_ViewZ,
    DlssOutput,
    Final,

    // History
    ComposedDiff_ViewZ,
    ComposedSpec_ViewZ,
    TaaHistory,
    TaaHistoryPrev,

    // SH
#if( NRD_MODE == SH )
    Unfiltered_DiffSh,
    Unfiltered_SpecSh,
    DiffSh,
    SpecSh,
#endif

    // Read-only
    NisData1,
    NisData2,
    MaterialTextures,

    MAX_NUM,

    // Aliases
    DlssInput = Unfiltered_Diff
};

enum class Pipeline : uint32_t
{
    AmbientRays,
    PrimaryRays,
    DirectLighting,
    IndirectRays,
    Composition,
    DeltaRays,
    Temporal,
    Upsample,
    UpsampleNis,
    PreDlss,
    AfterDlss,

    MAX_NUM,
};

enum class Descriptor : uint32_t
{
    World_AccelerationStructure,
    Light_AccelerationStructure,

    LinearMipmapLinear_Sampler,
    LinearMipmapNearest_Sampler,
    Linear_Sampler,
    Nearest_Sampler,

    PrimitiveData_Buffer,
    InstanceData_Buffer,

    Ambient_Texture,
    Ambient_StorageTexture,
    ViewZ_Texture,
    ViewZ_StorageTexture,
    Mv_Texture,
    Mv_StorageTexture,
    Normal_Roughness_Texture,
    Normal_Roughness_StorageTexture,
    PrimaryMipAndCurvature_Texture,
    PrimaryMipAndCurvature_StorageTexture,
    BaseColor_Metalness_Texture,
    BaseColor_Metalness_StorageTexture,
    DirectLighting_Texture,
    DirectLighting_StorageTexture,
    DirectEmission_Texture,
    DirectEmission_StorageTexture,
    Shadow_Texture,
    Shadow_StorageTexture,
    Diff_Texture,
    Diff_StorageTexture,
    Spec_Texture,
    Spec_StorageTexture,
    Unfiltered_ShadowData_Texture,
    Unfiltered_ShadowData_StorageTexture,
    Unfiltered_Diff_Texture,
    Unfiltered_Diff_StorageTexture,
    Unfiltered_Spec_Texture,
    Unfiltered_Spec_StorageTexture,
    Unfiltered_Shadow_Translucency_Texture,
    Unfiltered_Shadow_Translucency_StorageTexture,
    Validation_Texture,
    Validation_StorageTexture,
    Composed_ViewZ_Texture,
    Composed_ViewZ_StorageTexture,
    DlssOutput_Texture,
    DlssOutput_StorageTexture,
    Final_Texture,
    Final_StorageTexture,

    // History
    ComposedDiff_ViewZ_Texture,
    ComposedDiff_ViewZ_StorageTexture,
    ComposedSpec_ViewZ_Texture,
    ComposedSpec_ViewZ_StorageTexture,
    TaaHistory_Texture,
    TaaHistory_StorageTexture,
    TaaHistoryPrev_Texture,
    TaaHistoryPrev_StorageTexture,

    // SH
#if( NRD_MODE == SH )
    Unfiltered_DiffSh_Texture,
    Unfiltered_DiffSh_StorageTexture,
    Unfiltered_SpecSh_Texture,
    Unfiltered_SpecSh_StorageTexture,
    DiffSh_Texture,
    DiffSh_StorageTexture,
    SpecSh_Texture,
    SpecSh_StorageTexture,
#endif

    // Read-only
    NisData1,
    NisData2,
    MaterialTextures,

    MAX_NUM,

    // Aliases
    DlssInput_Texture = Unfiltered_Diff_Texture,
    DlssInput_StorageTexture = Unfiltered_Diff_StorageTexture
};

enum class DescriptorSet : uint32_t
{
    AmbientRays1,
    PrimaryRays1,
    DirectLighting1,
    IndirectRays1,
    Composition1,
    DeltaRays1,
    Temporal1a,
    Temporal1b,
    Upsample1a,
    Upsample1b,
    UpsampleNis1a,
    UpsampleNis1b,
    PreDlss1,
    AfterDlss1,
    RayTracing2,

    MAX_NUM
};

struct NRIInterface
    : public nri::CoreInterface
    , public nri::SwapChainInterface
    , public nri::RayTracingInterface
    , public nri::HelperInterface
{};

struct Frame
{
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
    nri::Descriptor* globalConstantBufferDescriptor;
    nri::DescriptorSet* globalConstantBufferDescriptorSet;
    uint64_t globalConstantBufferOffset;
};

struct GlobalConstantBufferData
{
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToView;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClip;
    float4x4 gWorldToClipPrev;
    float4 gHitDistParams;
    float4 gCameraFrustum;
    float4 gSunDirection_gExposure;
    float4 gCameraOrigin_gMipBias;
    float4 gTrimmingParams_gEmissionIntensity;
    float4 gViewDirection_gIsOrtho;
    float2 gWindowSize;
    float2 gInvWindowSize;
    float2 gOutputSize;
    float2 gInvOutputSize;
    float2 gRenderSize;
    float2 gInvRenderSize;
    float2 gRectSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gNearZ;
    float gAmbientAccumSpeed;
    float gAmbient;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gIndirectDiffuse;
    float gIndirectSpecular;
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gTransparent;
    float gReference;
    float gUsePrevFrame;
    float gMinProbability;
    uint32_t gDenoiserType;
    uint32_t gDisableShadowsAndEnableImportanceSampling;
    uint32_t gOnScreen;
    uint32_t gFrameIndex;
    uint32_t gForcedMaterial;
    uint32_t gUseNormalMap;
    uint32_t gIsWorldSpaceMotionEnabled;
    uint32_t gTracingMode;
    uint32_t gSampleNum;
    uint32_t gBounceNum;
    uint32_t gTAA;
    uint32_t gResolve;
    uint32_t gPSR;
    uint32_t gValidation;

    // NIS
    float gNisDetectRatio;
    float gNisDetectThres;
    float gNisMinContrastRatio;
    float gNisRatioNorm;
    float gNisContrastBoost;
    float gNisEps;
    float gNisSharpStartY;
    float gNisSharpScaleY;
    float gNisSharpStrengthMin;
    float gNisSharpStrengthScale;
    float gNisSharpLimitMin;
    float gNisSharpLimitScale;
    float gNisScaleX;
    float gNisScaleY;
    float gNisDstNormX;
    float gNisDstNormY;
    float gNisSrcNormX;
    float gNisSrcNormY;
    uint32_t gNisInputViewportOriginX;
    uint32_t gNisInputViewportOriginY;
    uint32_t gNisInputViewportWidth;
    uint32_t gNisInputViewportHeight;
    uint32_t gNisOutputViewportOriginX;
    uint32_t gNisOutputViewportOriginY;
    uint32_t gNisOutputViewportWidth;
    uint32_t gNisOutputViewportHeight;
};

struct Settings
{
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
    float       hitDistScale                       = 3.0f;
    float       disocclusionThreshold              = 1.0f;
    float       resolutionScale                    = 1.0f;
    float       sharpness                          = 0.15f;

    int32_t     maxAccumulatedFrameNum             = 31;
    int32_t     maxFastAccumulatedFrameNum         = 7;
    int32_t     onScreen                           = 0;
    int32_t     forcedMaterial                     = 0;
    int32_t     animatedObjectNum                  = 5;
    int32_t     activeAnimation                    = 0;
    int32_t     motionMode                         = 0;
    int32_t     denoiser                           = REBLUR;
    int32_t     rpp                                = 1;
    int32_t     bounceNum                          = 1;
    int32_t     tracingMode                        = RESOLUTION_HALF;
    int32_t     mvType                             = MV_25D;

    bool        cameraJitter                       = true;
    bool        limitFps                           = false;
    bool        ambient                            = true;
    bool        PSR                                = false;
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
    bool        linearMotion                       = true;
    bool        emissiveObjects                    = false;
    bool        importanceSampling                 = true;
    bool        specularLobeTrimming               = true;
    bool        ortho                              = false;
    bool        adaptiveAccumulation               = true;
    bool        usePrevFrame                       = true;
    bool        DLSS                               = false;
    bool        NIS                                = true;
    bool        adaptRadiusToResolution            = true;
    bool        windowAlignment                    = true;
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

struct AnimatedInstance
{
    float3 basePosition;
    float3 rotationAxis;
    float3 elipseAxis;
    float durationSec = 5.0f;
    float progressedSec = 0.0f;
    float inverseRotation = 1.0f;
    float inverseDirection = 1.0f;
    uint32_t instanceID = 0;

    float4x4 Animate(float elapsedSeconds, float scale, float3& position)
    {
        float angle = progressedSec / durationSec;
        angle = Pi(angle * 2.0f - 1.0f);

        float3 localPosition;
        localPosition.x = Cos(angle * inverseDirection);
        localPosition.y = Sin(angle * inverseDirection);
        localPosition.z = localPosition.y;

        position = basePosition + localPosition * elipseAxis * scale;

        float4x4 transform;
        transform.SetupByRotation(angle * inverseRotation, rotationAxis);
        transform.AddScale(scale);

        progressedSec += elapsedSeconds;
        progressedSec = (progressedSec >= durationSec) ? 0.0f : progressedSec;

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
    float curvature;
};

struct InstanceData
{
    float4 mWorldToWorldPrev0;
    float4 mWorldToWorldPrev1;
    float4 mWorldToWorldPrev2;

    uint32_t baseTextureIndex;
    uint32_t basePrimitiveIndex;
    uint32_t isStatic;
    float invScale;
};

class Sample : public SampleBase
{
public:
    Sample() :
        m_Reblur(BUFFERED_FRAME_MAX_NUM, "REBLUR"),
        m_Relax(BUFFERED_FRAME_MAX_NUM, "RELAX"),
        m_Sigma(BUFFERED_FRAME_MAX_NUM, "SIGMA"),
        m_Reference(BUFFERED_FRAME_MAX_NUM, "REFERENCE")
    {}

    ~Sample();

    void InitCmdLine(cmdline::parser& cmdLine) override
    {
        cmdLine.add<int32_t>("dlssQuality", 'd', "DLSS quality: [-1: 3]", false, -1, cmdline::range(-1, 3));
        cmdLine.add("debugNRD", 0, "enable NRD validation");
    }

    void ReadCmdLine(cmdline::parser& cmdLine) override
    {
        m_DlssQuality = cmdLine.get<int32_t>("dlssQuality");
        m_DebugNRD = cmdLine.exist("debugNRD");
    }

    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

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

    inline nri::Descriptor*& Get(Descriptor index)
    { return m_Descriptors[(uint32_t)index]; }

    inline nri::DescriptorSet*& Get(DescriptorSet index)
    { return m_DescriptorSets[(uint32_t)index]; }

    inline nri::AccelerationStructure*& Get(AccelerationStructure index)
    { return m_AccelerationStructures[(uint32_t)index]; }

    void LoadScene();
    void AddInnerGlassSurfaces();
    void GenerateAnimatedCubes();

    nri::Format CreateSwapChain();
    void CreateCommandBuffers();
    void CreatePipelineLayoutAndDescriptorPool();
    void CreatePipelines();
    void CreateAccelerationStructures();
    void CreateSamplers();
    void CreateResources(nri::Format swapChainFormat);
    void CreateDescriptorSets();

    void CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint16_t width, uint16_t height, uint16_t mipNum, uint16_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state);
    void CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage, nri::Format format = nri::Format::UNKNOWN);

    void UploadStaticData();
    void UpdateConstantBuffer(uint32_t frameIndex, float globalResetFactor);
    void RestoreBindings(nri::CommandBuffer& commandBuffer, const Frame& frame);
    void BuildTopLevelAccelerationStructure(nri::CommandBuffer& commandBuffer, uint32_t bufferedFrameIndex);
    uint32_t BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureTransitionBarrierDesc, MAX_TEXTURE_TRANSITION_NUM>& transitions);

    inline float3 GetSunDirection() const
    {
        float3 sunDirection;
        sunDirection.x = Cos( DegToRad(m_Settings.sunAzimuth) ) * Cos( DegToRad(m_Settings.sunElevation) );
        sunDirection.y = Sin( DegToRad(m_Settings.sunAzimuth) ) * Cos( DegToRad(m_Settings.sunElevation) );
        sunDirection.z = Sin( DegToRad(m_Settings.sunElevation) );

        return sunDirection;
    }

    inline float3 GetSpecularLobeTrimming() const
    { return m_Settings.specularLobeTrimming ? float3(0.95f, 0.04f, 0.11f) : float3(1.0f, 0.0f, 0.0001f); }

    inline float GetDenoisingRange() const
    { return 4.0f * m_Scene.aabb.GetRadius(); }

private:
    NrdIntegration m_Reblur;
    NrdIntegration m_Relax;
    NrdIntegration m_Sigma;
    NrdIntegration m_Reference;
    DlssIntegration m_DLSS;
    NRIInterface NRI = {};
    utils::Scene m_Scene;
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::CommandQueue* m_CommandQueue = nullptr;
    nri::Fence* m_FrameFence;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::TextureTransitionBarrierDesc> m_TextureStates;
    std::vector<nri::Format> m_TextureFormats;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Memory*> m_MemoryAllocations;
    std::vector<nri::Descriptor*> m_Descriptors;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::AccelerationStructure*> m_AccelerationStructures;
    std::vector<BackBuffer> m_SwapChainBuffers;
    std::vector<AnimatedInstance> m_AnimatedInstances;
    std::array<float, 256> m_FrameTimes = {};
    nrd::RelaxDiffuseSpecularSettings m_RelaxSettings = {};
    nrd::ReblurSettings m_ReblurSettings = {};
    nrd::ReferenceSettings m_ReferenceSettings = {};
    Settings m_Settings = {};
    Settings m_PrevSettings = {};
    Settings m_DefaultSettings = {};
    const std::vector<uint32_t>* m_checkMeTests = nullptr;
    const std::vector<uint32_t>* m_improveMeTests = nullptr;
    float3 m_PrevLocalPos = {};
    uint2 m_RenderResolution = {};
    uint64_t m_ConstantBufferSize = 0;
    uint32_t m_OpaqueObjectsNum = 0;
    uint32_t m_TransparentObjectsNum = 0;
    uint32_t m_EmissiveObjectsNum = 0;
    uint32_t m_ProxyInstancesNum = 0;
    uint32_t m_LastSelectedTest = uint32_t(-1);
    uint32_t m_TestNum = uint32_t(-1);
    int32_t m_DlssQuality = int32_t(-1);
    float m_UiWidth = 0.0f;
    float m_AmbientAccumFrameNum = 0.0f;
    float m_MinResolutionScale = 0.5f;
    bool m_HasTransparent = false;
    bool m_ShowUi = true;
    bool m_ForceHistoryReset = false;
    bool m_Resolve = true;
    bool m_DebugNRD = false;
    bool m_ShowValidationOverlay = true;
};

Sample::~Sample()
{
    if (!m_Device)
        return;

    NRI.WaitForIdle(*m_CommandQueue);

    m_DLSS.Shutdown();

    m_Reblur.Destroy();
    m_Relax.Destroy();
    m_Sigma.Destroy();
    m_Reference.Destroy();

    for (Frame& frame : m_Frames)
    {
        NRI.DestroyCommandBuffer(*frame.commandBuffer);
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

    for (uint32_t i = 0; i < m_AccelerationStructures.size(); i++)
    {
        if (m_AccelerationStructures[i])
            NRI.DestroyAccelerationStructure(*m_AccelerationStructures[i]);
    }

    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyFence(*m_FrameFence);
    NRI.DestroySwapChain(*m_SwapChain);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
    {
        if (m_MemoryAllocations[i])
            NRI.FreeMemory(*m_MemoryAllocations[i]);
    }

    DestroyUserInterface();

    nri::DestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI)
{
    Rand::Seed(106937, &m_FastRandState);

    nri::PhysicalDeviceGroup mostPerformantPhysicalDeviceGroup = {};
    uint32_t deviceGroupNum = 1;
    NRI_ABORT_ON_FAILURE(nri::GetPhysicalDevices(&mostPerformantPhysicalDeviceGroup, deviceGroupNum));

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.spirvBindingOffsets = SPIRV_BINDING_OFFSETS;
    deviceCreationDesc.physicalDeviceGroup = &mostPerformantPhysicalDeviceGroup;
    if (mostPerformantPhysicalDeviceGroup.vendor == nri::Vendor::NVIDIA)
        DlssIntegration::SetupDeviceExtensions(deviceCreationDesc);

    NRI_ABORT_ON_FAILURE( nri::CreateDevice(deviceCreationDesc, m_Device) );

    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) );

    NRI_ABORT_ON_FAILURE( NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, m_CommandQueue) );
    NRI_ABORT_ON_FAILURE( NRI.CreateFence(*m_Device, 0, m_FrameFence) );

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    m_ConstantBufferSize = helper::Align(sizeof(GlobalConstantBufferData), deviceDesc.constantBufferOffsetAlignment);
    m_RenderResolution = GetOutputResolution();

    if (m_DlssQuality != -1 && m_DLSS.InitializeLibrary(*m_Device, ""))
    {
        DlssSettings dlssSettings = {};
        DlssInitDesc dlssInitDesc = {};
        dlssInitDesc.outputResolution = { GetOutputResolution().x, GetOutputResolution().y };

        if (m_DLSS.GetOptimalSettings(dlssInitDesc.outputResolution, (DlssQuality)m_DlssQuality, dlssSettings))
        {
            dlssInitDesc.quality = (DlssQuality)m_DlssQuality;
            dlssInitDesc.isContentHDR = true;

            m_DLSS.Initialize(m_CommandQueue, dlssInitDesc);

            float sx = float(dlssSettings.minRenderResolution.Width) / float(dlssSettings.renderResolution.Width);
            float sy = float(dlssSettings.minRenderResolution.Height) / float(dlssSettings.renderResolution.Height);
            float minResolutionScale = sy > sx ? sy : sx;

            m_RenderResolution = {dlssSettings.renderResolution.Width, dlssSettings.renderResolution.Height};
            m_MinResolutionScale = minResolutionScale;

            printf("Render resolution (%u, %u)\n", m_RenderResolution.x, m_RenderResolution.y);

            m_Settings.sharpness = dlssSettings.sharpness;
            m_Settings.DLSS = true;
        }
        else
        {
            m_DLSS.Shutdown();

            printf("Unsupported DLSS mode!\n");
        }
    }

    #if 0
        // README "Memory requirements" table generator
        printf("| %10s | %36s | %16s | %16s | %16s |\n", "Resolution", "Denoiser", "Working set (Mb)", "Persistent (Mb)", "Aliasable (Mb)");
        printf("|------------|--------------------------------------|------------------|------------------|------------------|\n");

        for (uint32_t j = 0; j < 3; j++)
        {
            const char* resolution = "1080p";
            uint16_t w = 1920;
            uint16_t h = 1080;

            if (j == 1)
            {
                resolution = "1440p";
                w = 2560;
                h = 1440;
            }
            else if (j == 2)
            {
                resolution = "2160p";
                w = 3840;
                h = 2160;
            }

            for (uint32_t i = 0; i <= (uint32_t)nrd::Method::REFERENCE; i++)
            {
                nrd::Method method = (nrd::Method)i;
                const char* methodName = nrd::GetMethodString(method);

                const nrd::MethodDesc methodDesc = {method, w, h};

                nrd::DenoiserCreationDesc denoiserCreationDesc = {};
                denoiserCreationDesc.requestedMethods = &methodDesc;
                denoiserCreationDesc.requestedMethodsNum = 1;

                NrdIntegration denoiser(2);
                NRI_ABORT_ON_FALSE( denoiser.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
                printf("| %10s | %36s | %16.2f | %16.2f | %16.2f |\n", i == 0 ? resolution : "", methodName, denoiser.GetTotalMemoryUsageInMb(), denoiser.GetPersistentMemoryUsageInMb(), denoiser.GetAliasableMemoryUsageInMb());
                denoiser.Destroy();
            }

            if (j != 2)
                printf("| %10s | %36s | %16s | %16s | %16s |\n", "", "", "", "", "");
        }

        __debugbreak();
    #endif

    LoadScene();
    AddInnerGlassSurfaces();
    GenerateAnimatedCubes();

    nri::Format swapChainFormat = CreateSwapChain();
    CreateCommandBuffers();
    CreatePipelineLayoutAndDescriptorPool();
    CreatePipelines();
    CreateAccelerationStructures();
    CreateSamplers();
    CreateResources(swapChainFormat);
    CreateDescriptorSets();

    UploadStaticData();

    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);

    m_Scene.UnloadTextureData();
    m_Scene.UnloadGeometryData();

    { // REBLUR
        const nrd::MethodDesc methodDescs[] =
        {
    #if( NRD_MODE == OCCLUSION )
        #if( NRD_COMBINED == 1 )
            { nrd::Method::REBLUR_DIFFUSE_SPECULAR_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #else
            { nrd::Method::REBLUR_DIFFUSE_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            { nrd::Method::REBLUR_SPECULAR_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #endif
    #elif( NRD_MODE == SH )
        #if( NRD_COMBINED == 1 )
            { nrd::Method::REBLUR_DIFFUSE_SPECULAR_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #else
            { nrd::Method::REBLUR_DIFFUSE_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            { nrd::Method::REBLUR_SPECULAR_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #endif
    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
            { nrd::Method::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
    #else
        #if( NRD_COMBINED == 1 )
            { nrd::Method::REBLUR_DIFFUSE_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #else
            { nrd::Method::REBLUR_DIFFUSE, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            { nrd::Method::REBLUR_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #endif
    #endif
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Reblur.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    { // RELAX
        const nrd::MethodDesc methodDescs[] =
        {
            #if( NRD_MODE == SH )
                #if( NRD_COMBINED == 1 )
                    { nrd::Method::RELAX_DIFFUSE_SPECULAR_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                #else
                    { nrd::Method::RELAX_DIFFUSE_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                    { nrd::Method::RELAX_SPECULAR_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                #endif
            #else
                #if( NRD_COMBINED == 1 )
                    { nrd::Method::RELAX_DIFFUSE_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                #else
                    { nrd::Method::RELAX_DIFFUSE, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                    { nrd::Method::RELAX_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                #endif
            #endif
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Relax.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    { // SIGMA
        const nrd::MethodDesc methodDescs[] =
        {
            { nrd::Method::SIGMA_SHADOW_TRANSLUCENCY, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Sigma.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    { // REFERENCE
        const nrd::MethodDesc methodDescs[] =
        {
            { nrd::Method::REFERENCE, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Reference.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    m_DefaultSettings = m_Settings;

    return CreateUserInterface(*m_Device, NRI, NRI, swapChainFormat);
}

void Sample::PrepareFrame(uint32_t frameIndex)
{
    m_ForceHistoryReset = false;
    m_PrevSettings = m_Settings;
    m_Camera.SavePreviousState();

    PrepareUserInterface();

    if (IsKeyToggled(Key::Tab))
        m_ShowUi = !m_ShowUi;
    if (IsKeyToggled(Key::F1))
        m_Settings.debug = Step(0.5f, 1.0f - m_Settings.debug);
    if (IsKeyToggled(Key::F3))
        m_Settings.emission = !m_Settings.emission;
    if (IsKeyToggled(Key::Space))
        m_Settings.pauseAnimation = !m_Settings.pauseAnimation;
    if (IsKeyToggled(Key::PageDown) || IsKeyToggled(Key::Num3))
    {
        m_Settings.denoiser++;
        if (m_Settings.denoiser > DENOISER_MAX_NUM - 1)
            m_Settings.denoiser = 0;
    }
    if (IsKeyToggled(Key::PageUp) || IsKeyToggled(Key::Num9))
    {
        m_Settings.denoiser--;
        if (m_Settings.denoiser < 0)
            m_Settings.denoiser = DENOISER_MAX_NUM - 1;
    }

    if (!IsKeyPressed(Key::LAlt) && m_ShowUi)
    {
        ImGui::SetNextWindowPos(ImVec2(m_Settings.windowAlignment ? 5.0f : GetOutputResolution().x - m_UiWidth - 5.0f, 5.0f));
        ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f));
        ImGui::Begin("Settings [Tab]", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
        {
            float avgFrameTime = m_Timer.GetVerySmoothedFrameTime();

            char buf[256];
            snprintf(buf, sizeof(buf), "%.1f FPS (%.2f ms)", 1000.0f / avgFrameTime, avgFrameTime);

            ImVec4 colorFps = UI_GREEN;
            if (avgFrameTime > 1000.0f / 59.5f)
                colorFps = UI_YELLOW;
            if (avgFrameTime > 1000.0f / 29.5f)
                colorFps = UI_RED;

            float lo = avgFrameTime * 0.5f;
            float hi = avgFrameTime * 1.5f;

            const uint32_t N = helper::GetCountOf(m_FrameTimes);
            uint32_t head = frameIndex % N;
            m_FrameTimes[head] = m_Timer.GetFrameTime();
            ImGui::PushStyleColor(ImGuiCol_Text, colorFps);
                ImGui::PlotLines("", m_FrameTimes.data(), N, head, buf, lo, hi, ImVec2(0.0f, 70.0f));
            ImGui::PopStyleColor();

            if (IsButtonPressed(Button::Right))
            {
                ImGui::Text("Move - W/S/A/D");
                ImGui::Text("Accelerate - MOUSE SCROLL");
            }
            else
            {
                // "Camera" section
                ImGui::NewLine();
                ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                bool isUnfolded = ImGui::CollapsingHeader("CAMERA (press RIGHT MOUSE BOTTON for free-fly mode)", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();

                ImGui::PushID("CAMERA");
                if (isUnfolded)
                {
                    static const char* onScreenModes[] =
                    {
                #if( NRD_MODE == OCCLUSION )
                        "Diffuse occlusion",
                        "Specular occlusion",
                #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
                        "Diffuse occlusion",
                #else
                        "Final",
                        "Denoised diffuse",
                        "Denoised specular",
                        "Diffuse occlusion",
                        "Specular occlusion",
                        "Shadow",
                        "Base color",
                        "Normal",
                        "Roughness",
                        "Metalness",
                        "World units",
                        "Instance index",
                        "Mip level (primary)",
                        "Mip level (specular)",
                #endif
                    };

                    static const char* motionMode[] =
                    {
                        "Left / Right",
                        "Up / Down",
                        "Forward / Backward",
                        "Mixed",
                        "Pan",
                    };

                    static const char* mvType[] =
                    {
                        "2D",
                        "2.5D",
                        "3D",
                    };

                    ImGui::SliderFloat("FOV (deg)", &m_Settings.camFov, 1.0f, 160.0f, "%.1f");
                    ImGui::SliderFloat("Exposure", &m_Settings.exposure, 0.0f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat("Resolution scale (%)", &m_Settings.resolutionScale, m_MinResolutionScale, 1.0f, "%.3f");
                    ImGui::Combo("On screen", &m_Settings.onScreen, onScreenModes, helper::GetCountOf(onScreenModes));
                    ImGui::Checkbox("Ortho", &m_Settings.ortho);
                    ImGui::SameLine();
                    ImGui::Checkbox("FPS cap", &m_Settings.limitFps);
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, (!m_Settings.cameraJitter && (m_Settings.TAA || m_Settings.DLSS)) ? UI_RED : UI_DEFAULT);
                        ImGui::Checkbox("Jitter", &m_Settings.cameraJitter);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth( ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x );
                    ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.animatedObjects && !m_Settings.pauseAnimation && m_Settings.mvType == MV_2D) ? UI_RED : UI_DEFAULT);
                        ImGui::Combo("MV", &m_Settings.mvType, mvType, helper::GetCountOf(mvType));
                    ImGui::PopStyleColor();

                    ImGui::PushStyleColor(ImGuiCol_Text, m_Settings.motionStartTime > 0.0 ? UI_YELLOW : UI_DEFAULT);
                        bool isPressed = ImGui::Button("Animation");
                    ImGui::PopStyleColor();
                    if (isPressed)
                        m_Settings.motionStartTime = m_Settings.motionStartTime > 0.0 ? 0.0 : -1.0;
                    if (m_Settings.motionStartTime > 0.0)
                    {
                        ImGui::SameLine();
                        ImGui::Checkbox("Linear", &m_Settings.linearMotion);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth( ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x );
                        ImGui::Combo("Mode", &m_Settings.motionMode, motionMode, helper::GetCountOf(motionMode));
                        ImGui::SliderFloat("Slower / Faster", &m_Settings.emulateMotionSpeed, -10.0f, 10.0f);
                    }

                    if (m_Settings.limitFps)
                        ImGui::SliderFloat("Min / Max FPS", &m_Settings.maxFps, 30.0f, 120.0f, "%.0f");
                }
                ImGui::PopID();

                // "Antialiasing, upscaling & sharpening" section
                ImGui::NewLine();
                ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                isUnfolded = ImGui::CollapsingHeader("ANTIALIASING, UPSCALING & SHARPENING", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();

                ImGui::PushID("ANTIALIASING");
                if (isUnfolded)
                {
                    if (m_DLSS.IsInitialized())
                    {
                        ImGui::Checkbox("DLSS", &m_Settings.DLSS);
                        ImGui::SameLine();
                    }
                    if (!m_Settings.DLSS)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_ReblurSettings.enableReferenceAccumulation && m_Settings.TAA) ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Checkbox("TAA", &m_Settings.TAA);
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                    }
                    bool isNis = m_Settings.NIS && m_Settings.separator == 0.0f;
                    if (!m_Settings.DLSS)
                        ImGui::Checkbox("NIS", &m_Settings.NIS);
                    if (isNis || m_Settings.DLSS)
                    {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth( ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x );
                        ImGui::SliderFloat(m_Settings.DLSS ? "Sharpness" : "Sharpness", &m_Settings.sharpness, 0.0f, 1.0f, "%.2f");
                    }
                }
                ImGui::PopID();

                // "Materials" section
                ImGui::NewLine();
                ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                isUnfolded = ImGui::CollapsingHeader("MATERIALS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();

                ImGui::PushID("MATERIALS");
                if (isUnfolded)
                {
                    static const char* forcedMaterial[] =
                    {
                        "None",
                        "Gypsum",
                        "Cobalt",
                    };

                    ImGui::SliderFloat2("Roughness / Metalness", &m_Settings.roughnessOverride, 0.0f, 1.0f, "%.3f");
                    ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.emissiveObjects && !m_Settings.emission) ? UI_YELLOW : UI_DEFAULT);
                        ImGui::Checkbox("Emission [F3]", &m_Settings.emission);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth( ImGui::CalcItemWidth() - ImGui::GetCursorPosX() + ImGui::GetStyle().ItemSpacing.x );
                    ImGui::Combo("Material", &m_Settings.forcedMaterial, forcedMaterial, helper::GetCountOf(forcedMaterial));
                    if (m_Settings.emission)
                        ImGui::SliderFloat("Emission intensity", &m_Settings.emissionIntensity, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::PopID();

                if (m_Settings.onScreen == 10)
                    ImGui::SliderFloat("Units in 1 meter", &m_Settings.meterToUnitsMultiplier, 0.001f, 100.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
                else
                {
                    // "World" section
                    snprintf(buf, sizeof(buf) - 1, "WORLD%s", (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateCamera) ? (m_Settings.pauseAnimation ? " (SPACE - unpause)" : " (SPACE - pause)") : "");

                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader(buf, ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("WORLD");
                    if (isUnfolded)
                    {
                        ImGui::Checkbox("Animate sun", &m_Settings.animateSun);
                        if (!m_Scene.animations.empty() && m_Scene.animations[m_Settings.activeAnimation].cameraNode.animationNodeIndex != utils::InvalidIndex)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("Animate camera", &m_Settings.animateCamera);
                        }
                        if (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateCamera)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("Pause", &m_Settings.pauseAnimation);
                        }

                        ImGui::SliderFloat2("Sun position (deg)", &m_Settings.sunAzimuth, -180.0f, 180.0f, "%.2f");
                        ImGui::SliderFloat("Sun size (deg)", &m_Settings.sunAngularDiameter, 0.0f, 3.0f, "%.1f");
                        if (m_Settings.animateSun || m_Settings.animatedObjects || m_Settings.animateCamera)
                            ImGui::SliderFloat("Slower / Faster", &m_Settings.animationSpeed, -10.0f, 10.0f);

                        ImGui::Checkbox("Objects", &m_Settings.animatedObjects);
                        if (m_Settings.animatedObjects)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("9", &m_Settings.nineBrothers);
                            ImGui::SameLine();
                            ImGui::Checkbox("Blink", &m_Settings.blink);
                            ImGui::SameLine();
                            ImGui::Checkbox("Emissive", &m_Settings.emissiveObjects);
                            if (!m_Settings.nineBrothers)
                                ImGui::SliderInt("Object number", &m_Settings.animatedObjectNum, 1, (int32_t)MAX_ANIMATED_INSTANCE_NUM);
                            ImGui::SliderFloat("Object scale", &m_Settings.animatedObjectScale, 0.1f, 2.0f);
                        }

                        if (m_Settings.animateCamera && m_Scene.animations[m_Settings.activeAnimation].durationMs != 0.0f)
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
                    ImGui::PopID();

                    // "Indirect rays" section
                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("INDIRECT RAYS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("INDIRECT RAYS");
                    if (isUnfolded)
                    {
                        const float sceneRadiusInMeters = m_Scene.aabb.GetRadius() / m_Settings.meterToUnitsMultiplier;

                        static const char* resolution[] =
                        {
                            "Full",
                            "Full (probabilistic)",
                            "Half",
                        };

                    #if( NRD_MODE == NORMAL || NRD_MODE == SH )
                        ImGui::SliderInt2("Samples / Bounces", &m_Settings.rpp, 1, 8);
                    #else
                        ImGui::SliderInt("Samples", &m_Settings.rpp, 1, 8);
                    #endif
                        ImGui::SliderFloat("AO / SO range (m)", &m_Settings.hitDistScale, 0.01f, sceneRadiusInMeters, "%.2f");
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.denoiser == REFERENCE && m_Settings.tracingMode > RESOLUTION_FULL_PROBABILISTIC) ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Combo("Resolution", &m_Settings.tracingMode, resolution, helper::GetCountOf(resolution));
                        ImGui::PopStyleColor();

                        ImGui::Checkbox("Diffuse", &m_Settings.indirectDiffuse);
                        ImGui::SameLine();
                        ImGui::Checkbox("Specular", &m_Settings.indirectSpecular);
                        ImGui::SameLine();
                        ImGui::Checkbox("Trim lobe", &m_Settings.specularLobeTrimming);
                        ImGui::SameLine();
                        ImGui::Checkbox("Normal map", &m_Settings.normalMap);

                    #if( NRD_MODE == NORMAL || NRD_MODE == SH )
                        const float3& sunDirection = GetSunDirection();
                        bool cmp = sunDirection.z < 0.0f && m_Settings.importanceSampling;
                        if (cmp)
                            ImGui::PushStyleColor(ImGuiCol_Text, UI_RED);
                        ImGui::Checkbox("IS", &m_Settings.importanceSampling);
                        if (cmp)
                            ImGui::PopStyleColor();

                        ImGui::SameLine();
                        ImGui::Checkbox("Use prev frame", &m_Settings.usePrevFrame);

                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, (m_Settings.ambient && m_Settings.denoiser == RELAX) ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Checkbox("Ambient", &m_Settings.ambient);
                        ImGui::PopStyleColor();

                        if (m_Settings.tracingMode != RESOLUTION_HALF)
                        {
                            ImGui::SameLine();
                            ImGui::Checkbox("PSR", &m_Settings.PSR);
                        }
                    #else
                        if (m_Settings.tracingMode != RESOLUTION_HALF)
                            ImGui::Checkbox("PSR", &m_Settings.PSR);
                    #endif
                    }
                    ImGui::PopID();

                    // "NRD" section
                    static const char* denoiser[] =
                    {
                    #if( NRD_MODE == OCCLUSION )
                        "REBLUR_OCCLUSION",
                    #elif( NRD_MODE == SH )
                        "REBLUR_SH + SIGMA",
                    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
                        "REBLUR_DIRECTIONAL_OCCLUSION",
                    #else
                        "REBLUR + SIGMA",
                    #endif
                        "RELAX + SIGMA",
                        "REFERENCE",
                    };
                    const nrd::LibraryDesc& nrdLibraryDesc = nrd::GetLibraryDesc();
                    snprintf(buf, sizeof(buf) - 1, "NRD v%u.%u.%u - %s [PgDown / PgUp]", nrdLibraryDesc.versionMajor, nrdLibraryDesc.versionMinor, nrdLibraryDesc.versionBuild, denoiser[m_Settings.denoiser]);

                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader(buf, ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("NRD");
                    if (isUnfolded)
                    {
                        static const char* hitDistanceReconstructionMode[] =
                        {
                            "Off",
                            "3x3",
                            "5x5",
                        };

                        if (m_DebugNRD)
                        {
                            ImGui::PushStyleColor(ImGuiCol_Text, m_ShowValidationOverlay ? UI_YELLOW : UI_DEFAULT);
                            ImGui::Checkbox("Validation overlay", &m_ShowValidationOverlay);
                            ImGui::PopStyleColor();
                        }

                        if (ImGui::Button("<<"))
                        {
                            m_Settings.denoiser--;
                            if (m_Settings.denoiser < 0)
                                m_Settings.denoiser = DENOISER_MAX_NUM - 1;
                        }

                        ImGui::SameLine();
                        if (ImGui::Button(">>"))
                        {
                            m_Settings.denoiser++;
                            if (m_Settings.denoiser > DENOISER_MAX_NUM - 1)
                                m_Settings.denoiser = 0;
                        }

                        ImGui::SameLine();
                        m_ForceHistoryReset = ImGui::Button("Reset");

                        if (m_Settings.denoiser == REBLUR)
                        {
                            nrd::ReblurSettings defaults = {};

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                            {
                                defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
                                defaults.diffusePrepassBlurRadius = defaults.specularPrepassBlurRadius;
                            }

                            bool isSame = true;
                            if (m_ReblurSettings.historyFixFrameNum != defaults.historyFixFrameNum)
                                isSame = false;
                            else if (m_ReblurSettings.diffusePrepassBlurRadius != defaults.diffusePrepassBlurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.specularPrepassBlurRadius != defaults.specularPrepassBlurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.blurRadius != defaults.blurRadius)
                                isSame = false;
                            else if (m_ReblurSettings.historyFixStrideBetweenSamples != defaults.historyFixStrideBetweenSamples)
                                isSame = false;
                            else if (m_ReblurSettings.lobeAngleFraction != defaults.lobeAngleFraction)
                                isSame = false;
                            else if (m_ReblurSettings.roughnessFraction != defaults.roughnessFraction)
                                isSame = false;
                            else if (m_ReblurSettings.responsiveAccumulationRoughnessThreshold != defaults.responsiveAccumulationRoughnessThreshold)
                                isSame = false;
                            else if (m_ReblurSettings.stabilizationStrength != defaults.stabilizationStrength)
                                isSame = false;
                            else if (m_ReblurSettings.hitDistanceReconstructionMode != defaults.hitDistanceReconstructionMode)
                                isSame = false;
                            else if (m_ReblurSettings.enableAntiFirefly != defaults.enableAntiFirefly)
                                isSame = false;
                            else if (m_ReblurSettings.enableReferenceAccumulation != defaults.enableReferenceAccumulation)
                                isSame = false;
                            else if (m_ReblurSettings.enablePerformanceMode != defaults.enablePerformanceMode)
                                isSame = false;
                            else if (m_ReblurSettings.antilagHitDistanceSettings.enable != true)
                                isSame = false;
                            else if (m_ReblurSettings.antilagIntensitySettings.enable != true)
                                isSame = false;

                            ImGui::SameLine();
                            if (ImGui::Button("No spatial"))
                            {
                                m_ReblurSettings.blurRadius = 0.0f;
                                m_ReblurSettings.diffusePrepassBlurRadius = 0.0f;
                                m_ReblurSettings.specularPrepassBlurRadius = 0.0f;
                                m_ReblurSettings.antilagHitDistanceSettings.enable = false;
                                m_ReblurSettings.antilagIntensitySettings.enable = false;
                            }

                            ImGui::SameLine();
                            if (ImGui::Button(m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum ? "No fast" : "Fast"))
                            {
                                if (m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum)
                                    m_Settings.maxFastAccumulatedFrameNum = MAX_HISTORY_FRAME_NUM;
                                else
                                    m_Settings.maxFastAccumulatedFrameNum = defaults.maxFastAccumulatedFrameNum;
                            }

                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                            if (ImGui::Button("Defaults"))
                            {
                                m_ReblurSettings = defaults;
                                m_ReblurSettings.antilagIntensitySettings.enable = true;
                            }
                            ImGui::PopStyleColor();

                            ImGui::Checkbox("Adaptive radius", &m_Settings.adaptRadiusToResolution);
                            ImGui::SameLine();
                            ImGui::Checkbox("Adaptive accumulation", &m_Settings.adaptiveAccumulation);

                            ImGui::Checkbox("Ref accum", &m_ReblurSettings.enableReferenceAccumulation);
                            ImGui::SameLine();
                            ImGui::Checkbox("Anti-firefly", &m_ReblurSettings.enableAntiFirefly);
                            ImGui::SameLine();
                            ImGui::Checkbox("Perf mode", &m_ReblurSettings.enablePerformanceMode);
                        #if( NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION )
                            ImGui::SameLine();
                            ImGui::Checkbox("Resolve", &m_Resolve);
                        #endif

                            ImGui::SliderFloat("Disocclusion (%)", &m_Settings.disocclusionThreshold, 0.25f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                            ImGui::SliderInt2("History length (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d", ImGuiSliderFlags_Logarithmic);

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                            {
                                ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.hitDistanceReconstructionMode != nrd::HitDistanceReconstructionMode::OFF ? UI_GREEN : UI_RED);
                                {
                                    int32_t v = (int32_t)m_ReblurSettings.hitDistanceReconstructionMode;
                                    ImGui::Combo("HitT reconstruction", &v, hitDistanceReconstructionMode, helper::GetCountOf(hitDistanceReconstructionMode));
                                    m_ReblurSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)v;
                                }
                                ImGui::PopStyleColor();
                            }

                        #if( NRD_MODE != OCCLUSION )
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PushStyleColor(ImGuiCol_Text, m_ReblurSettings.diffusePrepassBlurRadius != 0.0f && m_ReblurSettings.specularPrepassBlurRadius != 0.0f ? UI_GREEN : UI_RED);
                            ImGui::SliderFloat2("Pre-pass radius (px)", &m_ReblurSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PopStyleColor();
                        #endif

                            ImGui::SliderFloat("Blur base radius (px)", &m_ReblurSettings.blurRadius, 0.0f, 60.0f, "%.1f");
                            ImGui::SliderFloat("Lobe fraction", &m_ReblurSettings.lobeAngleFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Roughness fraction", &m_ReblurSettings.roughnessFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("History fix stride", &m_ReblurSettings.historyFixStrideBetweenSamples, 0.0f, 20.0f, "%.1f");
                            ImGui::SliderInt("History fix frames", (int32_t*)&m_ReblurSettings.historyFixFrameNum, 0, 6);
                        #if( NRD_MODE != OCCLUSION )
                            ImGui::SliderFloat("Stabilization (%)", &m_ReblurSettings.stabilizationStrength, 0.0f, 1.0f, "%.2f");
                        #endif
                            ImGui::SetNextItemWidth( ImGui::CalcItemWidth() * 0.5f );
                            ImGui::SliderFloat("Responsive accumulation roughness threshold", &m_ReblurSettings.responsiveAccumulationRoughnessThreshold, 0.0f, 1.0f, "%.2f");

                        #if( NRD_MODE != OCCLUSION )
                            if (m_ReblurSettings.stabilizationStrength != 0.0f)
                            {
                                ImGui::Text("ANTI-LAG:");
                                ImGui::Checkbox("Intensity", &m_ReblurSettings.antilagIntensitySettings.enable);
                                ImGui::SameLine();
                                ImGui::Text("[%.1f%%; %.1f%%; %.1f]", m_ReblurSettings.antilagIntensitySettings.thresholdMin * 100.0, m_ReblurSettings.antilagIntensitySettings.thresholdMax * 100.0, m_ReblurSettings.antilagIntensitySettings.sigmaScale);

                                ImGui::SameLine();
                                ImGui::Checkbox("Hit dist", &m_ReblurSettings.antilagHitDistanceSettings.enable);
                                ImGui::SameLine();
                                ImGui::Text("[%.1f%%; %.1f%%; %.1f]", m_ReblurSettings.antilagHitDistanceSettings.thresholdMin * 100.0, m_ReblurSettings.antilagHitDistanceSettings.thresholdMax * 100.0, m_ReblurSettings.antilagHitDistanceSettings.sigmaScale);
                            }
                        #endif
                        }
                        else if (m_Settings.denoiser == RELAX)
                        {
                            nrd::RelaxDiffuseSpecularSettings defaults = {};

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                            {
                                defaults.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
                                defaults.diffusePrepassBlurRadius = defaults.specularPrepassBlurRadius;
                            }

                            bool isSame = true;
                            if (m_RelaxSettings.historyFixFrameNum != defaults.historyFixFrameNum)
                                isSame = false;
                            else if (m_RelaxSettings.diffusePrepassBlurRadius != defaults.diffusePrepassBlurRadius)
                                isSame = false;
                            else if (m_RelaxSettings.specularPrepassBlurRadius != defaults.specularPrepassBlurRadius)
                                isSame = false;
                            else if (m_RelaxSettings.diffusePhiLuminance != defaults.diffusePhiLuminance)
                                isSame = false;
                            else if (m_RelaxSettings.specularPhiLuminance != defaults.specularPhiLuminance)
                                isSame = false;
                            else if (m_RelaxSettings.diffuseLobeAngleFraction != defaults.diffuseLobeAngleFraction)
                                isSame = false;
                            else if (m_RelaxSettings.specularLobeAngleFraction != defaults.specularLobeAngleFraction)
                                isSame = false;
                            else if (m_RelaxSettings.specularVarianceBoost != defaults.specularVarianceBoost)
                                isSame = false;
                            else if (m_RelaxSettings.specularLobeAngleSlack != defaults.specularLobeAngleSlack)
                                isSame = false;
                            else if (m_RelaxSettings.historyFixEdgeStoppingNormalPower != defaults.historyFixEdgeStoppingNormalPower)
                                isSame = false;
                            else if (m_RelaxSettings.historyFixStrideBetweenSamples != defaults.historyFixStrideBetweenSamples)
                                isSame = false;
                            else if (m_RelaxSettings.historyClampingColorBoxSigmaScale != defaults.historyClampingColorBoxSigmaScale)
                                isSame = false;
                            else if (m_RelaxSettings.spatialVarianceEstimationHistoryThreshold != defaults.spatialVarianceEstimationHistoryThreshold)
                                isSame = false;
                            else if (m_RelaxSettings.atrousIterationNum != defaults.atrousIterationNum)
                                isSame = false;
                            else if (m_RelaxSettings.diffuseMinLuminanceWeight != defaults.diffuseMinLuminanceWeight)
                                isSame = false;
                            else if (m_RelaxSettings.specularMinLuminanceWeight != defaults.specularMinLuminanceWeight)
                                isSame = false;
                            else if (m_RelaxSettings.depthThreshold != defaults.depthThreshold)
                                isSame = false;
                            else if (m_RelaxSettings.roughnessFraction != defaults.roughnessFraction)
                                isSame = false;
                            else if (m_RelaxSettings.roughnessEdgeStoppingRelaxation != defaults.roughnessEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.normalEdgeStoppingRelaxation != defaults.normalEdgeStoppingRelaxation)
                                isSame = false;
                            else if (m_RelaxSettings.hitDistanceReconstructionMode != defaults.hitDistanceReconstructionMode)
                                isSame = false;
                            else if (m_RelaxSettings.enableAntiFirefly != defaults.enableAntiFirefly)
                                isSame = false;
                            else if (m_RelaxSettings.enableRoughnessEdgeStopping != defaults.enableRoughnessEdgeStopping)
                                isSame = false;

                            ImGui::SameLine();
                            if (ImGui::Button("No spatial"))
                            {
                                m_RelaxSettings.diffusePhiLuminance = 0.0f;
                                m_RelaxSettings.specularPhiLuminance = 0.0f;
                                m_RelaxSettings.diffusePrepassBlurRadius = 0.0f;
                                m_RelaxSettings.specularPrepassBlurRadius = 0.0f;
                                m_RelaxSettings.spatialVarianceEstimationHistoryThreshold = 0;
                            }

                            ImGui::SameLine();
                            if (ImGui::Button(m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum ? "No fast" : "Fast"))
                            {
                                if (m_Settings.maxFastAccumulatedFrameNum < m_Settings.maxAccumulatedFrameNum)
                                    m_Settings.maxFastAccumulatedFrameNum = MAX_HISTORY_FRAME_NUM;
                                else
                                    m_Settings.maxFastAccumulatedFrameNum = defaults.diffuseMaxFastAccumulatedFrameNum;
                            }

                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, isSame ? UI_DEFAULT : UI_YELLOW);
                            if (ImGui::Button("Defaults"))
                                m_RelaxSettings = defaults;
                            ImGui::PopStyleColor();

                            ImGui::Checkbox("Adaptive radius", &m_Settings.adaptRadiusToResolution);
                            ImGui::SameLine();
                            ImGui::Checkbox("Adaptive accumulation", &m_Settings.adaptiveAccumulation);

                            ImGui::Checkbox("Roughness edge stopping", &m_RelaxSettings.enableRoughnessEdgeStopping);
                            ImGui::SameLine();
                            ImGui::Checkbox("Anti-firefly", &m_RelaxSettings.enableAntiFirefly);
                        #if( NRD_MODE == SH)
                            ImGui::SameLine();
                            ImGui::Checkbox("Resolve", &m_Resolve);
                        #endif

                            ImGui::SliderFloat("Disocclusion (%)", &m_Settings.disocclusionThreshold, 0.25f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                            ImGui::SliderInt2("History length (frames)", &m_Settings.maxAccumulatedFrameNum, 0, MAX_HISTORY_FRAME_NUM, "%d", ImGuiSliderFlags_Logarithmic);

                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                            {
                                ImGui::PushStyleColor(ImGuiCol_Text, m_RelaxSettings.hitDistanceReconstructionMode != nrd::HitDistanceReconstructionMode::OFF ? UI_GREEN : UI_RED);
                                {
                                    int32_t v = (int32_t)m_RelaxSettings.hitDistanceReconstructionMode;
                                    ImGui::Combo("HitT reconstruction", &v, hitDistanceReconstructionMode, helper::GetCountOf(hitDistanceReconstructionMode));
                                    m_RelaxSettings.hitDistanceReconstructionMode = (nrd::HitDistanceReconstructionMode)v;
                                }
                                ImGui::PopStyleColor();
                            }

                        #if( NRD_MODE != OCCLUSION )
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PushStyleColor(ImGuiCol_Text, m_RelaxSettings.diffusePrepassBlurRadius != 0.0f && m_RelaxSettings.specularPrepassBlurRadius != 0.0f ? UI_GREEN : UI_RED);
                            ImGui::SliderFloat2("Pre-pass radius (px)", &m_RelaxSettings.diffusePrepassBlurRadius, 0.0f, 75.0f, "%.1f");
                            if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
                                ImGui::PopStyleColor();
                        #endif

                            ImGui::SliderInt("A-trous iterations", (int32_t*)&m_RelaxSettings.atrousIterationNum, 2, 8);
                            ImGui::SliderFloat2("Diff-Spec luma weight", &m_RelaxSettings.diffusePhiLuminance, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderFloat2("Min luma weight", &m_RelaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Spec lobe angle slack", &m_RelaxSettings.specularLobeAngleSlack, 0.0f, 89.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                            ImGui::SliderFloat("Depth threshold", &m_RelaxSettings.depthThreshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                            ImGui::Text("Diffuse lobe / Specular lobe / Roughness:");
                            ImGui::SliderFloat3("Fraction", &m_RelaxSettings.diffuseLobeAngleFraction, 0.0f, 1.0f, "%.2f");
                            ImGui::Text("Luminance / Normal / Roughness:");
                            ImGui::SliderFloat3("Relaxation", &m_RelaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f, "%.2f");
                            ImGui::SliderFloat("Spec variance boost", &m_RelaxSettings.specularVarianceBoost, 0.0f, 8.0f, "%.2f");
                            ImGui::SliderFloat("Clamping sigma scale", &m_RelaxSettings.historyClampingColorBoxSigmaScale, 0.0f, 10.0f, "%.1f");
                            ImGui::SliderInt("History threshold", (int32_t*)&m_RelaxSettings.spatialVarianceEstimationHistoryThreshold, 0, 10);

                            ImGui::Text("HISTORY FIX:");
                            ImGui::SliderFloat("Normal weight power", &m_RelaxSettings.historyFixEdgeStoppingNormalPower, 0.0f, 128.0f, "%.1f");
                            ImGui::SliderFloat("Stride", &m_RelaxSettings.historyFixStrideBetweenSamples, 0.0f, 20.0f, "%.1f");
                            ImGui::SliderInt("Frames", (int32_t*)&m_RelaxSettings.historyFixFrameNum, 0, 6);
                        }
                        else if (m_Settings.denoiser == REFERENCE)
                        {
                            float t = (float)m_ReferenceSettings.maxAccumulatedFrameNum;
                            ImGui::SliderFloat("History length (frames)", &t, 0.0f, 1024.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                            m_ReferenceSettings.maxAccumulatedFrameNum = (int32_t)t;
                        }
                    }
                    ImGui::PopID();

                    // "Other" section
                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("OTHER", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("OTHER");
                    if (isUnfolded)
                    {
                        ImGui::SliderFloat("Debug [F1]", &m_Settings.debug, 0.0f, 1.0f, "%.6f");
                        ImGui::SliderFloat("Input / Denoised", &m_Settings.separator, 0.0f, 1.0f, "%.2f");

                        if (ImGui::Button(m_Settings.windowAlignment ? ">>" : "<<"))
                            m_Settings.windowAlignment = !m_Settings.windowAlignment;

                        ImGui::SameLine();
                        if (ImGui::Button("Reload shaders"))
                        {
                            CreatePipelines();
                            printf("Ready!\n");
                        }

                        ImGui::SameLine();
                        if (ImGui::Button("Defaults"))
                        {
                            m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
                            m_Settings = m_DefaultSettings;
                            m_RelaxSettings = {};
                            m_ReblurSettings = {};
                            m_ReblurSettings.antilagIntensitySettings.enable = true;
                            m_ForceHistoryReset = true;
                        }
                    }
                    ImGui::PopID();

                    // "Tests" section
                    ImGui::NewLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
                    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
                    isUnfolded = ImGui::CollapsingHeader("TESTS [F2]", ImGuiTreeNodeFlags_CollapsingHeader);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleColor();

                    ImGui::PushID("TESTS");
                    if (isUnfolded)
                    {
                        float buttonWidth = 25.0f * float(GetWindowResolution().x) / float(GetOutputResolution().x);

                        char s[64];
                        std::string sceneName = std::string( utils::GetFileName(m_SceneFile) );
                        size_t dotPos = sceneName.find_last_of(".");
                        if (dotPos != std::string::npos)
                            sceneName = sceneName.substr(0, dotPos) + ".bin";
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
                        if (IsKeyToggled(Key::F2) && m_TestNum)
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

                            bool isColorChanged = false;
                            if(m_improveMeTests && std::find(m_improveMeTests->begin(), m_improveMeTests->end(), i + 1) != m_improveMeTests->end())
                            {
                                ImGui::PushStyleColor(ImGuiCol_Text, UI_RED);
                                isColorChanged = true;
                            }
                            else if(m_checkMeTests && std::find(m_checkMeTests->begin(), m_checkMeTests->end(), i + 1) != m_checkMeTests->end())
                            {
                                ImGui::PushStyleColor(ImGuiCol_Text, UI_YELLOW);
                                isColorChanged = true;
                            }

                            if (ImGui::Button(i == m_LastSelectedTest ? "*" : s, ImVec2(buttonWidth, 0.0f)) || isTestChanged)
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
                                    m_Settings.DLSS = m_DefaultSettings.DLSS;
                                    m_ForceHistoryReset = true;
                                }

                                if (fp)
                                    fclose(fp);

                                isTestChanged = false;
                            }

                            if (isColorChanged)
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
                    ImGui::PopID();
                }
            }
            m_UiWidth = ImGui::GetWindowWidth();
        }
        ImGui::End();
    }

    // Animate scene and update camera
    cBoxf cameraLimits = m_Scene.aabb;
    cameraLimits.Scale(2.0f);

    CameraDesc desc = {};
    desc.limits = cameraLimits;
    desc.aspectRatio = float(GetOutputResolution().x ) / float(GetOutputResolution().y );
    desc.horizontalFov = RadToDeg( Atan( Tan( DegToRad( m_Settings.camFov ) * 0.5f ) *  desc.aspectRatio * 9.0f / 16.0f ) * 2.0f ); // recalculate to ultra-wide if needed
    desc.nearZ = NEAR_Z * m_Settings.meterToUnitsMultiplier;
    desc.farZ = 10000.0f * m_Settings.meterToUnitsMultiplier;
    desc.isCustomMatrixSet = m_Settings.animateCamera;
    desc.isLeftHanded = CAMERA_LEFT_HANDED;
    desc.orthoRange = m_Settings.ortho ? Tan( DegToRad( m_Settings.camFov ) * 0.5f ) * 3.0f * m_Settings.meterToUnitsMultiplier : 0.0f;
    GetCameraDescFromInputDevices(desc);

    const float animationSpeed = m_Settings.pauseAnimation ? 0.0f : (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
    const float scale = m_Settings.animatedObjectScale * m_Settings.meterToUnitsMultiplier / 2.0f;
    const float animationDelta = animationSpeed * m_Timer.GetFrameTime() * 0.001f;

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

        if (m_Settings.motionMode == 4)
        {
            float3 axisX = m_Camera.state.mWorldToView.GetRow0().To3d();
            float3 axisY = m_Camera.state.mWorldToView.GetRow1().To3d();
            float2 v = Rotate(float2(1.0f, 0.0f), Mod(Pi(period * 2.0f), Pi(2.0f)));
            localPos = (axisX * v.x + axisY * v.y) * amplitude / Pi(1.0f);
        }
        else
            localPos *= amplitude * (m_Settings.linearMotion ? WaveTriangle(period) - 0.5f : Sin( Pi(period) ) * 0.5f);

        desc.dUser = localPos - m_PrevLocalPos;
        m_PrevLocalPos = localPos;
    }
    else if (m_Settings.motionStartTime == -1.0)
    {
        m_Settings.motionStartTime = m_Timer.GetTimeStamp();
        m_PrevLocalPos = float3::Zero();
    }

    if (!m_Scene.animations.empty())
        m_Scene.Animate(animationSpeed, m_Timer.GetFrameTime(), m_Settings.animationProgress, m_Settings.activeAnimation, m_Settings.animateCamera ? &desc.customMatrix : nullptr);

    m_Camera.Update(desc, frameIndex);

    // Animate sun
    if (m_Settings.animateSun)
    {
        m_Settings.sunElevation += animationDelta * 10.0f;
        if (m_Settings.sunElevation > 180.0f)
            m_Settings.sunElevation -= 360.0f;
    }

    // Animate objects
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
            float3 position;
            float4x4 transform = m_AnimatedInstances[i].Animate(animationDelta, scale, position);

            utils::Instance& instance = m_Scene.instances[ m_AnimatedInstances[i].instanceID ];
            instance.rotation = transform;
            instance.position = ToDouble(position);
        }
    }

    // Adjust settings if tracing mode has been changed to / from "probabilistic sampling"
    if (m_Settings.tracingMode != m_PrevSettings.tracingMode && (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC || m_PrevSettings.tracingMode == RESOLUTION_FULL_PROBABILISTIC))
    {
        nrd::ReblurSettings reblurDefaults = {};
        nrd::ReblurSettings relaxDefaults = {};

        if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
        {
            m_ReblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
            m_ReblurSettings.diffusePrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;
            m_ReblurSettings.specularPrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;

            m_RelaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
            m_RelaxSettings.diffusePrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
            m_RelaxSettings.specularPrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
        }
        else
        {
            m_ReblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
            m_ReblurSettings.diffusePrepassBlurRadius = reblurDefaults.diffusePrepassBlurRadius;
            m_ReblurSettings.specularPrepassBlurRadius = reblurDefaults.specularPrepassBlurRadius;

            m_RelaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
            m_RelaxSettings.diffusePrepassBlurRadius = relaxDefaults.diffusePrepassBlurRadius;
            m_RelaxSettings.specularPrepassBlurRadius = relaxDefaults.specularPrepassBlurRadius;
        }
    }

    // Print out information
    if (m_PrevSettings.resolutionScale != m_Settings.resolutionScale ||
        m_PrevSettings.tracingMode != m_Settings.tracingMode ||
        m_PrevSettings.rpp != m_Settings.rpp ||
        frameIndex == 0)
    {
        std::array<uint32_t, 4> rppScale = {2, 1, 2, 2};
        std::array<float, 4> wScale = {1.0f, 1.0f, 0.5f, 0.5f};
        std::array<float, 4> hScale = {1.0f, 1.0f, 1.0f, 0.5f};

        uint32_t pw = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
        uint32_t ph = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
        uint32_t iw = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale * wScale[m_Settings.tracingMode] + 0.5f);
        uint32_t ih = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale * hScale[m_Settings.tracingMode] + 0.5f);
        uint32_t rayNum = m_Settings.rpp * rppScale[m_Settings.tracingMode];
        float rpp = float( iw  * ih * rayNum ) / float( pw * ph );

        printf
        (
            "Output          : %ux%u\n"
            "  Primary rays  : %ux%u\n"
            "  Indirect rays : %ux%u x %u ray(s)\n"
            "  Indirect rpp  : %.2f\n",
            GetOutputResolution().x, GetOutputResolution().y,
            pw, ph,
            iw, ih, rayNum,
            rpp
        );
    }

    if (m_PrevSettings.denoiser != m_Settings.denoiser || frameIndex == 0)
    {
        m_checkMeTests = nullptr;
        m_improveMeTests = nullptr;

        if (m_SceneFile.find("BistroInterior") != std::string::npos)
        {
            m_checkMeTests = &interior_checkMeTests;
            if (m_Settings.denoiser == REBLUR)
                m_improveMeTests = &REBLUR_interior_improveMeTests;
            else if (m_Settings.denoiser == RELAX)
                m_improveMeTests = &RELAX_interior_improveMeTests;
        }
    }
}

void Sample::LoadScene()
{
    // Proxy geometry, which will be instancinated
    std::string sceneFile = utils::GetFullPath("Cubes/Cubes.obj", utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE( utils::LoadScene(sceneFile, m_Scene, !ALLOW_BLAS_MERGING) );

    m_ProxyInstancesNum = helper::GetCountOf(m_Scene.instances);

    // The scene
    sceneFile = utils::GetFullPath(m_SceneFile, utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE( utils::LoadScene(sceneFile, m_Scene, !ALLOW_BLAS_MERGING) );

    // Some scene dependent settings // TODO: improve
    m_ReblurSettings.antilagIntensitySettings.enable = true;

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

void Sample::AddInnerGlassSurfaces()
{
    // IMPORTANT: this is only valid for non-merged instances, when each instance represents a single object

    size_t instanceNum = m_Scene.instances.size();
    for (size_t i = 0; i < instanceNum; i++)
    {
        const utils::Instance& instance = m_Scene.instances[i];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];

        if (!material.IsTransparent())
            continue;

        const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];
        float3 size = mesh.aabb.vMax - mesh.aabb.vMin;
        size *= instance.rotation.GetScale();

        float minSize = Min(size.x, Min(size.y, size.z));
        if (minSize < GLASS_THICKNESS * 2.0f)
            continue;

        utils::Instance innerInstance = instance;
        innerInstance.scale = (size - GLASS_THICKNESS) / (size + 1e-15f);

        m_Scene.instances.push_back(innerInstance);
    }
}

void Sample::GenerateAnimatedCubes()
{
    for (uint32_t i = 0; i < MAX_ANIMATED_INSTANCE_NUM; i++)
    {
        float3 position = Lerp(m_Scene.aabb.vMin, m_Scene.aabb.vMax, Rand::uf3(&m_FastRandState));

        AnimatedInstance animatedInstance = {};
        animatedInstance.instanceID = helper::GetCountOf(m_Scene.instances);
        animatedInstance.basePosition = position;
        animatedInstance.durationSec = Rand::uf1(&m_FastRandState) * 10.0f + 5.0f;
        animatedInstance.progressedSec = animatedInstance.durationSec * Rand::uf1(&m_FastRandState);
        animatedInstance.rotationAxis = Normalize( Rand::sf3(&m_FastRandState) );
        animatedInstance.elipseAxis = Rand::sf3(&m_FastRandState) * 5.0f;
        animatedInstance.inverseDirection = Sign( Rand::sf1(&m_FastRandState) );
        animatedInstance.inverseRotation = Sign( Rand::sf1(&m_FastRandState) );
        m_AnimatedInstances.push_back(animatedInstance);

        utils::Instance instance = m_Scene.instances[i % m_ProxyInstancesNum];
        instance.allowUpdate = true;

        m_Scene.instances.push_back(instance);
    }
}

nri::Format Sample::CreateSwapChain()
{
    nri::SwapChainDesc swapChainDesc = {};
    swapChainDesc.windowSystemType = GetWindowSystemType();
    swapChainDesc.window = GetWindow();
    swapChainDesc.commandQueue = m_CommandQueue;
    swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
    swapChainDesc.verticalSyncInterval = m_VsyncInterval;
    swapChainDesc.width = (uint16_t)GetWindowResolution().x;
    swapChainDesc.height = (uint16_t)GetWindowResolution().y;
    swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;

    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

    nri::Format swapChainFormat = nri::Format::UNKNOWN;
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

        char name[32];
        snprintf(name, sizeof(name), "Texture::SwapChain#%u", i);
        NRI.SetTextureDebugName(*backBuffer.texture, name);

        nri::Texture2DViewDesc textureViewDesc = {backBuffer.texture, nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, backBuffer.colorAttachment));

        frameBufferDesc.colorAttachments = &backBuffer.colorAttachment;
        NRI_ABORT_ON_FAILURE(NRI.CreateFrameBuffer(*m_Device, frameBufferDesc, backBuffer.frameBufferUI));
    }

    return swapChainFormat;
}

void Sample::CreateCommandBuffers()
{
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, frame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer));
    }
}

void Sample::CreatePipelineLayoutAndDescriptorPool()
{
    const nri::DescriptorRangeDesc descriptorRanges0[] =
    {
        { 0, 1, nri::DescriptorType::CONSTANT_BUFFER, nri::ShaderStage::COMPUTE },
        { 0, 4, nri::DescriptorType::SAMPLER, nri::ShaderStage::COMPUTE },
    };

    const nri::DescriptorRangeDesc descriptorRanges1[] =
    {
        { 0, 10, nri::DescriptorType::TEXTURE, nri::ShaderStage::COMPUTE },
        { 0, 10, nri::DescriptorType::STORAGE_TEXTURE, nri::ShaderStage::COMPUTE },
    };

    const uint32_t textureNum = helper::GetCountOf(m_Scene.materials) * TEXTURES_PER_MATERIAL;
    nri::DescriptorRangeDesc descriptorRanges2[] =
    {
        { 0, 2, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::ShaderStage::COMPUTE },
        { 2, 2, nri::DescriptorType::STRUCTURED_BUFFER, nri::ShaderStage::COMPUTE },
        { 4, textureNum, nri::DescriptorType::TEXTURE, nri::ShaderStage::COMPUTE, nri::VARIABLE_DESCRIPTOR_NUM, nri::DESCRIPTOR_ARRAY },
    };

    const nri::DescriptorSetDesc descriptorSetDesc[] =
    {
        { 0, descriptorRanges0, helper::GetCountOf(descriptorRanges0) },
        { 1, descriptorRanges1, helper::GetCountOf(descriptorRanges1), nullptr, 0, nri::DescriptorSetBindingBits::PARTIALLY_BOUND },
        { 2, descriptorRanges2, helper::GetCountOf(descriptorRanges2) },
    };

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = descriptorSetDesc;
    pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDesc);
    pipelineLayoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;

    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

    nri::DescriptorPoolDesc descriptorPoolDesc = {};

    descriptorPoolDesc.descriptorSetMaxNum += BUFFERED_FRAME_MAX_NUM;
    descriptorPoolDesc.constantBufferMaxNum += descriptorRanges0[0].descriptorNum * BUFFERED_FRAME_MAX_NUM;
    descriptorPoolDesc.samplerMaxNum += descriptorRanges0[1].descriptorNum * BUFFERED_FRAME_MAX_NUM;

    descriptorPoolDesc.descriptorSetMaxNum += uint32_t(DescriptorSet::MAX_NUM);
    descriptorPoolDesc.textureMaxNum += descriptorRanges1[0].descriptorNum * uint32_t(DescriptorSet::MAX_NUM);
    descriptorPoolDesc.storageTextureMaxNum += descriptorRanges1[1].descriptorNum * uint32_t(DescriptorSet::MAX_NUM);

    descriptorPoolDesc.descriptorSetMaxNum += 1;
    descriptorPoolDesc.accelerationStructureMaxNum += descriptorRanges2[0].descriptorNum;
    descriptorPoolDesc.structuredBufferMaxNum += descriptorRanges2[1].descriptorNum;
    descriptorPoolDesc.textureMaxNum += descriptorRanges2[2].descriptorNum;

    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
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

    nri::ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = m_PipelineLayout;

    nri::Pipeline* pipeline = nullptr;
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    { // Pipeline::AmbientRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "AmbientRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::PrimaryRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "PrimaryRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::DirectLighting
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "DirectLighting.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::IndirectRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "IndirectRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Composition
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "Composition.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::DeltaRays
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "DeltaRays.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Temporal
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "Temporal.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::Upsample
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "Upsample.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::UpsampleNis
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "UpsampleNis.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::PreDlss
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "PreDlss.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }

    { // Pipeline::AfterDlss
        pipelineDesc.computeShader = utils::LoadShader(deviceDesc.graphicsAPI, "AfterDlss.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, pipelineDesc, pipeline));
        m_Pipelines.push_back(pipeline);
    }
}

void Sample::CreateAccelerationStructures()
{
    double stamp1 = m_Timer.GetTimeStamp();

    struct Parameters
    {
        nri::AccelerationStructure* accelerationStructure;
        uint64_t scratchOffset;
        uint32_t geometryObjectBase;
        uint32_t geometryObjectsNum;
    };

    uint64_t primitivesNum = 0;
    std::vector<Parameters> parameters;
    std::vector<nri::GeometryObject> geometryObjects;

    geometryObjects.reserve(m_Scene.instances.size()); // reallocation is NOT allowed!

    // Calculate temp memory size
    std::vector<uint32_t> dynamicMeshesIndices;
    uint64_t uploadSize = 0;
    uint64_t geometryOffset = 0;

    for (size_t i = m_ProxyInstancesNum; i < m_Scene.instances.size(); i++)
    {
        const utils::Instance& instance = m_Scene.instances[i];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];

        if (material.IsOff())
            continue;

        if (instance.allowUpdate)
        {
            if (std::find(dynamicMeshesIndices.begin(), dynamicMeshesIndices.end(), instance.meshIndex) != dynamicMeshesIndices.end())
                continue;
            else
                dynamicMeshesIndices.push_back(instance.meshIndex);
        }

        const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];
        uint64_t vertexDataSize = mesh.vertexNum * sizeof(float[3]);
        uint64_t indexDataSize = helper::Align(mesh.indexNum * sizeof(utils::Index), 4);
        uint64_t transformDataSize = instance.allowUpdate ? 0 : sizeof(float[12]);

        if (material.IsEmissive())
        {
            // Emissive meshes apper twice: in BLAS_StaticOpaque and in BLAS_StaticEmissive
            vertexDataSize *= 2;
            indexDataSize *= 2;
            transformDataSize *= 2;
        }

        uploadSize += vertexDataSize + indexDataSize + transformDataSize;
        geometryOffset += transformDataSize;
    }

    // Create temp buffer in UPLOAD heap
    nri::Buffer* uploadBuffer = nullptr;
    nri::Memory* uploadMemory = nullptr;
    {
        const nri::BufferDesc bufferDesc = {uploadSize, 0, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ};
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, uploadBuffer));

        nri::MemoryDesc memoryDesc = {};
        NRI.GetBufferMemoryInfo(*uploadBuffer, nri::MemoryLocation::HOST_UPLOAD, memoryDesc);

        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, uploadMemory));

        const nri::BufferMemoryBindingDesc memoryBindingDesc = {uploadMemory, uploadBuffer};
        NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &memoryBindingDesc, 1));
    }

    uint8_t* uploadData = (uint8_t*)NRI.MapBuffer(*uploadBuffer, 0, nri::WHOLE_SIZE);

    { // AccelerationStructure::TLAS_World
        nri::AccelerationStructureDesc accelerationStructureDesc = {};
        accelerationStructureDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
        accelerationStructureDesc.flags = TLAS_BUILD_BITS;
        accelerationStructureDesc.instanceOrGeometryObjectNum = helper::GetCountOf(m_Scene.instances);

        nri::AccelerationStructure* accelerationStructure = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureDesc, accelerationStructure));
        m_AccelerationStructures.push_back(accelerationStructure);

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*accelerationStructure, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {memory, accelerationStructure};
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        // Descriptor::World_AccelerationStructure
        nri::Descriptor* descriptor = nullptr;
        NRI.CreateAccelerationStructureDescriptor(*accelerationStructure, 0, descriptor);
        m_Descriptors.push_back(descriptor);
    }

    { // AccelerationStructure::TLAS_Emissive
        nri::AccelerationStructureDesc accelerationStructureDesc = {};
        accelerationStructureDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
        accelerationStructureDesc.flags = TLAS_BUILD_BITS;
        accelerationStructureDesc.instanceOrGeometryObjectNum = helper::GetCountOf(m_Scene.instances);

        nri::AccelerationStructure* accelerationStructure = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureDesc, accelerationStructure));
        m_AccelerationStructures.push_back(accelerationStructure);

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*accelerationStructure, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {memory, accelerationStructure};
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        // Descriptor::Light_AccelerationStructure
        nri::Descriptor* descriptor = nullptr;
        NRI.CreateAccelerationStructureDescriptor(*accelerationStructure, 0, descriptor);
        m_Descriptors.push_back(descriptor);
    }

    // Create BOTTOM_LEVEL acceleration structures for static geometry
    uint64_t scratchSize = 0;

    for (uint32_t mode = (uint32_t)AccelerationStructure::BLAS_StaticOpaque; mode <= (uint32_t)AccelerationStructure::BLAS_StaticEmissive; mode++)
    {
        size_t geometryObjectBase = geometryObjects.size();

        for (size_t i = m_ProxyInstancesNum; i < m_Scene.instances.size(); i++)
        {
            const utils::Instance& instance = m_Scene.instances[i];
            const utils::Material& material = m_Scene.materials[instance.materialIndex];

            if (material.IsOff())
                continue;

            if (instance.allowUpdate)
                continue;

            if (mode == (uint32_t)AccelerationStructure::BLAS_StaticOpaque)
            {
                if (material.IsTransparent())
                    continue;

                m_OpaqueObjectsNum++;
            }
            else if (mode == (uint32_t)AccelerationStructure::BLAS_StaticTransparent)
            {
                if (!material.IsTransparent())
                    continue;

                m_TransparentObjectsNum++;
            }
            else if (mode == (uint32_t)AccelerationStructure::BLAS_StaticEmissive)
            {
                if (!material.IsEmissive())
                    continue;

                m_EmissiveObjectsNum++;
            }

            const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];

            // Copy geometry to temp buffer
            uint64_t vertexDataSize = mesh.vertexNum * sizeof(float[3]);
            uint8_t* p = uploadData + geometryOffset;
            for (uint32_t v = 0; v < mesh.vertexNum; v++)
            {
                memcpy(p, m_Scene.vertices[mesh.vertexOffset + v].position, sizeof(float[3]));
                p += sizeof(float[3]);
            }

            uint64_t indexDataSize = mesh.indexNum * sizeof(utils::Index);
            memcpy(p, &m_Scene.indices[mesh.indexOffset], indexDataSize);

            // Copy transform to temp buffer
            float4x4 mObjectToWorld = instance.rotation;
            if (instance.scale != float3(1.0f))
            {
                float4x4 translation;
                translation.SetupByTranslation( ToFloat(instance.position) - mesh.aabb.GetCenter() );

                float4x4 translationInv = translation;
                translationInv.InvertOrtho();

                float4x4 scale;
                scale.SetupByScale(instance.scale);

                mObjectToWorld = mObjectToWorld * translationInv * scale * translation;
            }
            mObjectToWorld.AddTranslation( ToFloat(instance.position) );

            mObjectToWorld.Transpose3x4();

            uint64_t transformOffset = geometryObjects.size() * sizeof(float[12]);
            memcpy(uploadData + transformOffset, mObjectToWorld.a16, sizeof(float[12]));

            // Add geometry object
            nri::GeometryObject& geometryObject = geometryObjects.emplace_back();
            geometryObject = {};
            geometryObject.type = nri::GeometryType::TRIANGLES;
            geometryObject.flags = material.IsAlphaOpaque() ? nri::BottomLevelGeometryBits::NONE : nri::BottomLevelGeometryBits::OPAQUE_GEOMETRY;
            geometryObject.triangles.vertexBuffer = uploadBuffer;
            geometryObject.triangles.vertexOffset = geometryOffset;
            geometryObject.triangles.vertexNum = mesh.vertexNum;
            geometryObject.triangles.vertexStride = sizeof(float[3]);
            geometryObject.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
            geometryObject.triangles.indexBuffer = uploadBuffer;
            geometryObject.triangles.indexOffset = geometryOffset + vertexDataSize;
            geometryObject.triangles.indexNum = mesh.indexNum;
            geometryObject.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;
            geometryObject.triangles.transformBuffer = uploadBuffer;
            geometryObject.triangles.transformOffset = transformOffset;

            // Update geometry offset
            geometryOffset += vertexDataSize + helper::Align(indexDataSize, 4);
            primitivesNum += mesh.indexNum / 3;
        }

        uint32_t geometryObjectsNum = (uint32_t)(geometryObjects.size() - geometryObjectBase);
        if (geometryObjectsNum)
        {
            // Create BLAS
            nri::AccelerationStructureDesc accelerationStructureDesc = {};
            accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
            accelerationStructureDesc.flags = BLAS_BUILD_BITS;
            accelerationStructureDesc.instanceOrGeometryObjectNum = geometryObjectsNum;
            accelerationStructureDesc.geometryObjects = &geometryObjects[geometryObjectBase];

            nri::AccelerationStructure* accelerationStructure = nullptr;
            NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureDesc, accelerationStructure));
            m_AccelerationStructures.push_back(accelerationStructure);

            nri::MemoryDesc memoryDesc = {};
            NRI.GetAccelerationStructureMemoryInfo(*accelerationStructure, memoryDesc);

            nri::Memory* memory = nullptr;
            NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
            m_MemoryAllocations.push_back(memory);

            const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {memory, accelerationStructure};
            NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

            // Update parameters
            parameters.push_back( {accelerationStructure, scratchSize, (uint32_t)geometryObjectBase, geometryObjectsNum} );

            uint64_t size = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
            scratchSize += helper::Align(size, 256);
        }
        else
        {
            // Needed only to preserve order
            m_AccelerationStructures.push_back(nullptr);
            m_MemoryAllocations.push_back(nullptr);
        }
    }

    // Create BOTTOM_LEVEL acceleration structures for dynamic geometry
    for (uint32_t dynamicMeshIndex : dynamicMeshesIndices)
    {
        utils::Mesh& mesh = m_Scene.meshes[dynamicMeshIndex];
        mesh.blasIndex = (uint32_t)m_AccelerationStructures.size();

        // Copy geometry to temp buffer
        uint64_t vertexDataSize = mesh.vertexNum * sizeof(float[3]);
        uint8_t* p = uploadData + geometryOffset;
        for (uint32_t v = 0; v < mesh.vertexNum; v++)
        {
            memcpy(p, m_Scene.vertices[mesh.vertexOffset + v].position, sizeof(float[3]));
            p += sizeof(float[3]);
        }

        uint64_t indexDataSize = mesh.indexNum * sizeof(utils::Index);
        memcpy(p, &m_Scene.indices[mesh.indexOffset], indexDataSize);

        // Add geometry object
        nri::GeometryObject& geometryObject = geometryObjects.emplace_back();
        geometryObject = {};
        geometryObject.type = nri::GeometryType::TRIANGLES;
        geometryObject.flags = nri::BottomLevelGeometryBits::NONE; // will be set in TLAS instance
        geometryObject.triangles.vertexBuffer = uploadBuffer;
        geometryObject.triangles.vertexOffset = geometryOffset;
        geometryObject.triangles.vertexNum = mesh.vertexNum;
        geometryObject.triangles.vertexStride = sizeof(float[3]);
        geometryObject.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
        geometryObject.triangles.indexBuffer = uploadBuffer;
        geometryObject.triangles.indexOffset = geometryOffset + vertexDataSize;
        geometryObject.triangles.indexNum = mesh.indexNum;
        geometryObject.triangles.indexType = sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32;

        // Create BLAS
        nri::AccelerationStructureDesc accelerationStructureDesc = {};
        accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
        accelerationStructureDesc.flags = BLAS_BUILD_BITS;
        accelerationStructureDesc.instanceOrGeometryObjectNum = 1;
        accelerationStructureDesc.geometryObjects = &geometryObject;

        nri::AccelerationStructure* accelerationStructure = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureDesc, accelerationStructure));
        m_AccelerationStructures.push_back(accelerationStructure);

        nri::MemoryDesc memoryDesc = {};
        NRI.GetAccelerationStructureMemoryInfo(*accelerationStructure, memoryDesc);

        nri::Memory* memory = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, memory));
        m_MemoryAllocations.push_back(memory);

        const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {memory, accelerationStructure};
        NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

        // Update parameters
        parameters.push_back( {accelerationStructure, scratchSize, (uint32_t)(geometryObjects.size() - 1), 1} );

        uint64_t size = NRI.GetAccelerationStructureBuildScratchBufferSize(*accelerationStructure);
        scratchSize += helper::Align(size, 256);

        // Update geometry offset
        geometryOffset += vertexDataSize + helper::Align(indexDataSize, 4);
        primitivesNum += mesh.indexNum / 3;
    }

    // Allocate scratch memory
    const nri::BufferDesc bufferDesc = {scratchSize, 0, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE};
    nri::Buffer* scratchBuffer = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, scratchBuffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryInfo(*scratchBuffer, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::Memory* scratchMemory = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, nri::WHOLE_DEVICE_GROUP, memoryDesc.type, memoryDesc.size, scratchMemory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = { scratchMemory, scratchBuffer };
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));

    // Create command allocator and command buffer
    nri::CommandAllocator* commandAllocator = nullptr;
    NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, commandAllocator);

    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    double stamp2 = m_Timer.GetTimeStamp();

    // Write and execute commands
    NRI.BeginCommandBuffer(*commandBuffer, nullptr, 0);
    {
        for (const Parameters& params : parameters)
            NRI.CmdBuildBottomLevelAccelerationStructure(*commandBuffer, params.geometryObjectsNum, &geometryObjects[params.geometryObjectBase], BLAS_BUILD_BITS, *params.accelerationStructure, *scratchBuffer, params.scratchOffset);
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {&commandBuffer, 1};
    NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);

    // Wait idle
    NRI.WaitForIdle(*m_CommandQueue);

    double buildTime = m_Timer.GetTimeStamp() - stamp2;

    // Cleanup
    NRI.UnmapBuffer(*uploadBuffer);

    NRI.DestroyBuffer(*scratchBuffer);
    NRI.FreeMemory(*scratchMemory);

    NRI.DestroyBuffer(*uploadBuffer);
    NRI.FreeMemory(*uploadMemory);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    double totalTime = m_Timer.GetTimeStamp() - stamp1;

    printf(
        "Scene stats:\n"
        "  Instances     : %zu\n"
        "  Meshes        : %zu\n"
        "  Vertices      : %zu\n"
        "  Primitives    : %zu\n"
        "BVH stats:\n"
        "  Total time    : %.2f ms\n"
        "  Building time : %.2f ms\n"
        "  Scratch size  : %.2f Mb\n"
        "  BLAS num      : %zu\n"
        "  Geometries    : %zu\n"
        "  Primitives    : %zu\n"
        , m_Scene.instances.size()
        , m_Scene.meshes.size()
        , m_Scene.primitives.size()
        , m_Scene.vertices.size()
        , totalTime
        , buildTime
        , scratchSize / (1024.0 * 1024.0)
        , m_AccelerationStructures.size() - (size_t)AccelerationStructure::BLAS_StaticOpaque
        , geometryObjects.size()
        , primitivesNum
   );
}

void Sample::CreateSamplers()
{
    nri::Descriptor* descriptor = nullptr;

    { // Descriptor::LinearMipmapLinear_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDesc.minification = nri::Filter::LINEAR;
        samplerDesc.magnification = nri::Filter::LINEAR;
        samplerDesc.mip = nri::Filter::LINEAR;
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE( NRI.CreateSampler(*m_Device, samplerDesc, descriptor) );
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::LinearMipmapNearest_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDesc.minification = nri::Filter::LINEAR;
        samplerDesc.magnification = nri::Filter::LINEAR;
        samplerDesc.mip = nri::Filter::NEAREST;
        samplerDesc.mipMax = 16.0f;

        NRI_ABORT_ON_FAILURE( NRI.CreateSampler(*m_Device, samplerDesc, descriptor) );
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::Linear_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
        samplerDesc.minification = nri::Filter::LINEAR;
        samplerDesc.magnification = nri::Filter::LINEAR;

        NRI_ABORT_ON_FAILURE( NRI.CreateSampler(*m_Device, samplerDesc, descriptor) );
        m_Descriptors.push_back(descriptor);
    }

    { // Descriptor::Nearest_Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::CLAMP_TO_EDGE};
        samplerDesc.minification = nri::Filter::NEAREST;
        samplerDesc.magnification = nri::Filter::NEAREST;

        NRI_ABORT_ON_FAILURE( NRI.CreateSampler(*m_Device, samplerDesc, descriptor) );
        m_Descriptors.push_back(descriptor);
    }
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
        default:                                    return format;
    }
}

void Sample::CreateResources(nri::Format swapChainFormat)
{
    // TODO: DLSS doesn't support R16 UNORM/SNORM
#if( NRD_MODE == OCCLUSION )
    nri::Format dataFormat = m_DlssQuality != -1 ? nri::Format::R16_SFLOAT : nri::Format::R16_UNORM;
    nri::Format dlssDataFormat = nri::Format::R16_SFLOAT;
#elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
    nri::Format dataFormat = m_DlssQuality != -1 ? nri::Format::RGBA16_SFLOAT : nri::Format::RGBA16_SNORM;
    nri::Format dlssDataFormat = nri::Format::R16_SFLOAT;
#else
    nri::Format dataFormat = nri::Format::RGBA16_SFLOAT;
    nri::Format dlssDataFormat = nri::Format::R11_G11_B10_UFLOAT;
#endif

#if( NRD_NORMAL_ENCODING == 0 )
    nri::Format normalFormat = nri::Format::RGBA8_UNORM;
#elif( NRD_NORMAL_ENCODING == 1 )
    nri::Format normalFormat = nri::Format::RGBA8_SNORM;
#elif( NRD_NORMAL_ENCODING == 2 )
    nri::Format normalFormat = nri::Format::R10_G10_B10_A2_UNORM;
#elif( NRD_NORMAL_ENCODING == 3 )
    nri::Format normalFormat = nri::Format::RGBA16_UNORM;
#elif( NRD_NORMAL_ENCODING == 4 )
    nri::Format normalFormat = nri::Format::RGBA16_SNORM;
#endif

    const uint16_t w = (uint16_t)m_RenderResolution.x;
    const uint16_t h = (uint16_t)m_RenderResolution.y;
    const uint64_t instanceDataSize = m_Scene.instances.size() * sizeof(InstanceData);
    const uint64_t worldScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*Get(AccelerationStructure::TLAS_World));
    const uint64_t lightScratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(*Get(AccelerationStructure::TLAS_Emissive));

    std::vector<DescriptorDesc> descriptorDescs;

    // Buffers (HOST_UPLOAD)
    CreateBuffer(descriptorDescs, "Buffer::GlobalConstants", m_ConstantBufferSize * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::CONSTANT_BUFFER);
    CreateBuffer(descriptorDescs, "Buffer::InstanceDataStaging", instanceDataSize * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::NONE);
    CreateBuffer(descriptorDescs, "Buffer::WorldTlasDataStaging", m_Scene.instances.size() * sizeof(nri::GeometryObjectInstance) * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ);
    CreateBuffer(descriptorDescs, "Buffer::LightTlasDataStaging", m_Scene.instances.size() * sizeof(nri::GeometryObjectInstance) * BUFFERED_FRAME_MAX_NUM, 1, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ);

    // Buffers (DEVICE)
    CreateBuffer(descriptorDescs, "Buffer::PrimitiveData", m_Scene.primitives.size(), sizeof(PrimitiveData), nri::BufferUsageBits::SHADER_RESOURCE, nri::Format::UNKNOWN);
    CreateBuffer(descriptorDescs, "Buffer::InstanceData", instanceDataSize / sizeof(InstanceData), sizeof(InstanceData), nri::BufferUsageBits::SHADER_RESOURCE, nri::Format::UNKNOWN);
    CreateBuffer(descriptorDescs, "Buffer::WorldScratch", worldScratchBufferSize, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);
    CreateBuffer(descriptorDescs, "Buffer::LightScratch", lightScratchBufferSize, 1, nri::BufferUsageBits::RAY_TRACING_BUFFER | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE);

    // Textures (DEVICE)
    CreateTexture(descriptorDescs, "Texture::Ambient", nri::Format::RGBA16_SFLOAT, 2, 2, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ViewZ", nri::Format::R32_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Motion", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Normal_Roughness", normalFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::PrimaryMipAndCurvature", nri::Format::RG8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::BaseColor_Metalness", nri::Format::RGBA8_SRGB, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectLighting", nri::Format::R11_G11_B10_UFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DirectEmission", nri::Format::R11_G11_B10_UFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Shadow", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_ShadowData", nri::Format::RG16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Diff", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Spec", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_Shadow_Translucency", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Validation", nri::Format::RGBA8_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Composed_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::DlssOutput", dlssDataFormat, (uint16_t)GetOutputResolution().x, (uint16_t)GetOutputResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::Final", swapChainFormat, (uint16_t)GetWindowResolution().x, (uint16_t)GetWindowResolution().y, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::COPY_SOURCE);

    CreateTexture(descriptorDescs, "Texture::ComposedDiff_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::ComposedSpec_ViewZ", nri::Format::RGBA16_SFLOAT, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    CreateTexture(descriptorDescs, "Texture::TaaHistory", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::TaaHistoryPrev", nri::Format::R10_G10_B10_A2_UNORM, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE_STORAGE);

#if( NRD_MODE == SH )
    CreateTexture(descriptorDescs, "Texture::Unfiltered_DiffSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::Unfiltered_SpecSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::DiffSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
    CreateTexture(descriptorDescs, "Texture::SpecSh", dataFormat, w, h, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE, nri::AccessBits::SHADER_RESOURCE);
#endif

    CreateTexture(descriptorDescs, "Texture::NisData1", nri::Format::RGBA16_SFLOAT, kFilterSize / 4, kPhaseCount, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);
    CreateTexture(descriptorDescs, "Texture::NisData2", nri::Format::RGBA16_SFLOAT, kFilterSize / 4, kPhaseCount, 1, 1,
        nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);

    for (const utils::Texture* textureData : m_Scene.textures)
        CreateTexture(descriptorDescs, "", textureData->GetFormat(), textureData->GetWidth(), textureData->GetHeight(), textureData->GetMipNum(), textureData->GetArraySize(), nri::TextureUsageBits::SHADER_RESOURCE, nri::AccessBits::UNKNOWN);

    constexpr uint32_t offset = uint32_t(Buffer::UploadHeapBufferNum);

    // Bind memory
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

    // Create descriptors
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

void Sample::CreateDescriptorSets()
{
    nri::DescriptorSet* descriptorSet = nullptr;

    // Global constant buffer & samplers
    const nri::Descriptor* samplers[] =
    {
        Get(Descriptor::LinearMipmapLinear_Sampler),
        Get(Descriptor::LinearMipmapNearest_Sampler),
        Get(Descriptor::Linear_Sampler),
        Get(Descriptor::Nearest_Sampler),
    };

    for (Frame& frame : m_Frames)
    {
        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { &frame.globalConstantBufferDescriptor, 1 },
            { samplers, helper::GetCountOf(samplers) },
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &frame.globalConstantBufferDescriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));

        NRI.UpdateDescriptorRanges(*frame.globalConstantBufferDescriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::AmbientRays1
        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Ambient_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 1, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::PrimaryRays1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::ScramblingRanking1spp)),
            Get(Descriptor((uint32_t)Descriptor::MaterialTextures + utils::StaticTexture::SobolSequence)),
            Get(Descriptor::Ambient_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Mv_StorageTexture),
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
            Get(Descriptor::BaseColor_Metalness_StorageTexture),
            Get(Descriptor::PrimaryMipAndCurvature_StorageTexture),
            Get(Descriptor::DirectLighting_StorageTexture),
            Get(Descriptor::DirectEmission_StorageTexture),
            Get(Descriptor::Unfiltered_ShadowData_StorageTexture),
            Get(Descriptor::Unfiltered_Shadow_Translucency_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::DirectLighting1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::DirectEmission_Texture),
            Get(Descriptor::Shadow_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::DirectLighting_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::IndirectRays1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::PrimaryMipAndCurvature_Texture),
            Get(Descriptor::ComposedDiff_ViewZ_Texture),
            Get(Descriptor::ComposedSpec_ViewZ_Texture),
            Get(Descriptor::Ambient_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::DirectLighting_StorageTexture),
            Get(Descriptor::BaseColor_Metalness_StorageTexture),
            Get(Descriptor::Normal_Roughness_StorageTexture),
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Mv_StorageTexture),
            Get(Descriptor::Unfiltered_Diff_StorageTexture),
            Get(Descriptor::Unfiltered_Spec_StorageTexture),
#if( NRD_MODE == SH )
            Get(Descriptor::Unfiltered_DiffSh_StorageTexture),
            Get(Descriptor::Unfiltered_SpecSh_StorageTexture),
#endif
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Composition1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::ViewZ_Texture),
            Get(Descriptor::Normal_Roughness_Texture),
            Get(Descriptor::BaseColor_Metalness_Texture),
            Get(Descriptor::DirectLighting_Texture),
            Get(Descriptor::Ambient_Texture),
            Get(Descriptor::Diff_Texture),
            Get(Descriptor::Spec_Texture),
#if( NRD_MODE == SH )
            Get(Descriptor::DiffSh_Texture),
            Get(Descriptor::SpecSh_Texture),
#endif
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::ComposedDiff_ViewZ_StorageTexture),
            Get(Descriptor::ComposedSpec_ViewZ_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::DeltaRays1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::ViewZ_Texture),
            Get(Descriptor::ComposedDiff_ViewZ_Texture),
            Get(Descriptor::ComposedSpec_ViewZ_Texture),
            Get(Descriptor::Ambient_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Composed_ViewZ_StorageTexture),
            Get(Descriptor::Mv_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Temporal1a
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_ViewZ_Texture),
            Get(Descriptor::TaaHistoryPrev_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::TaaHistory_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Temporal1b
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_ViewZ_Texture),
            Get(Descriptor::TaaHistory_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::TaaHistoryPrev_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Upsample1a
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistory_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::Upsample1b
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistoryPrev_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::UpsampleNis1a
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistory_Texture),
            Get(Descriptor::NisData1),
            Get(Descriptor::NisData2),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::UpsampleNis1b
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::TaaHistoryPrev_Texture),
            Get(Descriptor::NisData1),
            Get(Descriptor::NisData2),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::PreDlss1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::Mv_Texture),
            Get(Descriptor::Composed_ViewZ_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::ViewZ_StorageTexture),
            Get(Descriptor::Unfiltered_ShadowData_StorageTexture),
            Get(Descriptor::DlssInput_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::AfterDlss1
        const nri::Descriptor* textures[] =
        {
            Get(Descriptor::DlssOutput_Texture),
            Get(Descriptor::Validation_Texture),
        };

        const nri::Descriptor* storageTextures[] =
        {
            Get(Descriptor::Final_StorageTexture),
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { textures, helper::GetCountOf(textures) },
            { storageTextures, helper::GetCountOf(storageTextures) },
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    { // DescriptorSet::RayTracing2
        std::vector<nri::Descriptor*> textures(m_Scene.materials.size() * TEXTURES_PER_MATERIAL);
        for (size_t i = 0; i < m_Scene.materials.size(); i++)
        {
            const size_t index = i * TEXTURES_PER_MATERIAL;
            const utils::Material& material = m_Scene.materials[i];

            textures[index] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.diffuseMapIndex) );
            textures[index + 1] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.specularMapIndex) );
            textures[index + 2] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.normalMapIndex) );
            textures[index + 3] = Get( Descriptor((uint32_t)Descriptor::MaterialTextures + material.emissiveMapIndex) );
        }

        const nri::Descriptor* structuredBuffers[] =
        {
            Get(Descriptor::InstanceData_Buffer),
            Get(Descriptor::PrimitiveData_Buffer)
        };

        const nri::Descriptor* accelerationStructures[] =
        {
            Get(Descriptor::World_AccelerationStructure),
            Get(Descriptor::Light_AccelerationStructure)
        };

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 2, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, helper::GetCountOf(textures)));
        m_DescriptorSets.push_back(descriptorSet);

        const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] =
        {
            { accelerationStructures, helper::GetCountOf(accelerationStructures) },
            { structuredBuffers, helper::GetCountOf(structuredBuffers) },
            { textures.data(), helper::GetCountOf(textures) }
        };

        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }
}

void Sample::CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint16_t width, uint16_t height, uint16_t mipNum, uint16_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state)
{
    const nri::TextureDesc textureDesc = nri::Texture2D(format, width, height, mipNum, arraySize, usage);

    nri::Texture* texture = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, texture));
    m_Textures.push_back(texture);

    if (state != nri::AccessBits::UNKNOWN)
    {
        nri::TextureTransitionBarrierDesc transition = nri::TextureTransitionFromUnknown(texture, state, state == nri::AccessBits::SHADER_RESOURCE ? nri::TextureLayout::SHADER_RESOURCE : nri::TextureLayout::GENERAL);
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

void Sample::UploadStaticData()
{
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

            float2 t0 = Packed::EncodeUnitVector( float3(v0.tangent), true );
            float2 t1 = Packed::EncodeUnitVector( float3(v1.tangent), true );
            float2 t2 = Packed::EncodeUnitVector( float3(v2.tangent), true );

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
            data.curvature = primitive.curvature;
        }
    }

    // Gather subresources for read-only textures
    std::vector<nri::TextureSubresourceUploadDesc> subresources;
    subresources.push_back( {coef_scale_fp16, 1, (kFilterSize / 4) * 8, (kFilterSize / 4) * kPhaseCount * 8} );
    subresources.push_back( {coef_usm_fp16, 1, (kFilterSize / 4) * 8, (kFilterSize / 4) * kPhaseCount * 8} );
    for (const utils::Texture* texture : m_Scene.textures)
    {
        for (uint32_t layer = 0; layer < texture->GetArraySize(); layer++)
        {
            for (uint32_t mip = 0; mip < texture->GetMipNum(); mip++)
            {
                nri::TextureSubresourceUploadDesc subresource;
                texture->GetSubresource(subresource, mip, layer);

                subresources.push_back(subresource);
            }
        }
    }

    // Gather upload data for read-only textures
    std::vector<nri::TextureUploadDesc> textureData;
    textureData.push_back( {&subresources[0], Get(Texture::NisData1), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE, 1, 1} );
    textureData.push_back( {&subresources[1], Get(Texture::NisData2), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE, 1, 1} );
    size_t subresourceOffset = 2;

    for (size_t i = 0; i < m_Scene.textures.size(); i++)
    {
        const utils::Texture* texture = m_Scene.textures[i];
        uint16_t mipNum = texture->GetMipNum();
        uint16_t arraySize = texture->GetArraySize();

        textureData.push_back( {&subresources[subresourceOffset], Get( (Texture)((size_t)Texture::MaterialTextures + i) ), nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE, mipNum, arraySize} );

        subresourceOffset += size_t(arraySize) * size_t(mipNum);
    }

    // Append textures without data to initialize initial state
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

    // Upload data and apply states
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_CommandQueue, textureData.data(), helper::GetCountOf(textureData), dataDescArray, helper::GetCountOf(dataDescArray)));
}

void Sample::BuildTopLevelAccelerationStructure(nri::CommandBuffer& commandBuffer, uint32_t bufferedFrameIndex)
{
    bool isAnimatedObjects = m_Settings.animatedObjects;
    if (m_Settings.blink)
    {
        double period = 0.0003 * m_Timer.GetTimeStamp() * (m_Settings.animationSpeed < 0.0f ? 1.0f / (1.0f + Abs(m_Settings.animationSpeed)) : (1.0f + m_Settings.animationSpeed));
        isAnimatedObjects &= WaveTriangle(period) > 0.5;
    }

    uint64_t tlasCount = m_Scene.instances.size();
    uint64_t tlasDataSize = tlasCount * sizeof(nri::GeometryObjectInstance);
    uint64_t tlasDataOffset = tlasDataSize * bufferedFrameIndex;
    uint64_t instanceDataSize = tlasCount * sizeof(InstanceData);
    uint64_t instanceDataOffset = instanceDataSize * bufferedFrameIndex;
    uint64_t staticInstanceCount = m_Scene.instances.size() - m_AnimatedInstances.size();
    uint64_t instanceCount = staticInstanceCount + (isAnimatedObjects ? m_Settings.animatedObjectNum : 0);

    auto instanceData = (InstanceData*)NRI.MapBuffer(*Get(Buffer::InstanceDataStaging), instanceDataOffset, instanceDataSize);
    auto worldTlasData = (nri::GeometryObjectInstance*)NRI.MapBuffer(*Get(Buffer::WorldTlasDataStaging), tlasDataOffset, tlasDataSize);
    auto lightTlasData = (nri::GeometryObjectInstance*)NRI.MapBuffer(*Get(Buffer::LightTlasDataStaging), tlasDataOffset, tlasDataSize);

    uint32_t instanceIndex = 0;
    uint32_t worldGeometryObjectsNum = 0;
    uint32_t lightGeometryObjectsNum = 0;

    float4x4 mCameraTranslation = float4x4::Identity();
    mCameraTranslation.AddTranslation( m_Camera.GetRelative(double3::Zero()) );
    mCameraTranslation.Transpose3x4();

    // Add static opaque (includes emissives)
    if (m_OpaqueObjectsNum)
    {
        uint32_t flags = FLAG_DEFAULT;
        uint32_t instanceIdAndFlags = instanceIndex | (flags << FLAG_FIRST_BIT);

        nri::GeometryObjectInstance geometryObjectInstance = {};
        memcpy(geometryObjectInstance.transform, mCameraTranslation.a16, sizeof(geometryObjectInstance.transform));
        geometryObjectInstance.instanceId = instanceIdAndFlags;
        geometryObjectInstance.mask = flags;
        geometryObjectInstance.shaderBindingTableLocalOffset = 0;
        geometryObjectInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE;
        geometryObjectInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*Get(AccelerationStructure::BLAS_StaticOpaque), 0);

        *worldTlasData++ = geometryObjectInstance;
        instanceIndex += m_OpaqueObjectsNum;
        worldGeometryObjectsNum++;
    }

    // Add static transparent
    if (m_TransparentObjectsNum)
    {
        uint32_t flags = FLAG_TRANSPARENT;
        uint32_t instanceIdAndFlags = instanceIndex | (flags << FLAG_FIRST_BIT);

        nri::GeometryObjectInstance geometryObjectInstance = {};
        memcpy(geometryObjectInstance.transform, mCameraTranslation.a16, sizeof(geometryObjectInstance.transform));
        geometryObjectInstance.instanceId = instanceIdAndFlags;
        geometryObjectInstance.mask = flags;
        geometryObjectInstance.shaderBindingTableLocalOffset = 0;
        geometryObjectInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE;
        geometryObjectInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*Get(AccelerationStructure::BLAS_StaticTransparent), 0);

        *worldTlasData++ = geometryObjectInstance;
        instanceIndex += m_TransparentObjectsNum;
        worldGeometryObjectsNum++;

        m_HasTransparent = m_TransparentObjectsNum ? true : false;
    }

    // Add static emissives (only emissives in a separate TLAS)
    if (m_EmissiveObjectsNum)
    {
        uint32_t flags = FLAG_DEFAULT;
        uint32_t instanceIdAndFlags = instanceIndex | (flags << FLAG_FIRST_BIT);

        nri::GeometryObjectInstance geometryObjectInstance = {};
        memcpy(geometryObjectInstance.transform, mCameraTranslation.a16, sizeof(geometryObjectInstance.transform));
        geometryObjectInstance.instanceId = instanceIdAndFlags;
        geometryObjectInstance.mask = flags;
        geometryObjectInstance.shaderBindingTableLocalOffset = 0;
        geometryObjectInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE;
        geometryObjectInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*Get(AccelerationStructure::BLAS_StaticEmissive), 0);

        *lightTlasData++ = geometryObjectInstance;
        instanceIndex += m_EmissiveObjectsNum;
        lightGeometryObjectsNum++;
    }

    // Gather instance data and add dynamic objects
    // IMPORTANT: instance data order must match geometry layout in BLAS-es
    for (uint32_t mode = (uint32_t)AccelerationStructure::BLAS_StaticOpaque; mode <= (uint32_t)AccelerationStructure::BLAS_Other; mode++)
    {
        for (size_t i = m_ProxyInstancesNum; i < instanceCount; i++)
        {
            utils::Instance& instance = m_Scene.instances[i];
            const utils::Material& material = m_Scene.materials[instance.materialIndex];

            if (material.IsOff())
                continue;

            if (mode == (uint32_t)AccelerationStructure::BLAS_StaticOpaque)
            {
                if (instance.allowUpdate || material.IsTransparent())
                    continue;
            }
            else if (mode == (uint32_t)AccelerationStructure::BLAS_StaticTransparent)
            {
                if (instance.allowUpdate || !material.IsTransparent())
                    continue;
            }
            else if (mode == (uint32_t)AccelerationStructure::BLAS_StaticEmissive)
            {
                if (instance.allowUpdate || !material.IsEmissive())
                    continue;
            }
            else
            {
                if (!instance.allowUpdate)
                    continue;
            }

            assert(instanceIndex <= INSTANCE_ID_MASK);

            float4x4 mObjectToWorld = float4x4::Identity();
            float4x4 mWorldToWorldPrev = float4x4::Identity();
            bool isLeftHanded = false;

            if (instance.allowUpdate)
            {
                // Current matrix
                mObjectToWorld = instance.rotation;
                if (instance.scale != float3(1.0f))
                {
                    const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];

                    float4x4 translation;
                    translation.SetupByTranslation( ToFloat(instance.position) - mesh.aabb.GetCenter() );

                    float4x4 translationInv = translation;
                    translationInv.InvertOrtho();

                    float4x4 scale;
                    scale.SetupByScale(instance.scale);

                    mObjectToWorld = mObjectToWorld * translationInv * scale * translation;
                }
                mObjectToWorld.AddTranslation( m_Camera.GetRelative(instance.position) );

                // Previous transform
                float4x4 mObjectToWorldPrev = instance.rotationPrev;
                mObjectToWorldPrev.AddTranslation( m_Camera.GetRelative(instance.positionPrev) );

                double4x4 dmWorldToObject = ToDouble(mObjectToWorld); // TODO: FP64 used to avoid imprecision problems on close up views (InvertOrtho can't be used due to scaling factors)
                dmWorldToObject.Invert();

                double4x4 dmObjectToWorldPrev = ToDouble(mObjectToWorldPrev);
                mWorldToWorldPrev = ToFloat(dmObjectToWorldPrev * dmWorldToObject);

                // Update previous state
                instance.positionPrev = instance.position;
                instance.rotationPrev = instance.rotation;
            }
            else
            {
                mObjectToWorld = mCameraTranslation;

                // Static geometry doesn't have "prev" transformation, reuse this matrix to pass object rotation needed for normals
                mWorldToWorldPrev = instance.rotation;

                // Transform can be left-handed (mirroring), in this case normals need flipping
                isLeftHanded = instance.rotation.IsLeftHanded();
            }

            mObjectToWorld.Transpose3x4();
            mWorldToWorldPrev.Transpose3x4();

            // Add instance data
            const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];
            uint32_t basePrimitiveIndex = mesh.indexOffset / 3;

            instanceData->baseTextureIndex = instance.materialIndex * TEXTURES_PER_MATERIAL;
            instanceData->basePrimitiveIndex = basePrimitiveIndex;
            instanceData->isStatic = instance.allowUpdate ? 0 : 1;
            instanceData->invScale = (isLeftHanded ? -1.0f : 1.0f) / Length( instance.rotation.GetScale() );
            instanceData->mWorldToWorldPrev0 = mWorldToWorldPrev.col0;
            instanceData->mWorldToWorldPrev1 = mWorldToWorldPrev.col1;
            instanceData->mWorldToWorldPrev2 = mWorldToWorldPrev.col2;
            instanceData++;

            // Add dynamic geometry
            if (instance.allowUpdate)
            {
                uint32_t flags = FLAG_DEFAULT;
                if (material.IsTransparent())
                {
                    flags = FLAG_TRANSPARENT;
                    m_HasTransparent = true;
                }
                else if (m_Settings.emission && m_Settings.emissiveObjects && i > staticInstanceCount && (i % 3 == 0))
                    flags = FLAG_FORCED_EMISSION;

                uint32_t instanceIdAndFlags = instanceIndex | (flags << FLAG_FIRST_BIT);
                instanceIndex++;

                nri::GeometryObjectInstance geometryObjectInstance = {};
                memcpy(geometryObjectInstance.transform, mObjectToWorld.a16, sizeof(geometryObjectInstance.transform));
                geometryObjectInstance.instanceId = instanceIdAndFlags;
                geometryObjectInstance.mask = flags;
                geometryObjectInstance.shaderBindingTableLocalOffset = 0;
                geometryObjectInstance.flags = nri::TopLevelInstanceBits::TRIANGLE_CULL_DISABLE | (material.IsAlphaOpaque() ? nri::TopLevelInstanceBits::NONE : nri::TopLevelInstanceBits::FORCE_OPAQUE);
                geometryObjectInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*m_AccelerationStructures[mesh.blasIndex], 0);

                *worldTlasData++ = geometryObjectInstance;
                worldGeometryObjectsNum++;

                if (flags == FLAG_FORCED_EMISSION || material.IsEmissive())
                {
                    *lightTlasData++ = geometryObjectInstance;
                    lightGeometryObjectsNum++;
                }
            }
        }
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
    NRI.CmdBuildTopLevelAccelerationStructure(commandBuffer, worldGeometryObjectsNum, *Get(Buffer::WorldTlasDataStaging), tlasDataOffset, TLAS_BUILD_BITS, *Get(AccelerationStructure::TLAS_World), *Get(Buffer::WorldScratch), 0);
    NRI.CmdBuildTopLevelAccelerationStructure(commandBuffer, lightGeometryObjectsNum, *Get(Buffer::LightTlasDataStaging), tlasDataOffset, TLAS_BUILD_BITS, *Get(AccelerationStructure::TLAS_Emissive), *Get(Buffer::LightScratch), 0);
}

void Sample::UpdateConstantBuffer(uint32_t frameIndex, float globalResetFactor)
{
    // Ambient accumulation
    const float maxSeconds = 0.5f;
    float maxAccumFrameNum = maxSeconds * 1000.0f / m_Timer.GetSmoothedFrameTime();
    m_AmbientAccumFrameNum = (m_AmbientAccumFrameNum + 1.0f) * globalResetFactor;
    m_AmbientAccumFrameNum = Min(m_AmbientAccumFrameNum, maxAccumFrameNum);

    const float3& sunDirection = GetSunDirection();

    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectWprev = uint32_t(m_RenderResolution.x * m_PrevSettings.resolutionScale + 0.5f);
    uint32_t rectHprev = uint32_t(m_RenderResolution.y * m_PrevSettings.resolutionScale + 0.5f);

    float emissionIntensity = m_Settings.emissionIntensity * float(m_Settings.emission);
    float baseMipBias = ((m_Settings.TAA || m_Settings.DLSS) ? -1.0f : 0.0f) + log2f(m_Settings.resolutionScale);

    float2 renderSize = float2(float(m_RenderResolution.x), float(m_RenderResolution.y));
    float2 outputSize = float2(float(GetOutputResolution().x), float(GetOutputResolution().y));
    float2 windowSize = float2(float(GetWindowResolution().x), float(GetWindowResolution().y));
    float2 rectSize = float2( float(rectW), float(rectH) );
    float2 rectSizePrev = float2( float(rectWprev), float(rectHprev) );
    float2 jitter = (m_Settings.cameraJitter ? m_Camera.state.viewportJitter : 0.0f) / rectSize;

    float3 viewDir = float3(m_Camera.state.mViewToWorld.GetCol2().xmm) * (CAMERA_LEFT_HANDED ? -1.0f : 1.0f);

    nrd::HitDistanceParameters hitDistanceParameters = {};
    hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;

    float minProbability = 0.0f;
    if (m_Settings.tracingMode == RESOLUTION_FULL_PROBABILISTIC)
    {
        nrd::HitDistanceReconstructionMode mode = nrd::HitDistanceReconstructionMode::OFF;
        if (m_Settings.denoiser == REBLUR)
            mode = m_ReblurSettings.hitDistanceReconstructionMode;
        else if (m_Settings.denoiser == RELAX)
            mode = m_RelaxSettings.hitDistanceReconstructionMode;

        // Min / max allowed probability to guarantee a sample in 3x3 or 5x5 area - https://godbolt.org/z/YGYo1rjnM
        if (mode == nrd::HitDistanceReconstructionMode::AREA_3X3)
            minProbability = 1.0f / 4.0f;
        else if (mode == nrd::HitDistanceReconstructionMode::AREA_5X5)
            minProbability = 1.0f / 16.0f;
    }

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const uint64_t rangeOffset = m_Frames[bufferedFrameIndex].globalConstantBufferOffset;
    nri::Buffer* globalConstants = Get(Buffer::GlobalConstants);
    auto data = (GlobalConstantBufferData*)NRI.MapBuffer(*globalConstants, rangeOffset, sizeof(GlobalConstantBufferData));
    {
        data->gViewToWorld                                  = m_Camera.state.mViewToWorld;
        data->gViewToClip                                   = m_Camera.state.mViewToClip;
        data->gWorldToView                                  = m_Camera.state.mWorldToView;
        data->gWorldToViewPrev                              = m_Camera.statePrev.mWorldToView;
        data->gWorldToClip                                  = m_Camera.state.mWorldToClip;
        data->gWorldToClipPrev                              = m_Camera.statePrev.mWorldToClip;
        data->gHitDistParams                                = float4(hitDistanceParameters.A, hitDistanceParameters.B, hitDistanceParameters.C, hitDistanceParameters.D);
        data->gCameraFrustum                                = m_Camera.state.frustum;
        data->gSunDirection_gExposure                       = sunDirection;
        data->gSunDirection_gExposure.w                     = m_Settings.exposure;
        data->gCameraOrigin_gMipBias                        = m_Camera.state.position;
        data->gCameraOrigin_gMipBias.w                      = baseMipBias + log2f(renderSize.x / outputSize.x);
        data->gTrimmingParams_gEmissionIntensity            = GetSpecularLobeTrimming();
        data->gTrimmingParams_gEmissionIntensity.w          = emissionIntensity;
        data->gViewDirection_gIsOrtho                       = float4( viewDir.x, viewDir.y, viewDir.z, m_Camera.m_IsOrtho );
        data->gWindowSize                                   = windowSize;
        data->gInvWindowSize                                = float2(1.0f, 1.0f) / windowSize;
        data->gOutputSize                                   = outputSize;
        data->gInvOutputSize                                = float2(1.0f, 1.0f) / outputSize;
        data->gRenderSize                                   = renderSize;
        data->gInvRenderSize                                = float2(1.0f, 1.0f) / renderSize;
        data->gRectSize                                     = rectSize;
        data->gInvRectSize                                  = float2(1.0f, 1.0f) / rectSize;
        data->gRectSizePrev                                 = rectSizePrev;
        data->gJitter                                       = jitter;
        data->gNearZ                                        = (CAMERA_LEFT_HANDED ? 1.0f : -1.0f) * NEAR_Z * m_Settings.meterToUnitsMultiplier;
        data->gAmbientAccumSpeed                            = 1.0f / (1.0f + m_AmbientAccumFrameNum);
        data->gAmbient                                      = m_Settings.ambient ? 1.0f : 0.0f;
        data->gSeparator                                    = m_Settings.separator;
        data->gRoughnessOverride                            = m_Settings.roughnessOverride;
        data->gMetalnessOverride                            = m_Settings.metalnessOverride;
        data->gUnitToMetersMultiplier                       = 1.0f / m_Settings.meterToUnitsMultiplier;
        data->gIndirectDiffuse                              = m_Settings.indirectDiffuse ? 1.0f : 0.0f;
        data->gIndirectSpecular                             = m_Settings.indirectSpecular ? 1.0f : 0.0f;
        data->gTanSunAngularRadius                          = Tan( DegToRad( m_Settings.sunAngularDiameter * 0.5f ) );
        data->gTanPixelAngularRadius                        = Tan( 0.5f * DegToRad(m_Settings.camFov) / outputSize.x );
        data->gDebug                                        = m_Settings.debug;
        data->gTransparent                                  = (m_HasTransparent && NRD_MODE != OCCLUSION && NRD_MODE != DIRECTIONAL_OCCLUSION && ( m_Settings.onScreen == 0 || m_Settings.onScreen == 2 )) ? 1.0f : 0.0f;
        data->gReference                                    = m_Settings.denoiser == REFERENCE ? 1.0f : 0.0f;
        data->gUsePrevFrame                                 = m_Settings.usePrevFrame;
        data->gMinProbability                               = minProbability;
        data->gDenoiserType                                 = (uint32_t)m_Settings.denoiser;
        data->gDisableShadowsAndEnableImportanceSampling    = (sunDirection.z < 0.0f && m_Settings.importanceSampling) ? 1 : 0;
        data->gOnScreen                                     = m_Settings.onScreen + ((NRD_MODE == OCCLUSION || NRD_MODE == DIRECTIONAL_OCCLUSION) ? 3 : 0); // preserve original mapping
        data->gFrameIndex                                   = frameIndex;
        data->gForcedMaterial                               = m_Settings.forcedMaterial;
        data->gUseNormalMap                                 = m_Settings.normalMap ? 1 : 0;
        data->gIsWorldSpaceMotionEnabled                    = m_Settings.mvType == MV_3D ? 1 : 0;
        data->gTracingMode                                  = m_Settings.tracingMode;
        data->gSampleNum                                    = m_Settings.rpp;
        data->gBounceNum                                    = m_Settings.bounceNum;
        data->gTAA                                          = (m_Settings.denoiser != REFERENCE && m_Settings.TAA) ? 1 : 0;
        data->gResolve                                      = m_Settings.denoiser == REFERENCE ? false : m_Resolve;
        data->gPSR                                          = m_Settings.PSR && m_Settings.tracingMode != RESOLUTION_HALF;
        data->gValidation                                   = m_DebugNRD && m_ShowValidationOverlay && m_Settings.denoiser != REFERENCE && m_Settings.separator != 1.0f;

        // NIS
        NISConfig config = {};
        NVScalerUpdateConfig
        (
            config, m_Settings.sharpness + Lerp( (1.0f - m_Settings.sharpness) * 0.25f, 0.0f, (m_Settings.resolutionScale - 0.5f) * 2.0f ),
            0, 0, rectW, rectH, m_RenderResolution.x, m_RenderResolution.y,
            0, 0, GetWindowResolution().x, GetWindowResolution().y, GetWindowResolution().x, GetWindowResolution().y,
            NISHDRMode::None
        );

        data->gNisDetectRatio                               = config.kDetectRatio;
        data->gNisDetectThres                               = config.kDetectThres;
        data->gNisMinContrastRatio                          = config.kMinContrastRatio;
        data->gNisRatioNorm                                 = config.kRatioNorm;
        data->gNisContrastBoost                             = config.kContrastBoost;
        data->gNisEps                                       = config.kEps;
        data->gNisSharpStartY                               = config.kSharpStartY;
        data->gNisSharpScaleY                               = config.kSharpScaleY;
        data->gNisSharpStrengthMin                          = config.kSharpStrengthMin;
        data->gNisSharpStrengthScale                        = config.kSharpStrengthScale;
        data->gNisSharpLimitMin                             = config.kSharpLimitMin;
        data->gNisSharpLimitScale                           = config.kSharpLimitScale;
        data->gNisScaleX                                    = config.kScaleX;
        data->gNisScaleY                                    = config.kScaleY;
        data->gNisDstNormX                                  = config.kDstNormX;
        data->gNisDstNormY                                  = config.kDstNormY;
        data->gNisSrcNormX                                  = config.kSrcNormX;
        data->gNisSrcNormY                                  = config.kSrcNormY;
        data->gNisInputViewportOriginX                      = config.kInputViewportOriginX;
        data->gNisInputViewportOriginY                      = config.kInputViewportOriginY;
        data->gNisInputViewportWidth                        = config.kInputViewportWidth;
        data->gNisInputViewportHeight                       = config.kInputViewportHeight;
        data->gNisOutputViewportOriginX                     = config.kOutputViewportOriginX;
        data->gNisOutputViewportOriginY                     = config.kOutputViewportOriginY;
        data->gNisOutputViewportWidth                       = config.kOutputViewportWidth;
        data->gNisOutputViewportHeight                      = config.kOutputViewportHeight;
    }
    NRI.UnmapBuffer(*globalConstants);
}

uint32_t Sample::BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureTransitionBarrierDesc, MAX_TEXTURE_TRANSITION_NUM>& transitions)
{
    uint32_t n = 0;

    for (uint32_t i = 0; i < stateNum; i++)
    {
        const TextureState& state = states[i];
        nri::TextureTransitionBarrierDesc& transition = GetState(state.texture);

        bool isStateChanged = transition.nextAccess != state.nextAccess || transition.nextLayout != state.nextLayout;
        bool isStorageBarrier = transition.nextAccess == nri::AccessBits::SHADER_RESOURCE_STORAGE && state.nextAccess == nri::AccessBits::SHADER_RESOURCE_STORAGE;
        if (isStateChanged || isStorageBarrier)
            transitions[n++] = nri::TextureTransitionFromState(transition, state.nextAccess, state.nextLayout);
    }

    return n;
}

void Sample::RestoreBindings(nri::CommandBuffer& commandBuffer, const Frame& frame)
{
    NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
    NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
    NRI.CmdSetDescriptorSet(commandBuffer, 0, *frame.globalConstantBufferDescriptorSet, nullptr);
    NRI.CmdSetDescriptorSet(commandBuffer, 2, *Get(DescriptorSet::RayTracing2), nullptr);
}

void Sample::RenderFrame(uint32_t frameIndex)
{
    std::array<nri::TextureTransitionBarrierDesc, MAX_TEXTURE_TRANSITION_NUM> optimizedTransitions = {};

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];
    const bool isEven = !(frameIndex & 0x1);
    nri::TransitionBarrierDesc transitionBarriers = {};
    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;

    if (frameIndex >= BUFFERED_FRAME_MAX_NUM)
    {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - BUFFERED_FRAME_MAX_NUM);
        NRI.ResetCommandAllocator(*frame.commandAllocator);
    }

    // Global history reset
    float sunCurr = Smoothstep( -0.9f, 0.05f, Sin( DegToRad(m_Settings.sunElevation) ) );
    float sunPrev = Smoothstep( -0.9f, 0.05f, Sin( DegToRad(m_PrevSettings.sunElevation) ) );
    float resetHistoryFactor = 1.0f - Smoothstep( 0.0f, 0.2f, Abs(sunCurr - sunPrev) );

    if (m_PrevSettings.denoiser != m_Settings.denoiser)
        m_ForceHistoryReset = true;
    if (m_PrevSettings.denoiser == REFERENCE && m_PrevSettings.tracingMode != m_Settings.tracingMode)
        m_ForceHistoryReset = true;
    if (m_PrevSettings.ortho != m_Settings.ortho)
        m_ForceHistoryReset = true;
    if (m_PrevSettings.onScreen != m_Settings.onScreen)
        m_ForceHistoryReset = true;

    // Sizes
    uint32_t rectW = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
    uint32_t rectH = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
    uint32_t rectGridW = (rectW + 15) / 16;
    uint32_t rectGridH = (rectH + 15) / 16;
    uint32_t windowGridW = (GetWindowResolution().x + 15) / 16;
    uint32_t windowGridH = (GetWindowResolution().y + 15) / 16;

    // NRD settings
    if (m_Settings.adaptiveAccumulation)
    {
        bool isFastHistoryEnabled = m_Settings.maxAccumulatedFrameNum > m_Settings.maxFastAccumulatedFrameNum;

        float fps = 1000.0f / m_Timer.GetSmoothedFrameTime();
        float maxAccumulatedFrameNum = Clamp(ACCUMULATION_TIME * fps, 5.0f, float(MAX_HISTORY_FRAME_NUM));
        float maxFastAccumulatedFrameNum = isFastHistoryEnabled ? (maxAccumulatedFrameNum / 5.0f) : float(MAX_HISTORY_FRAME_NUM);

        m_Settings.maxAccumulatedFrameNum = int32_t(maxAccumulatedFrameNum + 0.5f);
        m_Settings.maxFastAccumulatedFrameNum = int32_t(maxFastAccumulatedFrameNum + 0.5f);
    }

    uint32_t maxAccumulatedFrameNum = uint32_t(m_Settings.maxAccumulatedFrameNum * resetHistoryFactor + 0.5f);
    uint32_t maxFastAccumulatedFrameNum = uint32_t(m_Settings.maxFastAccumulatedFrameNum * resetHistoryFactor + 0.5f);

    nrd::CommonSettings commonSettings = {};
    memcpy(commonSettings.viewToClipMatrix, &m_Camera.state.mViewToClip, sizeof(m_Camera.state.mViewToClip));
    memcpy(commonSettings.viewToClipMatrixPrev, &m_Camera.statePrev.mViewToClip, sizeof(m_Camera.statePrev.mViewToClip));
    memcpy(commonSettings.worldToViewMatrix, &m_Camera.state.mWorldToView, sizeof(m_Camera.state.mWorldToView));
    memcpy(commonSettings.worldToViewMatrixPrev, &m_Camera.statePrev.mWorldToView, sizeof(m_Camera.statePrev.mWorldToView));
    commonSettings.motionVectorScale[0] = m_Settings.mvType == MV_3D ? 1.0f : 1.0f / float(rectW);
    commonSettings.motionVectorScale[1] = m_Settings.mvType == MV_3D ? 1.0f : 1.0f / float(rectH);
    commonSettings.motionVectorScale[2] = m_Settings.mvType != MV_2D ? 1.0f : 0.0f;
    commonSettings.cameraJitter[0] = m_Settings.cameraJitter ? m_Camera.state.viewportJitter.x : 0.0f;
    commonSettings.cameraJitter[1] = m_Settings.cameraJitter ? m_Camera.state.viewportJitter.y : 0.0f;
    commonSettings.resolutionScale[0] = m_Settings.resolutionScale;
    commonSettings.resolutionScale[1] = m_Settings.resolutionScale;
    commonSettings.denoisingRange = GetDenoisingRange();
    commonSettings.disocclusionThreshold = m_Settings.disocclusionThreshold * 0.01f;
    commonSettings.splitScreen = m_Settings.denoiser == REFERENCE ? 1.0f : m_Settings.separator;
    commonSettings.debug = m_Settings.debug;
    commonSettings.frameIndex = frameIndex;
    commonSettings.accumulationMode = m_ForceHistoryReset ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.isMotionVectorInWorldSpace = m_Settings.mvType == MV_3D;
    commonSettings.isBaseColorMetalnessAvailable = true;
    commonSettings.enableValidation = m_DebugNRD && m_ShowValidationOverlay;

    // NRD user pool
    NrdUserPool userPool = {};
    {
        // Common
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_MV, {&GetState(Texture::Mv), GetFormat(Texture::Mv)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_NORMAL_ROUGHNESS, {&GetState(Texture::Normal_Roughness), GetFormat(Texture::Normal_Roughness)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_VIEWZ, {&GetState(Texture::ViewZ), GetFormat(Texture::ViewZ)});

        // (Optional) Needed to allow IN_MV modification on the NRD side
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_BASECOLOR_METALNESS, {&GetState(Texture::BaseColor_Metalness), GetFormat(Texture::BaseColor_Metalness)});

        // (Optional) Validation
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_VALIDATION, {&GetState(Texture::Validation), GetFormat(Texture::Validation)});

        // Diffuse
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, {&GetState(Texture::Diff), GetFormat(Texture::Diff)});

        // Diffuse occlusion
    #if( NRD_MODE == OCCLUSION )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_HITDIST, {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_HITDIST, {&GetState(Texture::Diff), GetFormat(Texture::Diff)});
    #endif

        // Diffuse SH
    #if( NRD_MODE == SH )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_SH0, {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_SH1, {&GetState(Texture::Unfiltered_DiffSh), GetFormat(Texture::Unfiltered_DiffSh)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_SH0, {&GetState(Texture::Diff), GetFormat(Texture::Diff)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_SH1, {&GetState(Texture::DiffSh), GetFormat(Texture::DiffSh)});
    #endif

        // Diffuse directional occlusion
    #if( NRD_MODE == DIRECTIONAL_OCCLUSION )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_DIFF_DIRECTION_HITDIST, {&GetState(Texture::Unfiltered_Diff), GetFormat(Texture::Unfiltered_Diff)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_DIFF_DIRECTION_HITDIST, {&GetState(Texture::Diff), GetFormat(Texture::Diff)});
    #endif

        // Specular
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, {&GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, {&GetState(Texture::Spec), GetFormat(Texture::Spec)});

        // Specular occlusion
    #if( NRD_MODE == OCCLUSION )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_HITDIST, {&GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_HITDIST, {&GetState(Texture::Spec), GetFormat(Texture::Spec)});
    #endif

        // Specular SH
    #if( NRD_MODE == SH )
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_SH0, {&GetState(Texture::Unfiltered_Spec), GetFormat(Texture::Unfiltered_Spec)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SPEC_SH1, {&GetState(Texture::Unfiltered_SpecSh), GetFormat(Texture::Unfiltered_SpecSh)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_SH0, {&GetState(Texture::Spec), GetFormat(Texture::Spec)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SPEC_SH1, {&GetState(Texture::SpecSh), GetFormat(Texture::SpecSh)});
    #endif

        // SIGMA
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SHADOWDATA, {&GetState(Texture::Unfiltered_ShadowData), GetFormat(Texture::Unfiltered_ShadowData)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_SHADOW_TRANSLUCENCY, {&GetState(Texture::Unfiltered_Shadow_Translucency), GetFormat(Texture::Unfiltered_Shadow_Translucency)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, {&GetState(Texture::Shadow), GetFormat(Texture::Shadow)});

        // REFERENCE
        NrdIntegration_SetResource(userPool, nrd::ResourceType::IN_RADIANCE, {&GetState(Texture::Composed_ViewZ), GetFormat(Texture::Composed_ViewZ)});
        NrdIntegration_SetResource(userPool, nrd::ResourceType::OUT_RADIANCE, {&GetState(Texture::Composed_ViewZ), GetFormat(Texture::Composed_ViewZ)});
    }

    UpdateConstantBuffer(frameIndex, m_ForceHistoryReset ? 0.0f : resetHistoryFactor);

    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool, 0);
    {
        { // TLAS
            helper::Annotation annotation(NRI, commandBuffer, "TLAS");

            BuildTopLevelAccelerationStructure(commandBuffer, bufferedFrameIndex);
        }

        // All-in-one pipeline layout
        NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);

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
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            transitionBarriers.buffers = bufferTransitions;
            transitionBarriers.bufferNum = helper::GetCountOf(bufferTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
            transitionBarriers.bufferNum = 0;

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::AmbientRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 0, *frame.globalConstantBufferDescriptorSet, nullptr);
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::AmbientRays1), nullptr);
            NRI.CmdSetDescriptorSet(commandBuffer, 2, *Get(DescriptorSet::RayTracing2), nullptr);

            NRI.CmdDispatch(commandBuffer, 2, 2, 1);
        }

        { // Primary rays
            helper::Annotation annotation(NRI, commandBuffer, "Primary rays");

            const TextureState transitions[] =
            {
                // Input
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::PrimaryMipAndCurvature, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::DirectEmission, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Shadow_Translucency, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::PrimaryRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::PrimaryRays1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        { // Shadow denoising
            helper::Annotation annotation(NRI, commandBuffer, "Shadow denoising");

            nrd::SigmaSettings shadowSettings = {};

            m_Sigma.SetMethodSettings(nrd::Method::SIGMA_SHADOW_TRANSLUCENCY, &shadowSettings);
            m_Sigma.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);

            RestoreBindings(commandBuffer, frame);
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
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DirectLighting));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::DirectLighting1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        { // Indirect rays
            helper::Annotation annotation(NRI, commandBuffer, "Indirect rays");

            const TextureState transitions[] =
            {
                // Input
                {Texture::PrimaryMipAndCurvature, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::ComposedDiff_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Diff, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_Spec, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
#if( NRD_MODE == SH )
                {Texture::Unfiltered_DiffSh, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Unfiltered_SpecSh, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
#endif
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::IndirectRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::IndirectRays1), nullptr);

            uint32_t rectWmod = uint32_t(m_RenderResolution.x * m_Settings.resolutionScale + 0.5f);
            uint32_t rectHmod = uint32_t(m_RenderResolution.y * m_Settings.resolutionScale + 0.5f);
            uint32_t rectGridWmod = (rectWmod + 15) / 16;
            uint32_t rectGridHmod = (rectHmod + 15) / 16;

            NRI.CmdDispatch(commandBuffer, rectGridWmod, rectGridHmod, 1);
        }

        { // Denoising
            helper::Annotation annotation(NRI, commandBuffer, "Lighting denoising");

            float radiusResolutionScale = 1.0f;
            if (m_Settings.adaptRadiusToResolution)
                radiusResolutionScale = float( m_Settings.resolutionScale * m_RenderResolution.y ) / 1440.0f;

            if (m_Settings.denoiser == REBLUR || m_Settings.denoiser == REFERENCE)
            {
                nrd::HitDistanceParameters hitDistanceParameters = {};
                hitDistanceParameters.A = m_Settings.hitDistScale * m_Settings.meterToUnitsMultiplier;
                m_ReblurSettings.hitDistanceParameters = hitDistanceParameters;

                float resolutionScale = float( m_RenderResolution.x * m_Settings.resolutionScale ) / float( GetOutputResolution().x );
                float antilagMaxThresholdScale = 0.25f + 0.75f / (1.0f + (m_Settings.rpp - 1) * 0.25f);

                nrd::AntilagHitDistanceSettings defaultAntilagHitDistanceSettings = {};
                m_ReblurSettings.antilagHitDistanceSettings.thresholdMin = defaultAntilagHitDistanceSettings.thresholdMin / resolutionScale;
                m_ReblurSettings.antilagHitDistanceSettings.thresholdMax = defaultAntilagHitDistanceSettings.thresholdMax * antilagMaxThresholdScale;
                m_ReblurSettings.antilagHitDistanceSettings.sigmaScale = defaultAntilagHitDistanceSettings.sigmaScale / resolutionScale;

                nrd::AntilagIntensitySettings defaultAntilagIntensitySettings = {};
                m_ReblurSettings.antilagIntensitySettings.thresholdMin = defaultAntilagIntensitySettings.thresholdMin / resolutionScale;
                m_ReblurSettings.antilagIntensitySettings.thresholdMax = defaultAntilagIntensitySettings.thresholdMax * antilagMaxThresholdScale;
                m_ReblurSettings.antilagIntensitySettings.sigmaScale = defaultAntilagIntensitySettings.sigmaScale / resolutionScale;

                // IMPORTANT: needs to be manually tuned based on the input range
                m_ReblurSettings.antilagIntensitySettings.sensitivityToDarkness = 0.01f;

                m_ReblurSettings.maxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_ReblurSettings.maxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_ReblurSettings.checkerboardMode = m_Settings.tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                m_ReblurSettings.enableMaterialTestForDiffuse = true;
                m_ReblurSettings.enableMaterialTestForSpecular = false;

                nrd::ReblurSettings settings = m_ReblurSettings;
                settings.blurRadius *= radiusResolutionScale;
                settings.diffusePrepassBlurRadius *= radiusResolutionScale;
                settings.specularPrepassBlurRadius *= radiusResolutionScale;
                settings.historyFixStrideBetweenSamples *= radiusResolutionScale;

            #if( NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION )
                // High quality SG resolve allows to use more relaxed normal weights
                if (m_Resolve)
                    settings.lobeAngleFraction *= 1.333f;
            #endif

        #if( NRD_MODE == OCCLUSION )
            #if( NRD_COMBINED == 1 )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR_OCCLUSION, &m_ReblurSettings);
            #else
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_OCCLUSION, &m_ReblurSettings);
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR_OCCLUSION, &m_ReblurSettings);
            #endif
        #elif( NRD_MODE == SH )
            #if( NRD_COMBINED == 1 )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR_SH, &settings);
            #else
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SH, &m_ReblurSettings);
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR_SH, &m_ReblurSettings);
            #endif
        #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION, &settings);
        #else
            #if( NRD_COMBINED == 1 )
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE_SPECULAR, &settings);
            #else
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_DIFFUSE, &m_ReblurSettings);
                m_Reblur.SetMethodSettings(nrd::Method::REBLUR_SPECULAR, &m_ReblurSettings);
            #endif
        #endif

                m_Reblur.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);
            }
            else if (m_Settings.denoiser == RELAX)
            {
                m_RelaxSettings.diffuseMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_RelaxSettings.diffuseMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_RelaxSettings.specularMaxAccumulatedFrameNum = maxAccumulatedFrameNum;
                m_RelaxSettings.specularMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;
                m_RelaxSettings.checkerboardMode = m_Settings.tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::WHITE : nrd::CheckerboardMode::OFF;
                m_RelaxSettings.enableMaterialTestForDiffuse = true;
                m_RelaxSettings.enableMaterialTestForSpecular = false;

                nrd::RelaxDiffuseSpecularSettings settings = m_RelaxSettings;
                settings.diffusePrepassBlurRadius *= radiusResolutionScale;
                settings.specularPrepassBlurRadius *= radiusResolutionScale;
                settings.historyFixStrideBetweenSamples *= radiusResolutionScale;

                #if( NRD_COMBINED == 1 )
                    #if( NRD_MODE == SH )
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SPECULAR_SH, &m_RelaxSettings);
                    #else
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SPECULAR, &m_RelaxSettings);
                    #endif
                #else
                    nrd::RelaxDiffuseSettings diffuseSettings = {};
                    diffuseSettings.prepassBlurRadius                            = settings.diffusePrepassBlurRadius;
                    diffuseSettings.diffuseMaxAccumulatedFrameNum                = settings.diffuseMaxAccumulatedFrameNum;
                    diffuseSettings.diffuseMaxFastAccumulatedFrameNum            = settings.diffuseMaxFastAccumulatedFrameNum;
                    diffuseSettings.diffusePhiLuminance                          = settings.diffusePhiLuminance;
                    diffuseSettings.diffuseLobeAngleFraction                     = settings.diffuseLobeAngleFraction;
                    diffuseSettings.historyFixEdgeStoppingNormalPower            = settings.historyFixEdgeStoppingNormalPower;
                    diffuseSettings.historyFixStrideBetweenSamples               = settings.historyFixStrideBetweenSamples;
                    diffuseSettings.historyFixFrameNum                           = settings.historyFixFrameNum;
                    diffuseSettings.historyClampingColorBoxSigmaScale            = settings.historyClampingColorBoxSigmaScale;
                    diffuseSettings.spatialVarianceEstimationHistoryThreshold    = settings.spatialVarianceEstimationHistoryThreshold;
                    diffuseSettings.atrousIterationNum                           = settings.atrousIterationNum;
                    diffuseSettings.minLuminanceWeight                           = settings.diffuseMinLuminanceWeight;
                    diffuseSettings.depthThreshold                               = settings.depthThreshold;
                    diffuseSettings.checkerboardMode                             = settings.checkerboardMode;
                    diffuseSettings.hitDistanceReconstructionMode                = settings.hitDistanceReconstructionMode;
                    diffuseSettings.enableAntiFirefly                            = settings.enableAntiFirefly;
                    diffuseSettings.enableMaterialTest                           = settings.enableMaterialTestForDiffuse;

                    nrd::RelaxSpecularSettings specularSettings = {};
                    specularSettings.prepassBlurRadius                           = settings.specularPrepassBlurRadius;
                    specularSettings.specularMaxAccumulatedFrameNum              = settings.specularMaxAccumulatedFrameNum;
                    specularSettings.specularMaxFastAccumulatedFrameNum          = settings.specularMaxFastAccumulatedFrameNum;
                    specularSettings.specularPhiLuminance                        = settings.specularPhiLuminance;
                    specularSettings.diffuseLobeAngleFraction                    = settings.diffuseLobeAngleFraction;
                    specularSettings.specularLobeAngleFraction                   = settings.specularLobeAngleFraction;
                    specularSettings.roughnessFraction                           = settings.roughnessFraction;
                    specularSettings.specularVarianceBoost                       = settings.specularVarianceBoost;
                    specularSettings.specularLobeAngleSlack                      = settings.specularLobeAngleSlack;
                    specularSettings.historyFixEdgeStoppingNormalPower           = settings.historyFixEdgeStoppingNormalPower;
                    specularSettings.historyFixStrideBetweenSamples              = settings.historyFixStrideBetweenSamples;
                    specularSettings.historyFixFrameNum                          = settings.historyFixFrameNum;
                    specularSettings.historyClampingColorBoxSigmaScale           = settings.historyClampingColorBoxSigmaScale;
                    specularSettings.spatialVarianceEstimationHistoryThreshold   = settings.spatialVarianceEstimationHistoryThreshold;
                    specularSettings.atrousIterationNum                          = settings.atrousIterationNum;
                    specularSettings.minLuminanceWeight                          = settings.specularMinLuminanceWeight;
                    specularSettings.depthThreshold                              = settings.depthThreshold;
                    specularSettings.luminanceEdgeStoppingRelaxation             = settings.luminanceEdgeStoppingRelaxation;
                    specularSettings.normalEdgeStoppingRelaxation                = settings.normalEdgeStoppingRelaxation;
                    specularSettings.roughnessEdgeStoppingRelaxation             = settings.roughnessEdgeStoppingRelaxation;
                    specularSettings.checkerboardMode                            = m_Settings.tracingMode == RESOLUTION_HALF ? nrd::CheckerboardMode::BLACK : nrd::CheckerboardMode::OFF;
                    specularSettings.hitDistanceReconstructionMode               = settings.hitDistanceReconstructionMode;
                    specularSettings.enableAntiFirefly                           = settings.enableAntiFirefly;
                    specularSettings.enableRoughnessEdgeStopping                 = settings.enableRoughnessEdgeStopping;
                    specularSettings.enableMaterialTest                          = settings.enableMaterialTestForSpecular;

                    #if( NRD_MODE == SH )
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE_SH, &diffuseSettings);
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_SPECULAR_SH, &specularSettings);
                    #else
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_DIFFUSE, &diffuseSettings);
                        m_Relax.SetMethodSettings(nrd::Method::RELAX_SPECULAR, &specularSettings);
                    #endif
                #endif

                m_Relax.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);
            }

            RestoreBindings(commandBuffer, frame);
        }

        { // Composition
            helper::Annotation annotation(NRI, commandBuffer, "Composition");

            const TextureState transitions[] =
            {
                // Input
                {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Normal_Roughness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::BaseColor_Metalness, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::DirectLighting, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Ambient, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Diff, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Spec, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
            #if( NRD_MODE == SH )
                {Texture::DiffSh, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::SpecSh, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
            #endif
                // Output
                {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ComposedDiff_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::ComposedSpec_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Composition));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::Composition1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        { // Delta rays
            helper::Annotation annotation(NRI, commandBuffer, "Delta rays");

            const TextureState transitions[] =
            {
                // Input
                {Texture::Diff, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                {Texture::Spec, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                // Output
                {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                {Texture::Mv, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
            };
            transitionBarriers.textures = optimizedTransitions.data();
            transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
            NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

            NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::DeltaRays));
            NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::DeltaRays1), nullptr);

            NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
        }

        if (m_Settings.denoiser == REFERENCE)
        { // Reference
            helper::Annotation annotation(NRI, commandBuffer, "Reference denoising");

            commonSettings.resolutionScale[0] = 1.0f;
            commonSettings.resolutionScale[1] = 1.0f;
            commonSettings.splitScreen = m_Settings.separator;

            m_Reference.SetMethodSettings(nrd::Method::REFERENCE, &m_ReferenceSettings);
            m_Reference.Denoise(frameIndex, commandBuffer, commonSettings, userPool, true);

            RestoreBindings(commandBuffer, frame);
        }

        if (m_Settings.DLSS)
        {
            { // Pre
                helper::Annotation annotation(NRI, commandBuffer, "Pre Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::Mv, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                    {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                    {Texture::DlssInput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::PreDlss));
                NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::PreDlss1), nullptr);

                NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            }

            { // DLSS
                helper::Annotation annotation(NRI, commandBuffer, "Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Unfiltered_ShadowData, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::DlssInput, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                DlssDispatchDesc dlssDesc = {};
                dlssDesc.texOutput = {Get(Texture::DlssOutput), Get(Descriptor::DlssOutput_StorageTexture), GetFormat(Texture::DlssOutput), {GetOutputResolution().x, GetOutputResolution().y}};
                dlssDesc.texInput = {Get(Texture::DlssInput), Get(Descriptor::DlssInput_Texture), GetFormat(Texture::DlssInput), {m_RenderResolution.x, m_RenderResolution.y}};
                dlssDesc.texMv = {Get(Texture::Unfiltered_ShadowData), Get(Descriptor::Unfiltered_ShadowData_Texture), GetFormat(Texture::Unfiltered_ShadowData), {m_RenderResolution.x, m_RenderResolution.y}};
                dlssDesc.texDepth = {Get(Texture::ViewZ), Get(Descriptor::ViewZ_Texture), GetFormat(Texture::ViewZ), {m_RenderResolution.x, m_RenderResolution.y}};
                dlssDesc.sharpness = m_Settings.sharpness;
                dlssDesc.currentRenderResolution = {rectW, rectH};
                dlssDesc.motionVectorScale[0] = 1.0f;
                dlssDesc.motionVectorScale[1] = 1.0f;
                dlssDesc.jitter[0] = -m_Camera.state.viewportJitter.x;
                dlssDesc.jitter[1] = -m_Camera.state.viewportJitter.y;
                dlssDesc.reset = m_ForceHistoryReset;

                m_DLSS.Evaluate(&commandBuffer, dlssDesc);

                RestoreBindings(commandBuffer, frame); // TODO: is it needed?
            }

            { // After
                helper::Annotation annotation(NRI, commandBuffer, "After Dlss");

                const TextureState transitions[] =
                {
                    // Input
                    {Texture::DlssOutput, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Validation, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::AfterDlss));
                NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(DescriptorSet::AfterDlss1), nullptr);

                NRI.CmdDispatch(commandBuffer, windowGridW, windowGridH, 1);
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
                    {Texture::Mv, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Composed_ViewZ, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {taaSrc, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {taaDst, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Temporal));
                NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(isEven ? DescriptorSet::Temporal1a : DescriptorSet::Temporal1b), nullptr);

                NRI.CmdDispatch(commandBuffer, rectGridW, rectGridH, 1);
            }

            { // Upsample, copy and split screen
                helper::Annotation annotation(NRI, commandBuffer, "Upsample");

                const TextureState transitions[] =
                {
                    // Input
                    {taaDst, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    {Texture::Validation, nri::AccessBits::SHADER_RESOURCE, nri::TextureLayout::SHADER_RESOURCE},
                    // Output
                    {Texture::Final, nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::TextureLayout::GENERAL},
                };
                transitionBarriers.textures = optimizedTransitions.data();
                transitionBarriers.textureNum = BuildOptimizedTransitions(transitions, helper::GetCountOf(transitions), optimizedTransitions);
                NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

                bool isValidation = m_DebugNRD && m_ShowValidationOverlay;
                bool isNis = m_Settings.NIS && m_Settings.separator == 0.0f && !isValidation;
                if (isNis)
                {
                    NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::UpsampleNis));
                    NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(isEven ? DescriptorSet::UpsampleNis1a : DescriptorSet::UpsampleNis1b), nullptr);

                    // See NIS_Config.h
                    windowGridW = (GetWindowResolution().x + 31) / 32;
                    windowGridH = (GetWindowResolution().y + 31) / 32;
                }
                else
                {
                    NRI.CmdSetPipeline(commandBuffer, *Get(Pipeline::Upsample));
                    NRI.CmdSetDescriptorSet(commandBuffer, 1, *Get(isEven ? DescriptorSet::Upsample1a : DescriptorSet::Upsample1b), nullptr);
                }

                NRI.CmdDispatch(commandBuffer, windowGridW, windowGridH, 1);
            }
        }

        const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
        const BackBuffer* backBuffer = &m_SwapChainBuffers[backBufferIndex];

        { // Copy to back-buffer
            const nri::TextureTransitionBarrierDesc copyTransitions[] =
            {
                nri::TextureTransitionFromState(GetState(Texture::Final), nri::AccessBits::COPY_SOURCE, nri::TextureLayout::GENERAL),
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

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &frame.commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;
    NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);

    NRI.SwapChainPresent(*m_SwapChain);

    NRI.QueueSignal(*m_CommandQueue, *m_FrameFence, 1 + frameIndex);

    // Cap FPS if requested
    float msLimit = m_Settings.limitFps ? 1000.0f / m_Settings.maxFps : 0.0f;
    double lastFrameTimeStamp = m_Timer.GetLastFrameTimeStamp();

    while (m_Timer.GetTimeStamp() - lastFrameTimeStamp < msLimit)
        ;
}

SAMPLE_MAIN(Sample, 0);
