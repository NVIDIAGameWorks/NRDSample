#pragma once

// IMPORTANT: these files must be included beforehand
//    NRIDescs.hpp
//    Extensions/NRIWrapperVK.h
//    Extensions/NRIDeviceCreation.h

#include <assert.h>

#include <vulkan/vulkan.h>

#pragma warning(push)
#pragma warning(disable : 4100)

#include "NGX/include/nvsdk_ngx_helpers.h"
#include "NGX/include/nvsdk_ngx_helpers_vk.h"

#pragma warning(pop) 

#define DLSS_INTEGRATION_MAJOR 1
#define DLSS_INTEGRATION_MINOR 5
#define DLSS_INTEGRATION_DATE "23 March 2023"
#define DLSS_INTEGRATION 1

static_assert(NRI_VERSION_MAJOR >= 1 && NRI_VERSION_MINOR >= 93, "Unsupported NRI version!");

enum class DlssQuality
{
    ULTRA_PERFORMANCE,
    PERFORMANCE,
    BALANCED,
    QUALITY,

    MAX_NUM
};

// This is retrieved from DLSS
struct DlssSettings
{
    NVSDK_NGX_Dimensions renderResolution = {};
    NVSDK_NGX_Dimensions maxRenderResolution = {};
    NVSDK_NGX_Dimensions minRenderResolution = {};
    float sharpness = 0.0f;
};

struct DlssInitDesc
{
    NVSDK_NGX_Dimensions outputResolution = {};
    DlssQuality quality = DlssQuality::QUALITY;
    bool isContentHDR = false;
    bool isDepthInverted = false;
    bool isMotionVectorAtLowRes = true; // MVs match render resolution in 100% of cases
    bool enableAutoExposure = false;
};

struct DlssTexture
{
    nri::Texture* texture = nullptr;
    nri::Descriptor* descriptor = nullptr;
    nri::Format format = nri::Format::UNKNOWN;
    NVSDK_NGX_Dimensions dims = {};
};

struct DlssDispatchDesc
{
    // Output - required state SHADER_RESOURCE_STORAGE
    DlssTexture texOutput = {};

    // Inputs - required state SHADER_RESOURCE
    DlssTexture texInput = {};
    DlssTexture texMv = {};
    DlssTexture texDepth = {};
    DlssTexture texExposure = {}; // (optional) 1x1

    // Settings
    NVSDK_NGX_Dimensions currentRenderResolution = {};
    float jitter[2] = {0.0f, 0.0f};
    float motionVectorScale[2] = {1.0f, 1.0f};
    float sharpness = 0.0f;
    bool reset = false;
};

class DlssIntegration
{
public:
    inline DlssIntegration()
    { }

    inline ~DlssIntegration()
    { Shutdown(); };

    inline bool IsInitialized() const
    { return m_Initialized; }

    bool InitializeLibrary(nri::Device& device, const char* appDataPath, uint64_t applicationId = 231313132);
    bool GetOptimalSettings(const NVSDK_NGX_Dimensions& outputResolution, DlssQuality quality, DlssSettings& outSettings) const;
    bool Initialize(nri::CommandQueue* commandQueue, const DlssInitDesc& desc);
    void Evaluate(nri::CommandBuffer* commandBuffer, const DlssDispatchDesc& desc); // currently bound nri::DescriptorPool will be lost
    void Shutdown();

    static void SetupDeviceExtensions(nri::DeviceCreationDesc& desc);

private:

    inline NVSDK_NGX_Resource_VK SetupVulkanTexture(const DlssTexture& texture, bool isStorage);

private:

    struct NRIInterface
        : public nri::CoreInterface
        , public nri::WrapperVKInterface
    {};

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    NVSDK_NGX_Handle* m_DLSS = nullptr;
    NVSDK_NGX_Parameter* m_NgxParameters = nullptr;
    uint64_t m_ApplicationId = 0;
    bool m_Initialized = false;
};
