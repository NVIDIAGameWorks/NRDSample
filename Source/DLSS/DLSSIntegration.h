#pragma once

// IMPORTANT: these files must be included beforehand:
//    #include "NRI.h"
//    #include "Extensions/NRIHelper.h"
//    #include "Extensions/NRIWrapperVK.h"

#include <vulkan/vulkan.h>

#pragma warning(push)
#pragma warning(disable : 4100) // unreferenced formal parameter

// SR
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"

// RR
#include "nvsdk_ngx_helpers_dlssd.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

#pragma warning(pop)

#define DLSS_INTEGRATION_MAJOR 1
#define DLSS_INTEGRATION_MINOR 8
#define DLSS_INTEGRATION_DATE "5 February 2025"
#define DLSS_INTEGRATION 1

enum class DlssQuality
{
    ULTRA_PERFORMANCE,
    PERFORMANCE,
    BALANCED,
    QUALITY,
    AA,

    MAX_NUM
};

// Provided by DLSS
struct DlssSettings
{
    NVSDK_NGX_Dimensions optimalResolution = {};
    NVSDK_NGX_Dimensions dynamicResolutionMin = {};
    NVSDK_NGX_Dimensions dynamicResolutionMax = {};
};

struct DlssInitDesc
{
    NVSDK_NGX_Dimensions outputResolution = {};
    DlssQuality quality = DlssQuality::QUALITY;
    bool hasHdrContent = false;
    bool hasInvertedDepth = false;
    bool allowAutoExposure = false;
};

struct DlssTexture
{
    nri::Texture* resource = nullptr;
    nri::Descriptor* descriptor = nullptr;
};

struct DlssDispatchDesc
{
    // Output - required state SHADER_RESOURCE_STORAGE
    DlssTexture texOutput = {};

    // Inputs - required state SHADER_RESOURCE
    DlssTexture texInput = {};
    DlssTexture texMv = {};
    DlssTexture texDepth = {}; // HW for SR, linear for RR

    // RR specific inputs
    DlssTexture texDiffAlbedo = {};
    DlssTexture texSpecAlbedo = {};
    DlssTexture texNormalRoughness = {};
    DlssTexture texSpecHitDistance = {};

    // Settings
    NVSDK_NGX_Dimensions viewportDims = {};
    float jitter[2] = {0.0f, 0.0f};
    float mvScale[2] = {1.0f, 1.0f};
    bool reset = false;

    // RR specific settings
    float mWorldToView[16] = {};
    float mViewToClip[16] = {};
    bool useRR = false;
};

class DlssIntegration
{
public:
    inline DlssIntegration()
    { }

    inline ~DlssIntegration()
    { Shutdown(); };

    inline bool HasSR() const
    { return m_SR != nullptr; }

    inline bool HasRR() const
    { return m_RR != nullptr; }

    bool InitializeLibrary(nri::Device& device, const char* appDataPath, uint64_t applicationId = 231313132);
    bool GetOptimalSettings(const NVSDK_NGX_Dimensions& outputResolution, DlssQuality quality, DlssSettings& outSettings) const;
    bool Initialize(nri::Queue* queue, const DlssInitDesc& desc);
    void Evaluate(nri::CommandBuffer* commandBuffer, const DlssDispatchDesc& desc); // currently bound nri::DescriptorPool will be lost
    void Shutdown();

    static void SetupDeviceExtensions(nri::DeviceCreationDesc& desc);

private:
    inline NVSDK_NGX_Resource_VK SetupVulkanTexture(const DlssTexture& texture, bool isStorage = false);

private:
    struct NRIInterface
        : public nri::CoreInterface
        , public nri::WrapperVKInterface
        , public nri::HelperInterface
    {};

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    NVSDK_NGX_Handle* m_SR = nullptr;
    NVSDK_NGX_Handle* m_RR = nullptr;
    NVSDK_NGX_Parameter* m_NgxParameters = nullptr;
    uint64_t m_ApplicationId = 0;
};
