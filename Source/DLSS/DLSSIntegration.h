#pragma once

// IMPORTANT: these files must be included beforehand
//    NRIDescs.hpp
//    Extensions/NRIDeviceCreation.h
//    Extensions/NRIWrapperD3D11.h
//    Extensions/NRIWrapperD3D12.h
//    Extensions/NRIWrapperVK.h

#include <assert.h>

#include <vulkan/vulkan.h>

#include "NGX/include/nvsdk_ngx_helpers.h"
#include "NGX/include/nvsdk_ngx_helpers_vk.h"

#define DLSS_INTEGRATION 1
#define DLSS_INTEGRATION_MAJOR 2
#define DLSS_INTEGRATION_MINOR 0
#define DLSS_INTEGRATION_DATE "14 January 2022"

enum class DlssQuality
{
    ULTRA_PERFORMANCE,
    PERFORMANCE,
    BALANCED,
    QUALITY,
    ULTRA_QUALITY
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

struct DlssDispatchDesc
{
    // Inputs - required state SHADER_RESOURCE, render resolution
    nri::Texture* texInput = nullptr;
    nri::Texture* texMv = nullptr;
    nri::Texture* texDepth = nullptr;
    nri::Texture* texExposure = nullptr; // (optional) 1x1

    // Output - required state SHADER_RESOURCE_STORAGE, output resolution
    nri::Texture* texOutput = nullptr;

    // For VULKAN
    nri::Descriptor* descriptorInput = nullptr;
    nri::Descriptor* descriptorMv = nullptr;
    nri::Descriptor* descriptorDepth = nullptr;
    nri::Descriptor* descriptorExposure = nullptr;
    nri::Descriptor* descriptorOutput = nullptr;

    NVSDK_NGX_Dimensions renderOrScaledResolution = {};
    float jitter[2] = {0.0f, 0.0f};
    float motionVectorScale[2] = {1.0f, 1.0f};
    float sharpness = 0.0f;
    uint32_t physicalDeviceIndex = 0;
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

    static inline void SetupDeviceExtensions(nri::DeviceCreationDesc& desc)
    {
        const char** instanceExt;
        const char** deviceExt;
        NVSDK_NGX_VULKAN_RequiredExtensions(
            &desc.vulkanExtensions.instanceExtensionNum,
            &instanceExt,
            &desc.vulkanExtensions.deviceExtensionNum,
            &deviceExt);
        desc.vulkanExtensions.instanceExtensions = instanceExt;
        desc.vulkanExtensions.deviceExtensions = deviceExt;
    }

private:

    inline NVSDK_NGX_Resource_VK SetupVulkanTexture(nri::Texture* texture, nri::Descriptor* descriptor, uint32_t physicalDeviceIndex, bool isStorage);

private:

    struct NRIInterface
        : public nri::CoreInterface
        , public nri::WrapperD3D11Interface
        , public nri::WrapperD3D12Interface
        , public nri::WrapperVKInterface
    {};

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    NVSDK_NGX_Handle* m_DLSS = nullptr;
    NVSDK_NGX_Parameter* m_NgxParameters = nullptr;
    uint64_t m_ApplicationId = 0;
    bool m_Initialized = false;
};
