#include "DLSSIntegration.h"

#include <assert.h> // assert
#include <stdio.h> // printf

static_assert(NRI_VERSION_MAJOR >= 1 && NRI_VERSION_MINOR >= 151, "Unsupported NRI version!");

// An ugly temp workaround until DLSS fix the problem
#ifndef _WIN32

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Init(unsigned long long, const wchar_t*, ID3D11Device*, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version )
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Shutdown1(ID3D11Device*)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_CreateFeature(ID3D11DeviceContext*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_ReleaseFeature(NVSDK_NGX_Handle*)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_EvaluateFeature_C(ID3D11DeviceContext*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback_C)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_AllocateParameters(NVSDK_NGX_Parameter**)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_DestroyParameters(NVSDK_NGX_Parameter*)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetCapabilityParameters(NVSDK_NGX_Parameter**)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Init(unsigned long long, const wchar_t*, ID3D12Device*, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown1(ID3D12Device*)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle*)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback_C)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter**)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter*)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter**)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

#endif

static inline void DLSS_ConvertCharToWchar(const char* in, wchar_t* out, size_t outLength)
{
    if (outLength == 0)
        return;

    for (size_t i = 0; i < outLength - 1 && *in; i++)
        *out++ = *in++;

    *out = 0;
}

static inline NVSDK_NGX_PerfQuality_Value DLSS_ConvertQuality(DlssQuality quality)
{
    if (quality == DlssQuality::ULTRA_PERFORMANCE)
        return NVSDK_NGX_PerfQuality_Value_UltraPerformance;

    if (quality == DlssQuality::PERFORMANCE)
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;

    if (quality == DlssQuality::BALANCED)
        return NVSDK_NGX_PerfQuality_Value_Balanced;

    if (quality == DlssQuality::QUALITY)
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;

    if (quality == DlssQuality::AA)
        return NVSDK_NGX_PerfQuality_Value_DLAA;

    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

void DlssIntegration::SetupDeviceExtensions(nri::DeviceCreationDesc& desc)
{
    static const char* vulkanExts[] = {
        "VK_NVX_binary_import",
        "VK_NVX_image_view_handle",
        "VK_KHR_push_descriptor"
    };

    desc.vkExtensions.deviceExtensions = vulkanExts;
    desc.vkExtensions.deviceExtensionNum = 3;
}

inline NVSDK_NGX_Resource_VK DlssIntegration::SetupVulkanTexture(const DlssTexture& texture, bool isStorage)
{
    VkImage image = (VkImage)NRI.GetTextureNativeObject(*texture.resource);
    VkImageView view = (VkImageView)NRI.GetDescriptorNativeObject(*texture.descriptor);
    VkImageSubresourceRange subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkFormat format = (VkFormat)nri::nriConvertNRIFormatToVK(texture.format);

    return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, subresource, format, texture.dims.Width, texture.dims.Height, isStorage);
}

bool DlssIntegration::InitializeLibrary(nri::Device& device, const char* appDataPath, uint64_t applicationId)
{
    m_ApplicationId = applicationId;
    m_Device = &device;

    uint32_t nriResult = (uint32_t)nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
    nriResult |= (uint32_t)nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI);

    if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VK)
        nriResult |= (uint32_t)nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::WrapperVKInterface), (nri::WrapperVKInterface*)&NRI);

    if ((nri::Result)nriResult != nri::Result::SUCCESS)
        return false;

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    wchar_t path[512];
    DLSS_ConvertCharToWchar(appDataPath, path, 512);

    NVSDK_NGX_Result result = NVSDK_NGX_Result::NVSDK_NGX_Result_Fail;
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3d12Device = (ID3D12Device*)NRI.GetDeviceNativeObject(*m_Device);
        result = NVSDK_NGX_D3D12_Init(m_ApplicationId, path, d3d12Device);
        if (NVSDK_NGX_SUCCEED(result))
            result = NVSDK_NGX_D3D12_GetCapabilityParameters(&m_NgxParameters);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VK)
    {
        VkDevice vkDevice = (VkDevice)NRI.GetDeviceNativeObject(*m_Device);
        VkPhysicalDevice vkPhysicalDevice = (VkPhysicalDevice)NRI.GetPhysicalDeviceVK(*m_Device);
        VkInstance vkInstance = (VkInstance)NRI.GetInstanceVK(*m_Device);
        result = NVSDK_NGX_VULKAN_Init(m_ApplicationId, path, vkInstance, vkPhysicalDevice, vkDevice);
        if (NVSDK_NGX_SUCCEED(result))
            result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_NgxParameters);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        ID3D11Device* d3d11Device = (ID3D11Device*)NRI.GetDeviceNativeObject(*m_Device);
        result = NVSDK_NGX_D3D11_Init(m_ApplicationId, path, d3d11Device);
        if (NVSDK_NGX_SUCCEED(result))
            result = NVSDK_NGX_D3D11_GetCapabilityParameters(&m_NgxParameters);
    }

    if (!NVSDK_NGX_SUCCEED(result))
        Shutdown();

    return NVSDK_NGX_SUCCEED(result);
}

bool DlssIntegration::GetOptimalSettings(const NVSDK_NGX_Dimensions& outputResolution, DlssQuality quality, DlssSettings& outSettings) const
{
    NVSDK_NGX_PerfQuality_Value dlssQuality = DLSS_ConvertQuality(quality);

    float unused = 0.0f;
    NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        m_NgxParameters,
        outputResolution.Width, outputResolution.Height,
        dlssQuality,
        &outSettings.optimalResolution.Width, &outSettings.optimalResolution.Height,
        &outSettings.dynamicResolutionMax.Width, &outSettings.dynamicResolutionMax.Height,
        &outSettings.dynamicResolutionMin.Width, &outSettings.dynamicResolutionMin.Height,
        &unused);

    return NVSDK_NGX_SUCCEED(result);
}

bool DlssIntegration::Initialize(nri::Queue* queue, const DlssInitDesc& desc)
{
    const uint32_t creationNodeMask = 0x1;
    const uint32_t visibilityNodeMask = 0x1;
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    // Prepare
    nri::CommandAllocator* commandAllocator;
    NRI.CreateCommandAllocator(*queue, commandAllocator);

    nri::CommandBuffer* commandBuffer;
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    nri::Fence* fence;
    NRI.CreateFence(*m_Device, 0, fence);

    // Record
    int32_t flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    flags |= desc.hasHdrContent ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
    flags |= desc.hasInvertedDepth ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
    flags |= desc.allowAutoExposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;

    DlssSettings settings = {};
    if (!GetOptimalSettings(desc.outputResolution, desc.quality, settings))
        return false;

    NVSDK_NGX_Result result = NVSDK_NGX_Result_Success;
    NRI.BeginCommandBuffer(*commandBuffer, nullptr);
    {
        // SR
        if (NVSDK_NGX_SUCCEED(result))
        {
            NVSDK_NGX_DLSS_Create_Params srCreateParams = {};
            srCreateParams.Feature.InWidth = settings.optimalResolution.Width;
            srCreateParams.Feature.InHeight = settings.optimalResolution.Height;
            srCreateParams.Feature.InTargetWidth = desc.outputResolution.Width;
            srCreateParams.Feature.InTargetHeight = desc.outputResolution.Height;
            srCreateParams.Feature.InPerfQualityValue = DLSS_ConvertQuality(desc.quality);
            srCreateParams.InFeatureCreateFlags = flags;

            nri::VideoMemoryInfo videoMemoryInfo1 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo1);

            if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
            {
                ID3D12GraphicsCommandList* d3d12CommandList = (ID3D12GraphicsCommandList*)NRI.GetCommandBufferNativeObject(*commandBuffer);
                result = NGX_D3D12_CREATE_DLSS_EXT(d3d12CommandList, creationNodeMask, visibilityNodeMask, &m_SR, m_NgxParameters, &srCreateParams);
            }
            else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VK)
            {
                VkCommandBuffer vkCommandBuffer = (VkCommandBuffer)NRI.GetCommandBufferNativeObject(*commandBuffer);
                result = NGX_VULKAN_CREATE_DLSS_EXT(vkCommandBuffer, creationNodeMask, visibilityNodeMask, &m_SR, m_NgxParameters, &srCreateParams);
            }
            else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
            {
                ID3D11DeviceContext* d3d11DeviceContext = (ID3D11DeviceContext*)NRI.GetCommandBufferNativeObject(*commandBuffer);
                result = NGX_D3D11_CREATE_DLSS_EXT(d3d11DeviceContext, &m_SR, m_NgxParameters, &srCreateParams);
            }

            nri::VideoMemoryInfo videoMemoryInfo2 = {};
            NRI.QueryVideoMemoryInfo(*m_Device, nri::MemoryLocation::DEVICE, videoMemoryInfo2);

            printf("DLSS-SR: allocated %.2f Mb\n", (videoMemoryInfo2.usageSize - videoMemoryInfo1.usageSize) / (1024.0f * 1024.0f));
        }
    }
    NRI.EndCommandBuffer(*commandBuffer);

    // Submit & wait for completion
    if (NVSDK_NGX_SUCCEED(result))
    {
        nri::FenceSubmitDesc signalFence = {};
        signalFence.fence = fence;
        signalFence.value = 1;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;
        queueSubmitDesc.signalFences = &signalFence;
        queueSubmitDesc.signalFenceNum = 1;

        NRI.QueueSubmit(*queue, queueSubmitDesc);
        NRI.Wait(*fence, 1);
    }

    // Cleanup
    NRI.DestroyFence(*fence);
    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    return NVSDK_NGX_SUCCEED(result);
}

void DlssIntegration::Evaluate(nri::CommandBuffer* commandBuffer, const DlssDispatchDesc& desc)
{
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        ID3D12Resource* resourceInput = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texInput.resource);
        ID3D12Resource* resourceMv = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texMv.resource);
        ID3D12Resource* resourceDepth = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texDepth.resource);
        ID3D12Resource* resourceOutput = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texOutput.resource);

        ID3D12GraphicsCommandList* d3dCommandList = (ID3D12GraphicsCommandList*)NRI.GetCommandBufferNativeObject(*commandBuffer);
        {
            NVSDK_NGX_D3D12_DLSS_Eval_Params srParams = {};
            srParams.Feature.pInColor = resourceInput;
            srParams.Feature.pInOutput = resourceOutput;
            srParams.pInDepth = resourceDepth;
            srParams.pInMotionVectors = resourceMv;
            srParams.InJitterOffsetX = desc.jitter[0];
            srParams.InJitterOffsetY = desc.jitter[1];
            srParams.InRenderSubrectDimensions = desc.viewportDims;
            // Optional
            srParams.InReset = desc.reset;
            srParams.InMVScaleX = desc.mvScale[0];
            srParams.InMVScaleY = desc.mvScale[1];

            result = NGX_D3D12_EVALUATE_DLSS_EXT(d3dCommandList, m_SR, m_NgxParameters, &srParams);
        }
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VK)
    {
        NVSDK_NGX_Resource_VK resourceOutput = SetupVulkanTexture(desc.texOutput, true);
        NVSDK_NGX_Resource_VK resourceInput = SetupVulkanTexture(desc.texInput);
        NVSDK_NGX_Resource_VK resourceMv = SetupVulkanTexture(desc.texMv);
        NVSDK_NGX_Resource_VK resourceDepth = SetupVulkanTexture(desc.texDepth);

        VkCommandBuffer vkCommandbuffer = (VkCommandBuffer)NRI.GetCommandBufferNativeObject(*commandBuffer);
        {
            NVSDK_NGX_VK_DLSS_Eval_Params srParams = {};
            srParams.Feature.pInColor = &resourceInput;
            srParams.Feature.pInOutput = &resourceOutput;
            srParams.pInDepth = &resourceDepth;
            srParams.pInMotionVectors = &resourceMv;
            srParams.InJitterOffsetX = desc.jitter[0];
            srParams.InJitterOffsetY = desc.jitter[1];
            srParams.InRenderSubrectDimensions = desc.viewportDims;
            // Optional
            srParams.InReset = desc.reset;
            srParams.InMVScaleX = desc.mvScale[0];
            srParams.InMVScaleY = desc.mvScale[1];

            result = NGX_VULKAN_EVALUATE_DLSS_EXT(vkCommandbuffer, m_SR, m_NgxParameters, &srParams);
        }
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        ID3D11Resource* resourceInput = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texInput.resource);
        ID3D11Resource* resourceMv = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texMv.resource);
        ID3D11Resource* resourceDepth = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texDepth.resource);
        ID3D11Resource* resourceOutput = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texOutput.resource);

        ID3D11DeviceContext* d3d11DeviceContext = (ID3D11DeviceContext*)NRI.GetCommandBufferNativeObject(*commandBuffer);
        {
            NVSDK_NGX_D3D11_DLSS_Eval_Params srParams = {};
            srParams.Feature.pInColor = resourceInput;
            srParams.Feature.pInOutput = resourceOutput;
            srParams.pInDepth = resourceDepth;
            srParams.pInMotionVectors = resourceMv;
            srParams.InJitterOffsetX = desc.jitter[0];
            srParams.InJitterOffsetY = desc.jitter[1];
            srParams.InRenderSubrectDimensions = desc.viewportDims;
            // Optional
            srParams.InReset = desc.reset;
            srParams.InMVScaleX = desc.mvScale[0];
            srParams.InMVScaleY = desc.mvScale[1];

            result = NGX_D3D11_EVALUATE_DLSS_EXT(d3d11DeviceContext, m_SR, m_NgxParameters, &srParams);
        }
    }

    assert( NVSDK_NGX_SUCCEED(result) );
}

void DlssIntegration::Shutdown()
{
    if (!m_Device)
        return;

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        if (m_NgxParameters)
            NVSDK_NGX_D3D12_DestroyParameters(m_NgxParameters);

        if (m_SR)
            NVSDK_NGX_D3D12_ReleaseFeature(m_SR);

        if (m_RR)
            NVSDK_NGX_D3D12_ReleaseFeature(m_RR);

        NVSDK_NGX_D3D12_Shutdown1(nullptr);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VK)
    {
        if (m_NgxParameters)
            NVSDK_NGX_VULKAN_DestroyParameters(m_NgxParameters);

        if (m_SR)
            NVSDK_NGX_VULKAN_ReleaseFeature(m_SR);

        if (m_RR)
            NVSDK_NGX_VULKAN_ReleaseFeature(m_RR);

        NVSDK_NGX_VULKAN_Shutdown1(nullptr);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        if (m_NgxParameters)
            NVSDK_NGX_D3D11_DestroyParameters(m_NgxParameters);

        if (m_SR)
            NVSDK_NGX_D3D11_ReleaseFeature(m_SR);

        if (m_RR)
            NVSDK_NGX_D3D11_ReleaseFeature(m_RR);

        NVSDK_NGX_D3D11_Shutdown1(nullptr);
    }

    m_NgxParameters = nullptr;
    m_SR = nullptr;
    m_RR = nullptr;
    m_Device = nullptr;
}
