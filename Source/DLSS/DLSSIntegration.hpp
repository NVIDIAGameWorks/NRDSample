#include "DLSSIntegration.h"

static_assert(NRI_VERSION_MAJOR >= 1 && NRI_VERSION_MINOR >= 90, "Unsupported NRI version!");

// An ugly temp workaround until DLSS fix the problem
#ifndef _WIN32

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Init(unsigned long long InApplicationId, const wchar_t *InApplicationDataPath, ID3D11Device *InDevice, const NVSDK_NGX_FeatureCommonInfo *InFeatureInfo, NVSDK_NGX_Version InSDKVersion)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Shutdown1(ID3D11Device* device)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_CreateFeature(ID3D11DeviceContext *InDevCtx, NVSDK_NGX_Feature InFeatureID, NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_ReleaseFeature(NVSDK_NGX_Handle *InHandle)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_EvaluateFeature_C(ID3D11DeviceContext *InDevCtx, const NVSDK_NGX_Handle *InFeatureHandle, const NVSDK_NGX_Parameter *InParameters, PFN_NVSDK_NGX_ProgressCallback_C InCallback)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_AllocateParameters(NVSDK_NGX_Parameter** OutParameters)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_DestroyParameters(NVSDK_NGX_Parameter* InParameters)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Init(unsigned long long InApplicationId, const wchar_t *InApplicationDataPath, ID3D12Device *InDevice, const NVSDK_NGX_FeatureCommonInfo *InFeatureInfo, NVSDK_NGX_Version InSDKVersion)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* device)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList *InCmdList, NVSDK_NGX_Feature InFeatureID, NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle *InHandle)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList *InCmdList, const NVSDK_NGX_Handle *InFeatureHandle, const NVSDK_NGX_Parameter *InParameters, PFN_NVSDK_NGX_ProgressCallback_C InCallback)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** OutParameters)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* InParameters)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters)
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

    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

void DlssIntegration::SetupDeviceExtensions(nri::DeviceCreationDesc& desc)
{
    static const char* vulkanExts[] = {
        "VK_NVX_binary_import",
        "VK_NVX_image_view_handle",
        "VK_KHR_push_descriptor"
    };

    desc.vulkanExtensions.deviceExtensions = vulkanExts;
    desc.vulkanExtensions.deviceExtensionNum = 3;
}

inline NVSDK_NGX_Resource_VK DlssIntegration::SetupVulkanTexture(const DlssTexture& texture, bool isStorage)
{
    VkImage image = (VkImage)NRI.GetTextureNativeObject(*texture.texture, 0);
    VkImageView view = (VkImageView)NRI.GetDescriptorNativeObject(*texture.descriptor, 0);
    VkImageSubresourceRange subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkFormat format = (VkFormat)nri::ConvertNRIFormatToVK(texture.format);
    
    return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, subresource, format, texture.dims.Width, texture.dims.Height, isStorage);
}

bool DlssIntegration::InitializeLibrary(nri::Device& device, const char* appDataPath, uint64_t applicationId)
{
    m_ApplicationId = applicationId;
    m_Device = &device;

    uint32_t nriResult = (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
    if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VULKAN)
        nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::WrapperVKInterface), (nri::WrapperVKInterface*)&NRI);

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
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
    {
        VkDevice vkDevice = (VkDevice)NRI.GetDeviceNativeObject(*m_Device);
        VkPhysicalDevice vkPhysicalDevice = (VkPhysicalDevice)NRI.GetVkPhysicalDevice(*m_Device);
        VkInstance vkInstance = (VkInstance)NRI.GetVkInstance(*m_Device);
        result = NVSDK_NGX_VULKAN_Init(m_ApplicationId, path, vkInstance, vkPhysicalDevice, vkDevice);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        ID3D11Device* d3d11Device = (ID3D11Device*)NRI.GetDeviceNativeObject(*m_Device);
        result = NVSDK_NGX_D3D11_Init(m_ApplicationId, path, d3d11Device);
    }

    if (NVSDK_NGX_SUCCEED(result))
    {
        if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
            result = NVSDK_NGX_D3D12_GetCapabilityParameters(&m_NgxParameters);
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
            result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_NgxParameters);
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
            result = NVSDK_NGX_D3D11_GetCapabilityParameters(&m_NgxParameters);

        if (NVSDK_NGX_SUCCEED(result))
        {
            uint32_t needsUpdatedDriver = 1;
            result = m_NgxParameters->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);

            uint32_t dlssAvailable = 0;
            result = m_NgxParameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);

            m_Initialized = NVSDK_NGX_SUCCEED(result) && dlssAvailable && !needsUpdatedDriver;
        }
    }

    if (!m_Initialized)
        Shutdown();

    return m_Initialized;
}

bool DlssIntegration::GetOptimalSettings(const NVSDK_NGX_Dimensions& outputResolution, DlssQuality quality, DlssSettings& outSettings) const
{
    NVSDK_NGX_PerfQuality_Value dlssQuality = DLSS_ConvertQuality(quality);

    NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        m_NgxParameters,
        outputResolution.Width, outputResolution.Height,
        dlssQuality,
        &outSettings.renderResolution.Width, &outSettings.renderResolution.Height,
        &outSettings.maxRenderResolution.Width, &outSettings.maxRenderResolution.Height,
        &outSettings.minRenderResolution.Width, &outSettings.minRenderResolution.Height,
        &outSettings.sharpness);

    return NVSDK_NGX_SUCCEED(result) && outSettings.renderResolution.Width != 0 && outSettings.renderResolution.Height != 0;
}

bool DlssIntegration::Initialize(nri::CommandQueue* commandQueue, const DlssInitDesc& desc)
{
    assert(m_Initialized);

    nri::CommandAllocator* commandAllocator;
    NRI.CreateCommandAllocator(*commandQueue, 0, commandAllocator);

    nri::CommandBuffer* commandBuffer;
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    int32_t flags = NVSDK_NGX_DLSS_Feature_Flags_DoSharpening;
    flags |= desc.isMotionVectorAtLowRes ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
    flags |= desc.isContentHDR ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
    flags |= desc.isDepthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
    flags |= (desc.enableAutoExposure && desc.isContentHDR) ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;

    DlssSettings settings = {};
    if (!GetOptimalSettings(desc.outputResolution, desc.quality, settings))
        return false;

    NVSDK_NGX_DLSS_Create_Params dlssCreateParams = {};
    dlssCreateParams.Feature.InWidth = settings.renderResolution.Width;
    dlssCreateParams.Feature.InHeight = settings.renderResolution.Height;
    dlssCreateParams.Feature.InTargetWidth = desc.outputResolution.Width;
    dlssCreateParams.Feature.InTargetHeight = desc.outputResolution.Height;
    dlssCreateParams.Feature.InPerfQualityValue = DLSS_ConvertQuality(desc.quality);
    dlssCreateParams.InFeatureCreateFlags = flags;

    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    NRI.BeginCommandBuffer(*commandBuffer, nullptr, 0);
    {
        uint32_t creationNodeMask = 1;
        uint32_t visibilityNodeMask = 1;

        const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
        if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
        {
            ID3D12GraphicsCommandList* d3d12CommandList = (ID3D12GraphicsCommandList*)NRI.GetCommandBufferNativeObject(*commandBuffer);
            result = NGX_D3D12_CREATE_DLSS_EXT(d3d12CommandList, creationNodeMask, visibilityNodeMask, &m_DLSS, m_NgxParameters, &dlssCreateParams);
        }
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
        {
            VkCommandBuffer vkCommandBuffer = (VkCommandBuffer)NRI.GetCommandBufferNativeObject(*commandBuffer);
            result = NGX_VULKAN_CREATE_DLSS_EXT(vkCommandBuffer, creationNodeMask, visibilityNodeMask, &m_DLSS, m_NgxParameters, &dlssCreateParams);
        }
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
        {
            ID3D11DeviceContext* d3d11DeviceContext = (ID3D11DeviceContext*)NRI.GetCommandBufferNativeObject(*commandBuffer);
            result = NGX_D3D11_CREATE_DLSS_EXT(d3d11DeviceContext, &m_DLSS, m_NgxParameters, &dlssCreateParams);
        }
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;

    NRI.QueueSubmit(*commandQueue, queueSubmitDesc);

    nri::Fence* fence;
    NRI.CreateFence(*m_Device, 0, fence);
    NRI.QueueSignal(*commandQueue, *fence, 1);
    NRI.Wait(*fence, 1);
    NRI.DestroyFence(*fence);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    return NVSDK_NGX_SUCCEED(result);
}

void DlssIntegration::Evaluate(nri::CommandBuffer* commandBuffer, const DlssDispatchDesc& desc)
{
    assert(m_Initialized);

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        ID3D12Resource* resourceInput = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texInput.texture, 0);
        ID3D12Resource* resourceMv = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texMv.texture, 0);
        ID3D12Resource* resourceDepth = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texDepth.texture, 0);
        ID3D12Resource* resourceOutput = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texOutput.texture, 0);

        NVSDK_NGX_D3D12_DLSS_Eval_Params dlssEvalParams = {};
        dlssEvalParams.Feature.pInColor = resourceInput;
        dlssEvalParams.Feature.pInOutput = resourceOutput;
        dlssEvalParams.Feature.InSharpness = desc.sharpness;
        dlssEvalParams.pInDepth = resourceDepth;
        dlssEvalParams.pInMotionVectors = resourceMv;
        dlssEvalParams.InRenderSubrectDimensions = desc.currentRenderResolution;
        dlssEvalParams.InJitterOffsetX = desc.jitter[0];
        dlssEvalParams.InJitterOffsetY = desc.jitter[1];
        dlssEvalParams.InReset = desc.reset;
        dlssEvalParams.InMVScaleX = desc.motionVectorScale[0];
        dlssEvalParams.InMVScaleY = desc.motionVectorScale[1];

        if (desc.texExposure.texture)
        {
            ID3D12Resource* resourceExposure = (ID3D12Resource*)NRI.GetTextureNativeObject(*desc.texExposure.texture, 0);
            dlssEvalParams.pInExposureTexture = resourceExposure;
        }

        ID3D12GraphicsCommandList* d3dCommandList = (ID3D12GraphicsCommandList*)NRI.GetCommandBufferNativeObject(*commandBuffer);
        result = NGX_D3D12_EVALUATE_DLSS_EXT(d3dCommandList, m_DLSS, m_NgxParameters, &dlssEvalParams);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
    {
        NVSDK_NGX_Resource_VK resourceInput = SetupVulkanTexture(desc.texInput, false);
        NVSDK_NGX_Resource_VK resourceMv = SetupVulkanTexture(desc.texMv, false);
        NVSDK_NGX_Resource_VK resourceDepth = SetupVulkanTexture(desc.texDepth, false);
        NVSDK_NGX_Resource_VK resourceOutput = SetupVulkanTexture(desc.texOutput, true);

        NVSDK_NGX_VK_DLSS_Eval_Params dlssEvalParams = {};
        dlssEvalParams.Feature.pInColor = &resourceInput;
        dlssEvalParams.Feature.pInOutput = &resourceOutput;
        dlssEvalParams.Feature.InSharpness = desc.sharpness;
        dlssEvalParams.pInDepth = &resourceDepth;
        dlssEvalParams.pInMotionVectors = &resourceMv;
        dlssEvalParams.InRenderSubrectDimensions = desc.currentRenderResolution;
        dlssEvalParams.InJitterOffsetX = desc.jitter[0];
        dlssEvalParams.InJitterOffsetY = desc.jitter[1];
        dlssEvalParams.InReset = desc.reset;
        dlssEvalParams.InMVScaleX = desc.motionVectorScale[0];
        dlssEvalParams.InMVScaleY = desc.motionVectorScale[1];

        NVSDK_NGX_Resource_VK resourceExposure;
        if (desc.texExposure.texture)
        {
            resourceExposure = SetupVulkanTexture(desc.texExposure, false);
            dlssEvalParams.pInExposureTexture = &resourceExposure;
        }

        VkCommandBuffer vkCommandbuffer = (VkCommandBuffer)NRI.GetCommandBufferNativeObject(*commandBuffer);
        result = NGX_VULKAN_EVALUATE_DLSS_EXT(vkCommandbuffer, m_DLSS, m_NgxParameters, &dlssEvalParams);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        ID3D11Resource* resourceInput = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texInput.texture, 0);
        ID3D11Resource* resourceMv = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texMv.texture, 0);
        ID3D11Resource* resourceDepth = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texDepth.texture, 0);
        ID3D11Resource* resourceOutput = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texOutput.texture, 0);

        NVSDK_NGX_D3D11_DLSS_Eval_Params dlssEvalParams = {};
        dlssEvalParams.Feature.pInColor = resourceInput;
        dlssEvalParams.Feature.pInOutput = resourceOutput;
        dlssEvalParams.Feature.InSharpness = desc.sharpness;
        dlssEvalParams.pInDepth = resourceDepth;
        dlssEvalParams.pInMotionVectors = resourceMv;
        dlssEvalParams.InRenderSubrectDimensions = desc.currentRenderResolution;
        dlssEvalParams.InJitterOffsetX = desc.jitter[0];
        dlssEvalParams.InJitterOffsetY = desc.jitter[1];
        dlssEvalParams.InReset = desc.reset;
        dlssEvalParams.InMVScaleX = desc.motionVectorScale[0];
        dlssEvalParams.InMVScaleY = desc.motionVectorScale[1];

        if (desc.texExposure.texture)
        {
            ID3D11Resource* resourceExposure = (ID3D11Resource*)NRI.GetTextureNativeObject(*desc.texExposure.texture, 0);
            dlssEvalParams.pInExposureTexture = resourceExposure;
        }

        ID3D11DeviceContext* d3d11DeviceContext = (ID3D11DeviceContext*)NRI.GetCommandBufferNativeObject(*commandBuffer);
        result = NGX_D3D11_EVALUATE_DLSS_EXT(d3d11DeviceContext, m_DLSS, m_NgxParameters, &dlssEvalParams);
    }

    assert( NVSDK_NGX_SUCCEED(result) );
}

void DlssIntegration::Shutdown()
{
    if (!m_Device || !m_Initialized)
        return;

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        if (m_NgxParameters)
            NVSDK_NGX_D3D12_DestroyParameters(m_NgxParameters);

        if (m_DLSS)
            NVSDK_NGX_D3D12_ReleaseFeature(m_DLSS);

        NVSDK_NGX_D3D12_Shutdown1(nullptr);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
    {
        if (m_NgxParameters)
            NVSDK_NGX_VULKAN_DestroyParameters(m_NgxParameters);

        if (m_DLSS)
            NVSDK_NGX_VULKAN_ReleaseFeature(m_DLSS);

        NVSDK_NGX_VULKAN_Shutdown1(nullptr);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        if (m_NgxParameters)
            NVSDK_NGX_D3D11_DestroyParameters(m_NgxParameters);

        if (m_DLSS)
            NVSDK_NGX_D3D11_ReleaseFeature(m_DLSS);

        NVSDK_NGX_D3D11_Shutdown1(nullptr);
    }

    m_NgxParameters = nullptr;
    m_DLSS = nullptr;
    m_Device = nullptr;
    m_Initialized = false;
}
