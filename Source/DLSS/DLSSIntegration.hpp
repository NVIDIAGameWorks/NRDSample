#include "DLSSIntegration.h"

// An ugly temp workaround until DLSS fix the problem
#ifndef _WIN32

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Init(unsigned long long InApplicationId, const wchar_t *InApplicationDataPath, ID3D11Device *InDevice, const NVSDK_NGX_FeatureCommonInfo *InFeatureInfo, NVSDK_NGX_Version InSDKVersion)
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Shutdown()
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_CreateFeature(ID3D11DeviceContext *InDevCtx, NVSDK_NGX_Feature InFeatureID, const NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle)
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

NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown()
{ return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList *InCmdList, NVSDK_NGX_Feature InFeatureID, const NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle)
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

    if (quality == DlssQuality::ULTRA_QUALITY)
        return NVSDK_NGX_PerfQuality_Value_UltraQuality;

    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

inline NVSDK_NGX_Resource_VK DlssIntegration::SetupVulkanTexture(nri::Texture* texture, nri::Descriptor* descriptor, uint32_t physicalDeviceIndex, bool isStorage)
{
    nri::TextureVulkanDesc textureDesc = {};
    NRI.GetTextureVK(*texture, physicalDeviceIndex, textureDesc);

    VkImageSubresourceRange subresource = {};
    const VkImageView view = (VkImageView)NRI.GetTextureDescriptorVK(*descriptor, physicalDeviceIndex, subresource);

    return NVSDK_NGX_Create_ImageView_Resource_VK(view, (VkImage)textureDesc.vkImage, subresource, (VkFormat)textureDesc.vkFormat, textureDesc.size[0], textureDesc.size[1], isStorage);
}

bool DlssIntegration::InitializeLibrary(nri::Device& device, const char* appDataPath, uint64_t applicationId)
{
    m_ApplicationId = applicationId;
    m_Device = &device;

    uint32_t nriResult = (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
    if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
        nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::WrapperD3D12Interface), (nri::WrapperD3D12Interface*)&NRI);
    else if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VULKAN)
        nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::WrapperVKInterface), (nri::WrapperVKInterface*)&NRI);
    else if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D11)
        nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::WrapperD3D11Interface), (nri::WrapperD3D11Interface*)&NRI);

    if ((nri::Result)nriResult != nri::Result::SUCCESS)
        return false;

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    wchar_t path[512];
    DLSS_ConvertCharToWchar(appDataPath, path, 512);
    wchar_t* paths[1] = { path };

    NVSDK_NGX_FeatureCommonInfo commonInfo = {};
    commonInfo.PathListInfo.Length = 1;
    commonInfo.PathListInfo.Path = paths;

    NVSDK_NGX_Result result = NVSDK_NGX_Result::NVSDK_NGX_Result_Fail;
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        ID3D12Device* d3d12Device = NRI.GetDeviceD3D12(*m_Device);
        result = NVSDK_NGX_D3D12_Init(m_ApplicationId, path, d3d12Device, &commonInfo);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
    {
        VkDevice vkDevice = (VkDevice)NRI.GetDeviceVK(*m_Device);
        VkPhysicalDevice vkPhysicalDevice = (VkPhysicalDevice)NRI.GetPhysicalDeviceVK(*m_Device);
        VkInstance vkInstance = (VkInstance)NRI.GetInstanceVK(*m_Device);
        result = NVSDK_NGX_VULKAN_Init(m_ApplicationId, path, vkInstance, vkPhysicalDevice, vkDevice, &commonInfo);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        ID3D11Device* d3d11Device = NRI.GetDeviceD3D11(*m_Device);
        result = NVSDK_NGX_D3D11_Init(m_ApplicationId, path, d3d11Device, &commonInfo);
    }

    if (NVSDK_NGX_SUCCEED(result))
    {
        if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
        {
            result = NVSDK_NGX_D3D12_AllocateParameters(&m_NgxParameters);
            result = NVSDK_NGX_D3D12_GetCapabilityParameters(&m_NgxParameters);
        }
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
        {
            result = NVSDK_NGX_VULKAN_AllocateParameters(&m_NgxParameters);
            result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_NgxParameters);
        }
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
        {
            result = NVSDK_NGX_D3D11_AllocateParameters(&m_NgxParameters);
            result = NVSDK_NGX_D3D11_GetCapabilityParameters(&m_NgxParameters);
        }

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

    nri::DeviceSemaphore* semaphore;
    NRI.CreateDeviceSemaphore(*m_Device, false, semaphore);

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
            ID3D12GraphicsCommandList* d3d12CommandList = NRI.GetCommandBufferD3D12(*commandBuffer);
            result = NGX_D3D12_CREATE_DLSS_EXT(d3d12CommandList, creationNodeMask, visibilityNodeMask, &m_DLSS, m_NgxParameters, &dlssCreateParams);
        }
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
        {
            VkCommandBuffer vkCommandBuffer = (VkCommandBuffer)NRI.GetCommandBufferVK(*commandBuffer);
            result = NGX_VULKAN_CREATE_DLSS_EXT(vkCommandBuffer, creationNodeMask, visibilityNodeMask, &m_DLSS, m_NgxParameters, &dlssCreateParams);
        }
        else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
        {
            ID3D11DeviceContext* d3d11DeviceContext = NRI.GetCommandBufferD3D11(*commandBuffer);
            result = NGX_D3D11_CREATE_DLSS_EXT(d3d11DeviceContext, &m_DLSS, m_NgxParameters, &dlssCreateParams);
        }
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBuffers = &commandBuffer;
    workSubmissionDesc.commandBufferNum = 1;

    NRI.SubmitQueueWork(*commandQueue, workSubmissionDesc, semaphore);
    NRI.WaitForSemaphore(*commandQueue, *semaphore);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);
    NRI.DestroyDeviceSemaphore(*semaphore);

    return NVSDK_NGX_SUCCEED(result);
}

void DlssIntegration::Evaluate(nri::CommandBuffer* commandBuffer, const DlssDispatchDesc& desc)
{
    assert(m_Initialized);

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;
    if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D12)
    {
        ID3D12Resource* resourceInput = NRI.GetTextureD3D12(*desc.texInput);
        ID3D12Resource* resourceMv = NRI.GetTextureD3D12(*desc.texMv);
        ID3D12Resource* resourceDepth = NRI.GetTextureD3D12(*desc.texDepth);
        ID3D12Resource* resourceExposure = desc.texExposure ? NRI.GetTextureD3D12(*desc.texExposure) : nullptr;
        ID3D12Resource* resourceOutput = NRI.GetTextureD3D12(*desc.texOutput);

        NVSDK_NGX_D3D12_DLSS_Eval_Params d3d12DlssEvalParams = {};
        d3d12DlssEvalParams.Feature.pInColor = resourceInput;
        d3d12DlssEvalParams.Feature.pInOutput = resourceOutput;
        d3d12DlssEvalParams.Feature.InSharpness = desc.sharpness;
        d3d12DlssEvalParams.pInDepth = resourceDepth;
        d3d12DlssEvalParams.pInMotionVectors = resourceMv;
        d3d12DlssEvalParams.pInExposureTexture = resourceExposure;
        d3d12DlssEvalParams.InRenderSubrectDimensions = desc.renderOrScaledResolution;
        d3d12DlssEvalParams.InJitterOffsetX = desc.jitter[0];
        d3d12DlssEvalParams.InJitterOffsetY = desc.jitter[1];
        d3d12DlssEvalParams.InReset = desc.reset;
        d3d12DlssEvalParams.InMVScaleX = desc.motionVectorScale[0];
        d3d12DlssEvalParams.InMVScaleY = desc.motionVectorScale[1];

        ID3D12GraphicsCommandList* d3dCommandList = NRI.GetCommandBufferD3D12(*commandBuffer);
        result = NGX_D3D12_EVALUATE_DLSS_EXT(d3dCommandList, m_DLSS, m_NgxParameters, &d3d12DlssEvalParams);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
    {
        NVSDK_NGX_Resource_VK resourceInput = SetupVulkanTexture(desc.texInput, desc.descriptorInput, desc.physicalDeviceIndex, false);
        NVSDK_NGX_Resource_VK resourceMv = SetupVulkanTexture(desc.texMv, desc.descriptorMv, desc.physicalDeviceIndex, false);
        NVSDK_NGX_Resource_VK resourceDepth = SetupVulkanTexture(desc.texDepth, desc.descriptorDepth, desc.physicalDeviceIndex, false);
        NVSDK_NGX_Resource_VK resourceOutput = SetupVulkanTexture(desc.texOutput, desc.descriptorOutput, desc.physicalDeviceIndex, true);

        NVSDK_NGX_VK_DLSS_Eval_Params vkDlssEvalParams = {};
        vkDlssEvalParams.Feature.pInColor = &resourceInput;
        vkDlssEvalParams.Feature.pInOutput = &resourceOutput;
        vkDlssEvalParams.Feature.InSharpness = desc.sharpness;
        vkDlssEvalParams.pInDepth = &resourceDepth;
        vkDlssEvalParams.pInMotionVectors = &resourceMv;
        vkDlssEvalParams.InRenderSubrectDimensions = desc.renderOrScaledResolution;
        vkDlssEvalParams.InJitterOffsetX = desc.jitter[0];
        vkDlssEvalParams.InJitterOffsetY = desc.jitter[1];
        vkDlssEvalParams.InReset = desc.reset;
        vkDlssEvalParams.InMVScaleX = desc.motionVectorScale[0];
        vkDlssEvalParams.InMVScaleY = desc.motionVectorScale[1];

        if (desc.texExposure)
        {
            NVSDK_NGX_Resource_VK resourceExposure = SetupVulkanTexture(desc.texExposure, desc.descriptorExposure, desc.physicalDeviceIndex, false);
            vkDlssEvalParams.pInExposureTexture = &resourceExposure;
        }

        VkCommandBuffer vkCommandbuffer = (VkCommandBuffer)NRI.GetCommandBufferVK(*commandBuffer);
        result = NGX_VULKAN_EVALUATE_DLSS_EXT(vkCommandbuffer, m_DLSS, m_NgxParameters, &vkDlssEvalParams);
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        ID3D11Resource* resourceInput = NRI.GetTextureD3D11(*desc.texInput);
        ID3D11Resource* resourceMv = NRI.GetTextureD3D11(*desc.texMv);
        ID3D11Resource* resourceDepth = NRI.GetTextureD3D11(*desc.texDepth);
        ID3D11Resource* resourceExposure = desc.texExposure ? NRI.GetTextureD3D11(*desc.texExposure) : nullptr;
        ID3D11Resource* resourceOutput = NRI.GetTextureD3D11(*desc.texOutput);

        NVSDK_NGX_D3D11_DLSS_Eval_Params d3d11DlssEvalParams = {};
        d3d11DlssEvalParams.Feature.pInColor = resourceInput;
        d3d11DlssEvalParams.Feature.pInOutput = resourceOutput;
        d3d11DlssEvalParams.Feature.InSharpness = desc.sharpness;
        d3d11DlssEvalParams.pInDepth = resourceDepth;
        d3d11DlssEvalParams.pInMotionVectors = resourceMv;
        d3d11DlssEvalParams.pInExposureTexture = resourceExposure;
        d3d11DlssEvalParams.InRenderSubrectDimensions = desc.renderOrScaledResolution;
        d3d11DlssEvalParams.InJitterOffsetX = desc.jitter[0];
        d3d11DlssEvalParams.InJitterOffsetY = desc.jitter[1];
        d3d11DlssEvalParams.InReset = desc.reset;
        d3d11DlssEvalParams.InMVScaleX = desc.motionVectorScale[0];
        d3d11DlssEvalParams.InMVScaleY = desc.motionVectorScale[1];

        ID3D11DeviceContext* d3d11DeviceContext = NRI.GetCommandBufferD3D11(*commandBuffer);
        result = NGX_D3D11_EVALUATE_DLSS_EXT(d3d11DeviceContext, m_DLSS, m_NgxParameters, &d3d11DlssEvalParams);
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

        NVSDK_NGX_D3D12_Shutdown();
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::VULKAN)
    {
        if (m_NgxParameters)
            NVSDK_NGX_VULKAN_DestroyParameters(m_NgxParameters);

        if (m_DLSS)
            NVSDK_NGX_VULKAN_ReleaseFeature(m_DLSS);

        NVSDK_NGX_VULKAN_Shutdown();
    }
    else if (deviceDesc.graphicsAPI == nri::GraphicsAPI::D3D11)
    {
        if (m_NgxParameters)
            NVSDK_NGX_D3D11_DestroyParameters(m_NgxParameters);

        if (m_DLSS)
            NVSDK_NGX_D3D11_ReleaseFeature(m_DLSS);

        NVSDK_NGX_D3D11_Shutdown();
    }

    m_NgxParameters = nullptr;
    m_DLSS = nullptr;
    m_Device = nullptr;
    m_Initialized = false;
}
