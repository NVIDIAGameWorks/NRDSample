/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_Image, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_NisData1, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_NisData2, t, 2, 1 );

NRI_RESOURCE( RWTexture2D<float4>, gOut_Image, u, 0, 1 );

// Strange stuff #1
#define samplerLinearClamp          gLinearSampler
#define in_texture                  gIn_Image
#define coef_scaler                 gIn_NisData1
#define coef_usm                    gIn_NisData2
#define out_texture                 gOut_Image
#define kDetectRatio                gNisDetectRatio
#define kDetectThres                gNisDetectThres
#define kMinContrastRatio           gNisMinContrastRatio
#define kRatioNorm                  gNisRatioNorm
#define kContrastBoost              gNisContrastBoost
#define kEps                        gNisEps
#define kSharpStartY                gNisSharpStartY
#define kSharpScaleY                gNisSharpScaleY
#define kSharpStrengthMin           gNisSharpStrengthMin
#define kSharpStrengthScale         gNisSharpStrengthScale
#define kSharpLimitMin              gNisSharpLimitMin
#define kSharpLimitScale            gNisSharpLimitScale
#define kScaleX                     gNisScaleX
#define kScaleY                     gNisScaleY
#define kDstNormX                   gNisDstNormX
#define kDstNormY                   gNisDstNormY
#define kSrcNormX                   gNisSrcNormX
#define kSrcNormY                   gNisSrcNormY
#define kInputViewportOriginX       gNisInputViewportOriginX
#define kInputViewportOriginY       gNisInputViewportOriginY
#define kInputViewportWidth         gNisInputViewportWidth
#define kInputViewportHeight        gNisInputViewportHeight
#define kOutputViewportOriginX      gNisOutputViewportOriginX
#define kOutputViewportOriginY      gNisOutputViewportOriginY
#define kOutputViewportWidth        gNisOutputViewportWidth
#define kOutputViewportHeight       gNisOutputViewportHeight

// Strange stuff #2
#pragma warning( disable : 3557 )
#pragma warning( disable : 3203 )

// Main
#define NIS_SCALER                  1
#define NIS_HDR_MODE                0
#define NIS_BLOCK_WIDTH             32
#define NIS_BLOCK_HEIGHT            32
#define NIS_THREAD_GROUP_SIZE       128
#define NIS_USE_HALF_PRECISION      1
#define NIS_HLSL                    1
#define NIS_VIEWPORT_SUPPORT        0

#ifndef COMPILER_FXC
    #define NIS_HLSL_6_2            1
#endif

#include "../../External/NGX/NVIDIAImageScaling/NIS/NIS_Scaler.h"

[numthreads( NIS_THREAD_GROUP_SIZE, 1, 1 )]
void main( uint2 blockId : SV_GroupID, uint threadId : SV_GroupThreadID )
{
    NVScaler( blockId, threadId );
}
