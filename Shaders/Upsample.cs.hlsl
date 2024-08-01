/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_Image, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Validation, t, 1, 1 );

NRI_RESOURCE( RWTexture2D<float3>, gOut_Image, u, 0, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvWindowSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // DLSS upscales render resolution to output resolution
    float2 rectSize = gRectSize;
    float2 invRenderSize = gInvRenderSize;
    if( gSR || gRR )
    {
        rectSize = gOutputSize;
        invRenderSize = gInvOutputSize;
    }

    // Upsampling
    float2 pixelPosScaled = pixelUv * rectSize;
    float3 upsampled = BicubicFilterNoCorners( gIn_Image, gLinearSampler, pixelPosScaled, invRenderSize, 0.66 ).xyz;

    // Split screen - noisy input / denoised output
    float3 input = gIn_Image.SampleLevel( gNearestSampler, pixelPosScaled * invRenderSize, 0 ).xyz;
    float3 result = pixelUv.x < gSeparator ? input : upsampled;

    // Split screen - vertical line
    float verticalLine = saturate( 1.0 - abs( pixelUv.x - gSeparator ) * gWindowSize.x / 3.5 );
    verticalLine = saturate( verticalLine / 0.5 );
    verticalLine *= float( gSeparator != 0.0 );

    const float3 nvColor = float3( 118.0, 185.0, 0.0 ) / 255.0;
    result = lerp( result, nvColor * verticalLine, verticalLine );

    // Validation layer
    if( gValidation )
    {
        float4 validation = gIn_Validation.SampleLevel( gLinearSampler, pixelUv, 0 );
        result = lerp( result, validation.xyz, validation.w );
    }

    // Output
    gOut_Image[ pixelPos ] = result;
}