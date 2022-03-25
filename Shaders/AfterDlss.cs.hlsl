/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float3>, gIn_Image, t, 0, 1 );

NRI_RESOURCE( RWTexture2D<float3>, gOut_Image, u, 1, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = ( float2( pixelPos ) + 0.5 ) * gInvOutputSize;
    float3 Lsum = gIn_Image.SampleLevel( gLinearSampler, pixelUv, 0 );

    // Tonemap
    if( gOnScreen == SHOW_FINAL )
        Lsum = STL::Color::HdrToLinear_Uncharted( Lsum );

    // Conversion
    if( gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR )
        Lsum = STL::Color::LinearToSrgb( Lsum );

    gOut_Image[ pixelPos ] = Lsum;
}
