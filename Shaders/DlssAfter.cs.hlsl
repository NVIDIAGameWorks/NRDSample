/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

NRI_RESOURCE( RWTexture2D<float3>, gOut_Image, u, 0, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvOutputSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    float3 color = gOut_Image[ pixelPos ];

    color = ApplyTonemap( color );
    if( gIsSrgb )
        color = Color::ToSrgb( saturate( color ) );

    // Output
    gOut_Image[ pixelPos ] = color;
}
