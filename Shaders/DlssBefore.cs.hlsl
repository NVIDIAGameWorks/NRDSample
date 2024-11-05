/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

NRI_RESOURCE( RWTexture2D<float>, gInOut_ViewZ, u, 0, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = ( float2( pixelPos ) + 0.5 ) * gInvRenderSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    float viewZ = gInOut_ViewZ[ pixelPos ];
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZ, gOrthoMode );

    // Recalculate viewZ to depth ( needed for SR )
    if( gSR )
    {
        float4 clipPos = Geometry::ProjectiveTransform( gViewToClip, Xv );

        gInOut_ViewZ[ pixelPos ] = clipPos.z / clipPos.w;
    }
}