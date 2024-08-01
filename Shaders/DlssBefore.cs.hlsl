/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float4>, gIn_ComposedLighting_ViewZ, t, 0, 1 );

NRI_RESOURCE( RWTexture2D<float>, gOut_ViewZ, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gInOut_Mv, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_FinalImage, u, 2, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = ( float2( pixelPos ) + 0.5 ) * gInvRenderSize;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    float viewZ = gOut_ViewZ[ pixelPos ];
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZ, gOrthoMode );

    // Recalculate viewZ to depth ( needed for SR )
    if( gSR )
    {
        float4 clipPos = Geometry::ProjectiveTransform( gViewToClip, Xv );
        gOut_ViewZ[ pixelPos ] = clipPos.z / clipPos.w;
    }

    // Patch MV, because 2D MVs needed
    float3 mv = gInOut_Mv[ pixelPos ];
    if( gIsWorldSpaceMotionEnabled )
    {
        float3 Xprev = Geometry::AffineTransform( gViewToWorld, Xv ) + mv;
        float2 pixelUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, Xprev );
        mv.xy = ( pixelUvPrev - pixelUv ) * gRenderSize;
        gInOut_Mv[ pixelPos ] = mv;
    }

    // Apply exposure
    float3 Lsum = gIn_ComposedLighting_ViewZ[ pixelPos ].xyz;
    Lsum = ApplyExposure( Lsum );

    // Output
    gOut_FinalImage[ pixelPos ] = Lsum;
}