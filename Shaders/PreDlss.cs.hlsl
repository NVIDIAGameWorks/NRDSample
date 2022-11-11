/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float3>, gIn_Mv, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedLighting_ViewZ, t, 1, 1 );

NRI_RESOURCE( RWTexture2D<float>, gOut_ViewZ, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float2>, gOut_SurfaceMotion, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_FinalImage, u, 2, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = ( float2( pixelPos ) + 0.5 ) * gInvRenderSize;

    // ViewZ to depth
    float viewZ = gOut_ViewZ[ pixelPos ];
    float3 Xv = STL::Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZ, gOrthoMode );
    float4 clipPos = STL::Geometry::ProjectiveTransform( gViewToClip, Xv );
    gOut_ViewZ[ pixelPos ] = clipPos.z / clipPos.w;

    // Object to surface motion
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 motionVector = gIn_Mv[ pixelPos ] * ( gIsWorldSpaceMotionEnabled ? 1.0 : gInvRenderSize.xyy );
    float2 pixelUvPrev = STL::Geometry::GetPrevUvFromMotion( pixelUv, X, gWorldToClipPrev, motionVector, gIsWorldSpaceMotionEnabled );
    float2 pixelMotion = ( pixelUvPrev - pixelUv ) * gRenderSize;
    gOut_SurfaceMotion[ pixelPos ] = pixelMotion;

    // Post lighting composition
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    float3 Lsum = gIn_ComposedLighting_ViewZ[ pixelPos ].xyz;
    Lsum = ApplyExposure( Lsum, false );

    // Output
    gOut_FinalImage[ pixelPos ] = Lsum;
}