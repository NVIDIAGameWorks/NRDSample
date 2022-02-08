/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_DirectEmission, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Shadow, t, 1, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float4>, gInOut_DirectLighting, u, 2, 1 );

[numthreads( 16, 16, 1)]
void main( int2 pixelPos : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
        return;

    float4 shadowData = gIn_Shadow[ pixelPos ];
    shadowData = SIGMA_BackEnd_UnpackShadow( shadowData );
    float3 shadow = lerp( shadowData.yzw, 1.0, shadowData.x );

    float4 temp = gInOut_DirectLighting[ pixelPos ];
    float3 Ldirect = temp.xyz;

    float3 Lemi = gIn_DirectEmission[ pixelPos ];
    float3 Lsum = Ldirect * shadow + Lemi;

    // Debug
    if( gOnScreen == SHOW_SHADOW )
        Lsum = shadow;
    else if( gOnScreen >= SHOW_MESH )
        Lsum = Ldirect;

    // Output
    gInOut_DirectLighting[ pixelPos ] = float4( Lsum, temp.w );
}
