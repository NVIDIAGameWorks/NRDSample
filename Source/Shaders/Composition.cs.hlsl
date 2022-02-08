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
NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 0, 1 );
NRI_RESOURCE( Texture2D<float>, gIn_Downsampled_ViewZ, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 2, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 3, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_DirectLighting_ViewZ, t, 4, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 5, 1 );
NRI_RESOURCE( Texture2D<float2>, gIn_IntegratedBRDF, t, 6, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Diff, t, 7, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Spec, t, 8, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float4>, gOut_ComposedImage, u, 9, 1 );

float4 Upsample( Texture2D<float4> tex, float2 pixelUv, float zReal )
{
    // Set to 1 if you don't use a quarter part of the full texture
    float2 RESOLUTION_SCALE = 0.5 * gRectSize * gInvScreenSize;

    float4 zLow = gIn_Downsampled_ViewZ.GatherRed( gNearestMipmapNearestSampler, pixelUv * RESOLUTION_SCALE );
    float4 delta = abs( zReal - zLow ) / zReal;

    float4 offsets = float4( -1.0, 1.0, -1.0, 1.0 ) * gInvRectSize.xxyy;
    float3 d01 = float3( offsets.xw, delta.x );
    float3 d11 = float3( offsets.yw, delta.y );
    float3 d10 = float3( offsets.yz, delta.z );
    float3 d00 = float3( offsets.xz, delta.w );

    d00 = lerp( d01, d00, step( d00.z, d01.z ) );
    d00 = lerp( d11, d00, step( d00.z, d11.z ) );
    d00 = lerp( d10, d00, step( d00.z, d10.z ) );

    float2 invTexSize = 2.0 * gInvRectSize;
    float2 uvNearest = floor( ( pixelUv + d00.xy ) / invTexSize ) * invTexSize + gInvRectSize;

    float2 uv = pixelUv * gRectSize / gScreenSize;
    if( gTracingMode == RESOLUTION_QUARTER )
    {
        float4 cmp = step( 0.01, delta );
        cmp.x = saturate( dot( cmp, 1.0 ) );

        uv = lerp( pixelUv, uvNearest, cmp.x );
        uv *= RESOLUTION_SCALE;
    }

    return tex.SampleLevel( gLinearSampler, uv, 0 );
}

[numthreads( 16, 16, 1)]
void main( int2 pixelPos : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
        return;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Early out - sky
    float4 Ldirect = gIn_DirectLighting_ViewZ[ pixelPos ];
    float viewZ = gIn_ViewZ[ pixelPos ];

    if( abs( viewZ ) == INF )
    {
        Ldirect.xyz *= float( gOnScreen == SHOW_FINAL );
        gOut_ComposedImage[ pixelPos ] = Ldirect;

        return;
    }

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ] );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    float3 albedo, Rf0;
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo, Rf0 );

    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gIsOrtho );
    float3 X = STL::Geometry::RotateVector( gViewToWorld, Xv );
    float3 V = GetViewVector( X );

    // Denoised indirect lighting
    float4 diffIndirect = Upsample( gIn_Diff, pixelUv, viewZ );
    float4 specIndirect = Upsample( gIn_Spec, pixelUv, viewZ );

    [flatten]
    if( gOcclusionOnly )
    {
        diffIndirect = diffIndirect.xxxx;
        specIndirect = specIndirect.xxxx;
    }

    diffIndirect = gDenoiserType != REBLUR ? RELAX_BackEnd_UnpackRadianceAndHitDist( diffIndirect ) : REBLUR_BackEnd_UnpackRadianceAndHitDist( diffIndirect );
    diffIndirect.xyz *= gIndirectDiffuse;

    specIndirect = gDenoiserType != REBLUR ? RELAX_BackEnd_UnpackRadianceAndHitDist( specIndirect ) : REBLUR_BackEnd_UnpackRadianceAndHitDist( specIndirect );
    specIndirect.xyz *= gIndirectSpecular;

    // Environment ( pre-integrated ) specular term
    float NoV = abs( dot( N, V ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, roughness );

    // Add indirect lighting
    float3 Lsum = Ldirect.xyz;
    Lsum += diffIndirect.xyz * albedo;
    Lsum += specIndirect.xyz * F;

    // Add ambient // TODO: reduce if multi bounce?
    float3 ambient = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    ambient *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( X - gCameraOrigin ) );
    ambient *= gAmbientInComposition;

    float2 gg = gIn_IntegratedBRDF.SampleLevel( gLinearSampler, float2( NoV, roughness ), 0 );
    diffIndirect.w *= gg.x;
    specIndirect.w *= gg.y;

    float m = roughness * roughness;
    Lsum += ambient * diffIndirect.w * albedo * ( 1 - F );
    Lsum += ambient * specIndirect.w * m * F;

    // Debug
    if( gOnScreen == SHOW_DENOISED_DIFFUSE )
        Lsum = diffIndirect.xyz;
    else if( gOnScreen == SHOW_DENOISED_SPECULAR )
        Lsum = specIndirect.xyz;
    else if( gOnScreen == SHOW_AMBIENT_OCCLUSION )
        Lsum = diffIndirect.w;
    else if( gOnScreen == SHOW_SPECULAR_OCCLUSION )
        Lsum = specIndirect.w;
    else if( gOnScreen == SHOW_BASE_COLOR )
        Lsum = baseColorMetalness.xyz;
    else if( gOnScreen == SHOW_NORMAL )
        Lsum = N * 0.5 + 0.5;
    else if( gOnScreen == SHOW_ROUGHNESS )
        Lsum = roughness;
    else if( gOnScreen == SHOW_METALNESS )
        Lsum = baseColorMetalness.w;
    else if( gOnScreen == SHOW_WORLD_UNITS )
        Lsum = frac( X * gUnitToMetersMultiplier );
    else if( gOnScreen != SHOW_FINAL )
        Lsum = gOnScreen == SHOW_MIP_SPECULAR ? specIndirect.xyz : Ldirect.xyz;

    // Output
    gOut_ComposedImage[ pixelPos ] = float4( Lsum, abs( viewZ ) * NRD_FP16_VIEWZ_SCALE * STL::Math::Sign( dot( N, gSunDirection ) ) );
}
