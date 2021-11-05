/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 0, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_DirectLighting, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 2, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 3, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Shadow, t, 4, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Diff, t, 5, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Spec, t, 6, 1 );
NRI_RESOURCE( Texture2D<float2>, gIn_IntegratedBRDF, t, 7, 1 );

NRI_RESOURCE( RWTexture2D<float4>, gOut_ComposedImage, u, 8, 1 );

[numthreads( 16, 16, 1)]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    // Normal
    float4 normalAndRoughness = gIn_Normal_Roughness[ pixelPos ];
    float isGround = float( dot( normalAndRoughness.xyz, normalAndRoughness.xyz ) != SKY_MARK );
    float4 t = UnpackNormalAndRoughness( normalAndRoughness );
    float3 N = t.xyz;
    float roughness = t.w;

    // Material
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];

    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo, Rf0 );

    // To be used in indirect (!) lighting math
    albedo /= STL::ImportanceSampling::Cosine::GetPDF( );
    albedo /= STL::Math::Pi( 1.0 );

    // Denoised data
    float4 diffIndirect = gIn_Diff[ pixelPos ];
    float4 specIndirect = gIn_Spec[ pixelPos ];

    [flatten]
    if( gOcclusionOnly )
    {
        diffIndirect = diffIndirect.xxxx;
        specIndirect = specIndirect.xxxx;
    }

    diffIndirect = gDenoiserType != REBLUR ? RELAX_BackEnd_UnpackRadiance( diffIndirect ) : REBLUR_BackEnd_UnpackRadiance( diffIndirect );
    diffIndirect.xyz *= gIndirectDiffuse;

    specIndirect = gDenoiserType != REBLUR ? RELAX_BackEnd_UnpackRadiance( specIndirect ) : REBLUR_BackEnd_UnpackRadiance( specIndirect );
    specIndirect.xyz *= gIndirectSpecular;

    float4 shadowData = gIn_Shadow[ pixelPos ];
    shadowData = SIGMA_BackEnd_UnpackShadow( shadowData );
    float3 shadow = lerp( shadowData.yzw, 1.0, shadowData.x );

    // Good denoisers do nothing with sky...
    shadow = lerp( 1.0, shadow, isGround );
    diffIndirect *= isGround;
    specIndirect *= isGround;

    // Direct lighting and emission
    float3 directLighting = gIn_DirectLighting[ pixelPos ];
    float3 Lsum = directLighting * shadow;

    // Environment (pre-integrated) specular term
    float viewZ = gIn_ViewZ[ pixelPos ];
    float3 Vv = STL::Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZ, gIsOrtho );
    float3 V = -STL::Geometry::RotateVector( gViewToWorld, normalize( Vv ) );
    float NoV = abs( dot( N, V ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, roughness );

    // Add indirect lighting
    Lsum += diffIndirect.xyz * albedo;
    Lsum += specIndirect.xyz * F;

    // Add ambient
    float2 GG = gIn_IntegratedBRDF.SampleLevel( gLinearSampler, float2( NoV, roughness ), 0 );
    float m = roughness * roughness;

    diffIndirect.w *= GG.x;
    Lsum += gAmbientInComposition * gAmbient * diffIndirect.w * albedo;

    specIndirect.w *= GG.y; // Throughput can be applied during tracing to "normHitDist" but SO will get more blurry look, plus, it's not needed for specular virtual motion tracking
    Lsum += gAmbientInComposition * gAmbient * specIndirect.w * m * STL::BRDF::EnvironmentTerm_Unknown( Rf0, NoV, roughness ); // Works better for low roughness, than Ross

    // Debug
    if( gOnScreen == SHOW_AMBIENT_OCCLUSION )
        Lsum = diffIndirect.w;
    else if( gOnScreen == SHOW_SPECULAR_OCCLUSION )
        Lsum = specIndirect.w;
    else if( gOnScreen == SHOW_SHADOW )
        Lsum = shadow;
    else if( gOnScreen == SHOW_BASE_COLOR )
        Lsum = baseColorMetalness.xyz;
    else if( gOnScreen == SHOW_NORMAL )
        Lsum = N * 0.5 + 0.5;
    else if( gOnScreen == SHOW_ROUGHNESS )
        Lsum = roughness;
    else if( gOnScreen == SHOW_METALNESS )
        Lsum = baseColorMetalness.w;
    else if (gOnScreen == SHOW_DENOISED_DIFFUSE)
        Lsum = diffIndirect.xyz;
    else if (gOnScreen == SHOW_DENOISED_SPECULAR)
        Lsum = specIndirect.xyz;
    else if( gOnScreen >= SHOW_WORLD_UNITS )
        Lsum = gOnScreen == SHOW_MIP_SPECULAR ? specIndirect.xyz : ( directLighting * isGround );

    // Output
    gOut_ComposedImage[ pixelPos ] = float4( Lsum, abs( viewZ ) * NRD_FP16_VIEWZ_SCALE * STL::Math::Sign( dot( N, gSunDirection ) ) );
}
