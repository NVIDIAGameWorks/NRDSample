/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 2, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_DirectLighting, t, 3, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_DirectEmission, t, 4, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_PsrThroughput, t, 5, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 6, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Shadow, t, 7, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Diff, t, 8, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Spec, t, 9, 1 );
#if( NRD_MODE == SH )
    NRI_RESOURCE( Texture2D<float4>, gIn_DiffSh, t, 10, 1 );
    NRI_RESOURCE( Texture2D<float4>, gIn_SpecSh, t, 11, 1 );
#endif

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_ComposedDiff, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_ComposedSpec_ViewZ, u, 1, 1 );

[numthreads( 16, 16, 1)]
void main( int2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // ViewZ
    float viewZ = gIn_ViewZ[ pixelPos ];
    float3 Lemi = gIn_DirectEmission[ pixelPos ];

    // Normal, roughness and material ID
    float normMaterialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ], normMaterialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;

    // ( Trick ) Needed only to avoid back facing in "ReprojectIrradiance"
    float z = abs( viewZ ) * NRD_FP16_VIEWZ_SCALE;
    z *= STL::Math::Sign( dot( N, gSunDirection_gExposure.xyz ) );

    // Early out - sky
    if( abs( viewZ ) == INF )
    {
        gOut_ComposedDiff[ pixelPos ] = Lemi * float( gOnScreen == SHOW_FINAL );
        gOut_ComposedSpec_ViewZ[ pixelPos ] = float4( 0, 0, 0, z );

        return;
    }

    // Direct sun lighting * shadow + emission
    float4 shadowData = gIn_Shadow[ pixelPos ];
    shadowData = SIGMA_BackEnd_UnpackShadow( shadowData );
    float3 shadow = lerp( shadowData.yzw, 1.0, shadowData.x );

    float3 Ldirect = gIn_DirectLighting[ pixelPos ];
    if( gOnScreen < SHOW_INSTANCE_INDEX )
        Ldirect = Ldirect * shadow + Lemi;

    // G-buffer
    float3 albedo, Rf0;
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo, Rf0 );

    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gViewDirection_gOrthoMode.w );
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 V = gViewDirection_gOrthoMode.w == 0 ? normalize( gCameraOrigin_gMipBias.xyz - X ) : gViewDirection_gOrthoMode.xyz;

    // Sample NRD outputs
    float4 diff = gIn_Diff[ pixelPos ];
    float4 spec = gIn_Spec[ pixelPos ];

    #if( NRD_MODE == SH )
        float4 diff1 = gIn_DiffSh[ pixelPos ];
        float4 spec1 = gIn_SpecSh[ pixelPos ];
    #endif

    // Decode SH mode outputs
    #if( NRD_MODE == SH )
        NRD_SG diffSg = REBLUR_BackEnd_UnpackSh( diff, diff1 );
        NRD_SG specSg = REBLUR_BackEnd_UnpackSh( spec, spec1 );

        if( gDenoiserType == DENOISER_RELAX )
        {
            diffSg = RELAX_BackEnd_UnpackSh( diff, diff1 );
            specSg = RELAX_BackEnd_UnpackSh( spec, spec1 );
        }

        if( gResolve && pixelUv.x >= gSeparator )
        {
            // ( Optional ) replace "roughness" with "roughnessAA"
            roughness = NRD_SG_ExtractRoughnessAA( specSg );

            // Regain macro-details
            diff.xyz = NRD_SG_ResolveDiffuse( diffSg, N ) ; // or NRD_SH_ResolveDiffuse( sg, N )
            spec.xyz = NRD_SG_ResolveSpecular( specSg, N, V, roughness );

            // Regain micro-details & jittering // TODO: preload N and Z into SMEM
            float3 Ne = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2(  1,  0 ) ] ).xyz;
            float3 Nw = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( -1,  0 ) ] ).xyz;
            float3 Nn = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2(  0,  1 ) ] ).xyz;
            float3 Ns = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2(  0, -1 ) ] ).xyz;

            float Ze = gIn_ViewZ[ pixelPos + int2(  1,  0 ) ];
            float Zw = gIn_ViewZ[ pixelPos + int2( -1,  0 ) ];
            float Zn = gIn_ViewZ[ pixelPos + int2(  0,  1 ) ];
            float Zs = gIn_ViewZ[ pixelPos + int2(  0, -1 ) ];

            float2 scale = NRD_SG_ReJitter( diffSg, specSg, Rf0, V, roughness, viewZ, Ze, Zw, Zn, Zs, N, Ne, Nw, Nn, Ns );

            diff.xyz *= scale.x;
            spec.xyz *= scale.y;
        }
        else
        {
            diff.xyz = NRD_SG_ExtractColor( diffSg );
            spec.xyz = NRD_SG_ExtractColor( specSg );
        }

        // ( Optional ) AO / SO
        diff.w = diffSg.normHitDist;
        spec.w = specSg.normHitDist;
    // Decode OCCLUSION mode outputs
    #elif( NRD_MODE == OCCLUSION )
        diff.w = diff.x;
        spec.w = spec.x;
    // Decode DIRECTIONAL_OCCLUSION mode outputs
    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
        NRD_SG sg = REBLUR_BackEnd_UnpackDirectionalOcclusion( diff );

        if( gResolve )
        {
            // Regain macro-details
            diff.w = NRD_SG_ResolveDiffuse( sg, N ).x; // or NRD_SH_ResolveDiffuse( sg, N ).x

            // Regain micro-details // TODO: preload N and Z into SMEM
            float3 Ne = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( 1, 0 ) ] ).xyz;
            float3 Nw = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( -1, 0 ) ] ).xyz;
            float3 Nn = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( 0, 1 ) ] ).xyz;
            float3 Ns = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos + int2( 0, -1 ) ] ).xyz;

            float Ze = gIn_ViewZ[ pixelPos + int2( 1, 0 ) ];
            float Zw = gIn_ViewZ[ pixelPos + int2( -1, 0 ) ];
            float Zn = gIn_ViewZ[ pixelPos + int2( 0, 1 ) ];
            float Zs = gIn_ViewZ[ pixelPos + int2( 0, -1 ) ];

            float scale = NRD_SG_ReJitter( sg, sg, 0.0, V, 0.0, viewZ, Ze, Zw, Zn, Zs, N, Ne, Nw, Nn, Ns ).x;

            diff.w *= scale;
        }
        else
            diff.w = NRD_SG_ExtractColor( sg ).x;
    // Decode NORMAL mode outputs
    #else
        if( gDenoiserType == DENOISER_RELAX )
        {
            diff = RELAX_BackEnd_UnpackRadiance( diff );
            spec = RELAX_BackEnd_UnpackRadiance( spec );
        }
        else
        {
            diff = REBLUR_BackEnd_UnpackRadianceAndNormHitDist( diff );
            spec = REBLUR_BackEnd_UnpackRadianceAndNormHitDist( spec );
        }
    #endif

    // ( Optional ) RELAX doesn't support AO / SO
    if( gDenoiserType == DENOISER_RELAX )
    {
        diff.w = 1.0 / STL::Math::Pi( 1.0 );
        spec.w = 1.0 / STL::Math::Pi( 1.0 );
    }

    diff.xyz *= gIndirectDiffuse;
    spec.xyz *= gIndirectSpecular;

    // Environment ( pre-integrated ) specular term
    float NoV = abs( dot( N, V ) );
    float3 Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV, roughness );

    // Composition
    float3 diffDemod = ( 1.0 - Fenv ) * albedo * 0.99 + 0.01;
    float3 specDemod = Fenv * 0.99 + 0.01;

    float3 Ldiff = diff.xyz * diffDemod;
    float3 Lspec = spec.xyz * specDemod;

    // Ambient
    // TODO: drop ambient in the future and use a radiance cache instead, at least because in case of many RPP hit distance
    // is not averaged for specular, it's "min" across paths. It's needed for proper specular motion, but makes SO biased!
    float3 ambient = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    ambient *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( Xv ) );
    ambient *= gAmbient;

    float specAmbientAmount = gDenoiserType == DENOISER_RELAX ? roughness : GetSpecMagicCurve( roughness );

    Ldiff += ambient * diff.w * ( 1.0 - Fenv ) * albedo;
    Lspec += ambient * spec.w * Fenv * specAmbientAmount;

    // Apply PSR throughput ( primary surface material before replacement )
    #if( USE_PSR == 1 )
        float3 psrThroughput = gIn_PsrThroughput[ pixelPos ];
        Ldiff *= psrThroughput;
        Lspec *= psrThroughput;
        Ldirect *= psrThroughput;
    #else
        float3 psrThroughput = 1.0;
    #endif

    // IMPORTANT: we store diffuse and specular separately to be able to use the reprojection trick. Let's assume that direct lighting can always be reprojected as diffuse
    Ldiff += Ldirect;

    // Debug
    if( gOnScreen == SHOW_DENOISED_DIFFUSE )
        Ldiff = diff.xyz;
    else if( gOnScreen == SHOW_DENOISED_SPECULAR )
        Ldiff = spec.xyz;
    else if( gOnScreen == SHOW_AMBIENT_OCCLUSION )
        Ldiff = diff.w;
    else if( gOnScreen == SHOW_SPECULAR_OCCLUSION )
        Ldiff = spec.w;
    else if( gOnScreen == SHOW_SHADOW )
        Ldiff = shadow;
    else if( gOnScreen == SHOW_BASE_COLOR )
        Ldiff = baseColorMetalness.xyz;
    else if( gOnScreen == SHOW_NORMAL )
        Ldiff = N * 0.5 + 0.5;
    else if( gOnScreen == SHOW_ROUGHNESS )
        Ldiff = roughness;
    else if( gOnScreen == SHOW_METALNESS )
        Ldiff = baseColorMetalness.w;
    else if( gOnScreen == SHOW_MATERIAL_ID )
        Ldiff = normMaterialID;
    else if( gOnScreen == SHOW_PSR_THROUGHPUT )
        Ldiff = psrThroughput;
    else if( gOnScreen == SHOW_WORLD_UNITS )
        Ldiff = frac( X * gUnitToMetersMultiplier );
    else if( gOnScreen != SHOW_FINAL )
        Ldiff = gOnScreen == SHOW_MIP_SPECULAR ? spec.xyz : Ldirect.xyz;

    // Output
    gOut_ComposedDiff[ pixelPos ] = Ldiff;
    gOut_ComposedSpec_ViewZ[ pixelPos ] = float4( Lspec, z );
}
