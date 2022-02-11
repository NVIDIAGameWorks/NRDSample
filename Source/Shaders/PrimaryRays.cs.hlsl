/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if( !defined( COMPILER_FXC ) && !defined( VULKAN ) )

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<uint3>, gIn_Scrambling_Ranking_1spp, t, 0, 1 );
NRI_RESOURCE( Texture2D<uint4>, gIn_Sobol, t, 1, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 2, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_Motion, u, 3, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_ViewZ, u, 4, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 5, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_BaseColor_Metalness, u, 6, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_PrimaryMip, u, 7, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_DirectLighting, u, 8, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectEmission, u, 9, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_TransparentLayer, u, 10, 1 );
NRI_RESOURCE( RWTexture2D<float2>, gOut_ShadowData, u, 11, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Shadow_Translucency, u, 12, 1 );

float2 GetBlueNoise( Texture2D<uint3> texScramblingRanking, uint2 pixelPos, bool isCheckerboard, uint seed, uint sampleIndex, uint sppVirtual = 1, uint spp = 1 )
{
    // Final SPP - total samples per pixel ( there is a different "gIn_Scrambling_Ranking" texture! )
    // SPP - samples per pixel taken in a single frame ( must be POW of 2! )
    // Virtual SPP - "Final SPP / SPP" - samples per pixel distributed in time ( across frames )

    // Based on:
    //     https://eheitzresearch.wordpress.com/772-2/
    // Source code and textures can be found here:
    //     https://belcour.github.io/blog/research/publication/2019/06/17/sampling-bluenoise.html (but 2D only)

    // Sample index
    uint frameIndex = isCheckerboard ? ( gFrameIndex >> 1 ) : gFrameIndex;
    uint virtualSampleIndex = ( frameIndex + seed ) & ( sppVirtual - 1 );
    sampleIndex &= spp - 1;
    sampleIndex += virtualSampleIndex * spp;

    // The algorithm
    uint3 A = texScramblingRanking[ pixelPos & 127 ];
    uint rankedSampleIndex = sampleIndex ^ A.z;
    uint4 B = gIn_Sobol[ uint2( rankedSampleIndex & 255, 0 ) ];
    float4 blue = ( float4( B ^ A.xyxy ) + 0.5 ) * ( 1.0 / 256.0 );

    // Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    uint d = STL::Sequence::Bayer4x4ui( pixelPos, gFrameIndex );
    float2 dither = ( float2( d & 3, d >> 2 ) + 0.5 ) * ( 1.0 / 4.0 );
    blue += ( dither.xyxy - 0.5 ) * ( 1.0 / 256.0 );

    return saturate( blue.xy );
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
        return;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Primary ray
    float3 cameraRayOriginv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gIsOrtho );
    float3 cameraRayOrigin = STL::Geometry::AffineTransform( gViewToWorld, cameraRayOriginv );
    float3 cameraRayDirection = -GetViewVector( cameraRayOrigin );

    GeometryProps geometryProps0 = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, INF, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
    MaterialProps materialProps0 = GetMaterialProps( geometryProps0 );

    // Transparent layer
    // TODO: move after Composition to be able to use final frame for refractions
    if( gTransparent != 0.0 )
    {
        float2 mipAndCone = GetConeAngleFromRoughness( geometryProps0.mip, 0.0 );
        GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, geometryProps0.tmin, mipAndCone, gWorldTlas, GEOMETRY_ONLY_TRANSPARENT, 0 );

        float4 transparentLayer = 0;
        if( geometryPropsT.tmin < geometryProps0.tmin )
        {
            float3 origin = geometryPropsT.GetXoffset();
            float3 direction = reflect( cameraRayDirection, geometryPropsT.N );

            GeometryProps geometryPropsRefl = CastRay( origin, direction, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
            MaterialProps materialPropsRefl = GetMaterialProps( geometryPropsRefl );

            // Direct lighting at reflection
            float3 Lsum = materialPropsRefl.Ldirect;
            if( STL::Color::Luminance( Lsum ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
                Lsum *= CastVisibilityRay_AnyHit( geometryPropsRefl.GetXoffset( ), gSunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
            Lsum += materialPropsRefl.Lemi;

            // Ambient estimation at reflection
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialPropsRefl.baseColor, materialPropsRefl.metalness, albedo, Rf0 );

            float NoV = abs( dot( materialPropsRefl.N, -direction ) );
            float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialPropsRefl.roughness );

            float3 BRDF = albedo * ( 1 - F ) + F;
            BRDF *= STL::Math::Pi( 1.0 );
            BRDF *= float( !geometryPropsRefl.IsSky() );

            float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
            Lsum += Lamb * BRDF * gAmbient;

            // Apply glass material
            float NoV0 = abs( dot( geometryPropsT.N, direction ) );
            float F0 = STL::BRDF::FresnelTerm_Schlick( 0.05, NoV0 ).x;

            transparentLayer.xyz = Lsum * F0 * GLASS_TINT;
            transparentLayer.w = F0;
        }

        gOut_TransparentLayer[ pixelPos ] = transparentLayer;
    }

    // Early out - sky
    if( geometryProps0.IsSky( ) )
    {
        float4 clipPrev = STL::Geometry::ProjectiveTransform( gWorldToClipPrev, geometryProps0.X );
        float2 sampleUvPrev = ( clipPrev.xy / clipPrev.w ) * float2( 0.5, -0.5 ) + 0.5;
        float2 surfaceMotion = ( sampleUvPrev - sampleUv ) * gRectSize;
        float3 motion = gWorldSpaceMotion ? 0 : surfaceMotion.xyy;

        gOut_Motion[ pixelPos ] = motion * STL::Math::LinearStep( 0.0, 0.0000005, abs( motion ) ); // TODO: move LinearStep to NRD?
        gOut_ViewZ[ pixelPos ] = INF * STL::Math::Sign( gNearZ );
        gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi;
        gOut_DirectLighting[ pixelPos ] = float4( 0, 0, 0, NRD_FP16_MAX * STL::Math::Sign( gNearZ ) );

        return;
    }

    // Motion
    InstanceData instanceData = gIn_InstanceData[ geometryProps0.instanceIndex ];
    float3 Xprev = STL::Geometry::AffineTransform( instanceData.mWorldToWorldPrev, geometryProps0.X );
    float4 clipPrev = STL::Geometry::ProjectiveTransform( gWorldToClipPrev, Xprev );
    float2 sampleUvPrev = ( clipPrev.xy / clipPrev.w ) * float2( 0.5, -0.5 ) + 0.5;
    float2 surfaceMotion = ( sampleUvPrev - sampleUv ) * gRectSize;
    float3 motion = gWorldSpaceMotion ? ( Xprev - geometryProps0.X ) : surfaceMotion.xyy;

    gOut_Motion[ pixelPos ] = motion * STL::Math::LinearStep( 0.0, 0.0000005, abs( motion ) ); // TODO: move LinearStep to NRD?

    // G-buffer
    float viewZ = STL::Geometry::AffineTransform( gWorldToView, geometryProps0.X ).z;
    float mipNorm = STL::Math::Sqrt01( geometryProps0.mip / MAX_MIP_LEVEL );

    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps0.baseColor, materialProps0.metalness, albedo, Rf0 );
    uint noDiffuseFlag = STL::Color::Luminance( albedo ) < BRDF_ENERGY_THRESHOLD ? 1 : 0;

    gOut_ViewZ[ pixelPos ] = viewZ;
    gOut_PrimaryMip[ pixelPos ] = mipNorm;
    gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( materialProps0.N, materialProps0.roughness, noDiffuseFlag );
    gOut_BaseColor_Metalness[ pixelPos ] = float4( STL::Color::LinearToSrgb( materialProps0.baseColor ), materialProps0.metalness );

    // Debug & direct lighting
    if( gOnScreen == SHOW_MESH )
    {
        STL::Rng::Initialize( geometryProps0.instanceIndex, 0 );
        materialProps0.Ldirect = STL::Rng::GetFloat4().xyz;
    }
    else if( gOnScreen == SHOW_MIP_PRIMARY )
        materialProps0.Ldirect = STL::Color::ColorizeZucconi( mipNorm );

    gOut_DirectLighting[ pixelPos ] = float4( materialProps0.Ldirect, viewZ * NRD_FP16_VIEWZ_SCALE );
    gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi;

    // Sun shadow // TODO: move to a separate pass to unblock checkerboard
    float shadowHitDist = 0.0;
    float3 shadowTranslucency = 0.0;

    bool isShadowRayNeeded = STL::Color::Luminance( materialProps0.Ldirect ) != 0.0 && !gDisableShadowsAndEnableImportanceSampling;
    if( isShadowRayNeeded )
    {
        float2 rnd;
        if( gReference == 0.0 )
            rnd = GetBlueNoise( gIn_Scrambling_Ranking_1spp, pixelPos, false, 0, 0 );
        else
        {
            STL::Rng::Initialize( pixelPos, gFrameIndex );
            rnd = STL::Rng::GetFloat2( );
        }

        rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
        rnd *= gTanSunAngularRadius;

        float3x3 mSunBasis = STL::Geometry::GetBasis( gSunDirection ); // TODO: move to CB
        float3 sunDirection = normalize( mSunBasis[ 0 ] * rnd.x + mSunBasis[ 1 ] * rnd.y + mSunBasis[ 2 ] );
        float3 Xoffset = geometryProps0.GetXoffset( );
        float2 mipAndCone = GetConeAngleFromAngularRadius( geometryProps0.mip, gTanSunAngularRadius );

        shadowTranslucency = 1.0;
        while( STL::Color::Luminance( shadowTranslucency ) > 0.01 )
        {
            GeometryProps geometryPropsShadow = CastRay( Xoffset, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, 0, true );

            if( geometryPropsShadow.IsSky( ) )
            {
                // On immediate miss - return no shadow, otherwise - return accumulated data
                shadowTranslucency = shadowHitDist == 0.0 ? 0.0 : shadowTranslucency;
                shadowHitDist = shadowHitDist == 0.0 ? INF : shadowHitDist;
                break;
            }

            float NoV = abs( dot( geometryPropsShadow.N, sunDirection ) );
            shadowTranslucency *= lerp( 0.9, 0.0, STL::Math::Pow01( 1.0 - NoV, 2.5 ) ) * GLASS_TINT;
            shadowTranslucency *= float( geometryPropsShadow.IsTransparent( ) );

            float offset = geometryProps0.tmin * 0.0001 + 0.001;
            Xoffset = geometryPropsShadow.GetXoffset( ) + sunDirection * offset;

            shadowHitDist += geometryPropsShadow.tmin;
        }
    }

    float4 shadowData1;
    float2 shadowData0 = SIGMA_FrontEnd_PackShadow( viewZ, shadowHitDist == INF ? NRD_FP16_MAX : shadowHitDist, gTanSunAngularRadius, shadowTranslucency, shadowData1 );

    gOut_ShadowData[ pixelPos ] = shadowData0;
    gOut_Shadow_Translucency[ pixelPos ] = shadowData1;
}

#else

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // no TraceRayInline support, because of:
    //  - DXBC
    //  - SPIRV generation is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/4221
}

#endif
