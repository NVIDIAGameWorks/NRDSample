/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<uint3>, gIn_Scrambling_Ranking_1spp, t, 0, 1 );
NRI_RESOURCE( Texture2D<uint4>, gIn_Sobol, t, 1, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 2, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_Mv, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_ViewZ, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 2, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_BaseColor_Metalness, u, 3, 1 );
NRI_RESOURCE( RWTexture2D<float2>, gOut_PrimaryMipAndCurvature, u, 4, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectLighting, u, 5, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectEmission, u, 6, 1 );
NRI_RESOURCE( RWTexture2D<float2>, gOut_ShadowData, u, 7, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Shadow_Translucency, u, 8, 1 );

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
    float3 cameraRayOriginv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gOrthoMode );
    float3 cameraRayOrigin = STL::Geometry::AffineTransform( gViewToWorld, cameraRayOriginv );
    float3 cameraRayDirection = gOrthoMode == 0 ? normalize( STL::Geometry::RotateVector( gViewToWorld, cameraRayOriginv ) ) : -gViewDirection;

    GeometryProps geometryProps0 = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, INF, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, ( gOnScreen == SHOW_MESH || gOnScreen == SHOW_NORMAL ) ? GEOMETRY_ALL : GEOMETRY_IGNORE_TRANSPARENT, 0 );
    MaterialProps materialProps0 = GetMaterialProps( geometryProps0 );

    // ViewZ
    float viewZ = STL::Geometry::AffineTransform( gWorldToView, geometryProps0.X ).z;
    gOut_ViewZ[ pixelPos ] = geometryProps0.IsSky( ) ? STL::Math::Sign( viewZ ) * INF : viewZ;

    // Motion
    float3 Xprev = geometryProps0.X;
    if( !geometryProps0.IsSky( ) )
    {
        InstanceData instanceData = gIn_InstanceData[ geometryProps0.instanceIndex ];
        Xprev = STL::Geometry::AffineTransform( instanceData.mWorldToWorldPrev, geometryProps0.X );
    }

    float3 motion = Xprev - geometryProps0.X;
    if( !gIsWorldSpaceMotionEnabled )
    {
        float viewZprev = STL::Geometry::AffineTransform( gWorldToViewPrev, Xprev ).z;
        float2 sampleUvPrev = STL::Geometry::GetScreenUv( gWorldToClipPrev, Xprev );

        motion.xy = ( sampleUvPrev - sampleUv ) * gRectSize;
        motion.z = viewZprev - viewZ;
    }

    gOut_Mv[ pixelPos ] = motion;

    // Early out - sky
    if( geometryProps0.IsSky( ) )
    {
        gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi;
        gOut_DirectLighting[ pixelPos ] = 0;

        return;
    }

    // G-buffer
    float mipNorm = STL::Math::Sqrt01( geometryProps0.mip / MAX_MIP_LEVEL );
    float curvatureNorm = STL::Math::Sqrt01( materialProps0.curvature / 4.0 );

    float diffuseProbability = EstimateDiffuseProbability( geometryProps0, materialProps0 );
    uint materialID = diffuseProbability > BRDF_ENERGY_THRESHOLD ? 0 : 1;

    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( gDebug == 0.0 )
            materialID = materialID || ( frac( geometryProps0.X ).x < 0.05 ? 1 : 0 );
    #endif

    gOut_PrimaryMipAndCurvature[ pixelPos ] = float2( mipNorm, curvatureNorm );
    gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( materialProps0.N, materialProps0.roughness, materialID );
    gOut_BaseColor_Metalness[ pixelPos ] = float4( STL::Color::LinearToSrgb( materialProps0.baseColor ), materialProps0.metalness );

    // Debug & direct lighting
    if( gOnScreen == SHOW_MESH )
    {
        STL::Rng::Initialize( geometryProps0.instanceIndex, 0 );
        materialProps0.Ldirect = STL::Rng::GetFloat4().xyz;
    }
    else if( gOnScreen == SHOW_MIP_PRIMARY )
        materialProps0.Ldirect = STL::Color::ColorizeZucconi( mipNorm );

    gOut_DirectLighting[ pixelPos ] = materialProps0.Ldirect;
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
            GeometryProps geometryPropsShadow = CastRay( Xoffset, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, 0 );

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
