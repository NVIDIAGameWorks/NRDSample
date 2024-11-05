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

#define SHARC_QUERY 1
#include "SharcCommon.h"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_ComposedDiff, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedSpec_ViewZ, t, 1, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_Composed, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gInOut_Mv, u, 1, 1 );

//========================================================================================
// TRACE TRANSPARENT
//========================================================================================

struct TraceTransparentDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Pixel position
    uint2 pixelPos;

    // Number of bounces to trace ( up to )
    uint bounceNum;

    // Is reflection or refraction in first segment?
    bool isReflection;
};

// TODO: think about adding a specialized delta-event denoiser in NRD:
//  Inputs:
//      - Lsum ( delta events gathered across the path )
//      - reflections or refractions prevail?
//  Principle:
//      - add missing component (reflection or refraction) from neighboring pixels
float3 TraceTransparent( TraceTransparentDesc desc )
{
    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;

    GeometryProps geometryProps = desc.geometryProps;
    float pathThroughput = 1.0;
    bool isReflection = desc.isReflection;

    [loop]
    for( uint bounce = 1; bounce <= desc.bounceNum; bounce++ ) // TODO: stop if pathThroughput is low
    {
        // Reflection or refraction?
        float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
        float F = BRDF::FresnelTerm_Dielectric( eta, NoV );

        if( bounce == 1 )
            pathThroughput *= isReflection ? F : 1.0 - F;
        else
        {
            float rndBayer = Sequence::Bayer4x4( desc.pixelPos, gFrameIndex + bounce );
            float rndWhite = Rng::Hash::GetFloat( );
            float rnd = ( gDenoiserType == DENOISER_REFERENCE || gRR ) ? rndWhite : rndBayer + rndWhite / 16.0;

            isReflection = rnd < F;
        }

        // Compute ray
        float3 ray = reflect( -geometryProps.V, geometryProps.N );
        if( !isReflection )
        {
            float3 I = -geometryProps.V;
            float NoI = dot( geometryProps.N, I );
            float k = 1.0 - eta * eta * ( 1.0 - NoI * NoI );

            if( k < 0.0 )
                return 0.0; // should't be here

            ray = normalize( eta * I - ( eta * NoI + sqrt( k ) ) * geometryProps.N );
            eta = 1.0 / eta;
        }

        // Trace
        float3 Xoffset = geometryProps.GetXoffset( geometryProps.N * Math::Sign( dot( ray, geometryProps.N ) ), GLASS_RAY_OFFSET );
        uint flags = bounce == desc.bounceNum ? GEOMETRY_IGNORE_TRANSPARENT : GEOMETRY_ALL;

        geometryProps = CastRay( Xoffset, ray, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, 0.0 ), gWorldTlas, flags, 0 );

        // TODO: glass internal extinction?
        // ideally each "medium" should have "eta" and "extinction" parameters in "TraceTransparentDesc" and "TraceOpaqueDesc"
        if( !isReflection )
            pathThroughput *= 0.96;

        // Is opaque hit found?
        if( !geometryProps.Has( FLAG_TRANSPARENT ) )
        {
            MaterialProps materialProps = GetMaterialProps( geometryProps );

            // Lighting
            float4 Lcached = 0;
            if( !geometryProps.IsSky( ) )
            {
                // L1 cache - reproject previous frame, carefully treating specular
                float3 prevLdiff, prevLspec;
                float reprojectionWeight = ReprojectIrradiance( false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec );
                Lcached = float4( prevLdiff + prevLspec, reprojectionWeight );

                // L2 cache - SHARC
                GridParameters gridParameters = ( GridParameters )0;
                gridParameters.cameraPosition = gCameraGlobalPos.xyz;
                gridParameters.cameraPositionPrev = gCameraGlobalPosPrev.xyz;
                gridParameters.sceneScale = SHARC_SCENE_SCALE;
                gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;

                float3 Xglobal = GetGlobalPos( geometryProps.X );
                uint level = GetGridLevel( Xglobal, gridParameters );
                float voxelSize = GetVoxelSize( level, gridParameters );
                float smc = GetSpecMagicCurve( materialProps.roughness );

                float3x3 mBasis = Geometry::GetBasis( geometryProps.N );
                float2 rndScaled = ( Rng::Hash::GetFloat2( ) - 0.5 ) * voxelSize * USE_SHARC_DITHERING;
                Xglobal += mBasis[ 0 ] * rndScaled.x + mBasis[ 1 ] * rndScaled.y;

                SharcHitData sharcHitData = ( SharcHitData )0;
                sharcHitData.positionWorld = Xglobal;
                sharcHitData.normalWorld = geometryProps.N;

                SharcState sharcState;
                sharcState.gridParameters = gridParameters;
                sharcState.hashMapData.capacity = SHARC_CAPACITY;
                sharcState.hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;
                sharcState.voxelDataBuffer = gInOut_SharcVoxelDataBuffer;

                bool isSharcAllowed = gSHARC && NRD_MODE < OCCLUSION; // trivial
                isSharcAllowed &= geometryProps.hitT > voxelSize; // voxel angular size is acceptable
                isSharcAllowed &= Rng::Hash::GetFloat( ) > Lcached.w; // probabilistically estimate the need

                float3 sharcRadiance;
                if( isSharcAllowed && SharcGetCachedRadiance( sharcState, sharcHitData, sharcRadiance, false ) )
                    Lcached = float4( sharcRadiance, 1.0 );

                // Cache miss - compute lighting, if not found in caches
                if( Rng::Hash::GetFloat( ) > Lcached.w )
                {
                    float3 L = GetShadowedLighting( geometryProps, materialProps );
                    Lcached.xyz = max( Lcached.xyz, L );
                }
            }
            Lcached.xyz = max( Lcached.xyz, materialProps.Lemi );

            // Output
            return Lcached.xyz * pathThroughput;
        }
    }

    // Should't be here
    return 0.0;
}

//========================================================================================
// MAIN
//========================================================================================

[numthreads( 16, 16, 1 )]
void main( int2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // Initialize RNG
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Composed without glass
    float3 diff = gIn_ComposedDiff[ pixelPos ];
    float3 spec = gIn_ComposedSpec_ViewZ[ pixelPos ].xyz;
    float3 Lsum = diff + spec * float( gOnScreen == SHOW_FINAL );

    // Primary ray for transparent geometry only
    float3 cameraRayOrigin = ( float3 )0;
    float3 cameraRayDirection = ( float3 )0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    float viewZAndTaaMask = gInOut_Mv[ pixelPos ].w;
    float viewZ = Math::Sign( gNearZ ) * abs( viewZAndTaaMask ) / FP16_VIEWZ_SCALE; // viewZ before PSR
    float3 Xv = Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gOrthoMode );
    float tmin0 = gOrthoMode == 0 ? length( Xv ) : abs( Xv.z );

    GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, tmin0, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, gTransparent ? GEOMETRY_ONLY_TRANSPARENT : 0, 0 );

    // Trace delta events
    if( !geometryPropsT.IsSky( ) && geometryPropsT.hitT < tmin0 )
    {
        // Append "glass" mask to "hair" mask
        viewZAndTaaMask = viewZAndTaaMask < 0.0 ? viewZAndTaaMask : -viewZAndTaaMask;

        // Patch motion vectors replacing MV for the background with MV for the closest glass layer.
        // IMPORTANT: surface-based motion can be used only if the object is curved.
        // TODO: let's use the simplest heuristic for now, but better switch to some "smart" interpolation between
        // MVs for the primary opaque surface hit and the primary glass surface hit.
        float3 mvT = GetMotion( geometryPropsT.X, geometryPropsT.Xprev );
        gInOut_Mv[ pixelPos ] = float4( mvT, viewZAndTaaMask );

        // Trace transparent stuff
        TraceTransparentDesc desc = ( TraceTransparentDesc )0;
        desc.geometryProps = geometryPropsT;
        desc.pixelPos = pixelPos;
        desc.bounceNum = 10;

        // IMPORTANT: use 1 reflection path and 1 refraction path at the primary glass hit to significantly reduce noise
        // TODO: use probabilistic split at the primary glass hit when denoising becomes available
        desc.isReflection = true;
        float3 reflection = TraceTransparent( desc );
        Lsum = reflection;

        desc.isReflection = false;
        float3 refraction = TraceTransparent( desc );
        Lsum += refraction;
    }

    // Apply exposure
    Lsum = ApplyExposure( Lsum );

    // Output
    gOut_Composed[ pixelPos ] = Lsum;
}
