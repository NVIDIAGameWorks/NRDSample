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
NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 0, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_ComposedDiff, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedSpec_ViewZ, t, 2, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 3, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float4>, gOut_Composed, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gInOut_Mv, u, 1, 1 );

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
    float eta = STL::BRDF::IOR::Air / STL::BRDF::IOR::Glass;

    GeometryProps geometryProps = desc.geometryProps;
    float pathThroughput = 1.0;
    bool isReflection = desc.isReflection;

    [loop]
    for( uint bounce = 1; bounce <= desc.bounceNum; bounce++ ) // TODO: stop if pathThroughput is low
    {
        // Reflection or refraction?
        float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
        float F = STL::BRDF::FresnelTerm_Dielectric( eta, NoV );

        if( bounce == 1 )
            pathThroughput *= isReflection ? F : 1.0 - F;
        else
            isReflection = STL::Rng::Hash::GetFloat( ) < F;

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
        float3 origin = _GetXoffset( geometryProps.X, geometryProps.N * STL::Math::Sign( dot( ray, geometryProps.N ) ) );
        uint flags = bounce == desc.bounceNum ? GEOMETRY_IGNORE_TRANSPARENT : GEOMETRY_ALL;

        geometryProps = CastRay( origin, ray, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, 0.0 ), gWorldTlas, flags, 0 );

        // TODO: glass internal extinction?
        // ideally each "medium" should have "eta" and "extinction" parameters in "TraceTransparentDesc" and "TraceOpaqueDesc"
        if( !isReflection )
            pathThroughput *= 0.96;

        // Is opaque hit found?
        if( !geometryProps.IsTransparent( ) )
        {
            MaterialProps materialProps = GetMaterialProps( geometryProps );

            // L1 cache - reproject previous frame, carefully treating specular
            float3 prevLdiff, prevLspec;
            float reprojectionWeight = ReprojectIrradiance( false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec );
            float4 Lcached = float4( prevLdiff + prevLspec, reprojectionWeight );

            // Compute lighting at hit point
            if( Lcached.w != 1.0 )
            {
                float3 L = materialProps.Ldirect;
                if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
                    L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), gSunDirection_gExposure.xyz, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness ), gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );

                L += materialProps.Lemi;

                // Ambient estimation at the end of the path
                float3 BRDF = GetAmbientBRDF( geometryProps, materialProps );
                BRDF *= 1.0 + EstimateDiffuseProbability( geometryProps, materialProps, true );

                float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
                Lamb *= gAmbient;

                L += Lamb * BRDF;

                // Mix
                Lcached.xyz = lerp( L, Lcached.xyz, Lcached.w );
            }

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

[numthreads( 16, 16, 1)]
void main( int2 pixelPos : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
        return;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Initialize RNG
    STL::Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Composed without glass
    float3 diff = gIn_ComposedDiff[ pixelPos ];
    float4 spec = gIn_ComposedSpec_ViewZ[ pixelPos ];
    float3 Lsum = diff + spec.xyz * float( gOnScreen == SHOW_FINAL );

    // Primary ray for transparent geometry only
    float3 cameraRayOrigin = ( float3 )0;
    float3 cameraRayDirection = ( float3 )0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    float viewZ = gIn_ViewZ[ pixelPos ];
    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gViewDirection_gOrthoMode.w );
    float tmin0 = gViewDirection_gOrthoMode.w == 0 ? length( Xv ) : abs( Xv.z );

    GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, tmin0, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, gTransparent == 0.0 ? 0 : GEOMETRY_ONLY_TRANSPARENT, 0 );

    // Trace delta events
    if( !geometryPropsT.IsSky( ) && geometryPropsT.tmin < tmin0 )
    {
        // Patch motion vectors replacing MV for the background with MV for the closest glass layer.
        // IMPORTANT: surface-based motion can be used only if the object is curved.
        // TODO: let's use the simplest heuristic for now, but better switch to some "smart" interpolation between
        // MVs for the primary opaque surface hit and the primary glass surface hit.
        if( geometryPropsT.curvature != 0.0 )
        {
            float3 motion = GetMotion( geometryPropsT.X, geometryPropsT.Xprev );
            gInOut_Mv[ pixelPos ] = motion;
        }

        // Trace transparent stuff
        TraceTransparentDesc desc = ( TraceTransparentDesc )0;
        desc.geometryProps = geometryPropsT;
        desc.pixelPos = pixelPos;
        desc.bounceNum = 10;

        // IMPORTANT: use 1 reflection path and 1 refraction path at the primary glass hit to significantly reduce noise
        // TODO: use probabilistic split at the primary glass hit when denoising is available
        desc.isReflection = true;
        Lsum = TraceTransparent( desc );

        desc.isReflection = false;
        Lsum += TraceTransparent( desc );
    }

    // Output
    float z = spec.w;

    gOut_Composed[ pixelPos ] = float4( Lsum, z );
}
