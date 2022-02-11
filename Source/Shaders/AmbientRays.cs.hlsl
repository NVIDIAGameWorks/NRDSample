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

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_Ambient, u, 0, 1 );

#define CTA_W           16
#define CTA_H           16
#define DISPATCH_W      ( CTA_W * 2 )
#define DISPATCH_H      ( CTA_H * 2 )
#define NORM_VALUE      4096.0

groupshared uint g_Lsumi[ 3 ];

float3 GetBaseColor( MaterialProps materialProps, GeometryProps geometryProps )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );

    float scale = lerp( 1.0, 1.5, materialProps.metalness );
    float3 ambBRDF = albedo * ( 1 - F ) + F / scale;

    return ambBRDF;
}

[numthreads( CTA_W, CTA_H, 1 )]
void main( uint2 tilePos : SV_GroupId, uint2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    // Clear SMEM
    if( threadIndex < 3 )
        g_Lsumi[ threadIndex ] = 0;

    GroupMemoryBarrierWithGroupSync();

    // Cast rays in all directions
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    float2 rnd = STL::Sequence::Hammersley2D( pixelPos.y * DISPATCH_W + pixelPos.x, DISPATCH_W * DISPATCH_H );
    float3 rayDirection = STL::ImportanceSampling::Uniform::GetRay( float2( rnd.x, rnd.y * 2.0 - 1.0 ) );
    float2 mipAndCone = GetConeAngleFromRoughness( AMBIENT_MIP_BIAS, 0.0 );

    GeometryProps geometryProps = CastRay( gCameraOrigin, rayDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
    MaterialProps materialProps = GetMaterialProps( geometryProps );
    mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, 1.0 );

    // Accumulate 3 bounces
    float3 Lsum = 0.0; // skip direct
    float3 BRDF = GetBaseColor( materialProps, geometryProps );

    [loop]
    for( uint i = 0; i < AMBIENT_BOUNCE_NUM && !geometryProps.IsSky() && STL::Color::Luminance( BRDF ) > 0.02; i++ )
    {
        // Choose ray
        float2 rnd = STL::Rng::GetFloat2();
        float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay( rnd );
        float3x3 mLocalBasis = STL::Geometry::GetBasis( materialProps.N );
        float3 rayDirection = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

        // Update BRDF
        BRDF *= GetBaseColor( materialProps, geometryProps );

        // Cast ray
        geometryProps = CastRay( geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
        materialProps = GetMaterialProps( geometryProps );
        mipAndCone = GetConeAngleFromAngularRadius( geometryProps.mip, 1.0 );

        // Compute lighting
        float3 L = materialProps.Ldirect;
        if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
            L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), gSunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
        L += materialProps.Lemi;

        // Accumulate
        L *= BRDF;
        Lsum += L;
    }

    Lsum *= 0.5 / STL::ImportanceSampling::Uniform::GetPDF( );

    // Accumulate
    uint3 Lsumi = uint3( Lsum * NORM_VALUE );

    InterlockedAdd( g_Lsumi[ 0 ], Lsumi.x );
    InterlockedAdd( g_Lsumi[ 1 ], Lsumi.y );
    InterlockedAdd( g_Lsumi[ 2 ], Lsumi.z );

    GroupMemoryBarrierWithGroupSync();

    if( threadIndex == 0 )
    {
        uint3 Lsumi = uint3( g_Lsumi[ 0 ], g_Lsumi[ 1 ], g_Lsumi[ 2 ] );
        float3 currAmbient = float3( Lsumi ) / NORM_VALUE;
        currAmbient /= CTA_W * CTA_H;

        // Temporal stabilization
        float3 prevAmbient = gOut_Ambient[ tilePos ];
        currAmbient = lerp( prevAmbient, currAmbient, gAmbientAccumSpeed );

        // Store
        gOut_Ambient[ tilePos ] = currAmbient;
    }
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
