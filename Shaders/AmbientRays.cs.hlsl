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

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_Ambient, u, 0, 1 );

#define CTA_W           16
#define CTA_H           16
#define DISPATCH_W      ( CTA_W * 2 )
#define DISPATCH_H      ( CTA_H * 2 )
#define NORM_VALUE      4096.0

groupshared uint g_Lsumi[ 3 ];

[numthreads( CTA_W, CTA_H, 1 )]
void main( uint2 tilePos : SV_GroupId, uint2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    // Clear SMEM
    if( threadIndex < 3 )
        g_Lsumi[ threadIndex ] = 0;

    GroupMemoryBarrierWithGroupSync();

    // Cast rays in all directions ( find primary hit )
    STL::Rng::Hash::Initialize( pixelPos, gFrameIndex );

    float2 rnd = STL::Sequence::Hammersley2D( pixelPos.y * DISPATCH_W + pixelPos.x, DISPATCH_W * DISPATCH_H );
    float3 ray = STL::ImportanceSampling::Uniform::GetRay( float2( rnd.x, rnd.y * 2.0 - 1.0 ) );
    float2 mipAndCone = GetConeAngleFromRoughness( 0.0, tan( 1.0 / DISPATCH_W ) );

    GeometryProps geometryProps = CastRay( gCameraOrigin, ray, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
    MaterialProps materialProps = GetMaterialProps( geometryProps );
    mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, 1.0 );

    float3 BRDF = GetAmbientBRDF( geometryProps, materialProps, true );

    // Accumulate bounces not handled by the main tracer
    float3 Lsum = 0.0;

    [loop]
    for( uint i = 0; i < 9 && STL::Color::Luminance( BRDF ) > 0.001; i++ )
    {
        // Choose ray
        float2 rnd = STL::Rng::Hash::GetFloat2( );
        float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay( rnd );
        float3x3 mLocalBasis = STL::Geometry::GetBasis( materialProps.N );
        float3 ray = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

        // Cast ray
        geometryProps = CastRay( geometryProps.GetXoffset( ), ray, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );
        materialProps = GetMaterialProps( geometryProps );
        mipAndCone = GetConeAngleFromAngularRadius( geometryProps.mip, 1.0 );

        // Accumulate emissives
        if( i >= gBounceNum )
            Lsum += materialProps.Lemi * BRDF;

        // Update BRDF
        BRDF *= GetAmbientBRDF( geometryProps, materialProps, true );
    }

    Lsum /= STL::ImportanceSampling::Uniform::GetPDF( );
    Lsum *= 0.5;

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
