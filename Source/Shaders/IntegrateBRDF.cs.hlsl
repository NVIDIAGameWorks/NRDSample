/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "BindingBridge.hlsli"
#include "STL.hlsli"

NRI_RESOURCE( RWTexture2D<float2>, gOutput, u, 0, 0 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    const uint SAMPLE_NUM = 256;
    const float invTexSize = 1.0 / 256.0;

    float NoV = float( pixelPos.x + 0.5 ) * invTexSize;
    float roughness = float( pixelPos.y + 0.5 ) * invTexSize;

    float3 V;
    V.x = STL::Math::Sqrt01( 1.0 - NoV * NoV );
    V.y = 0.0;
    V.z = NoV;

    float2 GG = 0.0;

    [loop]
    for( uint i = 0; i < SAMPLE_NUM * SAMPLE_NUM; i++ )
    {
        float2 rnd = ( float2( i & ( SAMPLE_NUM - 1 ), uint( i / SAMPLE_NUM ) ) + 0.5 ) / ( SAMPLE_NUM - 1 );

        // Diffuse
        {
            float3 L = STL::ImportanceSampling::Cosine::GetRay( rnd );
            float3 H = normalize( V + L );

            float NoL = saturate( L.z );
            float VoH = saturate( dot( V, H ) );

            float F = STL::BRDF::Pow5( VoH );
            float Kdiff = STL::BRDF::DiffuseTerm( roughness, NoL, NoV, VoH );

            // NoL gets canceled by PDF
            GG.x += ( 1.0 - F ) * Kdiff / STL::ImportanceSampling::Cosine::GetPDF( );
        }

        // Specular
        {
            float3 H = STL::ImportanceSampling::VNDF::GetRay( rnd, roughness, V );
            float3 L = reflect( -V, H );

            float NoL = saturate( L.z );
            float VoH = saturate( dot( V, H ) );
            float NoH = saturate( H.z );

            // TODO: almost the same and simpler
            // throughput.x += STL::BRDF::GeometryTerm_Smith( roughness, NoL );

            float D = STL::BRDF::DistributionTerm( roughness, NoH );
            float G = STL::BRDF::GeometryTermMod( roughness, NoL, NoV, VoH, NoH );

            GG.y += D * G * NoL / STL::ImportanceSampling::VNDF::GetPDF( NoV, NoH, roughness );
        }
    }

    gOutput[ pixelPos ] = GG / float( SAMPLE_NUM * SAMPLE_NUM );
}
