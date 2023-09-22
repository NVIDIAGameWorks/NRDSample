/*
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( StructuredBuffer<MorphVertex>, gIn_MorphMeshVertices, t, 0, 3 );

// Outputs
NRI_RESOURCE( RWStructuredBuffer<float4>, gOut_MorphedPositions, u, 0, 3 );
NRI_RESOURCE( RWStructuredBuffer<MorphedAttributes>, gOut_MorphedAttributes, u, 1, 3 );

[numthreads( 256, 1, 1 )]
void main( uint vertexIndex : SV_DispatchThreadId )
{
    if( vertexIndex >= gNumVertices )
        return;

    float3 position = 0;
    float3 N = 0;
    float3 T = 0;

    // TODO: unroll to 4 weights at a time?
    uint maxWeights = min( MORPH_MAX_ACTIVE_TARGETS_NUM, gNumWeights );
    for( uint i = 0; i < maxWeights; i++ )
    {
        uint row = i / MORPH_ELEMENTS_PER_ROW_NUM;
        uint col = i % MORPH_ELEMENTS_PER_ROW_NUM;

        uint morphTargetIndex = gIndices[ row ][ col ];
        float weight = gWeights[ row ][ col ];
        uint morphTargetVertexIndex = morphTargetIndex + vertexIndex;

        MorphVertex v = gIn_MorphMeshVertices[ morphTargetVertexIndex ];

        position += v.position.xyz * weight;
        N += STL::Packing::DecodeUnitVector( ( float2 )v.N, true, true ) * weight;
        T += STL::Packing::DecodeUnitVector( ( float2 )v.T, true, true ) * weight;
    }

    gOut_MorphedPositions[ gPositionCurrFrameOffset + vertexIndex] = float4( position, 1.0 );

    MorphedAttributes attributes = ( MorphedAttributes )0;
    attributes.N = ( float16_t2 )STL::Packing::EncodeUnitVector( normalize( N ), true );
    attributes.T = ( float16_t2 )STL::Packing::EncodeUnitVector( normalize( T ), true );

    gOut_MorphedAttributes[ gAttributesOutputOffset + vertexIndex ] = attributes;
}
