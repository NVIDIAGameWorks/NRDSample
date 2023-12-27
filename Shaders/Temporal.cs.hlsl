/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float3>, gIn_Mv, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedLighting_ViewZ, t, 1, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_History, t, 2, 1 );

NRI_RESOURCE( RWTexture2D<float3>, gOut_Result, u, 0, 1 );

#define BORDER          1
#define GROUP_X         16
#define GROUP_Y         16
#define BUFFER_X        ( GROUP_X + BORDER * 2 )
#define BUFFER_Y        ( GROUP_Y + BORDER * 2 )

#define PRELOAD_INTO_SMEM \
    int2 groupBase = pixelPos - threadPos - BORDER; \
    uint stageNum = ( BUFFER_X * BUFFER_Y + GROUP_X * GROUP_Y - 1 ) / ( GROUP_X * GROUP_Y ); \
    [unroll] \
    for( uint stage = 0; stage < stageNum; stage++ ) \
    { \
        uint virtualIndex = threadIndex + stage * GROUP_X * GROUP_Y; \
        uint2 newId = uint2( virtualIndex % BUFFER_X, virtualIndex / BUFFER_X ); \
        if( stage == 0 || virtualIndex < BUFFER_X * BUFFER_Y ) \
            Preload( newId, groupBase + newId ); \
    } \
    GroupMemoryBarrierWithGroupSync( )

groupshared float4 s_Data[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );

    float4 color_viewZ = gIn_ComposedLighting_ViewZ[ globalPos ];
    color_viewZ.xyz = ApplyExposure( color_viewZ.xyz );
    color_viewZ.xyz = ApplyTonemap( color_viewZ.xyz );
    color_viewZ.w = abs( color_viewZ.w ) * STL::Math::Sign( gNearZ ) / NRD_FP16_VIEWZ_SCALE;

    s_Data[ sharedPos.y ][ sharedPos.x ] = color_viewZ;
}

#define TAA_HISTORY_SHARPNESS 0.66

[numthreads( GROUP_X, GROUP_Y, 1 )]
void main( int2 threadPos : SV_GroupThreadId, int2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    PRELOAD_INTO_SMEM;

    // Do not generate NANs for unused threads
    if( pixelUv.x > 1.0 || pixelUv.y > 1.0 )
        return;

    // Neighborhood
    float3 m1 = 0;
    float3 m2 = 0;
    float3 input = 0;

    float viewZ = s_Data[ threadPos.y + BORDER ][threadPos.x + BORDER ].w;
    float viewZnearest = viewZ;
    int2 offseti = int2( BORDER, BORDER );

    [unroll]
    for( int dy = 0; dy <= BORDER * 2; dy++ )
    {
        [unroll]
        for( int dx = 0; dx <= BORDER * 2; dx++ )
        {
            int2 t = int2( dx, dy );
            int2 smemPos = threadPos + t;
            float4 data = s_Data[ smemPos.y ][ smemPos.x ];

            if( dx == BORDER && dy == BORDER )
                input = data.xyz;
            else if( abs( data.w ) < abs( viewZnearest ) )
            {
                viewZnearest = data.w;
                offseti = t;
            }

            m1 += data.xyz;
            m2 += data.xyz * data.xyz;
        }
    }

    float invSum = 1.0 / ( ( BORDER * 2 + 1 ) * ( BORDER * 2 + 1 ) );
    m1 *= invSum;
    m2 *= invSum;

    float3 sigma = sqrt( abs( m2 - m1 * m1 ) );

    // Previous pixel position
    float3 Xv = STL::Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZnearest, gViewDirection_gOrthoMode.w );
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 mv = gIn_Mv[ pixelPos + offseti - BORDER ] * ( gIsWorldSpaceMotionEnabled ? 1.0 : gInvRectSize.xyy );
    float2 pixelUvPrev = pixelUv + mv.xy;

    if( gIsWorldSpaceMotionEnabled )
        pixelUvPrev = STL::Geometry::GetScreenUv( gWorldToClipPrev, X + mv );

    // History
    float2 pixelPosPrev = saturate( pixelUvPrev ) * gRectSizePrev;
    float3 history = BicubicFilterNoCorners( gIn_History, gLinearSampler, pixelPosPrev, gInvRenderSize, TAA_HISTORY_SHARPNESS ).xyz;

    bool isSrgb = gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR;
    if( isSrgb )
        history = STL::Color::SrgbToLinear( history );

    // History clamping
    // IMPORTANT: must be done in linear space
    float3 historyClamped = STL::Color::ClampAabb( m1, sigma, history );

    // History weight
    bool isInScreen = float( all( saturate( pixelUvPrev ) == pixelUvPrev ) );
    float historyWeight = gTAA;
    historyWeight *= float( gTAA != 0 );
    historyWeight *= float( isInScreen );

    // Final mix
    float3 result = lerp( input, historyClamped, historyWeight );

    // Split screen - noisy input / denoised output
    result = pixelUv.x < gSeparator ? input : result;

    // Split screen - vertical line
    float verticalLine = saturate( 1.0 - abs( pixelUv.x - gSeparator ) * gRectSize.x / 3.5 );
    verticalLine = saturate( verticalLine / 0.5 );
    verticalLine *= float( gSeparator != 0.0 );
    verticalLine *= float( gRenderSize.x == gRectSize.x );

    const float3 nvColor = float3( 118.0, 185.0, 0.0 ) / 255.0;
    result = lerp( result, nvColor * verticalLine, verticalLine );

    // Dithering
    STL::Rng::Hash::Initialize( pixelPos, gFrameIndex );

    float rnd = STL::Rng::Hash::GetFloat( );
    result += ( rnd * 2.0 - 1.0 ) / 1023.0;

    // Output
    if( isSrgb )
        result = STL::Color::LinearToSrgb( result );

    gOut_Result[ pixelPos ] = result;
}
