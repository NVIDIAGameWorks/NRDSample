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
NRI_RESOURCE( Texture2D<float4>, gIn_History, t, 2, 1 );

NRI_RESOURCE( RWTexture2D<float4>, gOut_Result, u, 0, 1 );

#define BORDER          2
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
    color_viewZ.w = abs( color_viewZ.w ) * Math::Sign( gNearZ ) / FP16_VIEWZ_SCALE;

    s_Data[ sharedPos.y ][ sharedPos.x ] = color_viewZ;
}

// TODO: move to ml?
// https://www.cs.rit.edu/~ncs/color/t_convert.html
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
float3 XyzToLab( float3 x )
{
    x /= float3( 95.0489, 100.0, 108.8840 );
    x = lerp( 7.787 * x + 16.0 / 116.0, pow( x, 0.333333 ), x > 0.008856 );

    float l = lerp( 903.3 * x.y, 116.0 * pow( x.y, 0.333333 ) - 16.0, x.y > 0.008856 );
    float a = 500.0 * ( x.x - x.y );
    float b = 200.0 * ( x.y - x.z );

    return float3( l, a, b );
}

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

    float3 sigma = sqrt( abs( m2 - m1 * m1 ) ); // TODO: increase sigma for hair and glass?

    // Previous pixel position
    float3 Xv = Geometry::ReconstructViewPosition( pixelUv, gCameraFrustum, viewZnearest, gOrthoMode );
    float3 X = Geometry::AffineTransform( gViewToWorld, Xv );
    float3 mv = gIn_Mv[ pixelPos + offseti - BORDER ] * ( gIsWorldSpaceMotionEnabled ? 1.0 : gInvRectSize.xyy );
    float2 pixelUvPrev = pixelUv + mv.xy;

    if( gIsWorldSpaceMotionEnabled )
        pixelUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, X + mv );

    // History
    float2 pixelPosPrev = saturate( pixelUvPrev ) * gRectSizePrev;
    float4 history = BicubicFilterNoCorners( gIn_History, gLinearSampler, pixelPosPrev, gInvRenderSize, TAA_HISTORY_SHARPNESS );

    history.xyz = max( history.xyz, 0.0 ); // yes, not "saturate"

    bool isSrgb = gIsSrgb && ( gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR );
    if( isSrgb )
        history.xyz = Color::FromSrgb( history.xyz );

    // Update mix rate
    float mixRate = saturate( history.w );
    mixRate /= 1.0 + mixRate;

    // Disocclusion #1
    bool isInScreen = all( saturate( pixelUvPrev ) == pixelUvPrev );
    mixRate = ( !isInScreen || pixelUv.x < gSeparator ) ? 1.0 : mixRate;

    // Disocclusion #2
    float3 clampedHistory = Color::ClampAabb( m1, sigma, history.xyz );
    #if 1 // good enough
        mixRate += length( clampedHistory - history.xyz ) * 0.75;
    #else
        float3 a = XyzToLab( Color::RgbToXyz( clampedHistory ) );
        float3 b = XyzToLab( Color::RgbToXyz( history.xyz ) );

        const float JND = 2.3; // just noticable difference
        mixRate += length( a - b ) / ( JND * 3.0 );
    #endif

    // Clamp mix rate
    mixRate = clamp( mixRate, gTAA, 1.0 );

    // Final mix
    float3 result = lerp( clampedHistory, input, mixRate );

    // Split screen - vertical line
    float verticalLine = saturate( 1.0 - abs( pixelUv.x - gSeparator ) * gRectSize.x / 3.5 );
    verticalLine = saturate( verticalLine / 0.5 );
    verticalLine *= float( gSeparator != 0.0 );
    verticalLine *= float( gRenderSize.x == gRectSize.x );

    const float3 nvColor = float3( 118.0, 185.0, 0.0 ) / 255.0;
    result = lerp( result, nvColor * verticalLine, verticalLine );

    // Dithering
    Rng::Hash::Initialize( pixelPos, gFrameIndex );

    float rnd = Rng::Hash::GetFloat( );
    result += ( rnd * 2.0 - 1.0 ) / 1023.0;

    // Output
    if( isSrgb )
        result = Color::ToSrgb( result );

    gOut_Result[ pixelPos ] = float4( result, mixRate );
}
