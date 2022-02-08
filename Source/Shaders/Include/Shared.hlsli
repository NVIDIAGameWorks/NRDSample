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

//===============================================================
// GLOSSARY
//===============================================================
/*
Names:
- V - view vector
- N - normal
- X - point position

Modifiers:
- v - view space
- 0..N - hit index ( 0 - primary ray )
*/

//===============================================================
// RESOURCES
//===============================================================

NRI_RESOURCE( cbuffer, globalConstants, b, 0, 0 )
{
    float4x4 gWorldToView;
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToClipPrev;
    float4x4 gWorldToClip;
    float4 gDiffHitDistParams;
    float4 gSpecHitDistParams;
    float4 gCameraFrustum;
    float3 gSunDirection;
    float gExposure;
    float3 gCameraOrigin;
    float gMipBias;
    float3 gTrimmingParams;
    float gEmissionIntensity;
    float3 gViewDirection;
    float gIsOrtho;
    float2 gOutputSize;
    float2 gInvOutputSize;
    float2 gScreenSize;
    float2 gInvScreenSize;
    float2 gRectSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gNearZ;
    float gAmbientAccumSpeed;
    float gAmbient;
    float gAmbientInComposition;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gIndirectDiffuse;
    float gIndirectSpecular;
    float gSunAngularRadius; // TODO: use gTanSunAngularRadius or fix the code where used
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gTransparent; // TODO: try to remove, casting a ray in an empty TLAS should be for free
    float gReference;
    uint gDenoiserType;
    uint gDisableShadowsAndEnableImportanceSampling; // TODO: remove - modify GetSunIntensity to return 0 if sun is below horizon
    uint gOnScreen;
    uint gFrameIndex;
    uint gForcedMaterial;
    uint gUseNormalMap;
    uint gWorldSpaceMotion;
    uint gTracingMode;
    uint gSampleNum;
    uint gBounceNum;
    uint gOcclusionOnly;
};

NRI_RESOURCE( SamplerState, gLinearMipmapLinearSampler, s, 1, 0 );
NRI_RESOURCE( SamplerState, gNearestMipmapNearestSampler, s, 2, 0 );
NRI_RESOURCE( SamplerState, gLinearSampler, s, 3, 0 );

//=============================================================================================
// DENOISER PART
//=============================================================================================

#include "NRD/Include/NRD.hlsli"

//=============================================================================================
// SETTINGS
//=============================================================================================

// Constants
#define REBLUR                              0
#define RELAX                               1

#define RESOLUTION_FULL                     0
#define RESOLUTION_HALF                     1
#define RESOLUTION_QUARTER                  2

#define SHOW_FINAL                          0
#define SHOW_DENOISED_DIFFUSE               1
#define SHOW_DENOISED_SPECULAR              2
#define SHOW_AMBIENT_OCCLUSION              3
#define SHOW_SPECULAR_OCCLUSION             4
#define SHOW_SHADOW                         5
#define SHOW_BASE_COLOR                     6
#define SHOW_NORMAL                         7
#define SHOW_ROUGHNESS                      8
#define SHOW_METALNESS                      9
#define SHOW_WORLD_UNITS                    10
#define SHOW_MESH                           11
#define SHOW_MIP_PRIMARY                    12
#define SHOW_MIP_SPECULAR                   13 // TODO: remove of fix visualization

#define FP16_MAX                            65504.0
#define INF                                 1e5

#define MAT_GYPSUM                          1
#define MAT_COBALT                          2

#define SKY_MARK                            0.0

// Settings
#define USE_SIMPLEX_LIGHTING_MODEL          0
#define USE_IMPORTANCE_SAMPLING             1
#define USE_MODULATED_IRRADIANCE            0 // an example, demonstrating how irradiance can be modulated before denoising and then de-modulated after denoising to avoid over-blurring
#define USE_SANITIZATION                    0 // NRD sample is NAN/INF free, but all relevant code is here to demonstrate usage

#define BRDF_ENERGY_THRESHOLD               0.003
#define AMBIENT_BOUNCE_NUM                  3
#define AMBIENT_MIP_BIAS                    2
#define AMBIENT_FADE                        ( -0.001 * gUnitToMetersMultiplier * gUnitToMetersMultiplier )
#define TAA_HISTORY_SHARPNESS               0.5 // [0; 1], 0.5 matches Catmull-Rom
#define TAA_MAX_HISTORY_WEIGHT              0.95
#define TAA_MIN_HISTORY_WEIGHT              0.1
#define TAA_MOTION_MAX_REUSE                0.1
#define MAX_MIP_LEVEL                       11.0
#define EMISSION_TEXTURE_MIP_BIAS           5.0
#define IMPORTANCE_SAMPLE_NUM               16
#define GLASS_TINT                          float3( 0.9, 0.9, 1.0 )

//=============================================================================================
// MISC
//=============================================================================================

float3 GetViewVector( float3 X )
{
    return gIsOrtho == 0.0 ? normalize( gCameraOrigin - X ) : gViewDirection;
}

float3 ApplyPostLightingComposition( uint2 pixelPos, float3 Lsum, Texture2D<float4> gIn_TransparentLayer, bool convertToLDR = true )
{
    // Transparent layer
    float4 transparentLayer = gTransparent ? gIn_TransparentLayer[ pixelPos ] : 0;
    Lsum = Lsum * ( 1.0 - transparentLayer.w ) * ( transparentLayer.w != 0.0 ? GLASS_TINT : 1.0 ) + transparentLayer.xyz;

    // Exposure
    if( gOnScreen <= SHOW_DENOISED_SPECULAR )
    {
        Lsum *= gExposure;

        // Dithering
        // IMPORTANT: requires STL::Rng::Initialize
        float rnd = STL::Rng::GetFloat2( ).x;
        float luma = STL::Color::Luminance( Lsum );
        float amplitude = lerp( 0.4, 1.0 / 1024.0, STL::Math::Sqrt01( luma ) );
        float dither = 1.0 + ( rnd - 0.5 ) * amplitude;
        Lsum *= dither;
    }

    if( convertToLDR )
    {
        // Tonemap
        if( gOnScreen == SHOW_FINAL )
            Lsum = STL::Color::HdrToLinear_Uncharted( Lsum );

        // Conversion
        if( gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR )
            Lsum = STL::Color::LinearToSrgb( Lsum );
    }

    return Lsum;
}

float3 BicubicFilterNoCorners( Texture2D<float3> tex, SamplerState samp, float2 samplePos, float2 invTextureSize, compiletime const float sharpness )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5;
    float2 f = samplePos - centerPos;
    float2 f2 = f * f;
    float2 f3 = f * f2;
    float2 w0 = -sharpness * f3 + 2.0 * sharpness * f2 - sharpness * f;
    float2 w1 = ( 2.0 - sharpness ) * f3 - ( 3.0 - sharpness ) * f2 + 1.0;
    float2 w2 = -( 2.0 - sharpness ) * f3 + ( 3.0 - 2.0 * sharpness ) * f2 + sharpness * f;
    float2 w3 = sharpness * f3 - sharpness * f2;
    float2 wl2 = w1 + w2;
    float2 tc2 = invTextureSize * ( centerPos + w2 * STL::Math::PositiveRcp( wl2 ) );
    float2 tc0 = invTextureSize * ( centerPos - 1.0 );
    float2 tc3 = invTextureSize * ( centerPos + 2.0 );

    float w = wl2.x * w0.y;
    float3 color = tex.SampleLevel( samp, float2( tc2.x, tc0.y ), 0 ) * w;
    float sum = w;

    w = w0.x  * wl2.y;
    color += tex.SampleLevel( samp, float2( tc0.x, tc2.y ), 0 ) * w;
    sum += w;

    w = wl2.x * wl2.y;
    color += tex.SampleLevel( samp, float2( tc2.x, tc2.y ), 0 ) * w;
    sum += w;

    w = w3.x  * wl2.y;
    color += tex.SampleLevel( samp, float2( tc3.x, tc2.y ), 0 ) * w;
    sum += w;

    w = wl2.x * w3.y;
    color += tex.SampleLevel( samp, float2( tc2.x, tc3.y ), 0 ) * w;
    sum += w;

    color *= STL::Math::PositiveRcp( sum );

    return color;
}

//=============================================================================================
// VERY SIMPLE SKY MODEL
//=============================================================================================

#define SKY_INTENSITY 1.0
#define SUN_INTENSITY 8.0

float3 GetSunIntensity( float3 v, float3 sunDirection, float angularRadius )
{
    float b = dot( v, sunDirection );
    float d = length( v - sunDirection * b );

    float glow = saturate( 1.015 - d );
    glow *= b * 0.5 + 0.5;
    glow *= 0.6;

    float a = sqrt( 2.0 ) * STL::Math::Sqrt01( 1.0 - b ); // acos approx
    float sun = 1.0 - STL::Math::SmoothStep( angularRadius * 0.9, angularRadius * 1.66, a );
    sun *= 1.0 - STL::Math::Pow01( 1.0 - v.z, 4.85 );
    sun *= STL::Math::SmoothStep( 0.0, 0.1, sunDirection.z );
    sun += glow;

    float3 sunColor = lerp( float3( 1.0, 0.6, 0.3 ), float3( 1.0, 0.9, 0.7 ), STL::Math::Sqrt01( sunDirection.z ) );
    sunColor *= saturate( sun );

    sunColor *= STL::Math::SmoothStep( -0.01, 0.05, sunDirection.z );

    return STL::Color::GammaToLinear( sunColor ) * SUN_INTENSITY;
}

float3 GetSkyIntensity( float3 v, float3 sunDirection, float angularRadius )
{
    float atmosphere = sqrt( 1.0 - saturate( v.z ) );

    float scatter = pow( saturate( sunDirection.z ), 1.0 / 15.0 );
    scatter = 1.0 - clamp( scatter, 0.8, 1.0 );

    float3 scatterColor = lerp( float3( 1.0, 1.0, 1.0 ), float3( 1.0, 0.3, 0.0 ) * 1.5, scatter );
    float3 skyColor = lerp( float3( 0.2, 0.4, 0.8 ), float3( scatterColor ), atmosphere / 1.3 );
    skyColor *= saturate( 1.0 + sunDirection.z );

    return STL::Color::GammaToLinear( saturate( skyColor ) ) * SKY_INTENSITY + GetSunIntensity( v, sunDirection, angularRadius );
}
