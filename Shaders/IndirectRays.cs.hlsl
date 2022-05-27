/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if( !defined( COMPILER_FXC ) )

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 2, 1 );
NRI_RESOURCE( Texture2D<float>, gIn_PrimaryMip, t, 3, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevFinalLighting_PrevViewZ, t, 4, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 5, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Motion, t, 6, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 7, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 8, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffDirectionPdf, u, 9, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecDirectionPdf, u, 10, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_Downsampled_ViewZ, u, 11, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_Downsampled_Motion, u, 12, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Downsampled_Normal_Roughness, u, 13, 1 );

/*
"TraceOpaque" traces "pathNum" paths, doing up to "bounceNum" bounces. The function
has not been designed to trace primary hits. But still can be used to trace
direct and indirect lighting.

Prerequisites:
    STL::Rng::Initialize( )

Derivation:
    Lsum = L0 + BRDF0 * ( L1 + BRDF1 * ( L2 + BRDF2 * ( L3 +  ... ) ) )

    Lsum = L0 +
        L1 * BRDF0 +
        L2 * BRDF0 * BRDF1 +
        L3 * BRDF0 * BRDF1 * BRDF2 +
        ...
*/

struct TraceOpaqueDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Material properties
    MaterialProps materialProps;

    // Ambient to be applied at the end of the path
    float3 Lamb;

    // Pixel position
    uint2 pixelPos;

    // Non-jittered pixel UV
    float2 pixelUv;

    // BRDF energy threshold
    float threshold;

    // Checkerboard
    uint checkerboard;

    // Number of paths to trace
    uint pathNum;

    // Number of bounces to trace ( up to )
    uint bounceNum;

    // Instance inclusion mask ( DXR )
    uint instanceInclusionMask;

    // Ray flags ( DXR )
    uint rayFlags;
};

struct TraceOpaqueResult
{
    // Accumulated lighting
    float4 diffRadianceAndHitDist;  // IN_DIFF_RADIANCE_HITDIST
    float4 specRadianceAndHitDist;  // IN_SPEC_RADIANCE_HITDIST

    // (Optional) for nrd::PrePass:ADVANCED
    float4 diffDirectionAndPdf;     // IN_DIFF_DIRECTION_PDF
    float4 specDirectionAndPdf;     // IN_SPEC_DIRECTION_PDF
};

float3 GetRadianceFromPreviousFrame( GeometryProps geometryProps, float2 pixelUv, inout float weight )
{
    float4 clipPrev = STL::Geometry::ProjectiveTransform( gWorldToClipPrev, geometryProps.X ); // Not Xprev because confidence is based on viewZ
    float2 uvPrev = ( clipPrev.xy / clipPrev.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;
    float4 prevLsum = gIn_PrevFinalLighting_PrevViewZ.SampleLevel( gNearestMipmapNearestSampler, uvPrev * gRectSizePrev * gInvScreenSize, 0 );
    float prevViewZ = abs( prevLsum.w ) / NRD_FP16_VIEWZ_SCALE;

    // Clear out bad values
    weight *= all( !isnan( prevLsum ) && !isinf( prevLsum ) );

    // Fade-out on screen edges
    float2 f = STL::Math::LinearStep( 0.0, 0.1, uvPrev ) * STL::Math::LinearStep( 1.0, 0.9, uvPrev );
    weight *= f.x * f.y;
    weight *= float( pixelUv.x > gSeparator );
    weight *= float( uvPrev.x > gSeparator );

    // Confidence - viewZ
    // No "abs" for clipPrev.w, because if it's negative we have a back-projection!
    float err = abs( prevViewZ - clipPrev.w ) * STL::Math::PositiveRcp( min( prevViewZ, abs( clipPrev.w ) ) );
    weight *= STL::Math::LinearStep( 0.02, 0.005, err );

    // Confidence - ignore back-facing
    // Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side
    float NoL = dot( geometryProps.N, gSunDirection );
    weight *= float( NoL * STL::Math::Sign( prevLsum.w ) > 0.0 );

    // Confidence - ignore too short rays
    float4 clip = STL::Geometry::ProjectiveTransform( gWorldToClip, geometryProps.X );
    float2 uv = ( clip.xy / clip.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;
    float d = length( ( uv - pixelUv ) * gRectSize );
    weight *= STL::Math::LinearStep( 1.0, 3.0, d );

    // Ignore sky
    weight *= float( !geometryProps.IsSky() );

    return weight ? prevLsum.xyz : 0;
}

float EstimateDiffuseMagicProbability( GeometryProps geometryProps, MaterialProps materialProps )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    // TODO: add missing "1 - F" energy conservation for diffuse?
    float smc = GetSpecMagicCurve( materialProps.roughness );
    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
    float3 Fenv = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );
    float lumDiff = lerp( STL::Color::Luminance( albedo ), 1.0, smc ); // boost diffuse if roughness is high
    float lumSpec = STL::Color::Luminance( Fenv ) * lerp( 10.0, 1.0, smc ); // boost specular if roughness is low
    float diffProb = lumDiff / ( lumDiff + lumSpec + 1e-6 );

    // TODO: bump if currDir ~= primary viewVector

    return diffProb;
}

TraceOpaqueResult TraceOpaque( TraceOpaqueDesc desc )
{
    TraceOpaqueResult result = ( TraceOpaqueResult )0;
    float diffSum = 0;
    float specSum = 0;

    [loop]
    for( uint i = 0; i < desc.pathNum; i++ )
    {
        GeometryProps geometryProps = desc.geometryProps;
        MaterialProps materialProps = desc.materialProps;
        float3 Lsum = 0;
        float3 BRDF = 1.0;
        float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );
        float accumulatedRoughness = 0;
        float pathLength = 0;
        uint bounceNum = 1 + desc.bounceNum;
        bool isDiffuse0 = true;

        float accumulatedPrevFrameWeight = 0.9 * gUsePrevFrame; // TODO: use F( bounceIndex ) bounceIndex = 99 => 1.0
        accumulatedPrevFrameWeight *= 1.0 - gAmbientAccumSpeed;

        [loop]
        for( uint bounceIndex = 1; bounceIndex < bounceNum && !geometryProps.IsSky(); bounceIndex++ )
        {
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

            // Estimate diffuse probability
            float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
            float3 Fenv = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness ); // TODO: use accumulated / max roughness to avoid caustics?
            float lumSpec = STL::Color::Luminance( Fenv ) * 0.999 + 0.001;
            float lumDiff = STL::Color::Luminance( albedo ) * ( 1.0 - lumSpec );
            float diffuseProbability = lumDiff / ( lumDiff + lumSpec );

            // These will be "set" ( not "averaged" ) because we choose a single direction
            float3 throughput = 0;
            float pdf = 0;
            float3 rayDirection = 0;

            // Diffuse probability:
            //  - 0 for metals ( or if albedo is 0 )
            //  - rescaled to sane values otherwise
            //  - not used for 1st bounce in the checkerboard and 1rpp modes
            //  - always used for 2nd+ bounces, since the split is probabilistic
            diffuseProbability = diffuseProbability != 0.0 ? clamp( diffuseProbability, 0.25, 0.75 ) : 1e-6;

            bool isDiffuse = STL::Rng::GetFloat2( ).x < diffuseProbability;
            if( bounceIndex == 1 )
            {
                [flatten]
                if( gTracingMode == RESOLUTION_HALF )
                    isDiffuse = desc.checkerboard;
                else if( gTracingMode == RESOLUTION_FULL || gTracingMode == RESOLUTION_QUARTER )
                    isDiffuse = ( i & 0x1 ) != 0;
                else
                    isDiffuse = STL::Sequence::Bayer4x4( desc.pixelPos, gFrameIndex ) + STL::Rng::GetFloat2( ).x / 16.0 < diffuseProbability;

                // Save "event type" at the primary hit
                isDiffuse0 = isDiffuse;
            }

            // Choose a ray
            float3x3 mLocalBasis = STL::Geometry::GetBasis( materialProps.N );
            float throughputWithImportanceSampling = 0;
            float sampleNum = 0;

            while( sampleNum < IMPORTANCE_SAMPLE_NUM && throughputWithImportanceSampling == 0 )
            {
                float2 rnd = STL::Rng::GetFloat2( );

                if( isDiffuse )
                {
                    float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay( rnd );
                    rayDirection = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

                    throughput = albedo;

                    // Optional
                    float NoL = saturate( dot( materialProps.N, rayDirection ) );
                    pdf = STL::ImportanceSampling::Cosine::GetPDF( NoL );
                }
                else
                {
                    float trimmingFactor = NRD_GetTrimmingFactor( materialProps.roughness, gTrimmingParams );
                    float3 Vlocal = STL::Geometry::RotateVector( mLocalBasis, -geometryProps.rayDirection );
                    float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay( rnd, materialProps.roughness, Vlocal, trimmingFactor );
                    float3 H = STL::Geometry::RotateVectorInverse( mLocalBasis, Hlocal );
                    rayDirection = reflect( geometryProps.rayDirection, H );

                    // It's a part of VNDF sampling - see http://jcgt.org/published/0007/04/01/paper.pdf ( paragraph "Usage in Monte Carlo renderer" )
                    float NoL = saturate( dot( materialProps.N, rayDirection ) );
                    throughput = STL::BRDF::GeometryTerm_Smith( materialProps.roughness, NoL );

                    float VoH = abs( dot( -geometryProps.rayDirection, H ) );
                    throughput *= STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );

                    // Optional
                    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
                    float NoH = saturate( dot( materialProps.N, H ) );
                    pdf = STL::ImportanceSampling::VNDF::GetPDF( NoV, NoH, materialProps.roughness );
                }

                // Allow low roughness specular to take data from the previous frame
                throughputWithImportanceSampling = STL::Color::Luminance( throughput );
                bool isImportanceSamplingNeeded = throughputWithImportanceSampling != 0 && ( isDiffuse || ( STL::Rng::GetFloat2( ).x < materialProps.roughness ) );

                if( gDisableShadowsAndEnableImportanceSampling && isImportanceSamplingNeeded )
                {
                    bool isMiss = CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gLightTlas, GEOMETRY_ONLY_EMISSIVE, desc.rayFlags );
                    throughputWithImportanceSampling *= float( !isMiss );
                }

                sampleNum += 1.0;
            }

            // Save sampling direction and PDF of the 1st bounce ( optional )
            if( bounceIndex == 1 )
            {
                if( isDiffuse0 )
                    result.diffDirectionAndPdf += float4( rayDirection, pdf );
                else
                    result.specDirectionAndPdf += float4( rayDirection, pdf );
            }

            // Update BRDF
            BRDF *= throughput / sampleNum;

            if( bounceIndex > 1 || gTracingMode == RESOLUTION_FULL_PROBABILISTIC )
                BRDF /= isDiffuse ? diffuseProbability : ( 1.0 - diffuseProbability );

            // Abort if expected contribution of the current bounce is low
            if( STL::Color::Luminance( BRDF ) < desc.threshold )
                break;

            float metalnessAtOrigin = materialProps.metalness;
            float diffProbAtOrigin = EstimateDiffuseMagicProbability( geometryProps, materialProps );

            // Cast ray and update result ( i.e. jump to next hit point )
            geometryProps = CastRay( geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
            materialProps = GetMaterialProps( geometryProps );
            mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );

            // Compute lighting at hit point
            float3 L = materialProps.Ldirect;

            if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling ) // TODO: hard shadows for simplicity
            {
                float2 rnd = STL::Rng::GetFloat2();
                rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
                rnd *= gTanSunAngularRadius;

                float3x3 mSunBasis = STL::Geometry::GetBasis( gSunDirection ); // TODO: move to CB
                float3 sunDirection = normalize( mSunBasis[ 0 ] * rnd.x + mSunBasis[ 1 ] * rnd.y + mSunBasis[ 2 ] );

                L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), sunDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
            }

            L += materialProps.Lemi;

            // Reuse previous frame ( carefully treating specular )
            float3 prevLsum = GetRadianceFromPreviousFrame( geometryProps, desc.pixelUv, accumulatedPrevFrameWeight );

            float diffProbAtHit = EstimateDiffuseMagicProbability( geometryProps, materialProps ); // TODO: better split previous frame data into diffuse and specular and apply the logic to specular only
            accumulatedPrevFrameWeight *= lerp( diffProbAtOrigin, diffProbAtHit, metalnessAtOrigin );

            float l1 = STL::Color::Luminance( L );
            float l2 = STL::Color::Luminance( prevLsum );
            accumulatedPrevFrameWeight *= l2 / ( l1 + l2 + 1e-6 );

            L = lerp( L, prevLsum, accumulatedPrevFrameWeight );

            // Accumulate lighting
            L *= BRDF;
            Lsum += L;

            // Accumulate path length
            float a = STL::Color::Luminance( L ) + 1e-6;
            float b = STL::Color::Luminance( Lsum ) + 1e-6;
            float importance = a / b;

            float hitDist = NRD_GetCorrectedHitDist( geometryProps.tmin, bounceIndex, accumulatedRoughness, importance );
            pathLength += hitDist;

            // Accumulate roughness along the path
            accumulatedRoughness += ( isDiffuse || materialProps.metalness < 0.5 ) ? 999.0 : materialProps.roughness; // TODO: fix / improve

            // Reduce contribution of next samples if previous frame is sampled ( already with multi-bounce information )
            BRDF *= 1.0 - accumulatedPrevFrameWeight;
        }

        // Ambient estimation at the end of the path
        BRDF *= GetAmbientBRDF( geometryProps, materialProps );
        BRDF *= 1.0 + EstimateDiffuseMagicProbability( geometryProps, materialProps ); // TODO: hack? divide by PDF of the last ray?

        float occlusion = REBLUR_FrontEnd_GetNormHitDist( geometryProps.tmin, 0.0, gHitDistParams, 1.0 );
        occlusion = lerp( 1.0 / STL::Math::Pi( 1.0 ), 1.0, occlusion );
        occlusion *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( geometryProps.X - gCameraOrigin ) );
        occlusion *= float( !geometryProps.IsSky() );
        occlusion *= isDiffuse0 ? 0.0 : ( 1.0 - GetSpecMagicCurve( desc.materialProps.roughness ) ); // balanced with ambient applied in Composition

        Lsum += BRDF * desc.Lamb * occlusion;

        // Debug visualization: specular mip level at the end of the path
        if( gOnScreen == SHOW_MIP_SPECULAR )
        {
            float mipNorm = STL::Math::Sqrt01( geometryProps.mip / MAX_MIP_LEVEL );
            Lsum = STL::Color::ColorizeZucconi( mipNorm );
        }

        // Accumulate diffuse and specular separately for denoising
        float w = NRD_GetSampleWeight( Lsum );

        float diffWeight = float( isDiffuse0 ) * w;
        result.diffRadianceAndHitDist.xyz += Lsum * diffWeight;
        result.diffRadianceAndHitDist.w += pathLength * diffWeight;
        diffSum += diffWeight;

        float specWeight = float( !isDiffuse0 ) * w;
        result.specRadianceAndHitDist.xyz += Lsum * specWeight;
        result.specRadianceAndHitDist.w += pathLength * specWeight;
        specSum += specWeight;
    }

    float invDiffSum = 1.0 / max( diffSum, 1e-6 );
    result.diffRadianceAndHitDist *= invDiffSum;
    result.diffDirectionAndPdf *= invDiffSum;

    float invSpecSum = 1.0 / max( specSum, 1e-6 );
    result.specRadianceAndHitDist *= invSpecSum;
    result.specDirectionAndPdf *= invSpecSum;

    return result;
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    float2 rectSize = gRectSize * ( gTracingMode == RESOLUTION_QUARTER ? 0.5 : 1.0 );
    if( pixelPos.x >= rectSize.x || pixelPos.y >= rectSize.y )
        return;

    uint2 outPixelPos = pixelPos;

    [branch]
    if( gTracingMode == RESOLUTION_QUARTER )
    {
        pixelPos <<= 1;

        // IMPORTANT: Don't do what is in commented out line! It bumps up entropy of the input signal.
        // If used, for REBLUR antilag "sigmaScale" should be set to 2.

        //pixelPos += uint2( gFrameIndex & 0x1, ( gFrameIndex >> 1 ) & 0x1 ) );

        gOut_Downsampled_ViewZ[ outPixelPos ] = gIn_ViewZ[ pixelPos ];
        gOut_Downsampled_Motion[ outPixelPos ] = gIn_Motion[ pixelPos ];
        gOut_Downsampled_Normal_Roughness[ outPixelPos ] = gIn_Normal_Roughness[ pixelPos ];
    }

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Early out
    float viewZ = gIn_ViewZ[ pixelPos ];
    if( abs( viewZ ) == INF )
    {
        gOut_Diff[ outPixelPos ] = 0;
        gOut_Spec[ outPixelPos ] = 0;

        return;
    }

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ] );
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];

    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gOrthoMode );
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 V = GetViewVector( X );
    float3 N = normalAndRoughness.xyz;
    float mip0 = gIn_PrimaryMip[ pixelPos ];

    float zScale = 0.0003 + abs( viewZ ) * 0.00005;
    float NoV0 = abs( dot( N, V ) );
    float3 Xoffset = _GetXoffset( X, N );
    Xoffset += V * zScale;
    Xoffset += N * STL::BRDF::Pow5( NoV0 ) * zScale;

    GeometryProps geometryProps0 = ( GeometryProps )0;
    geometryProps0.X = Xoffset;
    geometryProps0.rayDirection = -V;
    geometryProps0.N = N;
    geometryProps0.mip = mip0 * mip0 * MAX_MIP_LEVEL;

    MaterialProps materialProps0 = ( MaterialProps )0;
    materialProps0.N = N;
    materialProps0.baseColor = baseColorMetalness.xyz;
    materialProps0.roughness = normalAndRoughness.w;
    materialProps0.metalness = baseColorMetalness.w;

    // Secondary rays ( indirect and direct lighting from local light sources )
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;

    float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    Lamb *= gAmbient;

    TraceOpaqueDesc desc = ( TraceOpaqueDesc )0;
    desc.geometryProps = geometryProps0;
    desc.materialProps = materialProps0;
    desc.Lamb = Lamb;
    desc.pixelPos = pixelPos;
    desc.pixelUv = pixelUv;
    desc.threshold = BRDF_ENERGY_THRESHOLD;
    desc.checkerboard = checkerboard;
    desc.pathNum = gSampleNum << ( ( gTracingMode == RESOLUTION_FULL_PROBABILISTIC || gTracingMode == RESOLUTION_HALF ) ? 0 : 1 );
    desc.bounceNum = gBounceNum; // TODO: adjust by roughness
    desc.instanceInclusionMask = GEOMETRY_IGNORE_TRANSPARENT;
    desc.rayFlags = 0;

    TraceOpaqueResult result = TraceOpaque( desc );

    // De-modulate materials for denoising
    if( gOnScreen != SHOW_MIP_SPECULAR )
    {
        float3 albedo, Rf0;
        STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo, Rf0 );

        float roughness = materialProps0.roughness;
        float3 Fenv = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV0, roughness );

        float lobeAngleNorm = 2.0 * roughness * roughness / ( 1.0 + roughness * roughness );
        float roughnessMod = USE_ROUGHNESS_BASED_DEMODULATION ? lerp( 0.1, 1.0, lobeAngleNorm ) : 1.0;

        float3 diffDemod = gReference ? 1.0 : albedo;
        float3 specDemod = gReference ? 1.0 : Fenv * roughnessMod;

        result.diffRadianceAndHitDist.xyz *= rcp( max( diffDemod, 0.001 ) );
        result.specRadianceAndHitDist.xyz *= rcp( max( specDemod, 0.001 ) ); // actually, can't be 0
    }

    // Convert for NRD
    if( gDenoiserType == REBLUR )
    {
        result.diffRadianceAndHitDist.w = REBLUR_FrontEnd_GetNormHitDist( result.diffRadianceAndHitDist.w, viewZ, gHitDistParams, 1.0 );
        result.specRadianceAndHitDist.w = REBLUR_FrontEnd_GetNormHitDist( result.specRadianceAndHitDist.w, viewZ, gHitDistParams, materialProps0.roughness );

        result.diffRadianceAndHitDist = REBLUR_FrontEnd_PackRadianceAndHitDist( result.diffRadianceAndHitDist.xyz, result.diffRadianceAndHitDist.w, USE_SANITIZATION );
        result.specRadianceAndHitDist = REBLUR_FrontEnd_PackRadianceAndHitDist( result.specRadianceAndHitDist.xyz, result.specRadianceAndHitDist.w, USE_SANITIZATION );
    }
    else
    {
        result.diffRadianceAndHitDist = RELAX_FrontEnd_PackRadianceAndHitDist( result.diffRadianceAndHitDist.xyz, result.diffRadianceAndHitDist.w, USE_SANITIZATION );
        result.specRadianceAndHitDist = RELAX_FrontEnd_PackRadianceAndHitDist( result.specRadianceAndHitDist.xyz, result.specRadianceAndHitDist.w, USE_SANITIZATION );
    }

    result.diffDirectionAndPdf = NRD_FrontEnd_PackDirectionAndPdf( result.diffDirectionAndPdf.xyz, result.diffDirectionAndPdf.w );
    result.specDirectionAndPdf = NRD_FrontEnd_PackDirectionAndPdf( result.specDirectionAndPdf.xyz, result.specDirectionAndPdf.w );

    // Occlusion only
    [flatten]
    if( gOcclusionOnly )
    {
        result.diffRadianceAndHitDist = result.diffRadianceAndHitDist.wwww;
        result.specRadianceAndHitDist = result.specRadianceAndHitDist.wwww;
    }

    // Debug
    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( frac( X ).x < 0.05 )
            result.diffRadianceAndHitDist.xyz = float3( 0, 10, 0 ) * STL::Color::Luminance( result.diffRadianceAndHitDist.xyz );
    #endif

    // Output
    if( gTracingMode == RESOLUTION_HALF )
    {
        pixelPos.x >>= 1;

        if( checkerboard )
        {
            gOut_Diff[ pixelPos ] = result.diffRadianceAndHitDist;
            gOut_DiffDirectionPdf[ pixelPos ] = result.diffDirectionAndPdf;
        }
        else
        {
            gOut_Spec[ pixelPos ] = result.specRadianceAndHitDist;
            gOut_SpecDirectionPdf[ pixelPos ] = result.specDirectionAndPdf;
        }
    }
    else
    {
        gOut_Diff[ outPixelPos ] = result.diffRadianceAndHitDist;
        gOut_DiffDirectionPdf[ outPixelPos ] = result.diffDirectionAndPdf;

        gOut_Spec[ outPixelPos ] = result.specRadianceAndHitDist;
        gOut_SpecDirectionPdf[ outPixelPos ] = result.specDirectionAndPdf;
    }
}

#else

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // no TraceRayInline support
}

#endif
