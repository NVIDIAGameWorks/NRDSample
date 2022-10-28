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
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedDiff_PrevViewZ, t, 4, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedSpec_PrevViewZ, t, 5, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 6, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Motion, t, 7, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_TransparentLighting, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 2, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_Downsampled_ViewZ, u, 3, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_Downsampled_Motion, u, 4, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Downsampled_Normal_Roughness, u, 5, 1 );
#if( NRD_MODE == SH )
    NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffSh, u, 6, 1 );
    NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecSh, u, 7, 1 );
#endif

//========================================================================================
// MISC
//========================================================================================

float GetRadianceFromPreviousFrame( GeometryProps geometryProps, uint2 pixelPos, out float3 prevLdiff, out float3 prevLspec, bool isRefraction = false )
{
    float4 clipPrev = STL::Geometry::ProjectiveTransform( gWorldToClipPrev, geometryProps.X ); // Not Xprev because confidence is based on viewZ
    float2 uvPrev = ( clipPrev.xy / clipPrev.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;

    float4 data = gIn_PrevComposedDiff_PrevViewZ.SampleLevel( gNearestSampler, uvPrev * gRectSizePrev * gInvRenderSize, 0 );
    float prevViewZ = abs( data.w ) / NRD_FP16_VIEWZ_SCALE;

    prevLdiff = data.xyz;
    prevLspec = gIn_PrevComposedSpec_PrevViewZ.SampleLevel( gNearestSampler, uvPrev * gRectSizePrev * gInvRenderSize, 0 ).xyz;

    // Initial state
    float weight = 1.0;
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    if( isRefraction )
    {
        // Fade-out on screen edges
        weight *= all( saturate( uvPrev ) == uvPrev );
        weight *= float( pixelUv.x > gSeparator );
        weight *= float( uvPrev.x > gSeparator );
    }
    else
    {
        // Fade-out on screen edges
        float2 f = STL::Math::LinearStep( 0.0, 0.1, uvPrev ) * STL::Math::LinearStep( 1.0, 0.9, uvPrev );
        weight *= f.x * f.y;
        weight *= float( pixelUv.x > gSeparator );
        weight *= float( uvPrev.x > gSeparator );

        // Confidence - viewZ
        // No "abs" for clipPrev.w, because if it's negative we have a back-projection!
        float err = abs( prevViewZ - clipPrev.w ) * STL::Math::PositiveRcp( max( prevViewZ, abs( clipPrev.w ) ) );
        weight *= STL::Math::LinearStep( 0.02, 0.005, err );

        // Confidence - ignore back-facing
        // Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side
        float NoL = dot( geometryProps.N, gSunDirection );
        weight *= float( NoL * STL::Math::Sign( data.w ) > 0.0 );

        // Confidence - ignore too short rays
        float4 clip = STL::Geometry::ProjectiveTransform( gWorldToClip, geometryProps.X );
        float2 uv = ( clip.xy / clip.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;
        float d = length( ( uv - pixelUv ) * gRectSize );
        weight *= STL::Math::LinearStep( 1.0, 3.0, d );
    }

    // Ignore sky
    weight *= float( !geometryProps.IsSky() );

    // Clear out bad values
    [flatten]
    if( any( isnan( prevLdiff ) || isinf( prevLdiff ) || isnan( prevLspec ) || isinf( prevLspec ) ) )
    {
        prevLdiff = 0;
        prevLspec = 0;
        weight = 0;
    }

    return weight;
}

//========================================================================================
// TRACE OPAQUE
//========================================================================================

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
    float3 diffRadiance;
    float diffHitDist;

    float3 specRadiance;
    float specHitDist;

    // ( Optional ) Needed for SH and DIRECTIONAL_OCCLUSION
    float3 diffDirection;
    float3 specDirection;
};

TraceOpaqueResult TraceOpaque( TraceOpaqueDesc desc )
{
    TraceOpaqueResult result = ( TraceOpaqueResult )0;
    float diffSum = 1e-9;
    float specSum = 1e-9;
    uint pathNum = desc.pathNum << ( ( gTracingMode == RESOLUTION_FULL_PROBABILISTIC || gTracingMode == RESOLUTION_HALF ) ? 0 : 1 );

    [loop]
    for( uint i = 0; i < pathNum; i++ )
    {
        GeometryProps geometryProps = desc.geometryProps;
        MaterialProps materialProps = desc.materialProps;
        float3 Lsum = 0;
        float3 BRDF = 1.0;
        float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );
        float accumulatedRoughness = 0;
        float accumulatedHitDist = 0;
        uint bounceNum = 1 + desc.bounceNum;
        bool isDiffuse0 = true;

        [loop]
        for( uint bounceIndex = 1; bounceIndex < bounceNum && !geometryProps.IsSky(); bounceIndex++ )
        {
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

            // Estimate diffuse probability
            float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );

            // These will be "set" ( not "averaged" ) because we choose a single direction
            float3 throughput = 0;
            float3 rayDirection = 0;

            // Diffuse probability:
            //  - 0 for metals ( or if albedo is 0 )
            //  - rescaled to sane values otherwise to guarantee a sample in 3x3 or 5x5 area according to NRD settings
            //  - not used for 1st bounce in the checkerboard and 1rpp modes
            //  - always used for 2nd+ bounces, since the split is probabilistic
            diffuseProbability = clamp( diffuseProbability, gMinProbability, 1.0 - gMinProbability ) * float( diffuseProbability != 0.0 );

            bool isDiffuse = diffuseProbability != 0.0 && STL::Rng::GetFloat2( ).x < diffuseProbability;
            if( bounceIndex == 1 )
            {
                [flatten]
                if( gTracingMode == RESOLUTION_HALF )
                    isDiffuse = desc.checkerboard;
                else if( gTracingMode == RESOLUTION_FULL || gTracingMode == RESOLUTION_QUARTER )
                    isDiffuse = ( i & 0x1 ) != 0;
                else
                {
                    float rnd = STL::Sequence::Bayer4x4( desc.pixelPos, gFrameIndex ) + STL::Rng::GetFloat2( ).x / 16.0;
                    isDiffuse = diffuseProbability != 0.0 && rnd < diffuseProbability;
                }

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
                    // Generate a ray
                #if( NRD_MODE == DIRECTIONAL_OCCLUSION )
                    float3 rayLocal = STL::ImportanceSampling::Uniform::GetRay( rnd );
                    rayDirection = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

                    float NoL = saturate( dot( materialProps.N, rayDirection ) );
                    float pdf = STL::ImportanceSampling::Uniform::GetPDF( );
                #else
                    float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay( rnd );
                    rayDirection = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

                    float NoL = saturate( dot( materialProps.N, rayDirection ) );
                    float pdf = STL::ImportanceSampling::Cosine::GetPDF( NoL );
                #endif

                    // Throughput
                    throughput = NoL / ( STL::Math::Pi( 1.0 ) * pdf );

                #if( NRD_MODE != DIRECTIONAL_OCCLUSION )
                    throughput *= albedo;

                    float3 H = normalize( -geometryProps.rayDirection + rayDirection );
                    float VoH = abs( dot( -geometryProps.rayDirection, H ) );
                    float3 F = STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
                    throughput *= 1.0 - F;
                #endif
                }
                else
                {
                    // Generate a ray
                    float trimmingFactor = gTrimmingParams.x * STL::Math::SmoothStep( gTrimmingParams.y, gTrimmingParams.z, materialProps.roughness );
                    float3 Vlocal = STL::Geometry::RotateVector( mLocalBasis, -geometryProps.rayDirection );

                    float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay( rnd, materialProps.roughness, Vlocal, trimmingFactor );
                    float3 H = STL::Geometry::RotateVectorInverse( mLocalBasis, Hlocal );
                    rayDirection = reflect( geometryProps.rayDirection, H );

                    // Throughput ( see paragraph "Usage in Monte Carlo renderer" from http://jcgt.org/published/0007/04/01/paper.pdf )
                    float NoL = saturate( dot( materialProps.N, rayDirection ) );
                    throughput = STL::BRDF::GeometryTerm_Smith( materialProps.roughness, NoL );

                    float VoH = abs( dot( -geometryProps.rayDirection, H ) );
                    float3 F = STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
                    throughput *= F;
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

            // ( Optional ) Save sampling direction for the 1st bounce
            if( bounceIndex == 1 )
            {
                if( isDiffuse0 )
                    result.diffDirection += rayDirection;
                else
                    result.specDirection += rayDirection;
            }

            // Update BRDF
            BRDF *= throughput / sampleNum;

            if( bounceIndex > 1 || gTracingMode == RESOLUTION_FULL_PROBABILISTIC )
                BRDF /= isDiffuse ? diffuseProbability : ( 1.0 - diffuseProbability );

            // Abort if expected contribution of the current bounce is low
            if( STL::Color::Luminance( BRDF ) < desc.threshold )
                break;

            // Cast ray and update result ( i.e. jump to next hit point )
            geometryProps = CastRay( geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
            materialProps = GetMaterialProps( geometryProps );
            mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );

            // Compute lighting at hit point
            float3 L = materialProps.Ldirect;
            if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
            {
                float2 rnd = STL::Rng::GetFloat2();
                rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
                rnd *= gTanSunAngularRadius;

                float3x3 mSunBasis = STL::Geometry::GetBasis( gSunDirection ); // TODO: move to CB
                float3 sunDirection = normalize( mSunBasis[ 0 ] * rnd.x + mSunBasis[ 1 ] * rnd.y + mSunBasis[ 2 ] );

                L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), sunDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
            }

            L += materialProps.Lemi;

            // Estimate diffuse-like motion ( i.e. when specular moves like diffuse )
            float diffuseLikeMotionAmount = lerp( 1.0 - materialProps.metalness, 1.0, GetSpecMagicCurve( materialProps.roughness ) );
            diffuseLikeMotionAmount = lerp( diffuseLikeMotionAmount, 1.0, STL::Math::Sqrt01( materialProps.curvature ) );

            // Reuse previous frame carefully treating specular ( but specular reuse is still biased )
            float3 prevLdiff, prevLspec;
            float prevFrameWeight = GetRadianceFromPreviousFrame( geometryProps, desc.pixelPos, prevLdiff, prevLspec );

            prevFrameWeight *= isDiffuse ? 1.0 : diffuseLikeMotionAmount;
            prevFrameWeight *= 0.9 * gUsePrevFrame; // TODO: use F( bounceIndex ) bounceIndex = 99 => 1.0
            prevFrameWeight *= 1.0 - gAmbientAccumSpeed;

            float diffProbAtHit = EstimateDiffuseProbability( geometryProps, materialProps, true );
            float3 prevLsum = prevLdiff + prevLspec * diffProbAtHit; // if specular contribution is high we can't reuse it because it was computed for another view vector!

            float l1 = STL::Color::Luminance( L );
            float l2 = STL::Color::Luminance( prevLsum );
            prevFrameWeight *= l2 / ( l1 + l2 + 1e-6 );

            L = lerp( L, prevLsum, prevFrameWeight );

            // Accumulate lighting
            L *= BRDF;
            Lsum += L;

            // Accumulate path length for NRD
            /*
            IMPORTANT:
                - energy dissipation must be respected, i.e.
                    - diffuse - only 1st bounce hitT is needed
                    - specular - only 1st bounce hitT is needed for glossy+ specular
                    - sum of all segments can be ( but not necessary ) needed for 0 roughness only
                - for 0 roughness hitT must be clean like a tear! NRD specular tracking relies on it
                    - in case of probabilistic sampling there can be pixels with "no data" ( hitT = 0 ), but hitT reconstruction mode must be enabled in NRD
                - see how "diffuseLikeMotionAmount" is computed above:
                    - it's just a noise-free estimation
                    - curvature is important: it makes reflections closer to the surface making their motion diffuse-like
            */
            // TODO: apply thin lens equation to hitT? Seems not needed because "accumulatedRoughness" already includes it
            accumulatedHitDist += geometryProps.tmin * STL::Math::SmoothStep( 0.2, 0.0, accumulatedRoughness );

            float a = STL::Color::Luminance( L ) + 1e-6;
            float b = STL::Color::Luminance( Lsum ) + 1e-6;
            float importance = a / b;

            accumulatedRoughness += 1.0 - importance * ( 1.0 - diffuseLikeMotionAmount );

            // Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information ( biased )
            BRDF *= 1.0 - prevFrameWeight;
        }

        // Ambient estimation at the end of the path
        BRDF *= GetAmbientBRDF( geometryProps, materialProps );
        BRDF *= 1.0 + EstimateDiffuseProbability( geometryProps, materialProps, true ); // TODO: hack? divide by PDF of the last ray?

        float occlusion = REBLUR_FrontEnd_GetNormHitDist( geometryProps.tmin, 0.0, gHitDistParams, 1.0 );
        occlusion = lerp( 1.0 / STL::Math::Pi( 1.0 ), 1.0, occlusion );
        occlusion *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( geometryProps.X - gCameraOrigin ) );
        occlusion *= isDiffuse0 ? 0.0 : ( 1.0 - GetSpecMagicCurve( desc.materialProps.roughness ) ); // balanced with ambient applied in Composition

        Lsum += BRDF * desc.Lamb * occlusion;

        // Debug visualization: specular mip level at the end of the path
        if( gOnScreen == SHOW_MIP_SPECULAR )
        {
            float mipNorm = STL::Math::Sqrt01( geometryProps.mip / MAX_MIP_LEVEL );
            Lsum = STL::Color::ColorizeZucconi( mipNorm );
        }

        // Normalize hit distances for REBLUR and REFERENCE ( needed only for AO ) before averaging
        float diffNormHitDist = accumulatedHitDist;
        float specNormHitDist = accumulatedHitDist;

        if( gDenoiserType != RELAX )
        {
            float viewZ = STL::Geometry::AffineTransform( gWorldToView, desc.geometryProps.X ).z;

            diffNormHitDist = REBLUR_FrontEnd_GetNormHitDist( accumulatedHitDist, viewZ, gHitDistParams, 1.0 );
            specNormHitDist = REBLUR_FrontEnd_GetNormHitDist( accumulatedHitDist, viewZ, gHitDistParams, desc.materialProps.roughness );
        }

        // Accumulate diffuse and specular separately for denoising
        float w = NRD_GetSampleWeight( Lsum );

        float diffWeight = float( isDiffuse0 ) * w;
        result.diffRadiance += Lsum * diffWeight;
        result.diffHitDist += diffNormHitDist * diffWeight;
        diffSum += diffWeight;

        float specWeight = float( !isDiffuse0 ) * w;
        result.specRadiance += Lsum * specWeight;
        result.specHitDist += specNormHitDist * specWeight;
        specSum += specWeight;
    }

    // Radiance is already divided by sampling probability, we need to average across all paths
    result.diffRadiance /= float( desc.pathNum );
    result.specRadiance /= float( desc.pathNum );

    // Others are not divided by sampling probability, we need to average across diffuse / specular only paths
    result.diffHitDist /= diffSum;
    result.diffDirection /= diffSum;

    result.specHitDist /= specSum;
    result.specDirection /= specSum;

    return result;
}

//========================================================================================
// TRACE TRANSPARENT
//========================================================================================

struct TraceTransparentDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Ambient to be applied at the end of the path
    float3 Lamb;

    // Pixel position
    uint2 pixelPos;

    // Number of paths to trace
    uint pathNum;

    // Number of bounces to trace ( up to )
    uint bounceNum;

    // Is reflection or refraction in first segment?
    bool isReflection;
};

float3 TraceTransparent( TraceTransparentDesc desc )
{
    // TODO: think about spatial only denoiser in NRD. Input is L and BRDF ( transmittance )

    const float eta = STL::BRDF::IOR::Air / STL::BRDF::IOR::Glass;
    float3 Lsum = 0.0;

    [loop]
    for( uint path = 0; path < desc.pathNum; path++ )
    {
        float3 transmittance = 1.0;
        GeometryProps geometryProps = desc.geometryProps;
        bool isReflection = desc.isReflection;

        [loop]
        for( uint bounce = 0; bounce < desc.bounceNum; bounce++ ) // TODO: stop if transmittance is low
        {
            float NoV = abs( dot( geometryProps.N, -geometryProps.rayDirection ) );
            float F = STL::BRDF::FresnelTerm_Dielectric( eta, NoV );

            if( bounce == 0 )
                transmittance *= isReflection ? F : 1.0 - F;
            else
                isReflection = STL::Rng::GetFloat2().x < F;

            float3 origin = _GetXoffset( geometryProps.X, isReflection ? geometryProps.N : -geometryProps.N );
            float3 rayDirection = reflect( geometryProps.rayDirection, geometryProps.N );

            // Emulate thickness to avoid getting wrong refracted directions
            if( !isReflection )
            {
                // Glass is single sided in our scenes. If glass is not single sided, then better assume that primary ray is "air-glass",
                // the next bounce becomes "glass-air", the next switches to "air-glass" again, etc. ( if glass surface is hit, of course )
                rayDirection = refract( geometryProps.rayDirection, geometryProps.N, eta );

                float cosa = abs( dot( rayDirection, geometryProps.N ) );
                float d = GLASS_THICKNESS / max( cosa, 0.05 );
                origin += rayDirection * d; // TODO: since there is no RT, new origin can go under surface if a thin glass stands on it. It can lead to wrong shadows.

                rayDirection = refract( rayDirection, geometryProps.N, 1.0 / eta );

                transmittance *= GLASS_TINT;
            }

            uint flags = ( bounce == desc.bounceNum - 1 && !isReflection ) ? GEOMETRY_IGNORE_TRANSPARENT : GEOMETRY_ALL;
            geometryProps = CastRay( origin, rayDirection, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, 0.0 ), gWorldTlas, flags, 0 );

            // Is opaque hit found?
            if( !geometryProps.IsTransparent( ) )
            {
                MaterialProps materialProps = GetMaterialProps( geometryProps );

                // Compute lighting at hit point
                float3 L = materialProps.Ldirect;
                if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
                    L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), gSunDirection, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness ), gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0 );

                L += materialProps.Lemi;

                // Ambient lighting
                L += desc.Lamb * GetAmbientBRDF( geometryProps, materialProps );

                // Previous frame
                float3 prevLdiff, prevLspec;
                float prevFrameWeight = GetRadianceFromPreviousFrame( geometryProps, desc.pixelPos, prevLdiff, prevLspec, !isReflection );

                L = lerp( L, prevLdiff + prevLspec, prevFrameWeight );

                // Accumulate
                Lsum += L * transmittance;

                break;
            }
        }
    }

    return Lsum / float( desc.pathNum );
}

//========================================================================================
// MAIN
//========================================================================================

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

        // IMPORTANT: Don't do what is in the commented out line! It bumps up entropy of the input signal.
        // If used, for REBLUR antilag "sigmaScale" should be set to 2.
        //pixelPos += uint2( gFrameIndex & 0x1, ( gFrameIndex >> 1 ) & 0x1 ) );

        gOut_Downsampled_ViewZ[ outPixelPos ] = gIn_ViewZ[ pixelPos ];
        gOut_Downsampled_Motion[ outPixelPos ] = gIn_Motion[ pixelPos ];
        gOut_Downsampled_Normal_Roughness[ outPixelPos ] = gIn_Normal_Roughness[ pixelPos ];
    }
    else if( gTracingMode == RESOLUTION_HALF )
        outPixelPos.x >>= 1;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Initialize RNG
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    // Ambient level
    float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    Lamb *= gAmbient;

    // Transparent lighting // TODO: move after "Composition" to be able to use current final frame
    float viewZ = gIn_ViewZ[ pixelPos ];
    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gOrthoMode );

    float3 Ltransparent = 0.0;
    if( gTransparent != 0.0 )
    {
        // Primary ray
        float3 cameraRayOriginv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, gNearZ, gOrthoMode );
        float3 cameraRayOrigin = STL::Geometry::AffineTransform( gViewToWorld, cameraRayOriginv );
        float3 cameraRayDirection = gOrthoMode == 0 ? normalize( STL::Geometry::RotateVector( gViewToWorld, cameraRayOriginv ) ) : -gViewDirection;

        float tmin0 = length( Xv );
        GeometryProps geometryPropsT = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, tmin0, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, GEOMETRY_ONLY_TRANSPARENT, 0 );

        if( abs( viewZ ) != INF && geometryPropsT.tmin < tmin0 )
        {
            TraceTransparentDesc desc = ( TraceTransparentDesc )0;
            desc.geometryProps = geometryPropsT;
            desc.Lamb = Lamb;
            desc.pixelPos = pixelPos;
            desc.pathNum = 1;
            desc.bounceNum = 10;

            desc.isReflection = true;
            Ltransparent = TraceTransparent( desc );

            desc.isReflection = false;
            Ltransparent += TraceTransparent( desc );
        }
    }

    gOut_TransparentLighting[ pixelPos ] = Ltransparent;

    // Early out
    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;

    if( abs( viewZ ) == INF )
    {
        if( gTracingMode == RESOLUTION_HALF )
        {
            if( checkerboard )
            {
                gOut_Diff[ outPixelPos ] = 0;
                #if( NRD_MODE == SH )
                    gOut_DiffSh[ outPixelPos ] = 0;
                #endif
            }
            else
            {
                gOut_Spec[ outPixelPos ] = 0;
                #if( NRD_MODE == SH )
                    gOut_SpecSh[ outPixelPos ] = 0;
                #endif
            }
        }
        else
        {
            gOut_Diff[ outPixelPos ] = 0;
            gOut_Spec[ outPixelPos ] = 0;
            #if( NRD_MODE == SH )
                gOut_DiffSh[ outPixelPos ] = 0;
                gOut_SpecSh[ outPixelPos ] = 0;
            #endif
        }

        return;
    }

    // G-buffer
    float mip0 = gIn_PrimaryMip[ pixelPos ];

    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ] );
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];

    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 V = gOrthoMode == 0 ? normalize( gCameraOrigin - X ) : gViewDirection;
    float3 N = normalAndRoughness.xyz;

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
    TraceOpaqueDesc desc = ( TraceOpaqueDesc )0;
    desc.geometryProps = geometryProps0;
    desc.materialProps = materialProps0;
    desc.Lamb = Lamb;
    desc.pixelPos = pixelPos;
    desc.threshold = BRDF_ENERGY_THRESHOLD;
    desc.checkerboard = checkerboard;
    desc.pathNum = gSampleNum;
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
        float3 Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV0, roughness );

        float3 diffDemod = ( 1.0 - Fenv ) * albedo;
        float3 specDemod = Fenv;

        result.diffRadiance /= gReference ? 1.0 : ( diffDemod * 0.99 + 0.01 );
        result.specRadiance /= gReference ? 1.0 : ( specDemod * 0.99 + 0.01 );
    }

    // Debug
    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( frac( X ).x < 0.05 )
            result.diffRadiance = float3( 0, 10, 0 ) * STL::Color::Luminance( result.diffRadiance );
    #endif

    // Convert for NRD
    float4 outDiff = 0.0;
    float4 outSpec = 0.0;
    float4 outDiffSh = 0.0;
    float4 outSpecSh = 0.0;

    if( gDenoiserType == RELAX )
    {
        outDiff = RELAX_FrontEnd_PackRadianceAndHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackRadianceAndHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    }
    else
    {
    #if( NRD_MODE == OCCLUSION )
        outDiff = result.diffHitDist;
        outSpec = result.specHitDist;
    #elif( NRD_MODE == SH )
        outDiff = REBLUR_FrontEnd_PackSh( result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION );
        outSpec = REBLUR_FrontEnd_PackSh( result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION );
    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
        outDiff = REBLUR_FrontEnd_PackDirectionalOcclusion( result.diffDirection, result.diffHitDist, USE_SANITIZATION );
    #else
        outDiff = REBLUR_FrontEnd_PackRadianceAndNormHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = REBLUR_FrontEnd_PackRadianceAndNormHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    #endif
    }

    // Output
    if( gTracingMode == RESOLUTION_HALF )
    {
        if( checkerboard )
        {
            gOut_Diff[ outPixelPos ] = outDiff;
            #if( NRD_MODE == SH )
                gOut_DiffSh[ outPixelPos ] = outDiffSh;
            #endif
        }
        else
        {
            gOut_Spec[ outPixelPos ] = outSpec;
            #if( NRD_MODE == SH )
                gOut_SpecSh[ outPixelPos ] = outSpecSh;
            #endif
        }
    }
    else
    {
        gOut_Diff[ outPixelPos ] = outDiff;
        gOut_Spec[ outPixelPos ] = outSpec;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ outPixelPos ] = outDiffSh;
            gOut_SpecSh[ outPixelPos ] = outSpecSh;
        #endif
    }
}

#else

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // no TraceRayInline support
}

#endif
