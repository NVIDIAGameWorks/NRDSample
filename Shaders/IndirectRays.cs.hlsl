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
NRI_RESOURCE( Texture2D<float2>, gIn_PrimaryMipAndCurvature, t, 0, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_PrevComposedDiff, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedSpec_PrevViewZ, t, 2, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 3, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gInOut_DirectLighting, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gInOut_BaseColor_Metalness, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gInOut_Normal_Roughness, u, 2, 1 );
NRI_RESOURCE( RWTexture2D<float>, gInOut_ViewZ, u, 3, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gInOut_Mv, u, 4, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_PsrThroughput, u, 5, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 6, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 7, 1 );
#if( NRD_MODE == SH )
    NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffSh, u, 8, 1 );
    NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecSh, u, 9, 1 );
#endif

float4 GetRadianceFromPreviousFrame( GeometryProps geometryProps, MaterialProps materialProps, uint2 pixelPos, bool isDiffuse )
{
    // Reproject previous frame
    float3 prevLdiff, prevLspec;
    float prevFrameWeight = ReprojectRadiance( true, false, gIn_PrevComposedDiff, gIn_PrevComposedSpec_PrevViewZ, geometryProps, pixelPos, prevLdiff, prevLspec );
    prevFrameWeight *= gPrevFrameConfidence; // see C++ code for details

    // Estimate how strong lighting at hit depends on the view direction
    float diffuseProbabilityBiased = EstimateDiffuseProbability( geometryProps, materialProps, true );
    float3 prevLsum = prevLdiff + prevLspec * diffuseProbabilityBiased;

    float diffuseLikeMotion = lerp( diffuseProbabilityBiased, 1.0, STL::Math::Sqrt01( materialProps.curvature ) );
    prevFrameWeight *= isDiffuse ? 1.0 : diffuseLikeMotion;

    return float4( prevLsum, prevFrameWeight );
}

//========================================================================================
// TRACE OPAQUE
//========================================================================================

/*
"TraceOpaque" traces "pathNum" paths, doing up to "bounceNum" bounces. The function
has not been designed to trace primary hits. But still can be used to trace
direct and indirect lighting.

Prerequisites:
    STL::Rng::Hash::Initialize( )

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

    // Pixel position
    uint2 pixelPos;

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

    #if( NRD_MODE == DIRECTIONAL_OCCLUSION || NRD_MODE == SH )
        float3 diffDirection;
        float3 specDirection;
    #endif
};

bool IsPsrAllowed( MaterialProps materialProps )
{
    return materialProps.roughness < 0.044 && ( materialProps.metalness > 0.941 || STL::Color::Luminance( materialProps.baseColor ) < 0.005 ); // TODO: tweaked for some content?
}

TraceOpaqueResult TraceOpaque( TraceOpaqueDesc desc )
{
    // TODO: for RESOLUTION_FULL_PROBABILISTIC with 1 path per pixel the tracer can be significantly simplified

    //=====================================================================================================================================================================
    // Primary surface replacement ( PSR )
    //=====================================================================================================================================================================

    float viewZ = STL::Geometry::AffineTransform( gWorldToView, desc.geometryProps.X ).z;
    float3 psrLsum = 0.0;

    #if( USE_PSR == 1 )
        float3x3 mirrorMatrix = STL::Geometry::GetMirrorMatrix( 0 ); // identity
        float3 psrThroughput = 1.0;

        if( gPSR && IsPsrAllowed( desc.materialProps ) )
        {
            GeometryProps geometryProps = desc.geometryProps;
            MaterialProps materialProps = desc.materialProps;

            float accumulatedHitDist = 0.0;
            float accumulatedPseudoCurvature = 0.0; // TODO: is there a less empirical solution?

            [loop]
            for( uint bounceIndex = 1; bounceIndex <= desc.bounceNum; bounceIndex++ )
            {
                //=============================================================================================================================================================
                // Origin point
                //=============================================================================================================================================================

                float curvature = materialProps.curvature;
                accumulatedPseudoCurvature += STL::Math::Pow01( abs( curvature ), 0.25 );

                {
                    // Accumulate mirror matrix
                    mirrorMatrix = mul( STL::Geometry::GetMirrorMatrix( materialProps.N ), mirrorMatrix );

                    // Choose a ray
                    float3 ray = reflect( -geometryProps.V, materialProps.N );

                    // Update throughput
                    #if( NRD_MODE != OCCLUSION && NRD_MODE != DIRECTIONAL_OCCLUSION )
                        float3 albedo, Rf0;
                        STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

                        float NoV = abs( dot( materialProps.N, geometryProps.V ) );
                        float3 Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );

                        psrThroughput *= Fenv;
                    #endif

                    // Abort if expected contribution of the current bounce is low
                    if( PSR_THROUGHPUT_THRESHOLD != 0.0 && STL::Color::Luminance( psrThroughput ) < PSR_THROUGHPUT_THRESHOLD )
                        break;

                    //=========================================================================================================================================================
                    // Trace to the next hit
                    //=========================================================================================================================================================

                    float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );
                    geometryProps = CastRay( geometryProps.GetXoffset( ), ray, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
                    materialProps = GetMaterialProps( geometryProps );
                }

                //=============================================================================================================================================================
                // Hit point
                //=============================================================================================================================================================

                {
                    // Accumulate hit distance representing virtual point position ( see "README/NOISY INPUTS" )
                    // TODO: this heuristic works best so far
                    float hitDistFocused = geometryProps.tmin / ( 2.0 * curvature * geometryProps.tmin + 1.0 );
                    accumulatedHitDist += hitDistFocused * STL::Math::SmoothStep( 0.2, 0.0, accumulatedPseudoCurvature );

                    // If hit is not a delta-event ( or the last bounce, or a miss ) it's PSR
                    if( !IsPsrAllowed( materialProps ) || bounceIndex == desc.bounceNum || geometryProps.IsSky( ) )
                    {
                        // Update bounces
                        desc.bounceNum -= bounceIndex;
                        break;
                    }
                }
            }

            // L1 cache - reproject previous frame, carefully treating specular
            float4 Lcached = GetRadianceFromPreviousFrame( geometryProps, materialProps, desc.pixelPos, false );

            // Compute lighting at hit, if not found in caches
            if( Lcached.w != 1.0 )
            {
                float3 L = materialProps.Ldirect;
                if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
                {
                    float2 rnd = STL::Rng::Hash::GetFloat2( );
                    rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
                    rnd *= gTanSunAngularRadius;

                    float3x3 mSunBasis = STL::Geometry::GetBasis( gSunDirection ); // TODO: move to CB
                    float3 sunDirection = normalize( mSunBasis[ 0 ] * rnd.x + mSunBasis[ 1 ] * rnd.y + mSunBasis[ 2 ] );

                    float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );
                    L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), sunDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
                }

                L += materialProps.Lemi;

                // Mix
                Lcached.xyz = lerp( L, Lcached.xyz, Lcached.w );
            }

            // Throughput will be applied later
            psrLsum = Lcached.xyz;

            // Update materials and INF emission
            if( !geometryProps.IsSky( ) )
            {
                // IMPORTANT: use another set of materialID to avoid potential leaking, because PSR uses own de-modulation scheme
                float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );
                float materialID = diffuseProbability != 0.0 ? 2.0 : 3.0;

                float3 psrNormal = STL::Geometry::RotateVectorInverse( mirrorMatrix, materialProps.N );

                gInOut_BaseColor_Metalness[ desc.pixelPos ] = float4( STL::Color::LinearToSrgb( materialProps.baseColor ), materialProps.metalness );
                gInOut_Normal_Roughness[ desc.pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( psrNormal, materialProps.roughness, materialID );
            }
            else
            {
                // Composition pass doesn't apply "psrThroughput" to INF pixels
                gInOut_DirectLighting[ desc.pixelPos ] = psrLsum * psrThroughput;
            }

            // Update viewZ
            float3 Xvirtual = desc.geometryProps.X - desc.geometryProps.V * accumulatedHitDist;
            viewZ = STL::Geometry::AffineTransform( gWorldToView, Xvirtual ).z;

            gInOut_ViewZ[ desc.pixelPos ] = geometryProps.IsSky( ) ? STL::Math::Sign( viewZ ) * INF : viewZ;

            // Update motion
            float3 Xprev = geometryProps.X;
            if( !geometryProps.IsSky( ) && !geometryProps.IsStatic( ) )
            {
                InstanceData instanceData = gIn_InstanceData[ geometryProps.instanceIndex ];
                Xprev = STL::Geometry::AffineTransform( instanceData.mWorldToWorldPrev, geometryProps.X );
            }

            float3 XvirtualPrev = Xvirtual + Xprev - geometryProps.X;
            float3 motion = GetMotion( Xvirtual, XvirtualPrev );
            gInOut_Mv[ desc.pixelPos ] = motion;

            // Replace primary surface props with the replacement props
            desc.geometryProps = geometryProps;
            desc.materialProps = materialProps;
        }

        gOut_PsrThroughput[ desc.pixelPos ] = psrThroughput;
    #endif

    //=====================================================================================================================================================================
    // Tracing from the primary hit or PSR
    //=====================================================================================================================================================================

    TraceOpaqueResult result = ( TraceOpaqueResult )0;

    uint pathNum = desc.pathNum << ( gTracingMode == RESOLUTION_FULL ? 1 : 0 );
    uint diffPathsNum = 0;

    [loop]
    for( uint i = 0; i < pathNum; i++ )
    {
        GeometryProps geometryProps = desc.geometryProps;
        MaterialProps materialProps = desc.materialProps;

        float accumulatedHitDist = 0;
        float accumulatedDiffuseLikeMotion = 0;

        float3 Lsum = psrLsum;
        float3 pathThroughput = 1.0;

        bool isDiffusePath = gTracingMode == RESOLUTION_HALF ? desc.checkerboard : ( i & 0x1 );

        [loop]
        for( uint bounceIndex = 1; bounceIndex <= desc.bounceNum && !geometryProps.IsSky(); bounceIndex++ )
        {
            bool isDiffuse = isDiffusePath;

            //=============================================================================================================================================================
            // Origin point
            //=============================================================================================================================================================

            {
                // Estimate diffuse probability
                float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );

                // Clamp probability to a sane range to guarantee a sample in 3x3 ( or 5x5 ) area
                float rnd = STL::Rng::Hash::GetFloat( );
                if( gTracingMode == RESOLUTION_FULL_PROBABILISTIC && bounceIndex == 1 )
                {
                    diffuseProbability = float( diffuseProbability != 0.0 ) * clamp( diffuseProbability, gMinProbability, 1.0 - gMinProbability );
                    rnd = STL::Sequence::Bayer4x4( desc.pixelPos, gFrameIndex ) + rnd / 16.0;
                }

                // Diffuse or specular path?
                if( gTracingMode == RESOLUTION_FULL_PROBABILISTIC || bounceIndex > 1 )
                {
                    isDiffuse = rnd < diffuseProbability;
                    pathThroughput /= abs( float( !isDiffuse ) - diffuseProbability );

                    if( bounceIndex == 1 )
                        isDiffusePath = isDiffuse;
                }

                float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );

                // Choose a ray
                float3x3 mLocalBasis = STL::Geometry::GetBasis( materialProps.N );
                float3 Vlocal = STL::Geometry::RotateVector( mLocalBasis, geometryProps.V );
                float3 ray = 0;
                uint samplesNum = 0;
                uint maxSamplesNum = gDisableShadowsAndEnableImportanceSampling && bounceIndex == 1 && ( isDiffuse || STL::Rng::Hash::GetFloat( ) < materialProps.roughness ) ? IMPORTANCE_SAMPLE_NUM : 1;

                for( uint sampleIndex = 0; sampleIndex < maxSamplesNum; sampleIndex++ )
                {
                    float2 rnd = STL::Rng::Hash::GetFloat2( );

                    // Generate a ray in local space
                    float3 r;
                    if( isDiffuse )
                    {
                        #if( NRD_MODE == DIRECTIONAL_OCCLUSION )
                            r = STL::ImportanceSampling::Uniform::GetRay( rnd );
                        #else
                            r = STL::ImportanceSampling::Cosine::GetRay( rnd );
                        #endif
                    }
                    else
                    {
                        float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay( rnd, materialProps.roughness, Vlocal, SPEC_LOBE_ENERGY );
                        r = reflect( -Vlocal, Hlocal );
                    }

                    // Ignore sub-surface scattering // TODO: needed?
                    bool isMiss = r.z < 0.0;

                    // Transform to world space
                    r = STL::Geometry::RotateVectorInverse( mLocalBasis, r );

                    // Importance sampling for direct lighting
                    // TODO: move direct lighting tracing into a separate pass:
                    // - currently AO and SO get replaced with useless distances to closest lights if IS is on
                    // - better separate direct and indirect lighting denoising

                    //   1. If IS enabled, check the ray in LightBVH
                    if( !isMiss && maxSamplesNum != 1 )
                        isMiss = CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), r, 0.0, INF, mipAndCone, gLightTlas, GEOMETRY_ALL, desc.rayFlags );

                    //   2. Count rays hitting emissive surfaces
                    if( !isMiss )
                        samplesNum++;

                    //   3. Save either the first ray or the current ray hitting an emissive
                    if( !isMiss || sampleIndex == 0 )
                        ray = r;
                }

                // Adjust throughput by percentage of rays hitting any emissive surface ( do not modify throughput if there is no a hit, it's needed to cast a non-IS ray and get correct AO / SO at least )
                if( samplesNum != 0 )
                    pathThroughput *= float( samplesNum ) / float( maxSamplesNum );

                // ( Optional ) Save sampling direction for the 1st bounce
                #if( NRD_MODE == DIRECTIONAL_OCCLUSION || NRD_MODE == SH )
                    if( bounceIndex == 1 )
                    {
                        float3 psrRay = ray;
                        #if( USE_PSR == 1 )
                            psrRay = STL::Geometry::RotateVectorInverse( mirrorMatrix, ray );
                        #endif

                        if( isDiffuse )
                            result.diffDirection += psrRay;
                        else
                            result.specDirection += psrRay;
                    }
                #endif

                // Update path throughput
                #if( NRD_MODE != OCCLUSION && NRD_MODE != DIRECTIONAL_OCCLUSION )
                    float3 albedo, Rf0;
                    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

                    float3 H = normalize( geometryProps.V + ray );
                    float VoH = abs( dot( geometryProps.V, H ) );
                    float NoL = saturate( dot( materialProps.N, ray ) );

                    if( isDiffuse )
                    {
                        float NoV = abs( dot( materialProps.N, geometryProps.V ) );
                        pathThroughput *= STL::Math::Pi( 1.0 ) * albedo * STL::BRDF::DiffuseTerm_Burley( materialProps.roughness, NoL, NoV, VoH );
                    }
                    else
                    {
                        float3 F = STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
                        pathThroughput *= F;

                        // See paragraph "Usage in Monte Carlo renderer" from http://jcgt.org/published/0007/04/01/paper.pdf
                        pathThroughput *= STL::BRDF::GeometryTerm_Smith( materialProps.roughness, NoL );
                    }
                #endif

                // Abort if expected contribution of the current bounce is low
                #if( USE_RUSSIAN_ROULETTE == 1 )
                    /*
                    BAD PRACTICE:
                    Russian Roulette approach is here to demonstrate that it's a bad practice for real time denoising for the following reasons:
                    - increases entropy of the signal
                    - transforms radiance into non-radiance, which is strictly speaking not allowed to be processed spatially (who wants to get a high energy firefly
                    redistributed around surrounding pixels?)
                    - not necessarily converges to the right image, because we do assumptions about the future and approximate the tail of the path via a scaling factor
                    - this approach breaks denoising, especially REBLUR, which has been designed to work with pure radiance
                    */

                    // Nevertheless, RR can be used with caution: the code below tuned for good IQ / PERF tradeoff
                    float russianRouletteProbability = STL::Color::Luminance( pathThroughput );
                    russianRouletteProbability = STL::Math::Pow01( russianRouletteProbability, 0.25 );
                    russianRouletteProbability = max( russianRouletteProbability, 0.01 );

                    if( STL::Rng::Hash::GetFloat( ) > russianRouletteProbability )
                        break;

                    pathThroughput /= russianRouletteProbability;
                #else
                    /*
                    GOOD PRACTICE:
                    - terminate path if "pathThroughput" is smaller than some threshold
                    - approximate ambient at the end of the path
                    - re-use data from the previous frame
                    */

                    if( THROUGHPUT_THRESHOLD != 0.0 && STL::Color::Luminance( pathThroughput ) < THROUGHPUT_THRESHOLD )
                        break;
                #endif

                //=========================================================================================================================================================
                // Trace to the next hit
                //=========================================================================================================================================================

                geometryProps = CastRay( geometryProps.GetXoffset( ), ray, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
                materialProps = GetMaterialProps( geometryProps );
            }

            //=============================================================================================================================================================
            // Hit point
            //=============================================================================================================================================================

            {
                // L1 cache - reproject previous frame, carefully treating specular
                float4 Lcached = GetRadianceFromPreviousFrame( geometryProps, materialProps, desc.pixelPos, false );

                // Compute lighting at hit, if not found in caches
                if( Lcached.w != 1.0 )
                {
                    float3 L = materialProps.Ldirect;
                    if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
                    {
                        float2 rnd = STL::Rng::Hash::GetFloat2( );
                        rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
                        rnd *= gTanSunAngularRadius;

                        float3x3 mSunBasis = STL::Geometry::GetBasis( gSunDirection ); // TODO: move to CB
                        float3 sunDirection = normalize( mSunBasis[ 0 ] * rnd.x + mSunBasis[ 1 ] * rnd.y + mSunBasis[ 2 ] );

                        float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );
                        L *= CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), sunDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
                    }

                    L += materialProps.Lemi;

                    // Mix
                    Lcached.xyz = lerp( L, Lcached.xyz, Lcached.w );
                }

                // Accumulate lighting
                float3 L = Lcached.xyz * pathThroughput;
                Lsum += L;

                // ( Biased ) Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information
                pathThroughput *= 1.0 - Lcached.w;

                // Accumulate path length for NRD ( see "README/NOISY INPUTS" )
                /*
                REQUIREMENTS:
                    - hitT for the bounce after the primary hit or PSR must be provided "as is"
                    - hitT for subsequent bounces and for bounces before PSR must be adjusted by curvature and lobe energy dissipation on the application side

                IDEAL SOLUTION ( after PSR ):
                    hitDist = 0;
                    for( uint bounce = N; bounce > PSR + 1; bounce-- )
                    {
                        hitDist += hitT[ bounce ];
                        hitDist = ApplyThinLensEquation( hitDist, curvature[ bounce - 1 ] );
                        hitDist = ApplyLobeSpread( hitDist, bounce, isDiffuse ? 1.0 : roughness  );
                    }
                    hitDist += hitT[ PSR + 1 ];
                */
                float a = STL::Color::Luminance( L );
                float b = STL::Color::Luminance( Lsum ); // already includes L
                float importance = a / ( b + 1e-6 );

                importance *= 1.0 - STL::Color::Luminance( materialProps.Lemi ) / ( a + 1e-6 );

                float diffuseLikeMotion = EstimateDiffuseProbability( geometryProps, materialProps, true );
                diffuseLikeMotion = lerp( diffuseLikeMotion, 1.0, STL::Math::Sqrt01( materialProps.curvature ) );
                diffuseLikeMotion = isDiffuse ? 1.0 : diffuseLikeMotion;

                accumulatedHitDist += geometryProps.tmin * STL::Math::SmoothStep( 0.2, 0.0, accumulatedDiffuseLikeMotion );
                accumulatedDiffuseLikeMotion += 1.0 - importance * ( 1.0 - diffuseLikeMotion );
            }
        }

        // Ambient estimation at the end of the path ( balanced with ambient applied in Composition )
        pathThroughput *= GetAmbientBRDF( geometryProps, materialProps );
        pathThroughput *= 1.0 + EstimateDiffuseProbability( geometryProps, materialProps, true );

        float occlusion = REBLUR_FrontEnd_GetNormHitDist( geometryProps.tmin, 0.0, gHitDistParams, 1.0 );
        occlusion = lerp( 1.0 / STL::Math::Pi( 1.0 ), 1.0, occlusion );
        occlusion *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( geometryProps.X - gCameraOrigin ) );
        occlusion *= 1.0 - GetSpecMagicCurve( isDiffusePath ? 1.0 : desc.materialProps.roughness );

        float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
        Lamb *= gAmbient;

        Lsum += pathThroughput * Lamb * occlusion;

        // Debug visualization: specular mip level at the end of the path
        if( gOnScreen == SHOW_MIP_SPECULAR )
        {
            float mipNorm = STL::Math::Sqrt01( geometryProps.mip / MAX_MIP_LEVEL );
            Lsum = STL::Color::ColorizeZucconi( mipNorm );
        }

        // Normalize hit distances for REBLUR and REFERENCE ( needed only for AO ) before averaging
        float normHitDist = accumulatedHitDist;
        if( gDenoiserType != RELAX )
            normHitDist = REBLUR_FrontEnd_GetNormHitDist( accumulatedHitDist, viewZ, gHitDistParams, isDiffusePath ? 1.0 : desc.materialProps.roughness );

        // Accumulate diffuse and specular separately for denoising
        if( NRD_IsValidRadiance( Lsum ) )
        {
            if( isDiffusePath )
            {
                result.diffRadiance += Lsum;
                result.diffHitDist += normHitDist;
                diffPathsNum++;
            }
            else
            {
                result.specRadiance += Lsum;
                result.specHitDist += normHitDist;
            }
        }
    }

    // Material de-modulation ( convert irradiance into radiance )
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( desc.materialProps.baseColor, desc.materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( desc.materialProps.N, desc.geometryProps.V ) );
    float3 Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV, desc.materialProps.roughness );
    float3 diffDemod = ( 1.0 - Fenv ) * albedo * 0.99 + 0.01;
    float3 specDemod = Fenv * 0.99 + 0.01;

    if( gOnScreen != SHOW_MIP_SPECULAR )
    {
        result.diffRadiance /= diffDemod;
        result.specRadiance /= specDemod;
    }

    // Radiance is already divided by sampling probability, we need to average across all paths
    float radianceNorm = 1.0 / float( desc.pathNum );
    result.diffRadiance *= radianceNorm;
    result.specRadiance *= radianceNorm;

    // Others are not divided by sampling probability, we need to average across diffuse / specular only paths
    float diffNorm = diffPathsNum == 0 ? 0.0 : 1.0 / float( diffPathsNum );
    result.diffHitDist *= diffNorm;

    float specNorm = pathNum == diffPathsNum ? 0.0 : 1.0 / float( pathNum - diffPathsNum );
    result.specHitDist *= specNorm;

    #if( NRD_MODE == DIRECTIONAL_OCCLUSION || NRD_MODE == SH )
        result.diffDirection *= diffNorm;
        result.specDirection *= specNorm;
    #endif

    return result;
}

//========================================================================================
// MAIN
//========================================================================================

void WriteResult( uint checkerboard, uint2 outPixelPos, float4 diff, float4 spec, float4 diffSh, float4 specSh )
{
    if( gTracingMode == RESOLUTION_HALF )
    {
        if( checkerboard )
        {
            gOut_Diff[ outPixelPos ] = diff;
            #if( NRD_MODE == SH )
                gOut_DiffSh[ outPixelPos ] = diffSh;
            #endif
        }
        else
        {
            gOut_Spec[ outPixelPos ] = spec;
            #if( NRD_MODE == SH )
                gOut_SpecSh[ outPixelPos ] = specSh;
            #endif
        }
    }
    else
    {
        gOut_Diff[ outPixelPos ] = diff;
        gOut_Spec[ outPixelPos ] = spec;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ outPixelPos ] = diffSh;
            gOut_SpecSh[ outPixelPos ] = specSh;
        #endif
    }
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    const float NaN = sqrt( -1 );

    // Checkerboard
    uint2 outPixelPos = pixelPos;
    if( gTracingMode == RESOLUTION_HALF )
        outPixelPos.x >>= 1;

    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;

    // Do not generate NANs for unused threads
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
    {
        #if( USE_DRS_STRESS_TEST == 1 )
            WriteResult( checkerboard, outPixelPos, NaN, NaN, NaN, NaN );
        #endif

        return;
    }

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Early out
    float viewZ = gInOut_ViewZ[ pixelPos ];
    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gOrthoMode );

    if( abs( viewZ ) == INF )
    {
        #if( USE_INF_STRESS_TEST == 1 )
            WriteResult( checkerboard, outPixelPos, NaN, NaN, NaN, NaN );
        #endif

        return;
    }

    // Initialize RNG
    STL::Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gInOut_Normal_Roughness[ pixelPos ] );

    float2 primaryMipAndCurvature = gIn_PrimaryMipAndCurvature[ pixelPos ];
    primaryMipAndCurvature *= primaryMipAndCurvature;

    float4 baseColorMetalness = gInOut_BaseColor_Metalness[ pixelPos ];
    baseColorMetalness.xyz = STL::Color::SrgbToLinear( baseColorMetalness.xyz );

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
    geometryProps0.V = V;
    geometryProps0.N = N;
    geometryProps0.mip = primaryMipAndCurvature.x * MAX_MIP_LEVEL;

    MaterialProps materialProps0 = ( MaterialProps )0;
    materialProps0.N = N;
    materialProps0.baseColor = baseColorMetalness.xyz;
    materialProps0.roughness = normalAndRoughness.w;
    materialProps0.metalness = baseColorMetalness.w;
    materialProps0.curvature = primaryMipAndCurvature.y * 4.0;

    // Secondary rays ( indirect and direct lighting from local light sources )
    TraceOpaqueDesc desc = ( TraceOpaqueDesc )0;
    desc.geometryProps = geometryProps0;
    desc.materialProps = materialProps0;
    desc.pixelPos = pixelPos;
    desc.checkerboard = checkerboard;
    desc.pathNum = gSampleNum;
    desc.bounceNum = gBounceNum; // TODO: adjust by roughness?
    desc.instanceInclusionMask = GEOMETRY_IGNORE_TRANSPARENT;
    desc.rayFlags = 0;

    TraceOpaqueResult result = TraceOpaque( desc );

    // Debug
    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( frac( X ).x < 0.05 )
            result.diffRadiance = float3( 0, 10, 0 ) * STL::Color::Luminance( result.diffRadiance );
    #endif

    #if( USE_SIMULATED_FIREFLY_TEST == 1 )
        const float maxFireflyEnergyScaleFactor = 10000.0;
        result.diffRadiance /= lerp( 1.0 / maxFireflyEnergyScaleFactor, 1.0, STL::Rng::Hash::GetFloat( ) );
    #endif

    // Convert for NRD
    float4 outDiff = 0.0;
    float4 outSpec = 0.0;
    float4 outDiffSh = 0.0;
    float4 outSpecSh = 0.0;

    if( gDenoiserType == RELAX )
    {
    #if( NRD_MODE == SH )
        outDiff = RELAX_FrontEnd_PackSh( result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackSh( result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION );
    #else
        outDiff = RELAX_FrontEnd_PackRadianceAndHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackRadianceAndHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    #endif
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
    WriteResult( checkerboard, outPixelPos, outDiff, outSpec, outDiffSh, outSpecSh );
}
