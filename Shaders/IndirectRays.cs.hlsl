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
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedDiff_PrevViewZ, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedSpec_PrevViewZ, t, 2, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 3, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gInOut_DirectLighting, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gInOut_BaseColor_Metalness, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gInOut_Normal_Roughness, u, 2, 1 );
NRI_RESOURCE( RWTexture2D<float>, gInOut_ViewZ, u, 3, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gInOut_Mv, u, 4, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_TransparentLighting, u, 5, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 6, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 7, 1 );
#if( NRD_MODE == SH )
    NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffSh, u, 8, 1 );
    NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecSh, u, 9, 1 );
#endif

//========================================================================================
// MISC
//========================================================================================

// TODO: replace with currently denoised frame
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

    // Use only if radiance is on the screen
    weight *= float( gOnScreen < SHOW_AMBIENT_OCCLUSION );

    return weight;
}

float GetDiffuseLikeMotionAmount( GeometryProps geometryProps, MaterialProps materialProps )
{
    /*
    IMPORTANT / TODO: This is just a cheap estimation! Ideally curvature at hits should be applied in the reversed order:

        hitDist = hitT[ N ];
        for( uint i = N; i >= 1; i++ )
            hitDist = ApplyThinLensEquation( hitDist, curvature[ i - 1 ] ); // + lobe spread energy dissipation should be there too!

    Since NRD computes curvature at primary hit ( or PSR ), hit distances must respect the following rules:
    - For primary hits:
        - hitT for 1st bounce must be provided "as is" ( NRD applies thin lens equation )
        - hitT for 2nd+ bounces must be adjusted on the application side by taking into account:
            - energy dissipation due to lobe spread
            - curvature at bounces
    - For PSR:
        - hitT for bounces up to where PSR is found must be adjusted on the application side by taking into account curvature at each bounce ( it affects the virtual position location )
        - plus, same rules as for primary hits
    */

    float diffProb = EstimateDiffuseProbability( geometryProps, materialProps, true );
    diffProb = lerp( diffProb, 1.0, STL::Math::Sqrt01( materialProps.curvature ) );

    return diffProb;
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

    #if( NRD_MODE == DIRECTIONAL_OCCLUSION || NRD_MODE == SH )
        float3 diffDirection;
        float3 specDirection;
    #endif
};

TraceOpaqueResult TraceOpaque( TraceOpaqueDesc desc )
{
    // TODO: for RESOLUTION_FULL_PROBABILISTIC with 1 path per pixel the tracer can be significantly simplified:
    // - only one "float3" needed for radiance, hit distance and direction ( diff or spec )
    // - tracing to PSR can be embedded into the main loop
    // - temp vars "materialProps" and "geometryProps" not needed if "desc" modification is allowed ( or "desc" modification not needed )
    // - "startBRDF" and "startLsum" not needed

    //=====================================================================================================================================================================
    // Primary surface replacement ( PSR )
    //=====================================================================================================================================================================

    float viewZ = STL::Geometry::AffineTransform( gWorldToView, desc.geometryProps.X ).z;
    float3 startBRDF = 1.0;
    float3 startLsum = 0.0;

    #if( USE_PSR == 1 )
        float3x3 mirrorMatrix = STL::Geometry::GetMirrorMatrix( 0 ); // identity
        int psrBounceIndex = 0;

        bool canBePsr = gPSR && desc.materialProps.roughness < 0.005 && desc.materialProps.metalness > 0.995;
        if( canBePsr )
        {
            GeometryProps geometryProps = desc.geometryProps;
            MaterialProps materialProps = desc.materialProps;

            float accumulatedHitDistForTracking = 0.0;
            float accumulatedDiffuseLikeMotionForTracking = 0.0;

            [loop]
            for( uint bounceIndex = 1; bounceIndex <= desc.bounceNum; bounceIndex++ )
            {
                //=============================================================================================================================================================
                // Origin point
                //=============================================================================================================================================================

                {
                    // Accumulate mirror matrix
                    mirrorMatrix = mul( STL::Geometry::GetMirrorMatrix( materialProps.N ), mirrorMatrix );

                    // Choose a ray
                    float3 ray = reflect( -geometryProps.V, materialProps.N );

                    // Update throughput
                    #if( NRD_MODE != OCCLUSION && NRD_MODE != DIRECTIONAL_OCCLUSION )
                        float3 albedo, Rf0;
                        STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

                        float3 H = normalize( geometryProps.V + ray );
                        float VoH = abs( dot( geometryProps.V, H ) );
                        startBRDF *= STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
                    #endif

                    // Abort if expected contribution of the current bounce is low
                    if( STL::Color::Luminance( startBRDF ) < desc.threshold )
                        break;

                    // Estimate surface-based motion amount ( i.e. when specular moves like diffuse )
                    accumulatedDiffuseLikeMotionForTracking += GetDiffuseLikeMotionAmount( geometryProps, materialProps );

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
                    float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, materialProps.roughness );

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

                    // Reuse previous frame carefully treating specular ( if specular contribution is high it can't be reused because it was computed for a different view vector! )
                    float3 prevLdiff, prevLspec;
                    float prevFrameWeight = GetRadianceFromPreviousFrame( geometryProps, desc.pixelPos, prevLdiff, prevLspec );

                    float diffuseLikeMotion = GetDiffuseLikeMotionAmount( geometryProps, materialProps );
                    prevFrameWeight *= diffuseLikeMotion;
                    prevFrameWeight *= 0.9 * gUsePrevFrame; // TODO: use F( bounceIndex ) bounceIndex = 99 => 1.0
                    prevFrameWeight *= 1.0 - gAmbientAccumSpeed;

                    float diffuseProbabilityBiased = EstimateDiffuseProbability( geometryProps, materialProps, true );
                    float3 prevLsum = prevLdiff + prevLspec * diffuseProbabilityBiased;

                    float l1 = STL::Color::Luminance( L );
                    float l2 = STL::Color::Luminance( prevLsum );
                    prevFrameWeight *= l2 / ( l1 + l2 + 1e-6 );

                    L = lerp( L, prevLsum, prevFrameWeight );

                    // Accumulate lighting
                    L *= startBRDF;
                    startLsum += L;

                    // Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information ( biased )
                    startBRDF *= 1.0 - prevFrameWeight;

                    // Accumulate hit distance representing virtual point position
                    accumulatedHitDistForTracking += geometryProps.tmin * STL::Math::SmoothStep( 0.2, 0.0, accumulatedDiffuseLikeMotionForTracking );

                    // If hit is not a delta-event ( or the last bounce, or a miss ) it's PSR
                    if( !( materialProps.roughness < 0.005 && materialProps.metalness > 0.995 ) || bounceIndex == desc.bounceNum || geometryProps.IsSky( ) )
                    {
                        float3 Xvirtual = desc.geometryProps.X - desc.geometryProps.V * accumulatedHitDistForTracking;
                        viewZ = STL::Geometry::AffineTransform( gWorldToView, Xvirtual ).z;

                        float3 XvirtualPrev = Xvirtual;
                        if( !geometryProps.IsSky( ) )
                        {
                            float diffuseProbability = EstimateDiffuseProbability( geometryProps, materialProps );
                            uint materialID = diffuseProbability > BRDF_ENERGY_THRESHOLD ? 0 : 1;

                            InstanceData instanceData = gIn_InstanceData[ geometryProps.instanceIndex ];
                            XvirtualPrev = STL::Geometry::AffineTransform( instanceData.mWorldToWorldPrev, Xvirtual );

                            float3 psrNormal = STL::Geometry::RotateVectorInverse( mirrorMatrix, materialProps.N );

                            gInOut_BaseColor_Metalness[ desc.pixelPos ] = float4( STL::Color::LinearToSrgb( materialProps.baseColor ), materialProps.metalness );
                            gInOut_Normal_Roughness[ desc.pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( psrNormal, materialProps.roughness, materialID );
                        }
                        else
                            gInOut_DirectLighting[ desc.pixelPos ] = startLsum;

                        float3 motion = XvirtualPrev - Xvirtual;
                        if( !gIsWorldSpaceMotionEnabled )
                        {
                            float2 sampleUvPrev = STL::Geometry::GetScreenUv( gWorldToClipPrev, XvirtualPrev );
                            float2 sampleUv = STL::Geometry::GetScreenUv( gWorldToClip, Xvirtual );
                            motion.xy = ( sampleUvPrev - sampleUv ) * gRectSize;

                            float viewZprev = STL::Geometry::AffineTransform( gWorldToViewPrev, XvirtualPrev ).z;
                            motion.z = viewZprev - viewZ;
                        }

                        gInOut_ViewZ[ desc.pixelPos ] = geometryProps.IsSky( ) ? STL::Math::Sign( viewZ ) * INF : viewZ;
                        gInOut_Mv[ desc.pixelPos ] = motion;

                        desc.geometryProps = geometryProps;
                        desc.materialProps = materialProps;
                        desc.bounceNum -= bounceIndex;
                        psrBounceIndex = bounceIndex;

                        break;
                    }
                }
            }
        }
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

        float accumulatedHitDist = 0.0;
        float accumulatedDiffuseLikeMotion = 0.0;

        float3 Lsum = startLsum;
        float3 BRDF = startBRDF;

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
                float rnd = STL::Rng::GetFloat2( ).x;
                if( ( gTracingMode == RESOLUTION_FULL_PROBABILISTIC && bounceIndex == 1 ) )
                {
                    diffuseProbability = float( diffuseProbability != 0.0 ) * clamp( diffuseProbability, gMinProbability, 1.0 - gMinProbability );
                    rnd = STL::Sequence::Bayer4x4( desc.pixelPos, gFrameIndex ) + rnd / 16.0;
                }

                // Diffuse or specular path?
                if( gTracingMode == RESOLUTION_FULL_PROBABILISTIC || bounceIndex > 1 )
                {
                    isDiffuse = rnd < diffuseProbability;
                    BRDF /= abs( float( !isDiffuse ) - diffuseProbability );

                    if( bounceIndex == 1 )
                        isDiffusePath = isDiffuse;
                }

                float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );

                // Choose a ray
                float3x3 mLocalBasis = STL::Geometry::GetBasis( materialProps.N );
                float3 Vlocal = STL::Geometry::RotateVector( mLocalBasis, geometryProps.V );
                float3 ray = 0;
                float3 throughput = 0;
                float trimmingFactor = gTrimmingParams.x * STL::Math::SmoothStep( gTrimmingParams.y, gTrimmingParams.z, materialProps.roughness );
                float throughputWithImportanceSampling = 0;
                float VoH = 0;
                uint sampleNum = 0;

                while( sampleNum < IMPORTANCE_SAMPLE_NUM && throughputWithImportanceSampling == 0 )
                {
                    float2 rnd = STL::Rng::GetFloat2( );

                    if( isDiffuse )
                    {
                        // Generate a ray
                    #if( NRD_MODE == DIRECTIONAL_OCCLUSION )
                        ray = STL::ImportanceSampling::Uniform::GetRay( rnd );

                        float NoL = saturate( ray.z );
                        float pdf = STL::ImportanceSampling::Uniform::GetPDF( );
                    #else
                        ray = STL::ImportanceSampling::Cosine::GetRay( rnd );

                        float NoL = saturate( ray.z );
                        float pdf = STL::ImportanceSampling::Cosine::GetPDF( NoL );
                    #endif

                        // Throughput
                        throughput = NoL / ( STL::Math::Pi( 1.0 ) * pdf );

                        float3 Hlocal = normalize( Vlocal + ray );
                        VoH = abs( dot( Vlocal, Hlocal ) );
                    }
                    else
                    {
                        // Generate a ray
                        float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay( rnd, materialProps.roughness, Vlocal, trimmingFactor );
                        ray = reflect( -Vlocal, Hlocal );

                        // Throughput ( see paragraph "Usage in Monte Carlo renderer" from http://jcgt.org/published/0007/04/01/paper.pdf )
                        float NoL = saturate( ray.z );
                        throughput = STL::BRDF::GeometryTerm_Smith( materialProps.roughness, NoL );

                        VoH = abs( dot( Vlocal, Hlocal ) );
                    }

                    ray = STL::Geometry::RotateVectorInverse( mLocalBasis, ray );

                    // Allow low roughness specular to take data from the previous frame
                    throughputWithImportanceSampling = STL::Color::Luminance( throughput );
                    bool isImportanceSamplingNeeded = throughputWithImportanceSampling != 0 && ( isDiffuse || ( STL::Rng::GetFloat2( ).x < materialProps.roughness ) );

                    if( gDisableShadowsAndEnableImportanceSampling && isImportanceSamplingNeeded )
                    {
                        bool isMiss = CastVisibilityRay_AnyHit( geometryProps.GetXoffset( ), ray, 0.0, INF, mipAndCone, gLightTlas, GEOMETRY_ONLY_EMISSIVE, desc.rayFlags );
                        throughputWithImportanceSampling *= float( !isMiss );
                    }

                    sampleNum++;
                }

                // ( Optional ) Save sampling direction for the 1st bounce
                #if( NRD_MODE == DIRECTIONAL_OCCLUSION || NRD_MODE == SH )
                    if( bounceIndex == 1 )
                    {
                        float3 psrRay = ray;
                        #if( USE_PSR == 1 )
                            ray = STL::Geometry::RotateVectorInverse( mirrorMatrix, ray );
                        #endif

                        if( isDiffuse )
                            result.diffDirection += psrRay;
                        else
                            result.specDirection += psrRay;
                    }
                #endif

                // Update throughput
                #if( NRD_MODE != OCCLUSION && NRD_MODE != DIRECTIONAL_OCCLUSION )
                    float3 albedo, Rf0;
                    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

                    float3 F = STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
                    if( isDiffuse )
                        throughput *= ( 1.0 - F ) * albedo;
                    else
                        throughput *= F;
                #endif

                // Update BRDF
                BRDF *= throughput / float( sampleNum );

                // Abort if expected contribution of the current bounce is low
                if( STL::Color::Luminance( BRDF ) < desc.threshold )
                    break;

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
                float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );

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

                // Reuse previous frame carefully treating specular ( if specular contribution is high it can't be reused because it was computed for a different view vector! )
                float3 prevLdiff, prevLspec;
                float prevFrameWeight = GetRadianceFromPreviousFrame( geometryProps, desc.pixelPos, prevLdiff, prevLspec );

                float diffuseLikeMotion = isDiffuse ? 1.0 : GetDiffuseLikeMotionAmount( geometryProps, materialProps );
                prevFrameWeight *= diffuseLikeMotion;
                prevFrameWeight *= 0.9 * gUsePrevFrame; // TODO: use F( bounceIndex ) bounceIndex = 99 => 1.0
                prevFrameWeight *= 1.0 - gAmbientAccumSpeed;

                float diffuseProbabilityBiased = EstimateDiffuseProbability( geometryProps, materialProps, true );
                float3 prevLsum = prevLdiff + prevLspec * diffuseProbabilityBiased;

                float l1 = STL::Color::Luminance( L );
                float l2 = STL::Color::Luminance( prevLsum );
                prevFrameWeight *= l2 / ( l1 + l2 + 1e-6 );

                L = lerp( L, prevLsum, prevFrameWeight );

                // Accumulate lighting
                L *= BRDF;
                Lsum += L;

                // Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information ( biased )
                BRDF *= 1.0 - prevFrameWeight;

                // Accumulate path length for NRD
                /*
                IMPORTANT:
                    - energy dissipation must be respected, i.e.
                        - diffuse - only 1st bounce hitT is needed
                        - specular - only 1st bounce hitT is needed for glossy+ specular
                        - sum of all segments can be ( but not necessary ) needed for 0 roughness only
                    - for 0 roughness hitT must be clean like a tear! NRD specular tracking relies on it
                        - in case of probabilistic sampling there can be pixels with "no data" ( hitT = 0 ), but hitT reconstruction mode must be enabled in NRD
                    - see "GetDiffuseLikeMotionAmount":
                        - it's just a noise-free estimation
                        - curvature is important: it makes reflections closer to the surface making their motion diffuse-like
                */
                float a = STL::Color::Luminance( L ) + 1e-6;
                float b = STL::Color::Luminance( Lsum ) + 1e-6; // already includes L
                float importance = a / b;

                accumulatedHitDist += geometryProps.tmin * STL::Math::SmoothStep( 0.2, 0.0, accumulatedDiffuseLikeMotion );
                accumulatedDiffuseLikeMotion += 1.0 - importance * ( 1.0 - diffuseLikeMotion );
            }
        }

        // Ambient estimation at the end of the path
        BRDF *= GetAmbientBRDF( geometryProps, materialProps );
        BRDF *= 1.0 + EstimateDiffuseProbability( geometryProps, materialProps, true ); // TODO: hack? divide by PDF of the last ray?

        float occlusion = REBLUR_FrontEnd_GetNormHitDist( geometryProps.tmin, 0.0, gHitDistParams, 1.0 );
        occlusion = lerp( 1.0 / STL::Math::Pi( 1.0 ), 1.0, occlusion );
        occlusion *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( geometryProps.X - gCameraOrigin ) );
        occlusion *= isDiffusePath ? 0.0 : ( 1.0 - GetSpecMagicCurve( desc.materialProps.roughness ) ); // balanced with ambient applied in Composition

        Lsum += BRDF * desc.Lamb * occlusion;

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

    // Material de-modulation ( convert irradiance to radiance )
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( desc.materialProps.baseColor, desc.materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( desc.materialProps.N, desc.geometryProps.V ) );
    float3 Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV, desc.materialProps.roughness );
    float3 diffDemod = ( 1.0 - Fenv ) * albedo * 0.99 + 0.01;
    float3 specDemod = Fenv * 0.99 + 0.01;

    if( gOnScreen != SHOW_MIP_SPECULAR )
    {
        result.diffRadiance /= gReference ? 1.0 : diffDemod;
        result.specRadiance /= gReference ? 1.0 : specDemod;
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
            float NoV = abs( dot( geometryProps.N, geometryProps.V ) );
            float F = STL::BRDF::FresnelTerm_Dielectric( eta, NoV );

            if( bounce == 0 )
                transmittance *= isReflection ? F : 1.0 - F;
            else
                isReflection = STL::Rng::GetFloat2().x < F;

            float3 origin = _GetXoffset( geometryProps.X, isReflection ? geometryProps.N : -geometryProps.N );
            float3 ray = reflect( -geometryProps.V, geometryProps.N );

            // Emulate thickness to avoid getting wrong refracted directions
            if( !isReflection )
            {
                // Glass is single sided in our scenes. If glass is not single sided, then better assume that primary ray is "air-glass",
                // the next bounce becomes "glass-air", the next switches to "air-glass" again, etc. ( if glass surface is hit, of course )
                ray = refract( -geometryProps.V, geometryProps.N, eta );

                float cosa = abs( dot( ray, geometryProps.N ) );
                float d = GLASS_THICKNESS / max( cosa, 0.05 );
                origin += ray * d; // TODO: since there is no RT, new origin can go under surface if a thin glass stands on it. It can lead to wrong shadows.

                ray = refract( ray, geometryProps.N, 1.0 / eta );

                transmittance *= GLASS_TINT;
            }

            uint flags = ( bounce == desc.bounceNum - 1 && !isReflection ) ? GEOMETRY_IGNORE_TRANSPARENT : GEOMETRY_ALL;
            geometryProps = CastRay( origin, ray, 0.0, INF, GetConeAngleFromRoughness( geometryProps.mip, 0.0 ), gWorldTlas, flags, 0 );

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
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
        return;

    uint2 outPixelPos = pixelPos;
    if( gTracingMode == RESOLUTION_HALF )
        outPixelPos.x >>= 1;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Initialize RNG
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    // Ambient level
    float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    Lamb *= gAmbient;

    // Transparent lighting // TODO: move after "Composition" to be able to use current final frame
    float viewZ = gInOut_ViewZ[ pixelPos ];
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
    desc.Lamb = Lamb;
    desc.pixelPos = pixelPos;
    desc.threshold = BRDF_ENERGY_THRESHOLD;
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

    // Convert for NRD
    float4 outDiff = 0.0;
    float4 outSpec = 0.0;
    float4 outDiffSh = 0.0;
    float4 outSpecSh = 0.0;

    if( gDenoiserType == RELAX )
    {
    #if( NRD_MODE == SH )
        outDiff = RELAX_FrontEnd_PackSh(result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION);
        outSpec = RELAX_FrontEnd_PackSh(result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION);
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
