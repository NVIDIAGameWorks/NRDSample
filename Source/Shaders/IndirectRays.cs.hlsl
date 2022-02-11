/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if( !defined( COMPILER_FXC ) && !defined( VULKAN ) )

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
"TracePath" continues tracing from a given bounce, to start from the camera do the following:
    tracePathPayload.materialProps = ( MaterialProps )0;
    tracePathPayload.geometryProps = ( GeometryProps )0;
    tracePathPayload.geometryProps.X = LinePlaneIntersection( cameraPos, cameraView, nearPlane );

Prerequisites:
    STL::Rng::Initialize( )

Derivation:
    Lsum = L0 + BRDF0 * ( L1 + BRDF1 * ( L2 + BRDF2 * ( L3 +  ... ) ) )

    Lsum = L0 +
        L1 * BRDF0 +
        L2 * BRDF0 * BRDF1 +
        L3 * BRDF0 * BRDF1 * BRDF2 +
        ...

    for each bounce
    {
        Lsum += L[i] * BRDF
        pathLength += F( tmin[i], ... )
        BRDF *= BRDF[i]
    }
*/

struct TracePathDesc
{
    // Non-jittered pixel UV
    float2 pixelUv;

    // BRDF energy threshold
    float threshold;

    // Bounces to trace
    uint bounceNum;

    // Instance inclusion mask ( DXR )
    uint instanceInclusionMask;

    // Ray flags ( DXR )
    uint rayFlags;

    // A hint to use simplified materials ( flat colors, flat normals, etc. )
    bool useSimplifiedModel;

    // Some global ambient to be applied at the end of the path
    float3 Lamb;
};

struct TracePathPayload
{
    // Geometry properties
    GeometryProps geometryProps;

    // Material properties
    MaterialProps materialProps;

    // Left by bounce preceding input bounce ( 1 if starting from primary hits or from the camera )
    float3 BRDF;

    // Left by input bounce or 0
    float3 Lsum;

    // Accumulated previous frame weight
    float accumulatedPrevFrameWeight;

    // Left by input bounce or 0
    float pathLength;

    // Input bounce index ( 0 if tracing starts from the camera )
    uint bounceIndex;

    // Diffuse or specular path ( at this event, next event will be stochastically estimated )
    bool isDiffuse;
};

float3 GetRadianceFromPreviousFrame( GeometryProps geometryProps, MaterialProps materialProps, float2 pixelUv, inout float weight )
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

    // Confidence - ignore mirror specular
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );
    float diffusiness = materialProps.roughness;
    float lumDiff = lerp( STL::Color::Luminance( albedo ), 1.0, diffusiness );
    float lumSpec = lerp( STL::Color::Luminance( F ) * 10.0, 0.0, diffusiness ); // boost specular which can be hot // TODO: 10?
    float diffProb = lumDiff / ( lumDiff + lumSpec + 1e-6 );
    weight *= diffProb;

    // Ignore sky
    weight *= float( !geometryProps.IsSky() );

    return weight ? prevLsum.xyz : 0;
}

float GetBasePrevFrameWeight( )
{
    // Avoid "stuck in history" effect
    float weight = 0.9;
    weight *= 1.0 - gAmbientAccumSpeed;

    // Don't use in reference mode
    weight *= 1.0 - gReference;

    return weight;
}

bool IsNextEventDiffuse( GeometryProps geometryProps, MaterialProps materialProps )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );
    float lumDiff = STL::Color::Luminance( albedo ) + 1e-6;
    float lumSpec = STL::Color::Luminance( F ) + 1e-6;
    float diffProbability = lumDiff / ( lumDiff + lumSpec );
    float rnd = STL::Rng::GetFloat2( ).x;

    return rnd < diffProbability;
}

float4 TracePath( TracePathDesc desc, inout TracePathPayload payload, float primaryHitRoughness )
{
    float2 mipAndCone = GetConeAngleFromRoughness( payload.geometryProps.mip, payload.materialProps.roughness );
    float4 directionPdf = 0;

    [loop]
    for( uint i = 0; i < desc.bounceNum && !payload.geometryProps.IsSky(); i++ )
    {
        // Choose ray
        float3 rayDirection = 0;
        if( payload.bounceIndex != 0 )
        {
            // Not primary ray
            float3x3 mLocalBasis = STL::Geometry::GetBasis( payload.materialProps.N );
            float3 Vlocal = STL::Geometry::RotateVector( mLocalBasis, -payload.geometryProps.rayDirection );
            float trimmingFactor = NRD_GetTrimmingFactor( payload.materialProps.roughness, gTrimmingParams );

            float VoH = 0;
            float throughput = 0;
            float throughputWithImportanceSampling = 0;
            float pdf = 0;
            float sampleNum = 0;

            while( sampleNum < IMPORTANCE_SAMPLE_NUM && throughputWithImportanceSampling == 0 )
            {
                float2 rnd = STL::Rng::GetFloat2( );

                if( payload.isDiffuse )
                {
                    float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay( rnd );
                    rayDirection = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

                    throughput = 1.0; // = [ albedo / PI ] / STL::ImportanceSampling::Cosine::GetPDF( NoL );

                    float NoL = saturate( dot( payload.materialProps.N, rayDirection ) );
                    pdf = STL::ImportanceSampling::Cosine::GetPDF( NoL );
                }
                else
                {
                    float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay( rnd, payload.materialProps.roughness, Vlocal, trimmingFactor );
                    float3 H = STL::Geometry::RotateVectorInverse( mLocalBasis, Hlocal );
                    rayDirection = reflect( payload.geometryProps.rayDirection, H );

                    VoH = abs( dot( -payload.geometryProps.rayDirection, H ) );

                    // It's a part of VNDF sampling - see http://jcgt.org/published/0007/04/01/paper.pdf ( paragraph "Usage in Monte Carlo renderer" )
                    float NoL = saturate( dot( payload.materialProps.N, rayDirection ) );
                    throughput = STL::BRDF::GeometryTerm_Smith( payload.materialProps.roughness, NoL );

                    float NoV = abs( dot( payload.materialProps.N, -payload.geometryProps.rayDirection ) );
                    float NoH = saturate( dot( payload.materialProps.N, H ) );
                    pdf = STL::ImportanceSampling::VNDF::GetPDF( NoV, NoH, payload.materialProps.roughness );
                }

                throughputWithImportanceSampling = throughput;
                if( gDisableShadowsAndEnableImportanceSampling )
                {
                    bool isMiss = CastVisibilityRay_AnyHit( payload.geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gLightTlas, throughput != 0.0 ? GEOMETRY_ONLY_EMISSIVE : 0, desc.rayFlags );
                    throughputWithImportanceSampling *= float( !isMiss );
                }

                sampleNum += 1.0;
            }

            throughput /= sampleNum;
            directionPdf = payload.bounceIndex == 1 ? float4( rayDirection, pdf ) : directionPdf;

            // Update BRDF
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( payload.materialProps.baseColor, payload.materialProps.metalness, albedo, Rf0 );

            float3 F = STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
            payload.BRDF *= payload.isDiffuse ? albedo : F;
            payload.BRDF *= throughput;

            // Abort if expected contribution of the current bounce is low
            if( STL::Color::Luminance( payload.BRDF ) < desc.threshold )
                break;
        }
        else
        {
            // Primary ray
            rayDirection = -GetViewVector( payload.geometryProps.X );
        }

        // Cast ray and update payload ( i.e. jump to next point )
        payload.geometryProps = CastRay( payload.geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags, desc.useSimplifiedModel );
        payload.materialProps = GetMaterialProps( payload.geometryProps, desc.useSimplifiedModel );
        mipAndCone = GetConeAngleFromRoughness( payload.geometryProps.mip, payload.isDiffuse ? 1.0 : payload.materialProps.roughness );

        // Compute lighting
        float3 L = payload.materialProps.Ldirect;
        if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
            L *= CastVisibilityRay_AnyHit( payload.geometryProps.GetXoffset( ), gSunDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
        L += payload.materialProps.Lemi;

        // Reuse previous frame if possible
        float3 prevLsum = GetRadianceFromPreviousFrame( payload.geometryProps, payload.materialProps, desc.pixelUv, payload.accumulatedPrevFrameWeight );
        L = lerp( L, prevLsum, payload.accumulatedPrevFrameWeight );

        // Accumulate lighting
        L *= payload.BRDF;
        payload.Lsum += L;

        // Reduce contribution of next samples
        payload.BRDF *= 1.0 - payload.accumulatedPrevFrameWeight;

        // Accumulate path length
        float a = STL::Color::Luminance( L ) + 1e-6;
        float b = STL::Color::Luminance( payload.Lsum ) + 1e-6;
        float importance = a / b;
        payload.pathLength += NRD_GetCorrectedHitDist( payload.geometryProps.tmin, payload.bounceIndex, primaryHitRoughness, importance );

        // Estimate next event and go to next bounce
        payload.isDiffuse = IsNextEventDiffuse( payload.geometryProps, payload.materialProps );
        payload.bounceIndex++;
    }

    // Ambient estimation at the end of the path
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( payload.materialProps.baseColor, payload.materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( payload.materialProps.N, -payload.geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, payload.materialProps.roughness );

    float scale = lerp( 1.0, 1.5, payload.materialProps.metalness );
    float3 BRDF = albedo * ( 1 - F ) + F / scale;
    BRDF *= float( !payload.geometryProps.IsSky() );

    float occlusion = REBLUR_FrontEnd_GetNormHitDist( payload.geometryProps.tmin, 0.0, gDiffHitDistParams, 1.0 );
    occlusion = lerp( 1.0 / STL::Math::Pi( 1.0 ), 1.0, occlusion );
    occlusion *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( payload.geometryProps.X - gCameraOrigin ) );

    payload.Lsum += desc.Lamb * payload.BRDF * BRDF * occlusion;

    return directionPdf;
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
        pixelPos = ( pixelPos << 1 ) + uint2( gFrameIndex & 0x1, ( gFrameIndex >> 1 ) & 0x1 );

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

    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gIsOrtho );
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 V = GetViewVector( X );
    float3 N = normalAndRoughness.xyz;
    float mip0 = gIn_PrimaryMip[ pixelPos ];

    float3 Xoffset = _GetXoffset( X, N );
    Xoffset += V * ( 0.0003 + abs( viewZ ) * 0.00005 );

    float NoV0 = abs( dot( N, V ) );
    Xoffset += N * STL::BRDF::Pow5( NoV0 ) * ( 0.003 + abs( viewZ ) * 0.0005 );

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

    // Material de-modulation
    float3 albedo0, Rf00;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo0, Rf00 );

    albedo0 = max( albedo0, 0.001 );

    float3 envBRDF0 = STL::BRDF::EnvironmentTerm_Ross( Rf00, NoV0, materialProps0.roughness );
    envBRDF0 = max( envBRDF0, 0.001 );

    // Ambient
    float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    Lamb *= gAmbient;

    // Secondary rays
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    float4 diffIndirect = 0;
    float4 diffDirectionPdf = 0;
    float diffTotalWeight = 1e-6;

    float4 specIndirect = 0;
    float4 specDirectionPdf = 0;
    float specTotalWeight = 1e-6;

    uint sampleNum = gSampleNum << ( gTracingMode == RESOLUTION_HALF ? 0 : 1 );
    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;

    TracePathDesc tracePathDesc = ( TracePathDesc )0;
    tracePathDesc.pixelUv = pixelUv;
    tracePathDesc.bounceNum = gBounceNum; // TODO: adjust by roughness
    tracePathDesc.instanceInclusionMask = GEOMETRY_IGNORE_TRANSPARENT;
    tracePathDesc.rayFlags = 0;
    tracePathDesc.threshold = BRDF_ENERGY_THRESHOLD;

    for( uint i = 0; i < sampleNum; i++ )
    {
        bool isDiffuse = gTracingMode == RESOLUTION_HALF ? checkerboard : ( i < gSampleNum );

        // Trace
        tracePathDesc.useSimplifiedModel = isDiffuse; // TODO: adjust by roughness
        tracePathDesc.Lamb = Lamb * float( !isDiffuse );

        TracePathPayload tracePathPayload = ( TracePathPayload )0;
        tracePathPayload.BRDF = 1.0;
        tracePathPayload.Lsum = 0.0;
        tracePathPayload.accumulatedPrevFrameWeight = GetBasePrevFrameWeight( );
        tracePathPayload.pathLength = 0.0; // exclude primary ray length
        tracePathPayload.bounceIndex = 1; // starting from primary ray hit
        tracePathPayload.isDiffuse = isDiffuse;
        tracePathPayload.geometryProps = geometryProps0;
        tracePathPayload.materialProps = materialProps0;

        float4 directionPdf = TracePath( tracePathDesc, tracePathPayload, isDiffuse ? 1.0 : materialProps0.roughness );

        // De-modulate materials for denoising
        tracePathPayload.Lsum /= isDiffuse ? albedo0 : envBRDF0;

        // Convert for NRD
        directionPdf = NRD_FrontEnd_PackDirectionAndPdf( directionPdf.xyz, directionPdf.w );

        float normDist = REBLUR_FrontEnd_GetNormHitDist( tracePathPayload.pathLength, viewZ, isDiffuse ? gDiffHitDistParams : gSpecHitDistParams, isDiffuse ? 1.0 : materialProps0.roughness );
        float4 nrdData = REBLUR_FrontEnd_PackRadianceAndHitDist( tracePathPayload.Lsum, normDist, USE_SANITIZATION );
        if( gDenoiserType != REBLUR )
            nrdData = RELAX_FrontEnd_PackRadianceAndHitDist( tracePathPayload.Lsum, tracePathPayload.pathLength, USE_SANITIZATION );

        // Debug
        if( gOnScreen == SHOW_MIP_SPECULAR )
        {
            float mipNorm = STL::Math::Sqrt01( tracePathPayload.geometryProps.mip / MAX_MIP_LEVEL );
            nrdData.xyz = STL::Color::ColorizeZucconi( mipNorm );
        }

        // Accumulate
        float sampleWeight = NRD_GetSampleWeight( tracePathPayload.Lsum, USE_SANITIZATION );
        nrdData *= sampleWeight;
        directionPdf *= sampleWeight;

        diffIndirect += nrdData * float( isDiffuse );
        diffDirectionPdf += directionPdf * float( isDiffuse );
        diffTotalWeight += sampleWeight * float( isDiffuse );

        specIndirect += nrdData * float( !isDiffuse );
        specDirectionPdf += directionPdf * float( !isDiffuse );
        specTotalWeight += sampleWeight * float( !isDiffuse );
    }

    diffIndirect /= diffTotalWeight;
    diffDirectionPdf /= diffTotalWeight;

    specIndirect /= specTotalWeight;
    specDirectionPdf /= specTotalWeight;

    // Output
    [flatten]
    if( gOcclusionOnly )
    {
        diffIndirect = diffIndirect.wwww;
        specIndirect = specIndirect.wwww;
    }

    if( gTracingMode == RESOLUTION_HALF )
    {
        pixelPos.x >>= 1;

        if( checkerboard )
        {
            gOut_Diff[ pixelPos ] = diffIndirect;
            gOut_DiffDirectionPdf[ pixelPos ] = diffDirectionPdf;
        }
        else
        {
            gOut_Spec[ pixelPos ] = specIndirect;
            gOut_SpecDirectionPdf[ pixelPos ] = specDirectionPdf;
        }
    }
    else
    {
        gOut_Diff[ outPixelPos ] = diffIndirect;
        gOut_DiffDirectionPdf[ outPixelPos ] = diffDirectionPdf;

        gOut_Spec[ outPixelPos ] = specIndirect;
        gOut_SpecDirectionPdf[ outPixelPos ] = specDirectionPdf;
    }
}

#else

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // no TraceRayInline support, because of:
    //  - DXBC
    //  - SPIRV generation is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/4221
}

#endif
