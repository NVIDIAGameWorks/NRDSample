
NRI_RESOURCE( RaytracingAccelerationStructure, gWorldTlas, t, 0, 2 );
NRI_RESOURCE( RaytracingAccelerationStructure, gLightTlas, t, 1, 2 );
NRI_RESOURCE( StructuredBuffer<InstanceData>, gIn_InstanceData, t, 2, 2 );
NRI_RESOURCE( StructuredBuffer<PrimitiveData>, gIn_PrimitiveData, t, 3, 2 );
NRI_RESOURCE( StructuredBuffer<MorphedPrimitivePrevData>, gIn_MorphedPrimitivePrevPositions, t, 4, 2 );
NRI_RESOURCE( Texture2D<float4>, gIn_Textures[], t, 5, 2 );

#define TEX_SAMPLER gLinearMipmapLinearSampler

#include "HairBRDF.hlsli"

//====================================================================================================================================
// GEOMETRY & MATERIAL PROPERTIES
//====================================================================================================================================

float3 _GetXoffset( float3 X, float3 N )
{
    // RT Gems "A Fast and Robust Method for Avoiding Self-Intersection" ( updated version taken from Falcor )
    // Moves the ray origin further from surface to prevent self-intersections, minimizes the distance.
    // TODO: try out: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/

    const float origin = 1.0 / 16.0;
    const float fScale = 3.0 / 65536.0;
    const float iScale = 3.0 * 256.0;

    // Per-component integer offset to bit representation of FP32 position
    int3 iOff = int3( N * iScale );
    iOff.x = X.x < 0.0 ? -iOff.x : iOff.x;
    iOff.y = X.y < 0.0 ? -iOff.y : iOff.y;
    iOff.z = X.z < 0.0 ? -iOff.z : iOff.z;

    // Select per-component between small fixed offset or variable offset depending on distance to origin
    float3 Xi = asfloat( asint( X ) + iOff );
    float3 Xoff = X + N * fScale;

    X.x = abs( X.x ) < origin ? Xoff.x : Xi.x;
    X.y = abs( X.y ) < origin ? Xoff.y : Xi.y;
    X.z = abs( X.z ) < origin ? Xoff.z : Xi.z;

    return X;
}

struct GeometryProps
{
    float3 X;
    float3 Xprev;
    float3 V;
    float4 T;
    float3 N;
    float2 uv;
    float mip;
    float tmin;
    float curvature;
    uint textureOffsetAndFlags;
    uint instanceIndex;

    float3 GetXoffset( )
    { return _GetXoffset( X, N ); }

    bool IsStatic( )
    { return ( textureOffsetAndFlags & ( FLAG_STATIC << FLAG_FIRST_BIT ) ) != 0; }

    bool IsDeformable( )
    { return ( textureOffsetAndFlags & ( FLAG_DEFORMABLE << FLAG_FIRST_BIT ) ) != 0; }

    bool IsTransparent( )
    { return ( textureOffsetAndFlags & ( FLAG_TRANSPARENT << FLAG_FIRST_BIT ) ) != 0; }

    bool IsForcedEmission( )
    { return ( textureOffsetAndFlags & ( FLAG_FORCED_EMISSION << FLAG_FIRST_BIT ) ) != 0; }

    bool IsHair( )
    { return ( textureOffsetAndFlags & ( FLAG_HAIR << FLAG_FIRST_BIT ) ) != 0; }

    uint GetBaseTexture( )
    { return textureOffsetAndFlags & NON_FLAG_MASK; }

    float3 GetForcedEmissionColor( )
    { return ( ( textureOffsetAndFlags >> 2 ) & 0x1 ) ? float3( 1.0, 0.0, 0.0 ) : float3( 0.0, 1.0, 0.0 ); }

    bool IsSky( )
    { return tmin == INF; }
};

struct MaterialProps
{
    float3 Ldirect; // unshadowed
    float3 Lemi;
    float3 N;
    float3 T;
    float3 baseColor;
    float roughness;
    float metalness;
    float curvature;
};

float2 GetConeAngleFromAngularRadius( float mip, float tanConeAngle )
{
    // In any case, we are limited by the output resolution
    tanConeAngle = max( tanConeAngle, gTanPixelAngularRadius );

    return float2( mip, tanConeAngle );
}

float2 GetConeAngleFromRoughness( float mip, float roughness )
{
    float coneAngle = tan( STL::ImportanceSampling::GetSpecularLobeHalfAngle( roughness ) ); // TODO:  * 0.33333?

    return GetConeAngleFromAngularRadius( mip, coneAngle );
}

/*
Returns:
    .x - for visibility (emission, shadow)
        We must avoid using lower mips because it can lead to significant increase in AHS invocations. Mips lower than 128x128 are skipped!
    .y - for sampling (normals...)
        Negative MIP bias is applied
    .z - for sharp sampling
        Negative MIP bias is applied (can be more negative...)
*/
float3 GetRealMip( uint textureIndex, float mip )
{
    float w, h;
    gIn_Textures[ textureIndex ].GetDimensions( w, h ); // TODO: if I only had it as a constant...

    // Taking into account real dimensions of the current texture
    float mipNum = log2( w );
    float realMip = mip + mipNum - MAX_MIP_LEVEL;

    float3 mips;
    mips.x = min( realMip, mipNum - 7.0 );
    mips.y = realMip + gCameraOrigin_gMipBias.w * 0.5;
    mips.z = realMip + gCameraOrigin_gMipBias.w;

    return max( mips, 0.0 );
}

MaterialProps GetMaterialProps( GeometryProps geometryProps )
{
    MaterialProps props = ( MaterialProps )0;

    float3 Csky = GetSkyIntensity( -geometryProps.V, gSunDirection_gExposure.xyz, gTanSunAngularRadius );

    [branch]
    if( geometryProps.IsSky( ) )
    {
        props.Lemi = Csky;

        return props;
    }

    uint baseTexture = geometryProps.GetBaseTexture( );
    float3 mips = GetRealMip( baseTexture, geometryProps.mip );

    InstanceData instanceData = gIn_InstanceData[ geometryProps.instanceIndex ];

    // Base color
    float4 color = gIn_Textures[ NonUniformResourceIndex( baseTexture ) ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.z );
    color.xyz *= instanceData.baseColorAndMetalnessScale.xyz;
    color.xyz *= geometryProps.IsTransparent( ) ? 1.0 : STL::Math::PositiveRcp( color.w ); // Correct handling of BC1 with pre-multiplied alpha
    float3 baseColor = saturate( color.xyz );

    // Roughness and metalness
    float3 materialProps = gIn_Textures[ NonUniformResourceIndex( baseTexture + 1 ) ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.z ).xyz;
    float roughness = saturate( materialProps.y * instanceData.emissionAndRoughnessScale.w );
    float metalness = saturate( materialProps.z * instanceData.baseColorAndMetalnessScale.w );

    // Normal
    float2 packedNormal = gIn_Textures[ NonUniformResourceIndex( baseTexture + 2 ) ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.y ).xy;
    float3 N = gUseNormalMap ? STL::Geometry::TransformLocalNormal( packedNormal, geometryProps.T, geometryProps.N ) : geometryProps.N;
    float3 T = geometryProps.T.xyz;

    // Estimate curvature
    float curvature = length( STL::Geometry::UnpackLocalNormal( packedNormal ).xy ) * float( gUseNormalMap );

    // Emission
    float3 Lemi = gIn_Textures[ NonUniformResourceIndex( baseTexture + 3 ) ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.x ).xyz;
    Lemi *= instanceData.emissionAndRoughnessScale.xyz;
    Lemi *= ( baseColor + 0.01 ) / ( max( baseColor, max( baseColor, baseColor ) ) + 0.01 );

    [flatten]
    if( geometryProps.IsForcedEmission( ) )
    {
        Lemi = geometryProps.GetForcedEmissionColor( );
        baseColor = 0.0;
    }

    Lemi *= gEmissionIntensity;

    // Material overrides
    [flatten]
    if( gForcedMaterial == MATERIAL_GYPSUM )
    {
        roughness = 1.0;
        baseColor = 0.5;
        metalness = 0.0;
    }
    else if( gForcedMaterial == MATERIAL_COBALT )
    {
        roughness = pow( saturate( baseColor.x * baseColor.y * baseColor.z ), 0.33333 );
        baseColor = float3( 0.672411, 0.637331, 0.585456 );
        metalness = 1.0;

        #if( USE_ANOTHER_COBALT == 1 )
            roughness = pow( saturate( roughness - 0.1 ), 0.25 ) * 0.3 + 0.07;
        #endif
    }

    if( geometryProps.IsHair( ) )
    {
        roughness = gHairBetasOverride.x;
        metalness = gHairBetasOverride.y;
        baseColor = gHairBaseColorOverride.xyz;
    }
    else
    {
        metalness = gMetalnessOverride == 0.0 ? metalness : gMetalnessOverride;
        roughness = gRoughnessOverride == 0.0 ? roughness : gRoughnessOverride;
    }

    #if( USE_PUDDLES == 1 )
        roughness *= STL::Math::SmoothStep( 0.6, 0.8, length( frac( geometryProps.uv ) * 2.0 - 1.0 ) );
    #endif

    // Transform to diffuse material if emission is here
    float emissionLevel = STL::Color::Luminance( Lemi );
    emissionLevel = saturate( emissionLevel * 50.0 );

    metalness = lerp( metalness, 0.0, emissionLevel );
    roughness = lerp( roughness, 1.0, emissionLevel );

    // Direct lighting ( no shadow )
    float3 Ldirect = 0;
    float NoL = saturate( dot( geometryProps.N, gSunDirection_gExposure.xyz ) );
    float shadow = STL::Math::SmoothStep( 0.03, 0.1, NoL );

    [branch]
    if( shadow != 0.0 )
    {
        float3 Csun = GetSunIntensity( gSunDirection_gExposure.xyz, gSunDirection_gExposure.xyz, gTanSunAngularRadius );

        // Pseudo sky importance sampling
        float3 Cimp = lerp( Csky, Csun, STL::Math::SmoothStep( 0.0, 0.2, roughness ) );
        Cimp *= STL::Math::SmoothStep( -0.01, 0.05, gSunDirection_gExposure.z );

        if( geometryProps.IsHair( ) )
        {
            float3x3 mLocalBasis = HairGetBasis( N, T );
            float3 vLocal = STL::Geometry::RotateVector( mLocalBasis, geometryProps.V );

            HairSurfaceData hairSd = (HairSurfaceData)0;
            hairSd.N = float3( 0, 0, 1 );
            hairSd.T = float3( 1, 0, 0 );
            hairSd.V = vLocal;

            HairData hairData;
            hairData.baseColor = baseColor;
            hairData.betaM = roughness;
            hairData.betaN = metalness;

            HairContext hairBrdf = HairContextInit( hairSd, hairData );

            float3 sunLocal = STL::Geometry::RotateVector( mLocalBasis, gSunDirection_gExposure.xyz );

            // There isn't an easy separation for hair model, we could factor out
            // Ap[0] as specular, then Ap[1...k] as diffuse (TT, TRT paths)
            // Try using the pseudo sky color bit here.

            float3 throughput = HairEval( hairBrdf, vLocal, sunLocal );

            // sun only
            Ldirect = Cimp * throughput;
        }
        else
        {
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColor.xyz, metalness, albedo, Rf0 );

        #if( USE_SIMPLEX_LIGHTING_MODEL == 1 )
            // Very simple "diffuse-like" model
            float m = roughness * roughness;
            float3 C = albedo * Csun + Rf0 * m * Cimp;
            float NoL = dot( geometryProps.N, gSunDirection_gExposure.xyz );
            float Kdiff = NoL / STL::Math::Pi( 1.0 );

            Ldirect = Kdiff * C;
        #else
            float3 Cdiff, Cspec;
            STL::BRDF::DirectLighting( N, gSunDirection_gExposure.xyz, geometryProps.V, Rf0, roughness, Cdiff, Cspec );

            Ldirect = Cdiff * albedo * Csun + Cspec * Cimp;
        #endif
        }

        Ldirect *= shadow;
    }

    // Output
    props.Ldirect = Ldirect;
    props.Lemi = Lemi;
    props.N = N;
    props.T = T;
    props.baseColor = baseColor;
    props.roughness = roughness;
    props.metalness = metalness;
    props.curvature = geometryProps.curvature + curvature;

    return props;
}

//====================================================================================================================================
// MISC
//====================================================================================================================================

float3 GetAmbientBRDF( GeometryProps geometryProps, MaterialProps materialProps, bool approximate = false )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float3 Fenv = Rf0;
    if( !approximate )
    {
        float NoV = abs( dot( materialProps.N, geometryProps.V ) );
        Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );
    }

    float3 ambBRDF = albedo * ( 1.0 - Fenv ) + Fenv;
    ambBRDF *= float( !geometryProps.IsSky( ) );

    return ambBRDF;
}

float EstimateDiffuseProbability( GeometryProps geometryProps, MaterialProps materialProps, bool useMagicBoost = false )
{
    // IMPORTANT: can't be used for hair tracing, but applicable in other hair related calculations

    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, geometryProps.V ) );
    float3 Fenv = STL::BRDF::EnvironmentTerm_Rtg( Rf0, NoV, materialProps.roughness );

    float lumSpec = STL::Color::Luminance( Fenv );
    float lumDiff = STL::Color::Luminance( albedo * ( 1.0 - Fenv ) );

    float diffProb = lumDiff / ( lumDiff + lumSpec + 1e-6 );

    // Boost diffuse if roughness is high
    if( useMagicBoost )
        diffProb = lerp( diffProb, 1.0, GetSpecMagicCurve( materialProps.roughness ) );

    return diffProb < 0.005 ? 0.0 : diffProb;
}

float ReprojectIrradiance(
    bool isPrevFrame, bool isRefraction,
    Texture2D<float3> texDiff, Texture2D<float4> texSpecViewZ,
    GeometryProps geometryProps, uint2 pixelPos,
    out float3 prevLdiff, out float3 prevLspec
)
{
    float2 rescale = ( isPrevFrame ? gRectSizePrev : gRectSize ) * gInvRenderSize;

    // IMPORTANT: not Xprev because confidence is based on viewZ
    float4 clip = STL::Geometry::ProjectiveTransform( isPrevFrame ? gWorldToClipPrev : gWorldToClip, geometryProps.X );
    float2 uv = ( clip.xy / clip.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;

    float4 data = texSpecViewZ.SampleLevel( gNearestSampler, uv * rescale, 0 );
    float prevViewZ = abs( data.w ) / NRD_FP16_VIEWZ_SCALE;

    prevLdiff = texDiff.SampleLevel( gNearestSampler, uv * rescale, 0 );
    prevLspec = data.xyz;

    // Initial state
    float weight = 1.0;
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    // Ignore back-projection
    weight *= float( clip.w > 0.0 );

    // Ignore undenoised regions ( split screen mode is active )
    weight *= float( pixelUv.x > gSeparator );
    weight *= float( uv.x > gSeparator );

    // Relaxed checks for refractions
    float viewZ = abs( STL::Geometry::AffineTransform( isPrevFrame ? gWorldToViewPrev : gWorldToView, geometryProps.X ).z );
    float err = ( viewZ - prevViewZ ) * STL::Math::PositiveRcp( max( viewZ, prevViewZ ) );

    if( isRefraction )
    {
        // Confidence - viewZ ( PSR makes prevViewZ further than the original primary surface )
        weight *= STL::Math::LinearStep( 0.03, 0.01, saturate( err ) );

        // Fade-out on screen edges ( hard )
        weight *= all( saturate( uv ) == uv );
    }
    else
    {
        // Confidence - viewZ
        weight *= STL::Math::LinearStep( 0.03, 0.01, abs( err ) );

        // Fade-out on screen edges ( soft )
        float2 f = STL::Math::LinearStep( 0.0, 0.1, uv ) * STL::Math::LinearStep( 1.0, 0.9, uv );
        weight *= f.x * f.y;

        // Confidence - ignore back-facing
        // Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side
        float NoL = dot( geometryProps.N, gSunDirection_gExposure.xyz );
        weight *= float( NoL * STL::Math::Sign( data.w ) > 0.0 );

        // Confidence - ignore too short rays
        float4 clip = STL::Geometry::ProjectiveTransform( gWorldToClip, geometryProps.X );
        float2 uv = ( clip.xy / clip.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;
        float d = length( ( uv - pixelUv ) * gRectSize );
        weight *= STL::Math::LinearStep( 1.0, 3.0, d );
    }

    // Ignore sky
    weight *= float( !geometryProps.IsSky( ) );

    // Clear out bad values
    [flatten]
    if( any( isnan( prevLdiff ) | isinf( prevLdiff ) | isnan( prevLspec ) | isinf( prevLspec ) ) )
    {
        prevLdiff = 0;
        prevLspec = 0;
        weight = 0;
    }

    // Use only if radiance is on the screen
    weight *= float( gOnScreen < SHOW_AMBIENT_OCCLUSION );

    return weight;
}

//====================================================================================================================================
// TRACER
//====================================================================================================================================

#define CheckNonOpaqueTriangle( rayQuery, mipAndCone ) \
    { \
        /* Instance */ \
        uint instanceIndex = rayQuery.CandidateInstanceID( ) + rayQuery.CandidateGeometryIndex( ); \
        InstanceData instanceData = gIn_InstanceData[ instanceIndex ]; \
        \
        /* Transform */ \
        float3x3 mObjectToWorld = ( float3x3 )rayQuery.CandidateObjectToWorld3x4( ); \
        float3x4 mOverloaded = float3x4( instanceData.mOverloadedMatrix0, instanceData.mOverloadedMatrix1, instanceData.mOverloadedMatrix2 ); \
        if( instanceData.textureOffsetAndFlags & ( FLAG_STATIC << FLAG_FIRST_BIT ) ) \
            mObjectToWorld = ( float3x3 )mOverloaded; \
        \
        float flip = STL::Math::Sign( instanceData.invScale ) * ( rayQuery.CandidateTriangleFrontFace( ) ? -1.0 : 1.0 ); \
        \
        /* Primitive */ \
        uint primitiveIndex = instanceData.primitiveOffset + rayQuery.CandidatePrimitiveIndex( ); \
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ]; \
        \
        /* Barycentrics */ \
        float3 barycentrics; \
        barycentrics.yz = rayQuery.CandidateTriangleBarycentrics( ); \
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z; \
        \
        /* Uv */ \
        float2 uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2; \
        \
        /* Normal */ \
        float3 n0 = STL::Packing::DecodeUnitVector( primitiveData.n0, true ); \
        float3 n1 = STL::Packing::DecodeUnitVector( primitiveData.n1, true ); \
        float3 n2 = STL::Packing::DecodeUnitVector( primitiveData.n2, true ); \
        \
        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2; \
        N = STL::Geometry::RotateVector( mObjectToWorld, N ); \
        N = normalize( N * flip ); \
        \
        /* Mip level ( TODO: doesn't take into account integrated AO / SO - i.e. diffuse = lowest mip, but what if we see the sky through a tiny hole? ) */ \
        float NoR = abs( dot( rayQuery.WorldRayDirection( ), N ) ); \
        float a = rayQuery.CandidateTriangleRayT( ); \
        a *= mipAndCone.y; \
        a *= STL::Math::PositiveRcp( NoR ); \
        a *= primitiveData.worldToUvUnits * abs( instanceData.invScale ); \
        \
        float mip = log2( a ); \
        mip += MAX_MIP_LEVEL; \
        mip = max( mip, 0.0 ); \
        mip += mipAndCone.x; \
        \
        /* Alpha test */ \
        uint baseTexture = ( instanceData.textureOffsetAndFlags & NON_FLAG_MASK ) + 0; \
        float3 mips = GetRealMip( baseTexture, mip ); \
        float alpha = gIn_Textures[ baseTexture ].SampleLevel( TEX_SAMPLER, uv, mips.x ).w; \
        \
        if( alpha > 0.5 ) \
            rayQuery.CommitNonOpaqueTriangleHit( ); \
    }

bool CastVisibilityRay_AnyHit( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    return rayQuery.CommittedStatus( ) == COMMITTED_NOTHING;
}

float CastVisibilityRay_ClosestHit( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    return rayQuery.CommittedStatus( ) == COMMITTED_NOTHING ? INF : rayQuery.CommittedRayT( );
}

GeometryProps CastRay( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    // TODO: reuse data if committed == candidate (use T to check)
    GeometryProps props = ( GeometryProps )0;
    props.mip = mipAndCone.x;

    if( rayQuery.CommittedStatus( ) == COMMITTED_NOTHING )
    {
        props.tmin = INF;
        props.X = origin + direction * props.tmin;
        props.Xprev = props.X;
    }
    else
    {
        props.tmin = rayQuery.CommittedRayT( );

        // Instance
        uint instanceIndex = rayQuery.CommittedInstanceID( ) + rayQuery.CommittedGeometryIndex( );
        props.instanceIndex = instanceIndex;

        InstanceData instanceData = gIn_InstanceData[ instanceIndex ];

        // Texture offset and flags
        props.textureOffsetAndFlags = instanceData.textureOffsetAndFlags;

        // Transform
        float3x3 mObjectToWorld = ( float3x3 )rayQuery.CommittedObjectToWorld3x4( );
        float3x4 mOverloaded = float3x4( instanceData.mOverloadedMatrix0, instanceData.mOverloadedMatrix1, instanceData.mOverloadedMatrix2 ); \

        if( props.IsStatic( ) )
            mObjectToWorld = ( float3x3 )mOverloaded;

        float flip = STL::Math::Sign( instanceData.invScale ) * ( rayQuery.CommittedTriangleFrontFace( ) ? -1.0 : 1.0 );

        // Primitive
        uint primitiveIndex = instanceData.primitiveOffset + rayQuery.CommittedPrimitiveIndex( );
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ];

        // Barycentrics
        float3 barycentrics;
        barycentrics.yz = rayQuery.CommittedTriangleBarycentrics( );
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

        // Normal
        float3 n0 = STL::Packing::DecodeUnitVector( primitiveData.n0, true );
        float3 n1 = STL::Packing::DecodeUnitVector( primitiveData.n1, true );
        float3 n2 = STL::Packing::DecodeUnitVector( primitiveData.n2, true );

        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
        N = STL::Geometry::RotateVector( mObjectToWorld, N );
        N = normalize( N * flip );
        props.N = -N; // TODO: why negated?

        // Curvature
        props.curvature = barycentrics.x * primitiveData.curvature0_curvature1.x + barycentrics.y * primitiveData.curvature0_curvature1.y + barycentrics.z * primitiveData.curvature2_bitangentSign.x;
        props.curvature /= abs( instanceData.invScale );

        // Mip level (TODO: doesn't take into account integrated AO / SO - i.e. diffuse = lowest mip, but what if we see the sky through a tiny hole?)
        float NoR = abs( dot( direction, props.N ) );
        float a = props.tmin * mipAndCone.y;
        a *= STL::Math::PositiveRcp( NoR );
        a *= primitiveData.worldToUvUnits * abs( instanceData.invScale );

        float mip = log2( a );
        mip += MAX_MIP_LEVEL;
        mip = max( mip, 0.0 );
        props.mip += mip;

        // Uv
        props.uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2;

        // Tangent
        float3 t0 = STL::Packing::DecodeUnitVector( primitiveData.t0, true );
        float3 t1 = STL::Packing::DecodeUnitVector( primitiveData.t1, true );
        float3 t2 = STL::Packing::DecodeUnitVector( primitiveData.t2, true );

        float3 T = barycentrics.x * t0 + barycentrics.y * t1 + barycentrics.z * t2;
        T = STL::Geometry::RotateVector( mObjectToWorld, T );
        T = normalize( T );
        props.T = float4( T, primitiveData.curvature2_bitangentSign.y );

        props.X = origin + direction * props.tmin;
        if( props.IsDeformable( ) )
        {
            MorphedPrimitivePrevData prevData = gIn_MorphedPrimitivePrevPositions[ instanceData.morphedPrimitiveOffset + rayQuery.CommittedPrimitiveIndex( ) ];

            float3 XprevLocal = barycentrics.x * prevData.position0.xyz + barycentrics.y * prevData.position1.xyz + barycentrics.z * prevData.position2.xyz;
            props.Xprev = STL::Geometry::AffineTransform( mOverloaded, XprevLocal );
        }
        else if( !props.IsStatic( ) )
            props.Xprev = STL::Geometry::AffineTransform( mOverloaded, props.X );
        else
            props.Xprev = props.X;
    }

    props.V = -direction;

    return props;
}
