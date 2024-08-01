
// Chiang16 Hair model
// See https://benedikt-bitterli.me/pchfm/
// See https://www.pbrt.org/hair.pdf

struct HairData
{
    float3 baseColor;
    float betaM; // longitudinal roughness [0, 1]
    float betaN; // azimuthal roughness [0, 1]
};

static const uint kMaxScatterEvents = 3;
static const float kSqrtPiOver8 = 0.626657069f;

static const float k1_2Pi = 1.0 / Math::Pi(2.0);
static const float k2Pi = Math::Pi(2.0);
static const float kPi = Math::Pi(1.0);

static const float kHairAlpha = 2.0f; // hair scale angle 2 degrees
static const float kHairIoR = 1.55f; // hair IoR

struct HairContext
{
    float alpha, IoR;
    float3 sigmaA;
    float h;
    float eta;

    float gammaI;
    float v[kMaxScatterEvents + 1];
    float s;
    float sin2kAlpha[3], cos2kAlpha[3];
};

// V = view
// N = surface normal
// T = tangent
struct HairSurfaceData
{
    float3 V;
    float3 N;
    float3 T;
};

float3x3 HairGetBasis(float3 N, float3 T)
{
    // X+: Tangent T should point towards dp/du for the curve
    // Y+: is dp/dv for the curve
    // Z+: is ribbon normal
    float3 B = cross(N, T);

    return float3x3(T, B, N);
}

/*******************************************************************
                          Helper functions
*******************************************************************/

float I0(float x)
{
    float val = 0.f;
    float x2i = 1.f;
    float ifact = 1.f;
    uint i4 = 1;

    [unroll]
    for (uint i = 0; i < 10; i++)
    {
        if (i > 1) ifact *= i;
        val += x2i / (ifact * ifact * i4);
        x2i *= x * x;
        i4 *= 4;
    }

    return val;
}

float logI0(float x)
{
    if (x > 12)
        return x + 0.5f * (-log(k2Pi) + log(1.f / x) + 0.125f / x);
    else
        return log(I0(x));
}

float phiFunction(int p, float gammaI, float gammaT)
{
    return 2.f * p * gammaT - 2.f * gammaI + p * kPi;
}

float logistic(float x, float s)
{
    x = abs(x);
    float tmp = exp(-x / s);

    return tmp / (s * (1.f + tmp) * (1.f + tmp));
}

float logisticCDF(float x, float s)
{
    return 1.f / (1.f + exp(-x / s));
}

float trimmedLogistic(float x, float s, float a, float b)
{
    return logistic(x, s) / (logisticCDF(b, s) - logisticCDF(a, s));
}

float sampleTrimmedLogistic(float u, float s, float a, float b)
{
    float k = logisticCDF(b, s) - logisticCDF(a, s);
    float x = -s * log(1.f / (u * k + logisticCDF(a, s)) - 1.f);

    return clamp(x, a, b);
}

float sqrt01(float x)
{
    return max(sqrt(saturate(x)), 1e-7f);
}

float sqrt0(float x)
{
    return sqrt(max(x, 1e-7f));
}

float atan2safe(float x, float y)
{
    return (abs(x) + abs(y)) < 1e-7 ? 0.0f : atan2(x, y);
}

// Mapping from color to sigmaA
float3 sigmaAFromColor(float3 color, float betaN)
{
    float tmp = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * pow(betaN, 3) + 5.574f * pow(betaN, 4) + 0.245f * pow(betaN, 5);
    float3 sqrtSigmaA = log(max(color, 1e-4f)) / tmp;

    return sqrtSigmaA * sqrtSigmaA;
}

// Attenuation function Ap
void Ap(HairContext c, float cosThetaI, float3 T, out float3 ap[kMaxScatterEvents + 1])
{
    float cosGammaI = sqrt01(1.f - c.h * c.h);
    float cosTheta = cosThetaI * cosGammaI;
    float f = BRDF::FresnelTerm_Dielectric(c.eta, cosTheta);

    ap[0] = f;
    ap[1] = T * (1 - f) * (1 - f);

    [unroll]
    for (uint p = 2; p < kMaxScatterEvents; p++)
        ap[p] = ap[p - 1] * T * f;

    // Compute attenuation term accounting for remaining orders of scattering
    ap[kMaxScatterEvents] = ap[kMaxScatterEvents - 1] * T * f / (1.f - T * f);
}

// Compute a discrete pdf for sampling Ap (which BCSDF lobe)
void computeApPdf(HairContext c, float cosThetaI, out float apPdf[kMaxScatterEvents + 1])
{
    float sinThetaI = sqrt01(1.f - cosThetaI * cosThetaI);

    // Compute refracted ray.
    float sinThetaT = sinThetaI / c.IoR;
    float cosThetaT = sqrt01(1.f - sinThetaT * sinThetaT);

    float etap = sqrt0(c.IoR * c.IoR - sinThetaI * sinThetaI) / cosThetaI;
    float sinGammaT = c.h / etap;
    float cosGammaT = sqrt01(1.f - sinGammaT * sinGammaT);

    // Compute the transmittance T of a single path through the cylinder
    float tmp = -2.f * cosGammaT / cosThetaT;
    float3 T = exp(c.sigmaA * tmp);

    float3 ap[kMaxScatterEvents + 1];
    Ap(c, cosThetaI, T, ap);

    // Compute apPdf from individal ap terms
    float sumY = 0.f;
    {
        [unroll]
        for (uint p = 0; p <= kMaxScatterEvents; p++)
        {
            apPdf[p] = Color::Luminance(ap[p]);
            sumY += apPdf[p];
        }
    }

    float invSumY = 1.f / sumY;
    {
        [unroll]
        for (uint p = 0; p <= kMaxScatterEvents; p++)
            apPdf[p] *= invSumY;
    }

}

// Longitudinal scattering function Mp
float Mp(float cosThetaI, float cosThetaO, float sinThetaI, float sinThetaO, float v)
{
    float a = cosThetaI * cosThetaO / v;
    float b = sinThetaI * sinThetaO / v;
    float mp = (v <= 0.1f) ? exp(logI0(a) - b - 1.f / v + 0.6931f + log(0.5f / v)) : (exp(-b) * I0(a)) / (sinh(1.f / v) * 2.f * v);

    return mp;
}


// Azimuthal scattering function Np
float Np(float phi, int p, float s, float gammaI, float gammaT)
{
    float dphi = phi - phiFunction(p, gammaI, gammaT);

    // Remap dphi to [-pi, pi].
    dphi = fmod(dphi, k2Pi);
    if (dphi > kPi)
        dphi -= k2Pi;
    if (dphi < -kPi)
        dphi += k2Pi;

    return trimmedLogistic(dphi, s, -kPi, kPi);
}

HairContext HairContextInit(HairSurfaceData sd, HairData data)
{
    HairContext context = (HairContext)0;
    context.sigmaA = sigmaAFromColor(data.baseColor, data.betaN);
    context.alpha = kHairAlpha;
    context.IoR = kHairIoR;

    // Falcor tracks eta = IoR Incident / IoR Transmitted.
    // In our case we assume IoR Incident is 1.0.
    // so eta = 1.0 / IoR
    context.eta = 1.0 / context.IoR;

    // Compute offset h azimuthally with the unit circle cross section
    float3 wiProj = normalize(sd.V - dot(sd.V, sd.T) * sd.T);   // Project wi to the (B, N) plane
    float3 wiProjPerp = cross(wiProj, sd.T);
    context.h = dot(sd.N, wiProjPerp);

    // precompute()
    context.gammaI = asin(clamp(context.h, -1.f, 1.f));

    float tmp = 0.726f * data.betaM + 0.812f * data.betaM * data.betaM + 3.7f * pow(data.betaM, 20.f);
    context.v[0] = tmp * tmp;
    context.v[1] = 0.25f * context.v[0];
    context.v[2] = 4 * context.v[0];
    [unroll]
    for (uint p = 3; p <= kMaxScatterEvents; p++)
        context.v[p] = context.v[2];

    // Compute azimuthal logistic scale factor
    context.s = kSqrtPiOver8 * (0.265f * data.betaN + 1.194f * data.betaN * data.betaN + 5.372f * pow(data.betaN, 22.f));

    // Compute alpha terms for hair scales
    context.sin2kAlpha[0] = sin(context.alpha / 180.f * kPi);
    context.cos2kAlpha[0] = sqrt01(1.f - context.sin2kAlpha[0] * context.sin2kAlpha[0]);
    [unroll]
    for (uint i = 1; i < 3; i++)
    {
        context.sin2kAlpha[i] = 2 * context.cos2kAlpha[i - 1] * context.sin2kAlpha[i - 1];
        context.cos2kAlpha[i] = context.cos2kAlpha[i - 1] * context.cos2kAlpha[i - 1] - context.sin2kAlpha[i - 1] * context.sin2kAlpha[i - 1];
    }

    return context;
}

float3 HairEval(HairContext c, float3 wi, float3 wo)
{
    float sinThetaI = wi.x;
    float cosThetaI = sqrt01(1.f - sinThetaI * sinThetaI);
    float phiI = atan2safe(wi.z, wi.y);

    float sinThetaO = wo.x;
    float cosThetaO = sqrt01(1.f - sinThetaO * sinThetaO);
    float phiO = atan2safe(wo.z, wo.y);

    // Compute refracted ray
    float sinThetaT = sinThetaI / c.IoR;
    float cosThetaT = sqrt01(1.f - sinThetaT * sinThetaT);

    float etap = sqrt0(c.IoR * c.IoR - sinThetaI * sinThetaI) / cosThetaI;
    float sinGammaT = c.h / etap;
    float cosGammaT = sqrt01(1.f - sinGammaT * sinGammaT);
    float gammaT = asin(clamp(sinGammaT, -1.f, 1.f));

    // Compute the transmittance T of a single path through the cylinder
    float tmp = -2.f * cosGammaT / cosThetaT;
    float3 T = exp(c.sigmaA * tmp);

    // Evaluate hair BCSDF for each lobe
    float phi = phiO - phiI;
    float3 ap[kMaxScatterEvents + 1];
    Ap(c, cosThetaI, T, ap);
    float3 result = 0.f;

    [unroll]
    for (int p = 0; p < kMaxScatterEvents; p++)
    {
        float sinThetaIp, cosThetaIp;
        if (p == 0)
        {
            sinThetaIp = sinThetaI * c.cos2kAlpha[1] - cosThetaI * c.sin2kAlpha[1];
            cosThetaIp = cosThetaI * c.cos2kAlpha[1] + sinThetaI * c.sin2kAlpha[1];
        }
        else if (p == 1)
        {
            sinThetaIp = sinThetaI * c.cos2kAlpha[0] + cosThetaI * c.sin2kAlpha[0];
            cosThetaIp = cosThetaI * c.cos2kAlpha[0] - sinThetaI * c.sin2kAlpha[0];
        }
        else if (p == 2)
        {
            sinThetaIp = sinThetaI * c.cos2kAlpha[2] + cosThetaI * c.sin2kAlpha[2];
            cosThetaIp = cosThetaI * c.cos2kAlpha[2] - sinThetaI * c.sin2kAlpha[2];
        }
        else
        {
            sinThetaIp = sinThetaI;
            cosThetaIp = cosThetaI;
        }

        cosThetaIp = abs(cosThetaIp);
        result += ap[p] * Mp(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, c.v[p]) * Np(phi, p, c.s, c.gammaI, gammaT);
    }

    // Compute contribution of remaining terms after kMaxScatterEvents
    result += ap[kMaxScatterEvents] * Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, c.v[kMaxScatterEvents]) * k1_2Pi;

    return saturate( result );
}

float HairSampleRay(HairContext c, float3 wi, float4 rnd, out float3 wo)
{
    float sinThetaI = wi.x;
    float cosThetaI = sqrt01(1.f - sinThetaI * sinThetaI);
    float phiI = atan2safe(wi.z, wi.y);

    // Determine which term p to sample for hair scattering
    float apPdf[kMaxScatterEvents + 1];
    computeApPdf(c, cosThetaI, apPdf);

    uint p = 0;
    float vp = c.v[0];
    // Use compile-time for to avoid stack allocation
    // while (p < kMaxScatterEvents && rnd.x >= apPdf[p])
    // {
    //     rnd.x -= apPdf[p];
    //     p++;
    //     vp = v[p];
    // }
    {
        bool done = false;
        [unroll]
        for (uint i = 0; i < kMaxScatterEvents; i++)
        {
            if (!done && rnd.x >= apPdf[i])
            {
                rnd.x -= apPdf[i];
                p = i + 1;
                vp = c.v[i + 1];
            }
            else
                done = true;
        }
    }

    float sinThetaIp = sinThetaI;
    float cosThetaIp = cosThetaI;
    if (p == 0)
    {
        sinThetaIp = sinThetaI * c.cos2kAlpha[1] - cosThetaI * c.sin2kAlpha[1];
        cosThetaIp = cosThetaI * c.cos2kAlpha[1] + sinThetaI * c.sin2kAlpha[1];
    }
    else if (p == 1)
    {
        sinThetaIp = sinThetaI * c.cos2kAlpha[0] + cosThetaI * c.sin2kAlpha[0];
        cosThetaIp = cosThetaI * c.cos2kAlpha[0] - sinThetaI * c.sin2kAlpha[0];
    }
    else if (p == 2)
    {
        sinThetaIp = sinThetaI * c.cos2kAlpha[2] + cosThetaI * c.sin2kAlpha[2];
        cosThetaIp = cosThetaI * c.cos2kAlpha[2] - sinThetaI * c.sin2kAlpha[2];
    }

    // Sample Mp to compute thetaO
    //rnd.z = max(rnd.z, 1e-5f);
    float cosTheta = clamp(1.f + vp * log(rnd.z + (1.f - rnd.z) * exp(-2.f / vp)), -1.0, 1.0);
    float sinTheta = sqrt01(1.f - cosTheta * cosTheta);
    float cosPhi = cos(rnd.w * k2Pi);
    float sinThetaO = -cosTheta * sinThetaIp + sinTheta * cosPhi * cosThetaIp;
    float cosThetaO = sqrt01(1.f - sinThetaO * sinThetaO);

    // Sample Np to compute dphi
    float etap = sqrt0(c.IoR * c.IoR - sinThetaI * sinThetaI) / cosThetaI;
    float sinGammaT = c.h / etap;
    float gammaT = asin(clamp(sinGammaT, -1.f, 1.f));
    float dphi;
    if (p < kMaxScatterEvents)
        dphi = phiFunction(p, c.gammaI, gammaT) + sampleTrimmedLogistic(rnd.y, c.s, -kPi, kPi);
    else
        dphi = rnd.y * k2Pi;

    float phiO = phiI + dphi;
    wo = float3(sinThetaO, cosThetaO * cos(phiO), cosThetaO * sin(phiO));

    // Compute pdf.
    float pdf = 0;

    [unroll]
    for (uint i = 0; i < kMaxScatterEvents; i++)
    {
        float sinThetaIp, cosThetaIp;
        if (i == 0)
        {
            sinThetaIp = sinThetaI * c.cos2kAlpha[1] - cosThetaI * c.sin2kAlpha[1];
            cosThetaIp = cosThetaI * c.cos2kAlpha[1] + sinThetaI * c.sin2kAlpha[1];
        }
        else if (i == 1)
        {
            sinThetaIp = sinThetaI * c.cos2kAlpha[0] + cosThetaI * c.sin2kAlpha[0];
            cosThetaIp = cosThetaI * c.cos2kAlpha[0] - sinThetaI * c.sin2kAlpha[0];
        }
        else if (i == 2)
        {
            sinThetaIp = sinThetaI * c.cos2kAlpha[2] + cosThetaI * c.sin2kAlpha[2];
            cosThetaIp = cosThetaI * c.cos2kAlpha[2] - sinThetaI * c.sin2kAlpha[2];
        }
        else
        {
            sinThetaIp = sinThetaI;
            cosThetaIp = cosThetaI;
        }

        cosThetaIp = abs(cosThetaIp);
        pdf += Mp(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, c.v[i]) * apPdf[i] * Np(dphi, i, c.s, c.gammaI, gammaT);
    }

    pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, c.v[kMaxScatterEvents]) * apPdf[kMaxScatterEvents] * k1_2Pi;

    return pdf;
}