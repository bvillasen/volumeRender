// #include <cstdlib>
// using namespace std;
// Simple 3D volume render



#include <cutil_inline.h>
#include <cutil_math.h>


typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<float, 2, cudaReadModeElementType>         transferTex; // 1D transfer function texture

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
// __constant__ float c_invViewMatrix[12];

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


extern "C"{
__global__ void d_render(uint *d_output, uint imageW, uint imageH, float density, float brightness,  float transferOffset, float transferScale){
  const int maxSteps = 10000;
  const float tstep = 0.001f;
  const float opacityThreshold = 0.99f;
  const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
  const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  if ((x >= imageW) || (y >= imageH)) return;

  float u = (x / (float) imageW)*2.0f-1.0f;
  float v = (y / (float) imageH)*2.0f-1.0f;

  // calculate eye ray in world space
  Ray eyeRay;
  eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
  eyeRay.d = normalize(make_float3(u, v, -2.0f));
  eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

  // find intersection with box
  float tnear, tfar;
  int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
  if (!hit) return;
  if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

  // march along ray from front to back, accumulating color
  float4 sum = make_float4(0.0f);
  float t = tnear;
  float3 pos = eyeRay.o + eyeRay.d*tnear;
  float3 step = eyeRay.d*tstep;

  for(int i=0; i<maxSteps; i++) {
    // read from 3D texture
    // remap position to [0, 1] coordinates
    float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);

    //sample *= 64.0f;    // scale for 10-bit data

    // lookup in transfer function texture
    // float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
    float4 col;
    float nCol =  (sample-transferOffset)*transferScale;
    col.x = tex2D(transferTex, 0.f , nCol);
    col.y = tex2D(transferTex, 0.33f , nCol);
    col.z = tex2D(transferTex, 0.66f , nCol);
    col.w = tex2D(transferTex, 1.f , nCol);
    
    if (col.w == 1) col.w*=0.05; // Box borders frame uniform density
    else col.w *= density;
    
    // col.w *= sqrt(sample);
    col.w *= sample;
    // col.w *= sample*sample;

    // "under" operator for back-to-front blending
    // sum = lerp(sum, col, col.w);

    // pre-multiply alpha
    col.x *= col.w;
    col.y *= col.w;
    col.z *= col.w;
    // "over" operator for front-to-back blending
    sum = sum + col*(1.0f - sum.w);

    // exit early if opaque
    if (sum.w > opacityThreshold)
        break;

    t += tstep;
    if (t > tfar) break;

    pos += step;
  }
  
  sum *= brightness;
  // write output color
  d_output[y*imageW + x] = rgbaFloatToInt(sum);
}


} //extern C
