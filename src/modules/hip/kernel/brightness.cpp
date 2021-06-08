#include <hip/hip_runtime.h>

#if defined(STATIC)
#include "rpp_hip_host_decls.hpp"
#endif

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

typedef struct d_float8
{
    float data[8];
} d_float8;

__device__ unsigned char brighten(unsigned char input_pixel, float alpha, float beta)
{
    return saturate_8u(alpha * input_pixel + beta);
}

__device__ unsigned char brighten_fmaf(float input_pixel, float alpha, float beta)
{
    return (unsigned char)saturate_8u(fmaf(alpha, input_pixel, beta));
}

__device__ __forceinline__ uint rpp_hip_pack(float4 src)
{
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ float rpp_hip_unpack0(uint src)
{
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(uint src)
{
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(uint src)
{
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(uint src)
{
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack(uint src)
{
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
}









__device__ __forceinline__ ulong rpp_hip_pack_ulong(d_float8 src)
{
    ulong dst;
    dst |= ((ulong)src.data[1] & 0xFF) << 56;
    dst |= ((ulong)src.data[2] & 0xFF) << 48;
    dst |= ((ulong)src.data[3] & 0xFF) << 40;
    dst |= ((ulong)src.data[4] & 0xFF) << 32;
    dst |= ((ulong)src.data[5] & 0xFF) << 24;
    dst |= ((ulong)src.data[6] & 0xFF) << 16;
    dst |= ((ulong)src.data[7] & 0xFF) << 8;
    dst |= ((ulong)src.data[7] & 0xFF) << 0;

    return dst;
}

__device__ __forceinline__ float rpp_hip_unpack0_ulong(ulong src)
{
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1_ulong(ulong src)
{
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2_ulong(ulong src)
{
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3_ulong(ulong src)
{
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack4_ulong(ulong src)
{
    return (float)((src >> 32) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack5_ulong(ulong src)
{
    return (float)((src >> 40) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack6_ulong(ulong src)
{
    return (float)((src >> 48) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack7_ulong(ulong src)
{
    return (float)((src >> 56) & 0xFF);
}

__device__ __forceinline__ d_float8 rpp_hip_unpack_ulong(ulong src)
{
    d_float8 f8;
    f8.data[0] = rpp_hip_unpack0_ulong(src);
    f8.data[1] = rpp_hip_unpack1_ulong(src);
    f8.data[2] = rpp_hip_unpack2_ulong(src);
    f8.data[3] = rpp_hip_unpack3_ulong(src);
    f8.data[4] = rpp_hip_unpack4_ulong(src);
    f8.data[5] = rpp_hip_unpack5_ulong(src);
    f8.data[6] = rpp_hip_unpack6_ulong(src);
    f8.data[7] = rpp_hip_unpack7_ulong(src);

    return f8;
}




__device__ unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, unsigned int height, unsigned channel)
{
    return (id_x + id_y * width + id_z * width * height);
}

extern "C" __global__ void brightness(unsigned char *input,
                                      unsigned char *output,
                                      const float alpha,
                                      const int beta,
                                      const unsigned int height,
                                      const unsigned int width,
                                      const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = get_pln_index(id_x, id_y, id_z, width, height, channel);
    int res = input[pixIdx] * alpha + beta;
    output[pixIdx] = saturate_8u(res);
}










// 0 - WORKING - Original

// End to end time
// 2 224x224 images GPU Time - BatchPD : 0.000902s
// 32 224x224 images GPU Time - BatchPD : 0.001284s
// 32 3840x2160 images GPU Time - BatchPD : 0.015152s

// Profiler time
// 2 224x224 images Profiler Time - 0.013461ms (0.000013461s)
// 32 224x224 images Profiler Time - 0.087583ms (0.000087583s)
// 32 3840x2160 images Profiler Time - 7.012795ms (0.007012795s) PLN1
// 32 3840x2160 images Profiler Time - 11.915725ms (0.011915725) PLN3
// 32 3840x2160 images Profiler Time - 11.521102ms (0.011521102s) PKD3

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     float alphatmp = alpha[id_z], betatmp = beta[id_z];
//     long pixIdx = 0;

//     pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

//     if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
//     {
//         for(int indextmp = 0; indextmp < channel; indextmp++)
//         {
//             unsigned char valuergb = input[pixIdx];
//             output[pixIdx] = brighten(valuergb, alphatmp, betatmp);
//             pixIdx += inc[id_z];
//         }
//     }
//     else if((id_x < width[id_z]) && (id_y < height[id_z]))
//     {
//         for(int indextmp = 0; indextmp < channel; indextmp++)
//         {
//             output[pixIdx] = input[pixIdx];
//             pixIdx += inc[id_z];
//         }
//     }
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif










// 1 - Remove branching

// End to end time
// 2 224x224 images GPU Time - BatchPD : 0.000871s
// 32 224x224 images GPU Time - BatchPD : s
// 32 3840x2160 images GPU Time - BatchPD : s

// Profiler time
// 2 224x224 images Profiler Time - ms (s)
// 32 224x224 images Profiler Time - ms (s)
// 32 3840x2160 images Profiler Time - 5.879573ms (0.005879573s)

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     float alphatmp = alpha[id_z], betatmp = beta[id_z];
//     long pixIdx = 0;

//     pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

//     if ((id_x > width[id_z]) || (id_y > height[id_z]))
//     {
//         return;
//     }

//     for(int indextmp = 0; indextmp < channel; indextmp++)
//     {
//         unsigned char valuergb = input[pixIdx];
//         output[pixIdx] = brighten(valuergb, alphatmp, betatmp);
//         pixIdx += inc[id_z];
//     }
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif










// 2 - Use fused instruction ops
// GPU Time - BatchPD : 0.001022
// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     float alphatmp = alpha[id_z], betatmp = beta[id_z];
//     long pixIdx = 0;

//     pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

//     if ((id_x > width[id_z]) || (id_y > height[id_z]))
//     {
//         return;
//     }

//     for(int indextmp = 0; indextmp < channel; indextmp++)
//     {
//         unsigned char valuergb = input[pixIdx];
//         output[pixIdx] = brighten_fmaf(valuergb, alphatmp, betatmp);
//         pixIdx += inc[id_z];
//     }
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 32;
//     int localThreads_y = 32;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 31) & ~31;
//     int globalThreads_y = (max_height + 31) & ~31;
//     int globalThreads_z = handle.GetBatchSize();

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif










// 3 - Change to two dimensions with manual unroll

// End to end time
// 2 224x224 images GPU Time - BatchPD : s
// 32 224x224 images GPU Time - BatchPD : s
// 32 3840x2160 images GPU Time - BatchPD : s

// Profiler time
// 2 224x224 images Profiler Time - ms (s)
// 32 224x224 images Profiler Time - ms (s)
// 32 3840x2160 images Profiler Time - 2.446023ms (0.002446023s)

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned int *max_height,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

//     float alphatmp = alpha[id_y];
//     float betatmp = beta[id_y];
//     long pixIdx = 0;

//     pixIdx = (id_y * max_width[id_y] * max_height[id_y] * channel) + id_x;

//     output[pixIdx] = brighten_fmaf(input[pixIdx], alphatmp, betatmp);
//     output[pixIdx + 1] = brighten_fmaf(input[pixIdx + 1], alphatmp, betatmp);
//     output[pixIdx + 2] = brighten_fmaf(input[pixIdx + 2], alphatmp, betatmp);
//     output[pixIdx + 3] = brighten_fmaf(input[pixIdx + 3], alphatmp, betatmp);
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 256;
//     int localThreads_y = 1;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_height * max_width * channel + 3) >> 2;
//     int globalThreads_y = handle.GetBatchSize();
//     int globalThreads_z = 1;

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif










// 4 - WORKING - Replicate OpenVX style (PLN1 only)

// End to end time
// 2 224x224 images GPU Time - BatchPD : 0.000871s
// 32 224x224 images GPU Time - BatchPD : 0.001077s
// 32 3840x2160 images GPU Time - BatchPD : 0.004999s

// Profiler time
// 2 224x224 images Profiler Time - 0.01103ms (0.00001103s)
// 32 224x224 images Profiler Time - 0.038652ms (0.000038652s)
// 32 3840x2160 images Profiler Time - 1.898541ms (0.001898541s)

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_x > width[id_z]) || (id_y > height[id_z]))
//     {
//         return;
//     }

//     uint srcIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x);
//     uint dstIdx =  batch_index[id_z] + (id_y * max_width[id_z] + id_x);

//     uint2 src = *((uint2 *)(&input[srcIdx]));
//     uint2 dst;

//     // dst.x = 0xaa000b0a;
//     // dst.y = 0xffeeddcc;

//     float4 alpha4 = (float4)alpha[id_z];
//     float4 beta4 = (float4)beta[id_z];

//     dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
//     dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

//     *((uint2 *)(&output[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 16;
//     int localThreads_y = 16;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 7) >> 3;
//     int globalThreads_y = max_height;
//     int globalThreads_z = handle.GetBatchSize();

//     // start

//     // start = clock();
//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);
//     // end

//     return RPP_SUCCESS;
// }
// #endif









// 5 - WORKING - Replicate OpenVX style (PLN1 only) + fmaf

// End to end time
// 2 224x224 images GPU Time - BatchPD : s
// 32 224x224 images GPU Time - BatchPD : s
// 32 3840x2160 images GPU Time - BatchPD : s

// Profiler time
// 2 224x224 images Profiler Time - ms (s)
// 32 224x224 images Profiler Time - ms (s)
// 32 3840x2160 images Profiler Time - 1.886347ms (0.001886347s)

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_x > width[id_z]) || (id_y > height[id_z]))
//     {
//         return;
//     }

//     uint srcIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x);
//     uint dstIdx =  batch_index[id_z] + (id_y * max_width[id_z] + id_x);

//     uint2 src = *((uint2 *)(&input[srcIdx]));
//     uint2 dst;

//     // dst.x = 0xaa000b0a;
//     // dst.y = 0xffeeddcc;

//     // float4 alpha4 = (float4)alpha[id_z];
//     // float4 beta4 = (float4)beta[id_z];

//     // dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
//     // dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

//     float4 pixSet1 = rpp_hip_unpack(src.x);
//     float4 pixSet2 = rpp_hip_unpack(src.y);

//     // pixSet1.x = fmaf(pixSet1.x, alpha[id_z], beta[id_z]);
//     // pixSet1.y = fmaf(pixSet1.y, alpha[id_z], beta[id_z]);
//     // pixSet1.z = fmaf(pixSet1.z, alpha[id_z], beta[id_z]);
//     // pixSet1.w = fmaf(pixSet1.w, alpha[id_z], beta[id_z]);

//     // pixSet2.x = fmaf(pixSet2.x, alpha[id_z], beta[id_z]);
//     // pixSet2.y = fmaf(pixSet2.y, alpha[id_z], beta[id_z]);
//     // pixSet2.z = fmaf(pixSet2.z, alpha[id_z], beta[id_z]);
//     // pixSet2.w = fmaf(pixSet2.w, alpha[id_z], beta[id_z]);

//     pixSet1.x = pixSet1.x * alpha[id_z] + beta[id_z]);
//     pixSet1.y = pixSet1.y * alpha[id_z] + beta[id_z]);
//     pixSet1.z = pixSet1.z * alpha[id_z] + beta[id_z]);
//     pixSet1.w = pixSet1.w * alpha[id_z] + beta[id_z]);

//     pixSet2.x = pixSet2.x * alpha[id_z] + beta[id_z]);
//     pixSet2.y = pixSet2.y * alpha[id_z] + beta[id_z]);
//     pixSet2.z = pixSet2.z * alpha[id_z] + beta[id_z]);
//     pixSet2.w = pixSet2.w * alpha[id_z] + beta[id_z]);

//     dst.x = rpp_hip_pack(pixSet1);
//     dst.y = rpp_hip_pack(pixSet2);

//     // dst.x = rpp_hip_pack(fmaf(rpp_hip_unpack(src.x), alpha4, beta4));
//     // dst.y = rpp_hip_pack(fmaf(rpp_hip_unpack(src.y), alpha4, beta4));

//     *((uint2 *)(&output[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 16;
//     int localThreads_y = 16;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 7) >> 3;
//     int globalThreads_y = max_height;
//     int globalThreads_z = handle.GetBatchSize();

//     // start

//     // start = clock();
//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);
//     // end

//     return RPP_SUCCESS;
// }
// #endif









// 6 - WORKING - Replicate OpenVX style (PLN3)

// End to end time
// 2 224x224 images GPU Time - BatchPD : s
// 32 224x224 images GPU Time - BatchPD : s
// 32 3840x2160 images GPU Time - BatchPD : s

// Profiler time
// 2 224x224 images Profiler Time - ms (s)
// 32 224x224 images Profiler Time - ms (s)
// 32 3840x2160 images Profiler Time -  5.144345ms (0.005144345s) PLN3

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned int image_size,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_x > width[id_z]) || (id_y > height[id_z]))
//     {
//         return;
//         // if (channel == 3)
//         // {
//         //     if ((id_y > height[id_z] * 3))
//         //     {
//         //         return;
//         //     }
//         // }
//         // else
//         // {
//         //     return;
//         // }
//     }

//     uint srcIdx1 = batch_index[id_z] + (id_y * max_width[id_z] + id_x);
//     uint srcIdx2 = srcIdx1 + image_size;
//     uint srcIdx3 = srcIdx2 + image_size;

//     uint dstIdx1 =  batch_index[id_z] + (id_y * max_width[id_z] + id_x);
//     uint dstIdx2 =  dstIdx1 + image_size;
//     uint dstIdx3 =  dstIdx2 + image_size;

//     uint2 src1 = *((uint2 *)(&input[srcIdx1]));
//     uint2 src2 = *((uint2 *)(&input[srcIdx2]));
//     uint2 src3 = *((uint2 *)(&input[srcIdx3]));
//     uint2 dst1, dst2, dst3;

//     float4 alpha4 = (float4)alpha[id_z];
//     float4 beta4 = (float4)beta[id_z];

//     dst1.x = rpp_hip_pack(rpp_hip_unpack(src1.x) * alpha4 + beta4);
//     dst1.y = rpp_hip_pack(rpp_hip_unpack(src1.y) * alpha4 + beta4);

//     dst2.x = rpp_hip_pack(rpp_hip_unpack(src2.x) * alpha4 + beta4);
//     dst2.y = rpp_hip_pack(rpp_hip_unpack(src2.y) * alpha4 + beta4);

//     dst3.x = rpp_hip_pack(rpp_hip_unpack(src3.x) * alpha4 + beta4);
//     dst3.y = rpp_hip_pack(rpp_hip_unpack(src3.y) * alpha4 + beta4);

//     *((uint2 *)(&output[dstIdx1])) = dst1;
//     *((uint2 *)(&output[dstIdx2])) = dst2;
//     *((uint2 *)(&output[dstIdx3])) = dst3;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 16;
//     int localThreads_y = 16;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_width + 7) >> 3;
//     int globalThreads_y = max_height;
//     int globalThreads_z = handle.GetBatchSize();

//     // if (channel == 3)
//     // {
//     //     globalThreads_y *= channel;
//     // }

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        max_width * max_height,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif









// // 7 - WORKING - Replicate OpenVX style (PLN1, PLN3, PKD3) and reduce dimensions

// // End to end time
// // 2 224x224 images GPU Time - BatchPD : s
// // 32 224x224 images GPU Time - BatchPD : s
// // 32 3840x2160 images GPU Time - BatchPD : s

// // Profiler time
// // 2 224x224 images Profiler Time - ms (s)
// // 32 224x224 images Profiler Time - ms (s)
// // 32 3840x2160 images Profiler Time - 1.702551ms (0.001702551s) PLN1
// // 32 3840x2160 images Profiler Time - 4.565886ms (0.004565886s) PLN3
// // 32 3840x2160 images Profiler Time - 4.602853ms (0.004602853s) PKD3

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned int *max_height,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

//     uint srcIdx = batch_index[id_y] + id_x;
//     uint dstIdx =  batch_index[id_y] + id_x;

//     uint2 src = *((uint2 *)(&input[srcIdx]));
//     uint2 dst;

//     float4 alpha4 = (float4)alpha[id_y];
//     float4 beta4 = (float4)beta[id_y];

//     dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
//     dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

//     *((uint2 *)(&output[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 256;
//     int localThreads_y = 1;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_height * max_width * channel + 7) >> 3;
//     int globalThreads_y = handle.GetBatchSize();
//     int globalThreads_z = 1;

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif









// 8 - WORKING - Replicate OpenVX style (PLN1, PLN3, PKD3) and reduce dimensions and increase vectorization length

// End to end time
// 2 224x224 images GPU Time - BatchPD : s
// 32 224x224 images GPU Time - BatchPD : s
// 32 3840x2160 images GPU Time - BatchPD : s

// Profiler time
// 2 224x224 images Profiler Time - ms (s)
// 32 224x224 images Profiler Time - ms (s)
// 32 3840x2160 images Profiler Time - 1.702551ms (0.001702551s) PLN1
// 32 3840x2160 images Profiler Time - 4.565886ms (0.004565886s) PLN3
// 32 3840x2160 images Profiler Time - 4.602853ms (0.004602853s) PKD3

// extern "C" __global__ void brightness_batch(unsigned char *input,
//                                             unsigned char *output,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned int *xroi_begin,
//                                             unsigned int *xroi_end,
//                                             unsigned int *yroi_begin,
//                                             unsigned int *yroi_end,
//                                             unsigned int *height,
//                                             unsigned int *width,
//                                             unsigned int *max_width,
//                                             unsigned int *max_height,
//                                             unsigned long long *batch_index,
//                                             const unsigned int channel,
//                                             unsigned int *inc, // use width * height for pln and 1 for pkd
//                                             const int plnpkdindex) // use 1 pln 3 for pkd
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 16;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     // if ((id_x > width[id_y]) || (id_y > height[id_y]))
//     // {
//     //     if (channel == 3)
//     //     {
//     //         if ((id_y > height[id_y] * 3))
//     //         {
//     //             return;
//     //         }
//     //     }
//     //     else
//     //     {
//     //         return;
//     //     }
//     // }

//     uint srcIdx = batch_index[id_y] + id_x;
//     uint dstIdx =  batch_index[id_y] + id_x;

//     ulong2 src = *((ulong2 *)(&input[srcIdx]));
//     ulong2 dst;

//     d_float8 src1 = rpp_hip_unpack_ulong(src.x);
//     d_float8 src2 = rpp_hip_unpack_ulong(src.y);

//     src1.data[0] = fmaf(src1.data[0], alpha[id_y], beta[id_y]);
//     src1.data[1] = fmaf(src1.data[1], alpha[id_y], beta[id_y]);
//     src1.data[2] = fmaf(src1.data[2], alpha[id_y], beta[id_y]);
//     src1.data[3] = fmaf(src1.data[3], alpha[id_y], beta[id_y]);
//     src1.data[4] = fmaf(src1.data[4], alpha[id_y], beta[id_y]);
//     src1.data[5] = fmaf(src1.data[5], alpha[id_y], beta[id_y]);
//     src1.data[6] = fmaf(src1.data[6], alpha[id_y], beta[id_y]);
//     src1.data[7] = fmaf(src1.data[7], alpha[id_y], beta[id_y]);

//     src2.data[0] = fmaf(src2.data[0], alpha[id_y], beta[id_y]);
//     src2.data[1] = fmaf(src2.data[1], alpha[id_y], beta[id_y]);
//     src2.data[2] = fmaf(src2.data[2], alpha[id_y], beta[id_y]);
//     src2.data[3] = fmaf(src2.data[3], alpha[id_y], beta[id_y]);
//     src2.data[4] = fmaf(src2.data[4], alpha[id_y], beta[id_y]);
//     src2.data[5] = fmaf(src2.data[5], alpha[id_y], beta[id_y]);
//     src2.data[6] = fmaf(src2.data[6], alpha[id_y], beta[id_y]);
//     src2.data[7] = fmaf(src2.data[7], alpha[id_y], beta[id_y]);

//     // float4 alpha4 = (float4)alpha[id_y];
//     // float4 beta4 = (float4)beta[id_y];

//     dst.x = rpp_hip_pack_ulong(src1);
//     dst.y = rpp_hip_pack_ulong(src2);

//     *((ulong2 *)(&output[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int localThreads_x = 256;
//     int localThreads_y = 1;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_height * max_width * channel + 7) >> 3;
//     int globalThreads_y = handle.GetBatchSize();
//     int globalThreads_z = 1;

//     // if (channel == 3)
//     // {
//     //     globalThreads_y *= channel;
//     // }

//     hipLaunchKernelGGL(brightness_batch,
//                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                        dim3(localThreads_x, localThreads_y, localThreads_z),
//                        0,
//                        handle.GetStream(),
//                        srcPtr,
//                        dstPtr,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
//                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
//                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.height,
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
//                        channel,
//                        handle.GetInitHandle()->mem.mgpu.inc,
//                        plnpkdind);

//     return RPP_SUCCESS;
// }
// #endif















// 9 - WORKING - Replicate OpenVX style (PLN1, PLN3, PKD3) and reduce dimensions, double unrolling

// End to end time
// 2 224x224 images GPU Time - BatchPD : s
// 32 224x224 images GPU Time - BatchPD : s
// 32 3840x2160 images GPU Time - BatchPD : s

// Profiler time
// 2 224x224 images Profiler Time - ms (s)
// 32 224x224 images Profiler Time - ms (s)
// 32 3840x2160 images Profiler Time - 1.699089ms (0.001699089s) PLN1
// 32 3840x2160 images Profiler Time - 4.540544ms (0.004540544s) PLN3
// 32 3840x2160 images Profiler Time - 4.565369ms (0.004565369s) PKD3

extern "C" __global__ void brightness_batch(unsigned char *input,
                                            unsigned char *output,
                                            float *alpha,
                                            float *beta,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned int *max_height,
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 16;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    uint srcIdx = batch_index[id_y] + id_x;
    uint dstIdx =  batch_index[id_y] + id_x;

    uint2 src1 = *((uint2 *)(&input[srcIdx]));
    uint2 src2 = *((uint2 *)(&input[srcIdx + 8]));
    uint2 dst1, dst2;

    float4 alpha4 = (float4)alpha[id_y];
    float4 beta4 = (float4)beta[id_y];

    dst1.x = rpp_hip_pack(rpp_hip_unpack(src1.x) * alpha4 + beta4);
    dst1.y = rpp_hip_pack(rpp_hip_unpack(src1.y) * alpha4 + beta4);

    dst2.x = rpp_hip_pack(rpp_hip_unpack(src2.x) * alpha4 + beta4);
    dst2.y = rpp_hip_pack(rpp_hip_unpack(src2.y) * alpha4 + beta4);

    *((uint2 *)(&output[dstIdx])) = dst1;
    *((uint2 *)(&output[dstIdx + 8])) = dst2;
}

#if defined(STATIC)
RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = (max_height * max_width * channel + 15) >> 4;
    int globalThreads_y = handle.GetBatchSize();
    int globalThreads_z = 1;

    hipLaunchKernelGGL(brightness_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}
#endif