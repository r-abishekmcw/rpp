#include <hip/hip_runtime.h>

#if defined(STATIC)
#include "rpp_hip_host_decls.hpp"
#endif

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ unsigned char brighten(unsigned char input_pixel, float alpha, float beta)
{
    return saturate_8u(alpha * input_pixel + beta);
}

__device__ unsigned char brighten_fmaf(float input_pixel, float alpha, float beta)
{
    return (unsigned char)saturate_8u(fmaf(alpha, input_pixel, beta));
}

__device__ __forceinline__ uint rpp_hip_pack(float4 src) {
    return __builtin_amdgcn_cvt_pk_u8_f32(src.w, 3,
           __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2,
           __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1,
           __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
}

__device__ __forceinline__ float rpp_hip_unpack0(uint src) {
    return (float)(src & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack1(uint src) {
    return (float)((src >> 8) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack2(uint src) {
    return (float)((src >> 16) & 0xFF);
}

__device__ __forceinline__ float rpp_hip_unpack3(uint src) {
    return (float)((src >> 24) & 0xFF);
}

__device__ __forceinline__ float4 rpp_hip_unpack(uint src) {
    return make_float4(rpp_hip_unpack0(src), rpp_hip_unpack1(src), rpp_hip_unpack2(src), rpp_hip_unpack3(src));
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










// Original

// End to end time
// 2 224x224 images GPU Time - BatchPD : 0.000902s
// 32 224x224 images GPU Time - BatchPD : 0.001284s
// 32 3840x2160 images GPU Time - BatchPD : 0.015152s

// Profiler time
// 2 224x224 images Profiler Time - 0.013461ms (0.000013461s)
// 32 224x224 images Profiler Time - 0.087583ms (0.000087583s)
// 32 3840x2160 images Profiler Time - 7.012795ms (0.007012795s)



// OLD TIMES
// rpp-unittests times
// 2 224x224 images - GPU Time - BatchPD : 0.001071
// 32 224x224 images - GPU Time - BatchPD : 0.001272
// 32 3840x2160 images - GPU Time - BatchPD : 0.001329
// 256 224x224 images - GPU Time - BatchPD : 0.001059
// 256 3840x2160 images - hangs
// rpp-performancetests times
// 32 224x224 images - max/min/avg of 100 runs - 0.001173        0.000957        0.001004
// 32 3840x2160 images - max/min/avg of 100 runs - 0.044470        0.001213        0.023843

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










// 4 - Replicate OpenVX style (PLN1 only)

// End to end time
// 2 224x224 images GPU Time - BatchPD : 0.000871s
// 32 224x224 images GPU Time - BatchPD : 0.001077s
// 32 3840x2160 images GPU Time - BatchPD : 0.004999s

// Profiler time
// 2 224x224 images Profiler Time - 0.01103ms (0.00001103s)
// 32 224x224 images Profiler Time - 0.038652ms (0.000038652s)
// 32 3840x2160 images Profiler Time - 1.898541ms (0.001898541s)

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
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_x > width[id_z]) || (id_y > height[id_z]))
    {
        return;
    }

    uint srcIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x);
    uint dstIdx =  batch_index[id_z] + (id_y * max_width[id_z] + id_x);

    uint2 src = *((uint2 *)(&input[srcIdx]));
    uint2 dst;

    // dst.x = 0xaa000b0a;
    // dst.y = 0xffeeddcc;

    float4 alpha4 = (float4)alpha[id_z];
    float4 beta4 = (float4)beta[id_z];

    dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
    dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

    *((uint2 *)(&output[dstIdx])) = dst;
}

#if defined(STATIC)
RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 7) >> 3;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    // start

    // start = clock();
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
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);
    // end

    return RPP_SUCCESS;
}
#endif