#include <hip/hip_runtime.h>

#if defined(STATIC)
#include "rpp_hip_host_decls.hpp"
#endif

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

typedef struct d_float8 {
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

// 1 - WORKING - Original

// End to end time [ Format - Vega10 time(s) | Vega 20 time(s) | Type ]

// 8 224x224 images GPU Time - BatchPD : 0.000617s | 0.000319s | PKD3
// 8 224x224 images GPU Time - BatchPD : 0.000612s | 0.00034s | PLN3
// 8 224x224 images GPU Time - BatchPD : 0.000491s | 0.000321s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : 0.023746s | 0.017946s | PKD3
// 32 3840x2160 images GPU Time - BatchPD : 0.024411s | 0.018769s | PLN3
// 32 3840x2160 images GPU Time - BatchPD : 0.014549s | 0.011078s | PLN1

// Profiler time

// 8 224x224 images GPU Time - BatchPD : 0.042507ms = 0.000042507s | 0.00001946s | PKD3
// 8 224x224 images GPU Time - BatchPD : 0.044719ms = 0.000044719s | 0.000019907s | PLN3
// 8 224x224 images GPU Time - BatchPD : 0.027146ms = 0.000027146s | 0.00001311s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : 11.28056ms = 0.01128056s | ? | PKD3
// 32 3840x2160 images GPU Time - BatchPD : 11.910476ms = 0.011910476s | ? | PLN3
// 32 3840x2160 images GPU Time - BatchPD : 7.001131ms = 0.007001131s | ? | PLN1

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








// 7 - WORKING - Replicate OpenVX style (PLN1, PLN3, PKD3), reduce dimensions [Should fail for non-multiple of 8]

// End to end time

// 8 224x224 images GPU Time - BatchPD : 0.000552s PKD3
// 8 224x224 images GPU Time - BatchPD : 0.000510s PLN3
// 8 224x224 images GPU Time - BatchPD : 0.000449s PLN1

// 32 3840x2160 images GPU Time - BatchPD : 0.009684s PKD3
// 32 3840x2160 images GPU Time - BatchPD : 0.009622s PLN3
// 32 3840x2160 images GPU Time - BatchPD : 0.003894s PLN1

// Profiler time

// 8 224x224 images GPU Time - BatchPD : 0.024407ms = 0.000024407s PKD3
// 8 224x224 images GPU Time - BatchPD : 0.022826ms = 0.000022826s PLN3
// 8 224x224 images GPU Time - BatchPD : 0.011283ms = 0.000011283s PLN1

// 32 3840x2160 images GPU Time - BatchPD : 4.557237ms = 0.004557237s PKD3
// 32 3840x2160 images GPU Time - BatchPD : 4.518855ms = 0.004518855s PLN3
// 32 3840x2160 images GPU Time - BatchPD : 1.666841ms = 0.001666841s PLN1

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












// 10 - WORKING - Vectorization | Remove unnecessary arguments | Replicate OpenVX style (PLN1, PLN3, PKD3), Reduce dimensions

// End to end time [ Format - Vega10 time(s) | Vega 20 time(s) | Type ]

// 8 224x224 images GPU Time - BatchPD : 0.000552s | 0.000301s | PKD3
// 8 224x224 images GPU Time - BatchPD : 0.00051s | 0.000289s | PLN3
// 8 224x224 images GPU Time - BatchPD : 0.000449s | 0.000236s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : 0.009684s | 0.004431s | PKD3
// 32 3840x2160 images GPU Time - BatchPD : 0.009622s | 0.004509s | PLN3
// 32 3840x2160 images GPU Time - BatchPD : 0.003894s | 0.001819s | PLN1

// Profiler time

// 8 224x224 images GPU Time - BatchPD : = 0.000024407s | 0.000005158s | PKD3
// 8 224x224 images GPU Time - BatchPD : = 0.000022826s | 0.0000052s | PLN3
// 8 224x224 images GPU Time - BatchPD : = 0.000011283s | 0.000003299s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : = 0.004557237s | ? | PKD3
// 32 3840x2160 images GPU Time - BatchPD : = 0.004518855s | ? | PLN3
// 32 3840x2160 images GPU Time - BatchPD : = 0.001666841s | ? | PLN1

// extern "C" __global__ void brightness_batch(unsigned char *srcPtr,
//                                             unsigned char *dstPtr,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned long long *batch_index)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

//     uint srcIdx = batch_index[id_y] + id_x;
//     uint dstIdx = batch_index[id_y] + id_x;

//     uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
//     uint2 dst;

//     float4 alpha4 = (float4)alpha[id_y];
//     float4 beta4 = (float4)beta[id_y];

//     dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
//     dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

//     *((uint2 *)(&dstPtr[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int max_image_size = max_height * max_width * channel;

//     int localThreads_x = 256;
//     int localThreads_y = 1;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_image_size + 7) >> 3;
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
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex);

//     return RPP_SUCCESS;
// }
// #endif
















// 11 - WORKING - Scale back to id_x, id_y, id_z | Separate PLN/PKD | Vectorization | Remove unnecessary arguments | Replicate OpenVX style (PLN1, PLN3, PKD3)

// End to end time [ Format - Vega10 time(s) | Vega 20 time(s) | Type ]

// 8 224x224 images GPU Time - BatchPD : 0.000546s | 0.00033s | PKD3
// 8 224x224 images GPU Time - BatchPD : 0.000569s | 0.000344s | PLN3
// 8 224x224 images GPU Time - BatchPD : 0.00046s | 0.000296s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : 0.010457s | 0.005449s | PKD3
// 32 3840x2160 images GPU Time - BatchPD : 0.010458s | 0.005569s | PLN3
// 32 3840x2160 images GPU Time - BatchPD : 0.004218s | 0.002196s | PLN1

// Profiler time

// 8 224x224 images GPU Time - BatchPD : = 0.000024583s | 0.000007067s | PKD3
// 8 224x224 images GPU Time - BatchPD : = 0.000028493s | 0.000009019s | PLN3
// 8 224x224 images GPU Time - BatchPD : = 0.00001566s | 0.000005657s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : = 0.0049321s | ? | PKD3
// 32 3840x2160 images GPU Time - BatchPD : = 0.004909427s | ? | PLN3
// 32 3840x2160 images GPU Time - BatchPD : = 0.001806313s | ? | PLN1

// extern "C" __global__ void brightness_pkd_batch(uchar *srcPtr,
//                                                 uchar *dstPtr,
//                                                 float *alpha,
//                                                 float *beta,
//                                                 uint *srcWidth,
//                                                 uint *srcHeight,
//                                                 uint srcStride,
//                                                 uint dstStride,
//                                                 unsigned long long *batch_index)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     int srcElementsInRow = srcWidth[id_z] * 3;

//     if ((id_y >= srcHeight[id_z]) || (id_x >= srcElementsInRow))
//     {
//         return;
//     }

//     uint srcIdx = batch_index[id_z] + id_y * srcStride + id_x;
//     uint dstIdx = batch_index[id_z] + id_y * dstStride + id_x;

//     if (id_x + 7 >= srcElementsInRow)
//     {
//         int diff = srcElementsInRow - id_x;
//         for (int x = 0; x < diff; x++)
//         {
//             dstPtr[dstIdx] = brighten_fmaf((float)srcPtr[srcIdx], alpha[id_z], beta[id_z]);
//             srcIdx++;
//             dstIdx++;
//         }

//         return;
//     }

//     uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
//     uint2 dst;

//     float4 alpha4 = (float4)alpha[id_z];
//     float4 beta4 = (float4)beta[id_z];

//     dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
//     dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

//     *((uint2 *)(&dstPtr[dstIdx])) = dst;
// }

// extern "C" __global__ void brightness_pln_batch(uchar *srcPtr,
//                                                 uchar *dstPtr,
//                                                 float *alpha,
//                                                 float *beta,
//                                                 uint *srcWidth,
//                                                 uint *srcHeight,
//                                                 uint srcStride,
//                                                 uint dstStride,
//                                                 uint srcImageSizeMax,
//                                                 uint dstImageSizeMax,
//                                                 uint channel,
//                                                 unsigned long long *batch_index)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     int currentChannel = id_z % channel;
//     id_z /= channel;

//     int srcElementsInRow = srcWidth[id_z];

//     if ((id_y >= srcHeight[id_z]) || (id_x >= srcElementsInRow))
//     {
//         return;
//     }

//     uint srcIdx = batch_index[id_z] + (srcImageSizeMax * currentChannel) + id_y * srcStride + id_x;
//     uint dstIdx = batch_index[id_z] + (dstImageSizeMax * currentChannel) + id_y * dstStride + id_x;

//     if (id_x + 7 >= srcElementsInRow)
//     {
//         int diff = srcElementsInRow - id_x;
//         for (int x = 0; x < diff; x++)
//         {
//             dstPtr[dstIdx] = brighten_fmaf((float)srcPtr[srcIdx], alpha[id_z], beta[id_z]);
//             srcIdx++;
//             dstIdx++;
//         }

//         return;
//     }

//     uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
//     uint2 dst;

//     float4 alpha4 = (float4)alpha[id_z];
//     float4 beta4 = (float4)beta[id_z];

//     dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
//     dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

//     *((uint2 *)(&dstPtr[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     if (plnpkdind == 3)
//     {
//         uint srcStride = max_width * channel;
//         uint dstStride = max_width * channel;

//         int localThreads_x = 16;
//         int localThreads_y = 16;
//         int localThreads_z = 1;
//         int globalThreads_x = (int)(max_width * 3 + 7) >> 3;
//         int globalThreads_y = max_height;
//         int globalThreads_z = handle.GetBatchSize();

//         hipLaunchKernelGGL(brightness_pkd_batch,
//                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                            dim3(localThreads_x, localThreads_y, localThreads_z),
//                            0,
//                            handle.GetStream(),
//                            srcPtr,
//                            dstPtr,
//                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                            handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                            handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                            srcStride,
//                            dstStride,
//                            handle.GetInitHandle()->mem.mgpu.srcBatchIndex);
//     }
//     else
//     {
//         uint srcStride = max_width;
//         uint dstStride = max_width;

//         uint srcImageSizeMax = max_width * max_height;
//         uint dstImageSizeMax = max_width * max_height;

//         int localThreads_x = 16;
//         int localThreads_y = 16;
//         int localThreads_z = 1;
//         int globalThreads_x = (max_width + 7) >> 3;
//         int globalThreads_y = max_height;
//         int globalThreads_z = channel * handle.GetBatchSize();

//         hipLaunchKernelGGL(brightness_pln_batch,
//                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
//                            dim3(localThreads_x, localThreads_y, localThreads_z),
//                            0,
//                            handle.GetStream(),
//                            srcPtr,
//                            dstPtr,
//                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
//                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
//                            handle.GetInitHandle()->mem.mgpu.srcSize.width,
//                            handle.GetInitHandle()->mem.mgpu.srcSize.height,
//                            srcStride,
//                            dstStride,
//                            srcImageSizeMax,
//                            dstImageSizeMax,
//                            channel,
//                            handle.GetInitHandle()->mem.mgpu.srcBatchIndex);
//     }

//     return RPP_SUCCESS;
// }
// #endif
















// 12 - WORKING - fmaf | Remove unnecessary arguments | Replicate OpenVX style (PLN1, PLN3, PKD3), Reduce dimensions

// End to end time [ Format - Vega10 time(s) | Vega 20 time(s) | Type ]

// 8 224x224 images GPU Time - BatchPD : 0.000609s | --s | PKD3
// 8 224x224 images GPU Time - BatchPD : 0.000555s | --s | PLN3
// 8 224x224 images GPU Time - BatchPD : 0.000448s | --s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : 0.009687s | --s | PKD3
// 32 3840x2160 images GPU Time - BatchPD : 0.009727s | --s | PLN3
// 32 3840x2160 images GPU Time - BatchPD : 0.003947s | --s | PLN1

// Profiler time

// 8 224x224 images GPU Time - BatchPD : = 0.000039028s | --s | PKD3
// 8 224x224 images GPU Time - BatchPD : = 0.000038271s | --s | PLN3
// 8 224x224 images GPU Time - BatchPD : = 0.000010519s | --s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : = 0.004577556s | ? | PKD3
// 32 3840x2160 images GPU Time - BatchPD : = 0.004539694s | ? | PLN3
// 32 3840x2160 images GPU Time - BatchPD : = 0.001687579s | ? | PLN1

// extern "C" __global__ void brightness_batch(unsigned char *srcPtr,
//                                             unsigned char *dstPtr,
//                                             float *alpha,
//                                             float *beta,
//                                             unsigned long long *batch_index)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

//     uint srcIdx = batch_index[id_y] + id_x;
//     uint dstIdx = batch_index[id_y] + id_x;

//     uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
//     uint2 dst;

//     float4 f4_1, f4_2;
//     f4_1 = rpp_hip_unpack(src.x);
//     f4_2 = rpp_hip_unpack(src.y);

//     f4_1.x = fmaf(alpha[id_y], f4_1.x, beta[id_y]);
//     f4_1.y = fmaf(alpha[id_y], f4_1.y, beta[id_y]);
//     f4_1.z = fmaf(alpha[id_y], f4_1.z, beta[id_y]);
//     f4_1.w = fmaf(alpha[id_y], f4_1.w, beta[id_y]);

//     f4_2.x = fmaf(alpha[id_y], f4_2.x, beta[id_y]);
//     f4_2.y = fmaf(alpha[id_y], f4_2.y, beta[id_y]);
//     f4_2.z = fmaf(alpha[id_y], f4_2.z, beta[id_y]);
//     f4_2.w = fmaf(alpha[id_y], f4_2.w, beta[id_y]);

//     dst.x = rpp_hip_pack(f4_1);
//     dst.y = rpp_hip_pack(f4_2);

//     *((uint2 *)(&dstPtr[dstIdx])) = dst;
// }

// #if defined(STATIC)
// RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
// {
//     int max_image_size = max_height * max_width * channel;

//     int localThreads_x = 256;
//     int localThreads_y = 1;
//     int localThreads_z = 1;
//     int globalThreads_x = (max_image_size + 7) >> 3;
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
//                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex);

//     return RPP_SUCCESS;
// }
// #endif












// 13 - WORKING - Add exit condition | Vectorization | Remove unnecessary arguments | Replicate OpenVX style (PLN1, PLN3, PKD3), Reduce dimensions

// End to end time [ Format - Vega10 time(s) | Vega 20 time(s) | Type ]

// 8 224x224 images GPU Time - BatchPD : 0.000552s | 0.000301s | PKD3
// 8 224x224 images GPU Time - BatchPD : 0.00051s | 0.000289s | PLN3
// 8 224x224 images GPU Time - BatchPD : 0.000449s | 0.000236s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : 0.009684s | 0.004431s | PKD3
// 32 3840x2160 images GPU Time - BatchPD : 0.009622s | 0.004509s | PLN3
// 32 3840x2160 images GPU Time - BatchPD : 0.003894s | 0.001819s | PLN1

// Profiler time

// 8 224x224 images GPU Time - BatchPD : = 0.000024407s | 0.000005158s | PKD3
// 8 224x224 images GPU Time - BatchPD : = 0.000022826s | 0.0000052s | PLN3
// 8 224x224 images GPU Time - BatchPD : = 0.000011283s | 0.000003299s | PLN1

// 32 3840x2160 images GPU Time - BatchPD : = 0.004557237s | ? | PKD3
// 32 3840x2160 images GPU Time - BatchPD : = 0.004518855s | ? | PLN3
// 32 3840x2160 images GPU Time - BatchPD : = 0.001666841s | ? | PLN1

extern "C" __global__ void brightness_batch(unsigned char *srcPtr,
                                            unsigned char *dstPtr,
                                            float *alpha,
                                            float *beta,
                                            unsigned int max_image_size,
                                            unsigned long long *batch_index)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= max_image_size)
    {
        return;
    }

    uint srcIdx = batch_index[id_y] + id_x;
    uint dstIdx = batch_index[id_y] + id_x;

    if (id_x + 7 >= max_image_size)
    {
        int diff = max_image_size - id_x;
        for (int x = 0; x < diff; x++)
        {
            dstPtr[dstIdx] = brighten_fmaf((float)srcPtr[srcIdx], alpha[id_y], beta[id_y]);
            srcIdx++;
            dstIdx++;
        }

        return;
    }

    uint2 src = *((uint2 *)(&srcPtr[srcIdx]));
    uint2 dst;

    float4 alpha4 = (float4)alpha[id_y];
    float4 beta4 = (float4)beta[id_y];

    dst.x = rpp_hip_pack(rpp_hip_unpack(src.x) * alpha4 + beta4);
    dst.y = rpp_hip_pack(rpp_hip_unpack(src.y) * alpha4 + beta4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

#if defined(STATIC)
RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int max_image_size = max_height * max_width * channel;

    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = (max_image_size + 7) >> 3;
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
                       max_image_size,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex);

    return RPP_SUCCESS;
}
#endif