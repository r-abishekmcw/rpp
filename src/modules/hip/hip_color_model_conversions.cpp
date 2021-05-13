#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include "kernel/rpp_hip_host_decls.hpp"

/******************** color_temperature ********************/

RppStatus
color_temperature_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32s adjustmentValue, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "color_temperature.cpp", "temperature_planar", vld, vgd, "")(srcPtr,
                                                                                              dstPtr,
                                                                                              srcSize.height,
                                                                                              srcSize.width,
                                                                                              channel,
                                                                                              adjustmentValue);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "color_temperature.cpp", "temperature_packed", vld, vgd, "")(srcPtr,
                                                                                              dstPtr,
                                                                                              srcSize.height,
                                                                                              srcSize.width,
                                                                                              channel,
                                                                                              adjustmentValue);
    }

    return RPP_SUCCESS;
}

RppStatus
color_temperature_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "color_temperature.cpp", "color_temperature_batch", vld, vgd, "")(srcPtr,
                                                                                               dstPtr,
                                                                                               handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
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

#elif defined(STATIC)

    hip_exec_color_temperature_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** vignette ********************/

RppStatus
vignette_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, float stdDev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
vignette_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "vignette.cpp", "vignette_batch", vld, vgd, "")(srcPtr,
                                                                             dstPtr,
                                                                             handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                             handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                             handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                             handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                             handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                             channel,
                                                                             handle.GetInitHandle()->mem.mgpu.inc,
                                                                             plnpkdind);

#elif defined(STATIC)

    hip_exec_vignette_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** channel_extract ********************/

RppStatus
channel_extract_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u extractChannelNumber, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
channel_extract_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(max_width + 31) & ~31, (max_height + 31) & ~31, handle.GetBatchSize()};

    handle.AddKernel("", "", "channel_extract.cpp", "channel_extract_batch", vld, vgd, "")(srcPtr,
                                                                                           dstPtr,
                                                                                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                                           handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                           handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                           handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                           handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                           channel,
                                                                                           handle.GetInitHandle()->mem.mgpu.inc,
                                                                                           plnpkdind);

#elif defined(STATIC)

    hip_exec_channel_extract_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** channel_combine ********************/

RppStatus
channel_combine_hip(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* srcPtr3, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "channel_combine.cpp", "channel_combine_pln", vld, vgd, "")(srcPtr1,
                                                                                             srcPtr2,
                                                                                             srcPtr3,
                                                                                             dstPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, 1};
        handle.AddKernel("", "", "channel_combine.cpp", "channel_combine_pkd", vld, vgd, "")(srcPtr1,
                                                                                             srcPtr2,
                                                                                             srcPtr3,
                                                                                             dstPtr,
                                                                                             srcSize.height,
                                                                                             srcSize.width,
                                                                                             channel);
    }

    return RPP_SUCCESS;
}

RppStatus
channel_combine_hip_batch(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* srcPtr3, Rpp8u* dstPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "channel_combine.cpp", "channel_combine_batch", vld, vgd, "")(srcPtr1,
                                                                                           srcPtr2,
                                                                                           srcPtr3,
                                                                                           dstPtr,
                                                                                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                                           handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                           handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                           handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                           handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                           channel,
                                                                                           handle.GetInitHandle()->mem.mgpu.inc,
                                                                                           plnpkdind);

#elif defined(STATIC)

    hip_exec_channel_combine_batch(srcPtr1, srcPtr2, srcPtr3, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** hueRGB ********************/

RppStatus
hueRGB_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float hue_factor, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float sat = 0.0;
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "hue.cpp", "huergb_pln", vld, vgd, "")(srcPtr,
                                                                       dstPtr,
                                                                       hue_factor,
                                                                       sat,
                                                                       srcSize.height,
                                                                       srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "hue.cpp", "huergb_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        hue_factor,
                                                                        sat,
                                                                        srcSize.height,
                                                                        srcSize.width);
    }

    return RPP_SUCCESS;
}

RppStatus
hueRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "hue.cpp", "hue_batch", vld, vgd, "")(srcPtr,
                                                                   dstPtr,
                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                   plnpkdind);


#elif defined(STATIC)

    hip_exec_hueRGB_batch(srcPtr, dstPtr, handle, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** saturationRGB ********************/

RppStatus
saturationRGB_hip(Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    float hue_factor = 0.0;
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "hue.cpp", "huergb_pln", vld, vgd, "")(srcPtr,
                                                                       dstPtr,
                                                                       hue_factor,
                                                                       sat,
                                                                       srcSize.height,
                                                                       srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "hue.cpp", "huergb_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        hue_factor,
                                                                        sat,
                                                                        srcSize.height,
                                                                        srcSize.width);
    }

    return RPP_SUCCESS;
}

RppStatus
saturationRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "hue.cpp", "saturation_batch", vld, vgd, "")(srcPtr,
                                                                          dstPtr,
                                                                          handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                          handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                          handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                          handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                          handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                          handle.GetInitHandle()->mem.mgpu.inc,
                                                                          plnpkdind);


#elif defined(STATIC)

    hip_exec_saturationRGB_batch(srcPtr, dstPtr, handle, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** look_up_table ********************/

RppStatus
look_up_table_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr,Rpp8u* lutPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
look_up_table_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lutPtr,rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp8u* hipLutPtr;
    hipMalloc(&hipLutPtr, sizeof(Rpp8u) * 256 * channel * handle.GetBatchSize());
    hipMemcpy(hipLutPtr, lutPtr, sizeof(Rpp8u) * 256 * channel * handle.GetBatchSize(), hipMemcpyHostToDevice);
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "look_up_table.cpp", "look_up_table_batch", vld, vgd, "")(srcPtr,
                                                                                       dstPtr,
                                                                                       hipLutPtr,
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

    hipFree(&hipLutPtr);


#elif defined(STATIC)

    hip_exec_look_up_table_batch(srcPtr, dstPtr, hipLutPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** tensor_look_up_table ********************/

RppStatus
tensor_look_up_table_hip(Rpp32u tensorDimension, Rpp32u* tensorDimensionValues, Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lutPtr, rpp::Handle& handle)
{
    Rpp8u* hipLutPtr;
    hipMalloc(&hipLutPtr, sizeof(Rpp8u) * 256);
    hipMemcpy(hipLutPtr, lutPtr, sizeof(Rpp8u) * 256, hipMemcpyHostToDevice);
    size_t gDim3[3];
    if(tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if(tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for(int i = 2 ; i < tensorDimension ; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1,dim2,dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];

#if defined (HIPRTC)

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cpp", "tensor_look_up_table", vld, vgd, "")(tensorDimension,
                                                                                 srcPtr,
                                                                                 dstPtr,
                                                                                 dim1,
                                                                                 dim2,
                                                                                 dim3,
                                                                                 hipLutPtr);

#elif defined(STATIC)

    hip_exec_tensor_look_up_table_batch(tensorDimension, srcPtr, dstPtr, hipLutPtr, handle, dim1, dim2, dim3);

#endif

    return RPP_SUCCESS;
}