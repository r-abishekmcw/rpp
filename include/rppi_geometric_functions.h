#ifndef RPPI_GEOMETRIC_FUNCTIONS_H
#define RPPI_GEOMETRIC_FUNCTIONS_H


/**
 * \file rppi_geometry_functions.h
 * Image Geometry Transform Primitives.
 */
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * 1 channel 8-bit unsigned image flip function.
 *
 * \param srcPtr \ref source_image_pointer.
 * \param rSrcStep \ref source_image_line_step.
 * \param dstPtr \ref destination_image_pointer.
 * \param rDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
 */



// --------------------
// Flip
// --------------------

// Host function declarations

RppStatus 
rppi_flip_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis);

// Gpu function declarations

RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

RppStatus
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

RppStatus
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

//Resize--------------------------
//GPU--------
RppStatus
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         RppHandle_t rppHandle);

RppStatus
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         RppHandle_t rppHandle);

RppStatus
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         RppHandle_t rppHandle);
//--------GPU

//CPU--------
RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize);

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize);

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize);
//--------CPU
//-----------------------------Resize

//Resize Crop --------------------------
//GPU--------
RppStatus
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  RppHandle_t rppHandle);

RppStatus
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  RppHandle_t rppHandle);

RppStatus
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,  RppHandle_t rppHandle);
//--------GPU

//CPU-------
RppStatus
rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);

RppStatus
rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);

RppStatus
rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);
//-------CPU

//---------------------------Resize Crop

//Rotate-----------------------------------
//GPU--------
RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle);

RppStatus
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle);

RppStatus
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle);
//--------GPU

//CPU--------
RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);
//--------CPU
//-----------------------------------Rotate

#ifdef __cplusplus
}
#endif
#endif /* RPP_FILTERING_FUNCTIONS_H */
