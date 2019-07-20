#include <rppi_filter_operations.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_filter_operations.hpp" 
 
// ----------------------------------------
// Host bilateral_filter functions calls 
// ----------------------------------------


RppStatus
rppi_bilateral_filter_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, filterSize);
 	 validate_float_range( 0, 20, sigmaI);
 	 validate_float_range( 0, 20, sigmaS);
	 bilateral_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			filterSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 1);
	return RPP_SUCCESS;
}

RppStatus
rppi_bilateral_filter_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, filterSize);
 	 validate_float_range( 0, 20, sigmaI);
 	 validate_float_range( 0, 20, sigmaS);
	 bilateral_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			filterSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 3);
	return RPP_SUCCESS;
}

RppStatus
rppi_bilateral_filter_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS)
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, filterSize);
 	 validate_float_range( 0, 20, sigmaI);
 	 validate_float_range( 0, 20, sigmaS);
	 bilateral_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), 
			srcSize,
			static_cast<Rpp8u*>(dstPtr), 
			filterSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PACKED, 3);
	return RPP_SUCCESS;
}
 
// ----------------------------------------
// Host box_filter functions calls 
// ----------------------------------------



RppStatus
rppi_box_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
    box_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PLANAR, 1);
    return RPP_SUCCESS;
}

RppStatus
rppi_box_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
    box_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PLANAR, 3);
    return RPP_SUCCESS;
}

RppStatus
rppi_box_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize)
{
    box_filter_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr),
                     kernelSize,
                     RPPI_CHN_PACKED, 3);
    return RPP_SUCCESS;
}
 
// ----------------------------------------
// GPU bilateral_filter functions  calls 
// ----------------------------------------


RppStatus
rppi_bilateral_filter_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, filterSize);
 	 validate_float_range( 0, 20, sigmaI);
 	 validate_float_range( 0, 20, sigmaS);

#ifdef OCL_COMPILE
 	 {
 	 bilateral_filter_cl(static_cast<cl_mem>(srcPtr1), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			filterSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 1,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_bilateral_filter_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, filterSize);
 	 validate_float_range( 0, 20, sigmaI);
 	 validate_float_range( 0, 20, sigmaS);

#ifdef OCL_COMPILE
 	 {
 	 bilateral_filter_cl(static_cast<cl_mem>(srcPtr1), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			filterSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PLANAR, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}

RppStatus
rppi_bilateral_filter_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS, RppHandle_t rppHandle) 
{

 	 validate_image_size(srcSize);
 	 validate_int_min(0, filterSize);
 	 validate_float_range( 0, 20, sigmaI);
 	 validate_float_range( 0, 20, sigmaS);

#ifdef OCL_COMPILE
 	 {
 	 bilateral_filter_cl(static_cast<cl_mem>(srcPtr1), 
			srcSize,
			static_cast<cl_mem>(dstPtr), 
			filterSize,
			sigmaI,
			sigmaS,
			RPPI_CHN_PACKED, 3,
			static_cast<cl_command_queue>(rppHandle));
 	 } 
#elif defined (HIP_COMPILE) 
 	 { 
 	 } 
#endif //BACKEND 
		return RPP_SUCCESS;
}