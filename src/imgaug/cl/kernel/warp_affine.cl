__kernel void warp_affine_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            __global  float* affine,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int minX,
                            const unsigned int minY,
                            const unsigned int channel
)
{
   int id_x = get_global_id(0);
   int id_y = get_global_id(1);
   int id_z = get_global_id(2);
   float affine_inv[6];
   float det; //for Deteminent
   det = (affine[0] * affine [4])  - (affine[1] * affine[3]);
   affine_inv[0] = affine[4]/ det;
   affine_inv[1] = (- 1 * affine[1])/ det;
   affine_inv[2] = -1 * affine[2];
   affine_inv[3] = (-1 * affine[3]) /det ;
   affine_inv[4] = affine[0]/det;
   affine_inv[5] = -1 * affine[5];

   int xc = id_x - dest_width/2;
   int yc = id_y - dest_height/2;

   int k ;
   int l ;

   k = (int)((affine[0] * xc )+ (affine[1] * yc)) + affine[2];
   l = (int)((affine[3] * xc) + (affine[4] * yc)) + affine[5];
   k = k + source_width/2;
   l = l + source_height/2;
   if (l < source_height && l >=0 && k < source_width && k >=0 )
   dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] =
                           srcPtr[(id_z * source_height * source_width) + (l * source_width) + k];
   else
   dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = 0;

}

__kernel void warp_affine_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            __global  float* affine,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int minX,
                            const unsigned int minY,
                            const unsigned int channel
)
{

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    int k = (Rpp32s)((affine[0] * id_y) + (affine[1] * id_x) + (affine[2] * 1));
    int l = (Rpp32s)((affine[3] * id_y) + (affine[4] * id_x) + (affine[5] * 1));
    k -= (Rpp32s)minX;
    l -= (Rpp32s)minY;

    dstPtr[id_z + (channel * k * dstSize.width) + (channel * l)] =
                             srcPtr[id_z + (channel * id_y * srcSize.width) + (channel * id_x)];

}
