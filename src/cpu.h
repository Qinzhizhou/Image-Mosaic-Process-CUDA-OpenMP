#ifndef __cpu_h__
#define __cpu_h__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The initialisation function for the CPU Mosaic implementation
 * Memory allocation and initialisation occurs here, so that it can be timed separate to the algorithm
 * @param input_image Pointer to a constant struct containing the image to be processed
 */
void cpu_begin(const Image *input_image);
/**
 * Calculate the sum of pixel values for each individual tile
 */
void cpu_stage1();
/**
 * Convert the sums of tile pixel values to averages and calculate the whole image pixel average
 * @param output_global_average This parameter points to a buffer where the whole image pixel average is returned
 */
void cpu_stage2(unsigned char *output_global_average);
/**
 * Broadcast the tile averages out to produce the output mosaic'd image
 */
void cpu_stage3();
/**
 * The cleanup and return function for the CPU Mosaic implementation
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be output, output_image->data is pre-allocated
 */
void cpu_end(Image *output_image);

#ifdef __cplusplus
}
#endif

#endif // __cpu_h__
