#ifndef __cuda_cuh__
#define __cuda_cuh__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

/**
 * The initialisation function for the CUDA Mosaic implementation
 * Memory allocation and initialisation occurs here, so that it can be timed separate to the algorithm
 * @param input_image Pointer to a constant struct containing the image to be processed
 */
void cuda_begin(const Image *input_image);
/**
 * Calculate the sum of pixel values for each individual tile
 */
void cuda_stage1();
/**
 * Convert the sums of tile pixel values to averages and calculate the whole image pixel average
 * @param output_global_average This parameter points to a buffer where the whole image pixel average is returned
 */
void cuda_stage2(unsigned char* output_global_average);
/**
 * Broadcast the tile averages out to produce the output mosaic'd image
 */
void cuda_stage3();
/**
 * The cleanup and return function for the CUDA Mosaic implementation
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be output, output_image->data is pre-allocated
 */
void cuda_end(Image *output_image);


/**
 * Error check function for safe CUDA API calling
 * Wrap all calls to CUDA API functions with CUDA_CALL() to catch errors on failure
 * e.g. CUDA_CALL(cudaFree(myPtr));
 * CUDA_CHECk() can also be used to perform error checking after kernel launches and async methods
 * e.g. CUDA_CHECK()
 */
#if defined(_DEBUG) || defined(D_DEBUG)
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK(location) { gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__); }
#else
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK(location) { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        if (line >= 0) {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        } else {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        }
        exit(EXIT_FAILURE);
    }
}

#endif // __cuda_cuh__
