#include "cuda.cuh"

#include <cstring>

#include "helper.h"

///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;

// device copy of image's width and height
__device__ int d_width;
__device__ int d_height;
// device copy of nums of TILES in x-direction and y-direction
__device__ int d_TILES_X;
__device__ int d_TILES_Y;

//defien the function globally
__global__ void calMosaicSum(unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum);

// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference
    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;
    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // zhou:  Allocate device copy of width and height
    CUDA_CALL(cudaMemcpyToSymbol(d_width,(void*)&input_image->width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(d_height,(void*)&input_image->height, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(d_TILES_X,(void*)&cuda_TILES_X, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(d_TILES_Y,(void*)&cuda_TILES_Y, sizeof(int)));
    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));
    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));
    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));
}

// zhou: a global function for calculating mosaic sum

__global__ void calMosaicSum(unsigned char* d_input_image_data, unsigned long long*d_mosaic_sum){
    // use the indexes of threads to obtain tile_index and tile_offset. Here,
    // threadIdx.x is similar with t_x in cpu_stage1()
    // blockIdx.x is similar with t_y in cpu_stage1()
    // blockDim.x is similar with cpu_TILES_X in cpu_stage1()
    
    // Therefor, one thread in used for control one tile_index
    const unsigned int tile_index = (blockDim.x * blockIdx.x + threadIdx.x) * CHANNELS; // Tile_id
    const unsigned int tile_offset = (blockDim.x * blockIdx.x * TILE_SIZE * TILE_SIZE +  threadIdx.x * TILE_SIZE) * CHANNELS;// Offset
    
    // skip the threads which are out of index 
    if (tile_index >= d_TILES_X * d_TILES_Y * CHANNELS)
        return;
    
    // For each pixel within the tile
    for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
        for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
            // For each colour channel
            const unsigned int pixel_offset = (p_y * d_width + p_x) * CHANNELS;
            for (int ch = 0; ch < CHANNELS; ++ch) {
                // Load pixel
                const unsigned char pixel = d_input_image_data[tile_offset + pixel_offset + ch];
                d_mosaic_sum[tile_index + ch] += pixel;
            }
        }
    }
    return;
}

void cuda_stage1() { // 529.995 seconds in CPU 293.218 in GPU
    // initialize sizes of block and gird
    dim3 block = cuda_TILES_X; // 
    dim3 grid = ((cuda_TILES_X * cuda_TILES_Y + block.x - 1) / block.x);
    
    // Kernal function 
    calMosaicSum<<<block, grid>>>(d_input_image_data, d_mosaic_sum); // <<< 32, (32*32 + x - 1)/x 

#ifdef VALIDATION
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // delcare and allocate memory of test_mosaic_sum for validation
    unsigned long long* test_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
    
    // copy the data back to host
    CUDA_CALL(cudaMemcpy(test_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
   validate_tile_sum(&cuda_input_image, test_mosaic_sum);
   #endif
}


void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);
    // Calculate the average of each tile, and sum these to produce a whole image average.
  


#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, mosaic_value, output_global_average);
#endif    
}






void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(cuda_input_image, compact_mosaic, d_output_image);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_broadcast(&input_image, mosaic_value, &output_image);
#endif    
}


void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
}
