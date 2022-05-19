#include "openmp.h"
#include "helper.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

Image openmp_input_image;
Image openmp_output_image;

unsigned int openmp_TILES_X, openmp_TILES_Y;
unsigned long long* openmp_mosaic_sum; // 储存tile_sum 的地址 
unsigned char* openmp_mosaic_value; // 最终mosaic_value 的地址

// TILE_SIZE = 32,  CHANNELS 3

void openmp_begin(const Image *input_image) {
    openmp_TILES_X = input_image->width / TILE_SIZE;
    openmp_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic 分配内存
    openmp_mosaic_sum = (unsigned long long*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned long long));
    // Allocate buffer for storing the output pixel value of each tile 输出图片的内存
    openmp_mosaic_value = (unsigned char*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned char));
    
    // Allocate copy of input image 输入图片的内存
    openmp_input_image = *input_image;
    openmp_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(openmp_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    
    // Allocate output image 输出图片的内存
    openmp_output_image = *input_image;
    openmp_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    
}
//serial: 55.212
// Openmp 4 and 1: 21.903ms

void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);
    // Reset sum memory to 0
    memset(openmp_mosaic_sum, 0, openmp_TILES_X * openmp_TILES_Y * openmp_input_image.channels * sizeof(unsigned long long));
    // Sum pixel data within each tile

    // Sum pixel data within each tile
    int t_x, t_y, p_x, p_y, ch;  // looper index 
    int tile_index, tile_offset, pixel_offset ;
    char pixel;
    
#pragma omp parallel for collapse(4)  default (none) private(t_x, t_y,  tile_index, tile_offset, pixel_offset, p_x, p_y, ch) shared (openmp_mosaic_sum) schedule(static,12) // reduction( +: openmp_mosaic_sum[tile_index + ch])
    for (t_x = 0; t_x < openmp_TILES_X; ++t_x) 
    for (t_y = 0; t_y < openmp_TILES_Y; ++t_y)
    {
            tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
            
            // all indexes in the cell
            tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;
            
            // For each pixel within the tile
            //#pragma omp parallel for collapse(2) private(p_x, p_y)// shared (openmp_mosaic_sum) schedule(static, 8)
            for (p_x = 0; p_x < TILE_SIZE; ++p_x) 
            for (p_y = 0; p_y < TILE_SIZE; ++p_y) 
            {
                    // For each colour channel
                    pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;
                    for (ch = 0; ch < openmp_input_image.channels; ++ch) 
                    {
                        // Load pixel
                        const unsigned char pixel = openmp_input_image.data[tile_offset + pixel_offset + ch];
//#pragma omp critical
                        openmp_mosaic_sum[tile_index + ch] += pixel;
                    }
                }
            }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_tile_sum(&openmp_input_image, openmp_mosaic_sum);
#endif
}

// seriel 0.141

void openmp_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(openmp_TILES_X, openmp_TILES_Y, openmp_mosaic_sum, compact_mosaic, global_pixel_average);
     // Calculate the average of each tile, and sum these to produce a whole image average.

    unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    int t, ch;
    
    #pragma omp parallel for private(ch) schedule(dynamic) // why not static
    //#pragma omp parallel  for default (none) private(t, ch)
    for (t = 0; t < openmp_TILES_X * openmp_TILES_Y; ++t) {
        for (ch = 0; ch < openmp_input_image.channels; ++ch) {
            openmp_mosaic_value[t * openmp_input_image.channels + ch] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels + ch] / TILE_PIXELS);  //average of each tile
            whole_image_sum[ch] += openmp_mosaic_value[t * openmp_input_image.channels + ch];// sum these to produce a whole image average.
        }
    }
    // Reduce the whole image sum to whole image average for the return value
    //#pragma omp critical
    for (int ch = 0; ch < openmp_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (openmp_TILES_X * openmp_TILES_Y));
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    validate_compact_mosaic(openmp_TILES_X, openmp_TILES_Y, openmp_mosaic_sum, openmp_mosaic_value, output_global_average);
#endif    
}


// serial 154.966
// collapse 
void openmp_stage3() { // 1.665

    int t_x, t_y, p_x, p_y, ch;  // looper index 
    int tile_index, tile_offset, pixel_offset;
    char pixel;

    // #pragma omp parallel  for num_threads(8) collapse(2)  default (none) private(t_x, t_y, tile_index, tile_offset, p_x, p_y, ch) shared (openmp_mosaic_sum) schedule(static, 1)
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);



#pragma omp parallel for default (none) private(t_x, t_y,  tile_index, tile_offset, pixel_offset, p_x, p_y, ch) shared (openmp_mosaic_sum) schedule(static,12)
    for (t_x = 0; t_x < openmp_TILES_X; ++t_x) {
        for (t_y = 0; t_y < openmp_TILES_Y; ++t_y) {
            tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
            tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;
            // For each pixel within the tile
            for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;
                    // Copy whole pixel
                    memcpy(openmp_output_image.data + tile_offset + pixel_offset, openmp_mosaic_value + tile_index, openmp_input_image.channels);
                }
            }
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_broadcast(&openmp_input_image, openmp_mosaic_value, &openmp_output_image);
#endif    
}
void openmp_end(Image *output_image) {
    // Store return value
    output_image->width = openmp_output_image.width;
    output_image->height = openmp_output_image.height;
    output_image->channels = openmp_output_image.channels;
    
    memcpy(output_image->data, openmp_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    
    free(openmp_output_image.data);
    free(openmp_input_image.data);
    free(openmp_mosaic_value);
    free(openmp_mosaic_sum);   
}