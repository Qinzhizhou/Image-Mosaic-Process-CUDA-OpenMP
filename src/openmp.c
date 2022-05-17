#include "cpu.h"
#include "helper.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
/* To enable OpenMP support in your project you will need to include the OpenMP header file `omp.h`
and enable the compiler to use the OpenMP runtime.
Set 'OpenMP Support' to 'Yes' (for both Debug and Release builds) in Project->Properties->C/C++->Language
Add `_CRT_SECURE_NO_WARNINGS` to 'Preprocessor Definitions' in Project->Properties->C/C++->Preprocessor */

///
/// Algorithm storage
///
Image cpu_input_image;
Image cpu_output_image;
unsigned int cpu_TILES_X, cpu_TILES_Y;
unsigned long long* cpu_mosaic_sum;
unsigned char* cpu_mosaic_value;

///
/// Implementation



void openmp_begin(const Image* input_image) {
    cpu_TILES_X = input_image->width / TILE_SIZE;
    cpu_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    cpu_mosaic_sum = (unsigned long long*)malloc(cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    cpu_mosaic_value = (unsigned char*)malloc(cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    cpu_input_image = *input_image;
    cpu_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(cpu_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    cpu_output_image = *input_image;
    cpu_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    
}

// 图像数据以从左到右、从上到下的像素顺序存储。每个像素由 3 个 unsigned char，代表 3 个颜色通道红色/绿色/蓝色。在整个过程中，这 3 个通道必须独立处理。 
//在此阶段，加载每个图块的像素并将它们的像素相加以产生 3 个颜色通道中的每一个的总和。
//方法 cpu_stage1() 提供了算法这一阶段的单线程实现。方法 validate_tile_sum(...) 和 skip_tile_sum(...) 可用于验证或跳过此阶段的输出。但是，这两个函数都需要指向与 CPU 参考实现的布局和顺序相匹配的主机内存的指针。



void openmp_stage1() {
    // Reset sum memory to 0
    memset(cpu_mosaic_sum, 0, cpu_TILES_X * cpu_TILES_Y * cpu_input_image.channels * sizeof(unsigned long long));

    // Sum pixel data within each tile
    int t_x, t_y, p_x, p_y, tile_index, tile_offset, ch;  // looper index 
    //#pragma omp parallel for  // reduction(+ : cpu_mosaic_sum ? )
    for (t_x = 0; t_x < cpu_TILES_X; ++t_x) {// from left to right tiles(outer loops)
        for (t_y = 0; t_y < cpu_TILES_Y; ++t_y)
        { // from top to bottom tiles(inner loops)
            #pragma omp critical
            tile_index = (t_y * cpu_TILES_X + t_x) * cpu_input_image.channels; // tile_index
            tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cpu_input_image.channels;
            for (p_x = 0; p_x < TILE_SIZE; ++p_x)  // pixel_value 
                for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    
                    // For each colour channel
                    const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x) * cpu_input_image.channels;
                    
                    // #pragma omp parallel for private(ch) 
                    for (ch = 0; ch < cpu_input_image.channels; ++ch) {
                    // Load pixel
                        const unsigned char pixel = cpu_input_image.data[tile_offset + pixel_offset + ch];
                        
                        #pragma omp critical
                        cpu_mosaic_sum[tile_index + ch] += pixel;
                    }
                }
            



        }
    }
            // For each pixel within the tile
           
            
  
    
#ifdef VALIDATION
    validate_tile_sum(&cpu_input_image, cpu_mosaic_sum);
#endif
}

void openmp_stage2(unsigned char* output_global_average) {
    // Calculate the average of each tile, and sum these to produce a whole image average.
    unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    for (unsigned int t = 0; t < cpu_TILES_X * cpu_TILES_Y; ++t) {
        for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
            cpu_mosaic_value[t * cpu_input_image.channels + ch] = (unsigned char)(cpu_mosaic_sum[t * cpu_input_image.channels + ch] / TILE_PIXELS);  // Integer division is fine here
            whole_image_sum[ch] += cpu_mosaic_value[t * cpu_input_image.channels + ch];
        }
    }
    // Reduce the whole image sum to whole image average for the return value
    for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (cpu_TILES_X * cpu_TILES_Y));
    }
#ifdef VALIDATION
    validate_compact_mosaic(cpu_TILES_X, cpu_TILES_Y, cpu_mosaic_sum, cpu_mosaic_value, output_global_average);
#endif
}
void openmp_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    // For each tile
    for (unsigned int t_x = 0; t_x < cpu_TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * cpu_TILES_X + t_x) * cpu_input_image.channels;
            const unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cpu_input_image.channels;
            
            // For each pixel within the tile
            for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x) * cpu_input_image.channels;
                    // Copy whole pixel
                    memcpy(cpu_output_image.data + tile_offset + pixel_offset, cpu_mosaic_value + tile_index, cpu_input_image.channels);
                }
            }
        }
    }
#ifdef VALIDATION
    validate_broadcast(&cpu_input_image, cpu_mosaic_value, &cpu_output_image);
#endif
}
void openmp_end(Image* output_image) {
    // Store return value
    output_image->width = cpu_output_image.width;
    output_image->height = cpu_output_image.height;
    output_image->channels = cpu_output_image.channels;
    memcpy(output_image->data, cpu_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(cpu_output_image.data);
    free(cpu_input_image.data);
    free(cpu_mosaic_value);
    free(cpu_mosaic_sum);
}