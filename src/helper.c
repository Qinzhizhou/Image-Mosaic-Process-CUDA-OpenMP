#include "helper.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"


int skip_tile_sum_used = -1;
void validate_tile_sum(const Image* input_image, unsigned long long* test_mosaic_sum) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // Allocate and generate our own internal mosaic sum
    unsigned long long*  mosaic_sum = (unsigned long long*)malloc(TILES_X * TILES_Y * input_image->channels * sizeof(unsigned long long));
    skip_tile_sum(input_image, mosaic_sum);
    skip_tile_sum_used--;
    // Validate and report result
    unsigned int bad_tiles = 0;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * TILES_X + t_x) * input_image->channels;
            for (int ch = 0; ch < input_image->channels; ++ch) {
                if (mosaic_sum[tile_index + ch] != test_mosaic_sum[tile_index + ch]) {
                    bad_tiles++;
                    break;
                }
            }
        }
    }
    if (bad_tiles) {
        fprintf(stderr, "validate_tile_sum() found %d/%u tiles contain atleast 1 invalid colour sum.\n", bad_tiles, TILES_X * TILES_Y);
    } else {
        fprintf(stderr, "validate_tile_sum() found no errors!\n");
    }
    // Release internal mosaic sum
    free(mosaic_sum);
}
void skip_tile_sum(const Image* input_image, unsigned long long* mosaic_sum) {
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    // Reset sum memory to 0
    memset(mosaic_sum, 0, TILES_X * TILES_Y * input_image->channels * sizeof(unsigned long long));
    // Sum pixel data within each tile
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * TILES_X + t_x) * input_image->channels;
            const unsigned int tile_offset = (t_y * TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * input_image->channels;
            // For each pixel within the tile
            for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    // For each colour channel
                    const unsigned int pixel_offset = (p_y * input_image->width + p_x) * input_image->channels;
                    for (int ch = 0; ch < input_image->channels; ++ch) {
                        // Load pixel
                        const unsigned char pixel = input_image->data[tile_offset + pixel_offset + ch];
                        mosaic_sum[tile_index + ch] += pixel;
                    }
                }
            }
        }
    }
    skip_tile_sum_used++;
}

int skip_compact_mosaic_used = -1;
void validate_compact_mosaic(unsigned int TILES_X, unsigned int TILES_Y,
    unsigned long long* mosaic_sum, unsigned char* test_compact_mosaic, unsigned char* test_global_pixel_average) {
    // Allocate our own internal compact_mosaic
    unsigned char* compact_mosaic = (unsigned char*)malloc(TILES_X * TILES_Y * CHANNELS * sizeof(unsigned char));
    unsigned char global_pixel_average[4] = { 0,0,0,0 };
    skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum,  compact_mosaic, global_pixel_average);
    skip_compact_mosaic_used--;
    // Validate and report result
    unsigned int bad_tiles = 0;
    for (unsigned int t = 0; t < TILES_X * TILES_Y; ++t) {
        for (int ch = 0; ch < CHANNELS; ++ch) {
            if (compact_mosaic[t * CHANNELS + ch] != test_compact_mosaic[t * CHANNELS + ch]) {
                ++bad_tiles;
                break;
            }
        }
    }
    unsigned int bad_global_averages = 0;
    for (int ch = 0; ch < CHANNELS; ++ch) {
        if (global_pixel_average[ch] != test_global_pixel_average[ch])
            ++bad_global_averages;
    }
    if (bad_tiles) {
        fprintf(stderr, "validate_compact_mosaic() found %u/%u incorrect mosaic colours.\n", bad_tiles, TILES_X * TILES_Y);
    }
    if (bad_global_averages) {
        fprintf(stderr, "validate_compact_mosaic() found %u/%d channels of the global pixel average were incorrect.\n", bad_global_averages, CHANNELS);
    }
    if (!bad_tiles && !bad_global_averages) {
        fprintf(stderr, "validate_compact_mosaic() found no errors!\n");
    }
    // Release internal buffers
    free(compact_mosaic);
}
void skip_compact_mosaic(unsigned int TILES_X, unsigned int TILES_Y,
    unsigned long long* mosaic_sum, unsigned char* compact_mosaic, unsigned char* global_pixel_average) {
    unsigned long long whole_image_sum[4] = { 0,0,0,0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    // Calculate the average of each tile, and sum these to produce a whole image average.
    for (unsigned int t = 0; t < TILES_X * TILES_Y; ++t) {
        for (int ch = 0; ch < CHANNELS; ++ch) {
            compact_mosaic[t * CHANNELS + ch] = (unsigned char)(mosaic_sum[t * CHANNELS + ch] / TILE_PIXELS);  // Integer division is fine here
            whole_image_sum[ch] += compact_mosaic[t * CHANNELS + ch];
        }
    }
    // Reduce the whole image sum to whole image average for the return value
    for (int ch = 0; ch < CHANNELS; ++ch) {
        global_pixel_average[ch] = (unsigned char)(whole_image_sum[ch] / (TILES_X * TILES_Y));
    }
    skip_compact_mosaic_used++;
}
int skip_broadcast_used = -1;
void validate_broadcast(const Image* input_image, unsigned char* compact_mosaic, Image* test_output_image) {
    // Allocate, copy and generate our own internal output image
    Image output_image = *input_image;
    output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    skip_broadcast(input_image, compact_mosaic, &output_image);
    skip_broadcast_used--;
    // Validate and report result
    unsigned int bad_pixels = 0;
    for (int t = 0; t < input_image->width * input_image->height; ++t) {
        for (int ch = 0; ch < input_image->channels; ++ch) {
            if (output_image.data[t * input_image->channels + ch] != test_output_image->data[t * input_image->channels + ch]) {
                ++bad_pixels;
                break;
            }
        }
    }
    if (bad_pixels) {
        fprintf(stderr, "validate_broadcast() found %d/%u incorrect final pixels.\n", bad_pixels, input_image->width * input_image->height);
    } else {
        fprintf(stderr, "validate_broadcast() found no errors!\n");
    }
    // Release internal buffers
    free(output_image.data);
}
void skip_broadcast(const Image* input_image, unsigned char* compact_mosaic, Image* output_image) {
    output_image->channels = input_image->channels;
    output_image->width = input_image->width;
    output_image->height = input_image->height;
    const unsigned int TILES_X = input_image->width / TILE_SIZE;
    const unsigned int TILES_Y = input_image->height / TILE_SIZE;
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * TILES_X + t_x) * input_image->channels;
            const unsigned int tile_offset = (t_y * TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * input_image->channels;

            // For each pixel within the tile
            for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    const unsigned int pixel_offset = (p_y * input_image->width + p_x) * input_image->channels;
                    // Copy whole pixel
                    memcpy(output_image->data + tile_offset + pixel_offset, compact_mosaic + tile_index, input_image->channels);
                }
            }
        }
    }
    skip_broadcast_used++;
}

int getSkipUsed() {
    return skip_tile_sum_used + skip_compact_mosaic_used + skip_broadcast_used;
}
int getStage1SkipUsed() {
    return skip_tile_sum_used;
}
int getStage2SkipUsed() {
    return skip_compact_mosaic_used;
}
int getStage3SkipUsed() {
    return skip_broadcast_used;
}
