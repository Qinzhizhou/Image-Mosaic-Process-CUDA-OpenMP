#ifndef __helper_h__
#define __helper_h__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These helper methods can be use to check validity of individual stages of your algorithm
 * Pass the required component to the validate_?() methods for it's validation status to be printed to console
 * Pass the required components to the skip_?() methods to have the reference implementation perform the step.
 * For the CUDA algorithm, this will require copying data back to the host. You CANNOT pass device pointers to these methods.
 *
 * Pointers passed to helper methods must point to memory which has been allocated in the same format as CPU.c
 *
 * Some images will have limited/no errors when performed incorrectly, it's best to validate with a wide range of images.
 *
 * Do not use these methods during benchmark runs, as they will invalidate the timing.
 */

///
/// Stage 1 helpers
///
/**
 * Validates whether the results of stage 1 have been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param test_mosaic_sum Host pointer to a 1-dimensional array containing 1 pixel (3 ulonglong) per tile to be checked
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void validate_tile_sum(const Image *input_image, unsigned long long* test_mosaic_sum);
/**
 * Calculate the results of stage 1 from the input_image
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param mosaic_sum Host pointer to a 1-dimensional array containing 1 pixel (3 ulonglong) per tile
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void skip_tile_sum(const Image *input_image, unsigned long long* mosaic_sum);

///
/// Stage 2 helpers
///
/**
 * Validates whether each histograms[][]->limited_histogram of stage 2 has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param mosaic_sum Host pointer to a 1-dimensional array containing 1 pixel (3 ulonglong) per tile
 * @param test_compact_mosaic Host pointer to a 1-dimensional array containing 1 pixel (3 uchar) per tile to be checked
 * @param test_global_pixel_average Host pointer to a pre-allocated 1-dimensional array containing 1 uchar per colour channel to be checked
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void validate_compact_mosaic(unsigned int TILES_X, unsigned int TILES_Y, unsigned long long* mosaic_sum, unsigned char* test_compact_mosaic, unsigned char* test_global_pixel_average);
/**
 * Calculate histograms[][]->limited_histogram of stage 2 using histograms[][]->histogram
 * The result is applied to the parameter histograms
 *
 * @param TILES_X The number of histograms in the first dimension of test_histograms
 *        Also known as, the horizontal number of tiles in the image (input_image->width / TILE_SIZE)
 * @param TILES_Y The number of histograms in the second dimension of test_histograms
 *        Also known as, the vertical number of tiles in the image (input_image->height / TILE_SIZE)
 * @param mosaic_sum Host pointer to a 1-dimensional array containing 1 pixel (3 ulonglong) per tile
 * @param compact_mosaic Host pointer to a pre-allocated 1-dimensional array containing 1 pixel (3 uchar) per tile
 * @param global_pixel_average Host pointer to a pre-allocated 1-dimensional array containing 1 uchar per colour channel
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void skip_compact_mosaic(unsigned int TILES_X, unsigned int TILES_Y, unsigned long long* mosaic_sum, unsigned char* compact_mosaic, unsigned char* global_pixel_average);

///
/// Stage 3 helpers
///
/**
 * Validates whether the output image of stage 3 has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param compact_mosaic Host pointer to a 1-dimensional array containing 1 pixel (3 uchar) per tile
 * @param test_output_image Host pointer to a pre-allocated image for output
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void validate_broadcast(const Image* input_image, unsigned char* compact_mosaic, Image* test_output_image);
/**
 * Calculate the output image of stage 3 using compact_mosaic from stage 2 and the input image
 * The result is applied to the parameter histograms
 *
 * @param input_image Host pointer to (a copy of) the input image provided to stage 1
 * @param compact_mosaic Host pointer to a 1-dimensional array containing 1 pixel (3 uchar) per tile
 * @param output_image Host pointer to a pre-allocated image for output
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void skip_broadcast(const Image *input_image, unsigned char* compact_mosaic, Image *output_image);

///
/// These are used for reporting whether timing is invalid due to helper use
///
int getSkipUsed();
int getStage1SkipUsed();
int getStage2SkipUsed();
int getStage3SkipUsed();

#ifdef __cplusplus
}
#endif

#if defined(_DEBUG) || defined(DEBUG)
#define VALIDATION
#endif

#endif  // __helper_h__
