#ifndef __common_h__
#define __common_h__

#include "config.h"

/**
 * This structure represents a multiple channel image (e.g. rgb, rgba)
 * It contains the data required by the stb image read/write functions
 */
struct Image {
   /**
    * Array of pixel data of the image, 1 unsigned char per pixel channel (e.g. r, g, b or a)
    * Pixels ordered left to right, top to bottom
    * There is no stride, this is a compact storage
    */
    unsigned char *data;
    /**
     * Image width and height
     */
    int width, height;
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     * For the purposes of the assignment this will always evaluate to 3 (e.g. RGB images)
     */
    int channels;
};
typedef struct Image Image;

#endif  // __common_h__
