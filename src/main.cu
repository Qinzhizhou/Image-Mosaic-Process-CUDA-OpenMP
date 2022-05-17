#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include "main.h"
#include "config.h"
#include "common.h"
#include "cpu.h"
#include "openmp.h"
#include "cuda.cuh"
#include "helper.h"

int main(int argc, char **argv)
{
    // Parse args
    Config config;
    parse_args(argc, argv, &config);

    // Load image
    Image user_image;
    {
        user_image.data = stbi_load(config.input_file, &user_image.width, &user_image.height, &user_image.channels, 0);
        if (!user_image.data) {
            printf("Unable to load image '%s', please try a different file.\n", config.input_file);
            return EXIT_FAILURE;
        }
        if (user_image.channels != CHANNELS) {
            printf("Only %d channel images are supported, please try a different file.\n", CHANNELS);
            return EXIT_FAILURE;
        }
    }

    // Format image (e.g. crop to multiple of block size)
    Image input_image;
    {
        input_image.width = (user_image.width / TILE_SIZE) * TILE_SIZE;
        input_image.height = (user_image.height / TILE_SIZE) * TILE_SIZE;
        input_image.channels = user_image.channels;
        input_image.data = (unsigned char *)malloc(input_image.width * input_image.height * input_image.channels * sizeof(unsigned char));
        const int user_row_width = user_image.width * user_image.channels;
        const int input_row_width = input_image.width * input_image.channels;
        // Copy cropped data across
        for (int y = 0; y < input_image.height; ++y) {
            memcpy(input_image.data + y*input_row_width, user_image.data + y*user_row_width, input_row_width);
        }
    }
    stbi_image_free(user_image.data);

    // Export cleaned input image
    {
        if (!stbi_write_png("input.png", input_image.width, input_image.height, input_image.channels, input_image.data, input_image.width * input_image.channels)) {
            printf("Unable to save clean input image to input.png.\n");
            // return EXIT_FAILURE;
        }
    }

    // Create result for validation
    Image validation_image;
    unsigned char validation_global_image_average[4] = { 0,0,0,0 };
    {
        const unsigned int TILES_X = input_image.width / TILE_SIZE;
        const unsigned int TILES_Y = input_image.height / TILE_SIZE;
        // Copy metadata
        validation_image.width = input_image.width;
        validation_image.height = input_image.height;
        // Allocate memory
        validation_image.data = (unsigned char*)malloc(input_image.width * input_image.height * input_image.channels * sizeof(unsigned char));
        unsigned long long* validation_mosaic_sum = (unsigned long long*)malloc(TILES_X * TILES_Y * input_image.channels * sizeof(unsigned long long));
        unsigned char* validation_compact_mosaic = (unsigned char*)malloc(TILES_X * TILES_Y * input_image.channels * sizeof(unsigned char));
        // Run algorithm
        skip_tile_sum(&input_image, validation_mosaic_sum);
        skip_compact_mosaic(TILES_X, TILES_Y, validation_mosaic_sum, validation_compact_mosaic, validation_global_image_average);
        skip_broadcast(&input_image, validation_compact_mosaic, &validation_image);
        // Free temporary resources
        free(validation_mosaic_sum);
        free(validation_compact_mosaic);
    }
       
    Image output_image;
    Runtimes timing_log;
    // Location to store result of stage1
    unsigned char global_image_average[4] = {0,0,0,0};
    const int TOTAL_RUNS = config.benchmark ? BENCHMARK_RUNS : 1;
    {
        //Init for run  
        cudaEvent_t startT, initT, stage1T, stage2T, stage3T, stopT;
        CUDA_CALL(cudaEventCreate(&startT));
        CUDA_CALL(cudaEventCreate(&initT));
        CUDA_CALL(cudaEventCreate(&stage1T));
        CUDA_CALL(cudaEventCreate(&stage2T));
        CUDA_CALL(cudaEventCreate(&stage3T));
        CUDA_CALL(cudaEventCreate(&stopT));

        // Run 1 or many times
        memset(&timing_log, 0, sizeof(Runtimes));
        for (int runs = 0; runs < TOTAL_RUNS; ++runs) {
            if (TOTAL_RUNS > 1)
                printf("\r%d/%d", runs + 1, TOTAL_RUNS);
            memset(&output_image, 0, sizeof(Image));
            output_image.data = (unsigned char*)malloc(input_image.width * input_image.height * input_image.channels * sizeof(unsigned char));
            // Run Adaptive Histogram algorithm
            CUDA_CALL(cudaEventRecord(startT));
            CUDA_CALL(cudaEventSynchronize(startT));
            switch (config.mode) {
            case CPU:
                {
                    cpu_begin(&input_image);
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    cpu_stage1();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    cpu_stage2(global_image_average);
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    cpu_stage3();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    cpu_end(&output_image);
                }
                break;
            case OPENMP:
                {
                    openmp_begin(&input_image);
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    openmp_stage1();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    openmp_stage2(global_image_average);
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    openmp_stage3();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    openmp_end(&output_image);
                }
                break;
            case CUDA:
                {
                    cuda_begin(&input_image);
                    CUDA_CHECK("cuda_begin()");
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    cuda_stage1();
                    CUDA_CHECK("cuda_stage1()");
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    cuda_stage2(global_image_average);
                    CUDA_CHECK("cuda_stage2()");
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    cuda_stage3();
                    CUDA_CHECK("cuda_stage3()");
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    cuda_end(&output_image);
                }
                break;
            }
            CUDA_CALL(cudaEventRecord(stopT));
            CUDA_CALL(cudaEventSynchronize(stopT));
            // Sum timing info
            float milliseconds = 0;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, initT));
            timing_log.init += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, initT, stage1T));
            timing_log.stage1 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage1T, stage2T));
            timing_log.stage2 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage2T, stage3T));
            timing_log.stage3 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage3T, stopT));
            timing_log.cleanup += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, stopT));
            timing_log.total += milliseconds;
            // Avoid memory leak
            if (runs + 1 < TOTAL_RUNS) {
                if (output_image.data)
                    free(output_image.data);
            }
        }
        // Convert timing info to average
        timing_log.init /= TOTAL_RUNS;
        timing_log.stage1 /= TOTAL_RUNS;
        timing_log.stage2 /= TOTAL_RUNS;
        timing_log.stage3 /= TOTAL_RUNS;
        timing_log.cleanup /= TOTAL_RUNS;
        timing_log.total /= TOTAL_RUNS;

        // Cleanup timing
        cudaEventDestroy(startT);
        cudaEventDestroy(initT);
        cudaEventDestroy(stage1T);
        cudaEventDestroy(stage2T);
        cudaEventDestroy(stage3T);
        cudaEventDestroy(stopT);
    }    

    // Validate and report    
    {
        printf("\rValidation Status: \n");
        printf("\tImage width: %s\n", validation_image.width == output_image.width ? "Pass" : "Fail");
        printf("\tImage height: %s\n", validation_image.height == output_image.height ? "Pass" : "Fail");
        int v_size = validation_image.width * validation_image.height;
        int o_size = output_image.width * output_image.height;
        int s_size = v_size < o_size ? v_size : o_size;
        int bad_pixels = 0;
        int close_pixels = 0;
        if (output_image.data) {
            for (int i = 0; i < s_size; ++i) {
                for (int ch = 0; ch < validation_image.channels; ++ch) {
                    if (output_image.data[i * validation_image.channels + ch] != validation_image.data[i * validation_image.channels + ch]) {
                        // Give a +-1 threshold for error (incase fast-math triggers a small difference in places)
                        if (output_image.data[i] + 1 == validation_image.data[i] || output_image.data[i] - 1 == validation_image.data[i]) {
                            close_pixels++;
                        } else {
                            bad_pixels++;
                            break;
                        }
                    }
                }
            }
            printf("\tImage pixels: %s (%d/%u wrong)\n", bad_pixels ?  "Fail": "Pass", bad_pixels, o_size);
        } else {
            printf("\tImage pixels: Fail, (output_image->data not set)\n");
        }
        int bad_global_average = 0;
        for (int i = 0; i < validation_image.channels; ++i){
            if (global_image_average[i] != validation_global_image_average[i]) {
                bad_global_average = 1;
                break;
            }
        }
        printf("\tGlobal Image Average Pixel Value: %s\n", bad_global_average ? "Fail": "Pass");
    }

    // Export output image
    if (config.output_file) {
        if (!stbi_write_png(config.output_file, output_image.width, output_image.height, output_image.channels, output_image.data, output_image.width * output_image.channels)) {
            printf("Unable to save image output to %s.\n", config.output_file);
            // return EXIT_FAILURE;
        }
    }


    // Report timing information    
    printf("%s Average execution timing from %d runs\n", mode_to_string(config.mode), TOTAL_RUNS);
    if (config.mode == CUDA) {
        int device_id = 0;
        CUDA_CALL(cudaGetDevice(&device_id));
        cudaDeviceProp props;
        memset(&props, 0, sizeof(cudaDeviceProp));
        CUDA_CALL(cudaGetDeviceProperties(&props, device_id));
        printf("Using GPU: %s\n", props.name);
    }
#ifdef _DEBUG
    printf("Code built as DEBUG, timing results are invalid!\n");
#endif
    printf("Init: %.3fms\n", timing_log.init);
    printf("Stage 1: %.3fms%s\n", timing_log.stage1, getStage1SkipUsed() ? " (helper method used, time invalid)" : "");
    printf("Stage 2: %.3fms%s\n", timing_log.stage2, getStage2SkipUsed() ? " (helper method used, time invalid)" : "");
    printf("Stage 3: %.3fms%s\n", timing_log.stage3, getStage3SkipUsed() ? " (helper method used, time invalid)" : "");
    printf("Free: %.3fms\n", timing_log.cleanup);
    printf("Total: %.3fms%s\n", timing_log.total, getSkipUsed() ? " (helper method used, time invalid)" : "");

    // Cleanup
    cudaDeviceReset();
    free(validation_image.data);
    free(output_image.data);
    free(input_image.data);
    free(config.input_file);
    if (config.output_file)
        free(config.output_file);
    return EXIT_SUCCESS;
}
void parse_args(int argc, char **argv, Config *config) {
    // Clear config struct
    memset(config, 0, sizeof(Config));
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Program expects 2-4 arguments, only %d provided.\n", argc-1);
        print_help(argv[0]);
    }
    // Parse first arg as mode
    {
        char lower_arg[7];  // We only care about first 6 characters
        // Convert to lower case
        int i = 0;
        for(; argv[1][i] && i < 6; i++){
            lower_arg[i] = tolower(argv[1][i]);
        }
        lower_arg[i] = '\0';
        // Check for a match
        if (!strcmp(lower_arg, "cpu")) {
            config->mode = CPU;
        } else if (!strcmp(lower_arg, "openmp")) {
            config->mode = OPENMP;
        } else if (!strcmp(lower_arg, "cuda") || !strcmp(lower_arg, "gpu")) {
            config->mode = CUDA;
        } else {
            fprintf(stderr, "Unexpected string provided as first argument: '%s' .\n", argv[1]);
            fprintf(stderr, "First argument expects a single mode as string: CPU, OPENMP, CUDA.\n");
            print_help(argv[0]);
        }
    }
    // Parse second arg as input file
    {
        // Find length of string
        const size_t input_name_len = strlen(argv[2]) + 1;  // Add 1 for null terminating character
        // Allocate memory and copy
        config->input_file = (char*)malloc(input_name_len);
        memcpy(config->input_file, argv[2], input_name_len);
    }
    
    // Iterate over remaining args    
    int i = 3;
    char * t_arg = 0;
    for (; i < argc; i++) {
        // Make a lowercase copy of the argument
        const size_t arg_len = strlen(argv[i]) + 1;  // Add 1 for null terminating character
        if (t_arg) 
            free(t_arg);
        t_arg = (char*)malloc(arg_len);
        int j = 0;
        for(; argv[i][j]; ++j){
            t_arg[j] = tolower(argv[i][j]);
        }
        t_arg[j] = '\0';
        // Decide which arg it is
        if (!strcmp("--bench", t_arg) || !strcmp("--benchmark", t_arg)|| !strcmp("-b", t_arg)) {
            config->benchmark = 1;
            continue;
        }
        if (!strcmp(t_arg + arg_len - 5, ".png")) {
            // Allocate memory and copy
            config->output_file = (char*)malloc(arg_len);
            memcpy(config->output_file, argv[i], arg_len);
            continue;
        }
        fprintf(stderr, "Unexpected optional argument: %s\n", argv[i]);
        print_help(argv[0]);
    }
    if (t_arg) 
        free(t_arg);
}
void print_help(const char *program_name) {
    fprintf(stderr, "%s <mode> <input image> (<output image>) (--bench)\n", program_name);
    
    const char *line_fmt = "%-18s %s\n";
    fprintf(stderr, "Required Arguments:\n");
    fprintf(stderr, line_fmt, "<mode>", "The algorithm to use: CPU, OPENMP, CUDA");
    fprintf(stderr, line_fmt, "<input image>", "Input image, .png, .jpg");
    fprintf(stderr, "Optional Arguments:\n");
    fprintf(stderr, line_fmt, "<output image>", "Output image, requires .png filetype");
    fprintf(stderr, line_fmt, "-b, --bench", "Enable benchmark mode");

    exit(EXIT_FAILURE);
}
const char *mode_to_string(Mode m) {
    switch (m)
    {
    case CPU:
      return "CPU";
    case OPENMP:
     return "OpenMP";
    case CUDA:
      return "CUDA";
    }
    return "?";
}