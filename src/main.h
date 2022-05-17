#ifndef __main_h__
#define __main_h__
// Chose the mode by CMD
enum Mode{CPU, OPENMP, CUDA};
typedef enum Mode Mode;

/**
 * Structure containing the options provided by runtime arguments
 */

struct Config {
    /**
     * Path to input image
     */
    char *input_file;
    /**
     * Path to output image (must be .png)
     */
    char *output_file;
    /**
     * Which algorithm to use CPU, OpenMP, CUDA
     */
    Mode mode;
    /**
     * Treated as boolean, program will operate in benchmark mode
     * This repeats the algorithm multiple times and returns an average time
     * It may also warn about incorrect settings
     */
    unsigned char benchmark;
}; typedef struct Config Config;

struct Runtimes
{
    float init;
    float stage1;
    float stage2;
    float stage3;
    float cleanup;
    float total;
};
typedef struct HSVImage HSVImage;
/**
 * Parse the runtime args into config
 * @param argc argc from main()
 * @param argv argv from main()]
 * @param config Pointer to config structure for return value
 */
void parse_args(int argc, char **argv, Config *config);
/**
 * Print runtime args and exit
 * @param program_name argv[0] should always be passed to this parameter
 */
void print_help(const char *program_name);
/**
 * Return the corresponding string for the provided Mode enum
 */
const char *mode_to_string(Mode m);
#endif  // __main_h__
