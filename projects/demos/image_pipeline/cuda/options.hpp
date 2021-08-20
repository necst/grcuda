#pragma once

#include <getopt.h>

#include <cstdlib>
#include <map>
#include <string>

#include "utils.hpp"

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define DEFAULT_BLOCK_SIZE_1D 32
#define DEFAULT_BLOCK_SIZE_2D 8
#define DEFAULT_NUM_BLOCKS 6
#define DEFAULT_POLICY "default"
#define DEFAULT_PREFETCH false
#define DEFAULT_STREAM_ATTACH false
#define DEFAULT_BLACK_AND_WHITE false

//////////////////////////////
//////////////////////////////

enum Policy {
    Sync,
    Async,
};

//////////////////////////////
//////////////////////////////

inline Policy get_policy(std::string policy) {
    if (policy == "sync")
        return Policy::Sync;
    else
        return Policy::Async;
}

struct Options {
    // Testing options;
    int debug = DEBUG;
    int block_size_1d = DEFAULT_BLOCK_SIZE_1D;
    int block_size_2d = DEFAULT_BLOCK_SIZE_2D;
    int num_blocks = DEFAULT_NUM_BLOCKS;
    bool prefetch = DEFAULT_PREFETCH;
    bool stream_attach = DEFAULT_STREAM_ATTACH;
    Policy policy_choice = get_policy(DEFAULT_POLICY);

    // Input image for the benchmark;
    std::string input_image;
    // Use black and white processing instead of color processing;
    bool black_and_white = DEFAULT_BLACK_AND_WHITE;

    // Used for printing;
    std::map<Policy, std::string> policy_map;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        map_init(policy_map)(Policy::Sync, "sync")(Policy::Async, "default");

        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"block_size_1d", required_argument, 0, 'b'},
                                               {"block_size_2d", required_argument, 0, 'c'},
                                               {"num_blocks", required_argument, 0, 'g'},
                                               {"policy", required_argument, 0, 'p'},
                                               {"prefetch", no_argument, 0, 'r'},
                                               {"attach", no_argument, 0, 'a'},
                                               {"input", required_argument, 0, 'i'},
                                               {"bw", no_argument, 0, 'w'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "db:c:g:p:rai:w", long_options, &option_index)) != EOF) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'b':
                    block_size_1d = atoi(optarg);
                    break;
                case 'c':
                    block_size_2d = atoi(optarg);
                    break;
                case 'g':
                    num_blocks = atoi(optarg);
                    break;
                case 'p':
                    policy_choice = get_policy(optarg);
                    break;
                case 'r':
                    prefetch = true;
                    break;
                case 'a':
                    stream_attach = true;
                    break;
                case 'i':
                    input_image = optarg;
                    break;
                case 'w':
                    black_and_white = true;
                    break;
                default:
                    break;
            }
        }
    }
};
