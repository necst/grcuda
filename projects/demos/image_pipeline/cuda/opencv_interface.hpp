#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>

#include "options.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

class OpenCVInterface {
   public:
    OpenCVInterface(Options &options) : debug(options.debug),
                                        image_name(options.input_image),
                                        black_and_white(options.black_and_white),
                                        image_width(options.resized_image_width) {
        if (debug) {
            std::cout << "------------------------------" << std::endl;
            std::cout << "- image name=" << options.input_image << std::endl;
            std::cout << "- image size=" << image_width << "x" << image_width << std::endl;
            std::cout << "- black and white? " << (options.black_and_white ? "no" : "yes") << std::endl;
            std::cout << "------------------------------" << std::endl;
        }
    }

    // Main execution functions;
    uchar* read_input();
    void write_output(unsigned char* buffer);
    int image_array_length;

   private:

    // Instance-specific settings;
    std::string image_name;  // Input image for the benchmark;
    bool black_and_white = DEFAULT_BLACK_AND_WHITE;  // Convert image to black and white;
    int image_width = DEFAULT_RESIZED_IMAGE_WIDTH;
    
    // General configuration settings;
    int debug = DEBUG;

    // OpenCV data;
    cv::Mat image_matrix;
    cv::Mat resized_image;
    cv::Mat output_matrix;

    // Utility functions;
    void write_output_inner(std::string kind, int resize_width);
};
