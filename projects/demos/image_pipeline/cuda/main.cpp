#include <string>
#include <iostream>
#include <ctime>    // For time()
#include "options.hpp"
#include "opencv_interface.hpp"
#include "image_pipeline.cuh"

int main(int argc, char *argv[])
{ 
    Options options = Options(argc, argv);
    OpenCVInterface interface = OpenCVInterface(options);
    auto* img = interface.read_input();
    ImagePipeline pipeline = ImagePipeline(options);
    interface.write_output(img);
}
