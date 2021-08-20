#include <string>
#include <iostream>
#include <ctime>    // For time()
#include "options.hpp"
#include "image_pipeline.cuh"

int main(int argc, char *argv[])
{ 
    Options options = Options(argc, argv);
    ImagePipeline img = ImagePipeline(options);
}
