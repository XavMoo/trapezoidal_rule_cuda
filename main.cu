#include <stdio.h>
#include <stdlib.h>
#include "reduce.h"
#include "utils.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <a> <b> <log2(n)> <block_size> <num_streams> <N>\n";
        return EXIT_FAILURE;
    }

    try {
        const double a = std::stod(argv[1]);
        const double b = std::stod(argv[2]);
        const uint64_t n = static_cast<uint64_t>(std::pow(2, std::stoi(argv[3])));
        const unsigned int block_size = std::stoul(argv[4]);
        const unsigned int num_streams = std::stoul(argv[5]);
        const unsigned int N = std::stoul(argv[6]);

        if (N < 2) {
            std::cerr << "Error: N must be 2 or greater.\n";
            return EXIT_FAILURE;
        }

        if (block_size != 128 && block_size != 256 && block_size != 512 && block_size != 1024) {
            std::cerr << "Error: block_size must be one of 128, 256, 512, or 1024.\n";
            return EXIT_FAILURE;
        }

        // Ensure that n / N / num_streams / block_size is an integer
        if ((n % N != 0) || 
            ((n / N) % num_streams != 0) ||
            (((n / N) / num_streams) % block_size != 0)) {

            std::cerr << "Error: n must be divisible by N, and then by num_streams, and then by block_size with no remainder.\n";
            return EXIT_FAILURE;
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nn: " << n << "\n" << std::endl;

        // GPU computation
        auto gpu_start = std::chrono::high_resolution_clock::now();
        double gpu_result = trapezoid_kernel_GPU(a, b, n, block_size, num_streams, N);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

        std::cout << "GPU time taken: " << gpu_duration.count() << " ms" << std::endl;
        std::cout << "GPU result    : " << gpu_result << "\n" << std::endl;

        // CPU computation
        auto cpu_start = std::chrono::high_resolution_clock::now();
        double cpu_result = trapezoid_kernel_CPU(a, b, n);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

        std::cout << "CPU time taken: " << cpu_duration.count() << " ms" << std::endl;
        std::cout << "CPU result    : " << cpu_result << "\n" << std::endl;

        // Speedup
        double speedup = static_cast<double>(cpu_duration.count()) / gpu_duration.count();
        std::cout << "Speedup (CPU / GPU): " << speedup << "\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
