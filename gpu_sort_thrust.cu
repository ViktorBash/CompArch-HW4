#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <time.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// Check for cuda errors
void check_cuda(cudaError_t result, char const *const func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(99);
    }
}


int main(int argc, char **argv) {
    // Timing stuff
    clock_t total_start_cpu, total_end_cpu;
    clock_t read_start_cpu, read_end_cpu;
    clock_t write_start_cpu, write_end_cpu;
    cudaEvent_t sort_start_gpu, sort_end_gpu;
    float sort_time_ms = 0;

    total_start_cpu = clock();

    check_cuda(cudaEventCreate(&sort_start_gpu), "cudaEventCreate sort_start_gpu");
    check_cuda(cudaEventCreate(&sort_end_gpu), "cudaEventCreate sort_end_gpu");


    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin>" << std::endl;
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    // Reading in our data
    read_start_cpu = clock();
    std::ifstream in(input_file, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Error opening input file: " << input_file << std::endl;
        return 1;
    }

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<uint32_t> h_data(size / sizeof(uint32_t));
    if (!in.read((char*)h_data.data(), size)) {
        std::cerr << "Error reading from input file." << std::endl;
        return 1;
    }
    in.close();
    read_end_cpu = clock();
    std::cout << "Time to read data: " << (double)(read_end_cpu - read_start_cpu) / CLOCKS_PER_SEC << " seconds" << std::endl;


    // COMMENCE THE SORTING
    cudaEventRecord(sort_start_gpu, 0);

    thrust::host_vector<uint32_t> thrust_h_data = h_data;
    thrust::device_vector<uint32_t> thrust_d_data = thrust_h_data;

    // This under the hood uses Radix Sort, which is O(N)
    thrust::sort(thrust_d_data.begin(), thrust_d_data.end());

    thrust::copy(thrust_d_data.begin(), thrust_d_data.end(), thrust_h_data.begin());

    cudaEventRecord(sort_end_gpu, 0);
    cudaEventSynchronize(sort_end_gpu);
    check_cuda(cudaEventElapsedTime(&sort_time_ms, sort_start_gpu, sort_end_gpu), "cudaEventElapsedTime");
    std::cout << "Time to sort data on GPU: " << sort_time_ms / 1000.0f << " seconds" << std::endl;


    // COMMENCE THE WRITING
    // This step takes hella time
    write_start_cpu = clock();
    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening output file: " << output_file << std::endl;
        return 1;
    }
    out.write((char*)thrust_h_data.data(), thrust_h_data.size() * sizeof(uint32_t));
    out.close();
    write_end_cpu = clock();
    std::cout << "Time to write data: " << (double)(write_end_cpu - write_start_cpu) / CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << "Sort complete. Written to " << output_file << std::endl;


    // Cleanup
    cudaEventDestroy(sort_start_gpu);
    cudaEventDestroy(sort_end_gpu);

    total_end_cpu = clock(); // End total CPU timer
    std::cout << "Total program time: " << (double)(total_end_cpu - total_start_cpu) / CLOCKS_PER_SEC << " seconds" << std::endl;

    return 0;
}
