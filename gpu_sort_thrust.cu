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
#include <windows.h>
#include <iostream>
#include <vector>

/*
 * Windows specific function that is essentially the same as mmap() on UNIX
 * Very sad that mmap() is UNIX only :C
 */
std::vector<uint32_t> read_memory_mapped(const std::string& filename) {
    HANDLE hFile = CreateFileA(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening file for reading." << std::endl;
        return {};
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        std::cerr << "Error getting file size." << std::endl;
        CloseHandle(hFile);
        return {};
    }

    HANDLE hMapping = CreateFileMappingA(
        hFile,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );
    if (hMapping == NULL) {
        std::cerr << "Error creating file mapping for reading." << std::endl;
        CloseHandle(hFile);
        return {};
    }

    void* mapped_ptr = MapViewOfFile(
        hMapping,
        FILE_MAP_READ,
        0,
        0,
        file_size.QuadPart
    );
    if (mapped_ptr == NULL) {
        std::cerr << "Error mapping view of file for reading." << std::endl;
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return {};
    }

    // Create a vector and copy the data
    size_t num_elements = file_size.QuadPart / sizeof(uint32_t);
    std::vector<uint32_t> data(num_elements);
    std::memcpy(data.data(), mapped_ptr, file_size.QuadPart);

    // Cleanup
    UnmapViewOfFile(mapped_ptr);
    CloseHandle(hMapping);
    CloseHandle(hFile);

    return data;
}

/*
 * Windows specific function that is essentially the same as mmap() on UNIX
 * Very sad that mmap() is UNIX only :C
 */
void write_memory_mapped(const std::string& filename, const thrust::host_vector<uint32_t>& data) {
    size_t data_size = data.size() * sizeof(uint32_t);

    // Open or create
    HANDLE hFile = CreateFileA(
        filename.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS, // Overwrite if exists
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "Error creating file." << std::endl;
        return;
    }

    HANDLE hMapping = CreateFileMappingA(
        hFile,
        NULL,
        PAGE_READWRITE,
        0,
        data_size, // Low-order DWORD of size
        NULL
    );

    if (hMapping == NULL) {
        std::cerr << "Error creating mapping." << std::endl;
        CloseHandle(hFile);
        return;
    }

    // Map the file, get ptr
    void* mapped_ptr = MapViewOfFile(
        hMapping,
        FILE_MAP_WRITE,
        0,
        0,
        data_size
    );

    if (mapped_ptr == NULL) {
        std::cerr << "Error mapping view." << std::endl;
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return;
    }

    // This is speedier
    std::memcpy(mapped_ptr, data.data(), data_size);

    // Cleanup and flushing
    UnmapViewOfFile(mapped_ptr);
    CloseHandle(hMapping);
    CloseHandle(hFile);
}

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
    std::vector<uint32_t> h_data = read_memory_mapped(input_file);
    if (h_data.empty()) {
        std::cerr << "Failed to read data from " << input_file << std::endl;
        return 1;
    }
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
    write_start_cpu = clock();

    // Using windows mmap() equivalent
    write_memory_mapped(output_file, thrust_h_data);

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
