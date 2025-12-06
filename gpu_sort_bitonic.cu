#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <time.h>

#define BLOCK_SIZE 256
/*
 * Bitonic sort requires you to sort 2^n items, so we inject some INT_MAX
 */

// Given an int, output the power of two above it
size_t next_power_of_two(size_t n) {
    size_t count = 0;
    // Check if already power of 2
    if (n && !(n & (n - 1))) return n;

    while(n != 0) {
        n >>= 1;
        count += 1;
    }
    return 1ULL << count;
}

// CUDA Kernel for bitonic sort :D
// n = padded size
__global__ void bitonic_sort_step(uint32_t *dev_values, size_t n, int j, int k) {
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    size_t ixj = i ^ j; // The partner index

    // Base case
    if (i >= n) return;

    // To ensure each pair is only processed once, we let the thread
    // with the lower index handle the comparison.
    if (ixj > i) {
        uint32_t val_i = dev_values[i];
        uint32_t val_ixj = dev_values[ixj];

        // Direction logic:
        // If (i & k) == 0, we sort Ascending.
        // If (i & k) != 0, we sort Descending.
        int ascending = ((i & k) == 0);

        if ((ascending && val_i > val_ixj) || (!ascending && val_i < val_ixj)) {
            // Swap
            dev_values[i] = val_ixj;
            dev_values[ixj] = val_i;
        }
    }
}

// See any cuda errors
void check_cuda(cudaError_t result, char const *const func) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", func, cudaGetErrorString(result));
        exit(99);
    }
}

int main(int argc, char **argv) {
    // Time our program
    clock_t total_start_cpu, total_end_cpu;
    clock_t read_start_cpu, read_end_cpu;
    clock_t write_start_cpu, write_end_cpu;
    cudaEvent_t sort_start_gpu, sort_end_gpu;
    float sort_time_ms = 0;

    total_start_cpu = clock();

    check_cuda(cudaEventCreate(&sort_start_gpu), "cudaEventCreate sort_start_gpu");
    check_cuda(cudaEventCreate(&sort_end_gpu), "cudaEventCreate sort_end_gpu");

    if (argc < 3) {
        printf("Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    // --- 1. Read Input File ---
    read_start_cpu = clock();
    FILE *fp = fopen(input_file, "rb");
    if (!fp) {
        perror("Error opening input file");
        return 1;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    size_t file_bytes = ftell(fp);
    rewind(fp);

    size_t num_elements = file_bytes / sizeof(uint32_t);
    size_t padded_n = next_power_of_two(num_elements);

    printf("Reading %zu elements. Padding to %zu for Bitonic Sort...\n", num_elements, padded_n);

    // Allocate HOST memory (Padded Size)
    uint32_t *h_data;
    check_cuda(cudaMallocHost((void **) &h_data, padded_n * sizeof(uint32_t)), "cudaMallocHost");

    // Read data
    size_t read_count = fread(h_data, sizeof(uint32_t), num_elements, fp);
    if (read_count != num_elements) {
        fprintf(stderr, "Error: Read mismatch.\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    // Fill padding with UINT32_MAX so they bubble to the end
    for (size_t i = num_elements; i < padded_n; i++) {
        h_data[i] = UINT32_MAX;
    }
    read_end_cpu = clock();
    printf("Time to read data: %f seconds\n", (double) (read_end_cpu - read_start_cpu) / CLOCKS_PER_SEC);


    // BEGIN THE SORTING PROCESS
    cudaEventRecord(sort_start_gpu, 0); // Start GPU timing

    // Allocate DEVICE memory
    uint32_t *d_data;
    check_cuda(cudaMalloc((void **) &d_data, padded_n * sizeof(uint32_t)), "cudaMalloc");

    // Copy Host -> Device
    check_cuda(cudaMemcpy(d_data, h_data, padded_n * sizeof(uint32_t), cudaMemcpyHostToDevice), "Memcpy H->D");

    // Bitonic Sort Loop
    size_t blocks = (padded_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int k = 2; k <= padded_n; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bitonic_sort_step<<<blocks, BLOCK_SIZE>>>(d_data, padded_n, j, k);
        }
    }

    // Check for kernel errors and synchronize
    check_cuda(cudaGetLastError(), "Kernel Launch");
    check_cuda(cudaDeviceSynchronize(), "Kernel Sync");

    // Copy Device -> Host
    check_cuda(cudaMemcpy(h_data, d_data, padded_n * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Memcpy D->H");

    cudaEventRecord(sort_end_gpu, 0); // End GPU timing
    cudaEventSynchronize(sort_end_gpu);
    check_cuda(cudaEventElapsedTime(&sort_time_ms, sort_start_gpu, sort_end_gpu), "cudaEventElapsedTime");
    printf("Time to sort data on GPU: %f seconds\n", sort_time_ms / 1000.0f);


    // --- 3. Write Output File ---
    write_start_cpu = clock();
    FILE *fp_out = fopen(output_file, "wb");
    if (!fp_out) {
        perror("Error opening output file");
        return 1;
    }
    fwrite(h_data, sizeof(uint32_t), num_elements, fp_out);
    fclose(fp_out);
    write_end_cpu = clock();
    printf("Time to write data: %f seconds\n", (double) (write_end_cpu - write_start_cpu) / CLOCKS_PER_SEC);

    printf("Sort complete. Written to %s\n", output_file);

    // Cleanup
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaEventDestroy(sort_start_gpu);
    cudaEventDestroy(sort_end_gpu);

    total_end_cpu = clock(); // End total CPU timer
    printf("Total program time: %f seconds\n", (double) (total_end_cpu - total_start_cpu) / CLOCKS_PER_SEC);

    return 0;
}