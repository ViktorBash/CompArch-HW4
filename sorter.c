#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "merge.h"
#include "parallel_merge.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>


int main(int argc, char *argv[]) {
    clock_t program_start = clock();
    char *filename = NULL;
    char *type = "merge";
    const char *USAGE_MSG = "Usage: %s -f <input_file> -type <parallel|merge>\n";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-type") == 0) {
            if (i + 1 < argc) {
                type = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, USAGE_MSG, argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-f") == 0) {
            if (i + 1 < argc) {
                filename = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, USAGE_MSG, argv[0]);
                return 1;
            }
        }
    }


    if (filename == NULL) {
        fprintf(stderr, USAGE_MSG, argv[0]);
        return 1;
    }

    clock_t start, end;
    double read_time, sort_time, write_time;

    start = clock();

    int fd;
    struct stat sb;
    uint32_t *sorted_arr = NULL;
    size_t file_size;

    // 1. Open the file
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    // 2. Get the file size
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size (fstat)");
        close(fd);
        return EXIT_FAILURE;
    }
    file_size = sb.st_size;
    long size = file_size / sizeof(uint32_t);

    // Check if file size is a multiple of uint32_t size
    if (file_size == 0 || (file_size % sizeof(uint32_t) != 0)) {
        fprintf(stderr, "File is empty or not a clean number of uint32_t integers.\n");
        close(fd);
        return EXIT_FAILURE;
    }

    // 3. Map the file into memory
    // --- CHANGE 2: Cast the return to uint32_t pointer ---
    sorted_arr = (uint32_t *)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // Check if mmap failed
    if (sorted_arr == MAP_FAILED) {
        perror("Error mapping file to memory (mmap)");
        close(fd);
        return EXIT_FAILURE;
    }


    end = clock();
    read_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    start = clock();

    if (strcmp(type, "parallel") == 0) {
        parallel_merge_sort_array(sorted_arr, size);
    } else {
        merge_sort_array(sorted_arr, size);
    }

    end = clock();
    sort_time = ((double) (end - start)) / CLOCKS_PER_SEC;


    start = clock();

    // This effectively forces write to the file
    if (msync(sorted_arr, file_size, MS_SYNC) == -1) {
        perror("Error synchronizing memory (msync)");
        // Decide whether to continue or exit on failure
    }

    // Unmap the memory region
    if (munmap(sorted_arr, file_size) == -1) {
        perror("Error unmapping memory (munmap)");
    }
    // Close fd
    close(fd);
    end = clock();
    write_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time to read data: %f seconds\n", read_time);
    printf("Time to sort data: %f seconds\n", sort_time);
    printf("Time to write data: %f seconds\n", write_time);

    clock_t program_end = clock();
    double total_program_time = ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("Total program time: %f seconds\n", total_program_time);
    return 0;
}
