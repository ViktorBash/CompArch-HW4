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


int main(int argc, char *argv[]) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    int max_depth = (cores > 0) ?  (int)cores : 4;

    struct timespec program_start, program_end;
    clock_gettime(CLOCK_MONOTONIC, &program_start);
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

    struct timespec start, end;
    double read_time, sort_time, write_time;

    clock_gettime(CLOCK_MONOTONIC, &start);

    int fd;
    struct stat sb;
    uint32_t *sorted_arr = NULL;
    size_t file_size;

    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size (fstat)");
        close(fd);
        return EXIT_FAILURE;
    }
    file_size = sb.st_size;
    size_t size = file_size / sizeof(uint32_t);

    if (file_size == 0 || (file_size % sizeof(uint32_t) != 0)) {
        fprintf(stderr, "File is empty or not a clean number of uint32_t integers.\n");
        close(fd);
        return EXIT_FAILURE;
    }

    sorted_arr = (uint32_t *)mmap(NULL, file_size,
                                  PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, 0);

    if (sorted_arr == MAP_FAILED) {
        perror("Error mapping file to memory (mmap)");
        close(fd);
        return EXIT_FAILURE;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    read_time = (end.tv_sec - start.tv_sec) +
                (end.tv_nsec - start.tv_nsec) / 1e9;

    // ===== Build list if using linked-list merge sort =====
    Node *head = array_to_list(sorted_arr, size);
    if (head == NULL) {
        fprintf(stderr, "Failed to build linked list from array\n");
        munmap(sorted_arr, file_size);
        close(fd);
        return EXIT_FAILURE;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    if (strcmp(type, "parallel") == 0) {
        head = parallel_merge_sort_ll(head, max_depth);
    } else {
        head = merge_sort_ll(head);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    sort_time = (end.tv_sec - start.tv_sec) +
                (end.tv_nsec - start.tv_nsec) / 1e9;

    // ===== Copy sorted data back into the mmap’d array =====
    // list_to_array(head, sorted_arr, size);

    // Free the linked list
    free_list(head);

    clock_gettime(CLOCK_MONOTONIC, &start);

    if (msync(sorted_arr, file_size, MS_SYNC) == -1) {
        perror("Error synchronizing memory (msync)");
    }

    if (munmap(sorted_arr, file_size) == -1) {
        perror("Error unmapping memory (munmap)");
    }
    close(fd);

    clock_gettime(CLOCK_MONOTONIC, &end);
    write_time = (end.tv_sec - start.tv_sec) +
                 (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Time to read data: %f seconds\n", read_time);
    printf("Time to sort data: %f seconds\n", sort_time);
    printf("Time to write data: %f seconds\n", write_time);

    clock_gettime(CLOCK_MONOTONIC, &program_end);
    double total_program_time =
        (program_end.tv_sec - program_start.tv_sec) +
        (program_end.tv_nsec - program_start.tv_nsec) / 1e9;
    printf("Total program time: %f seconds\n", total_program_time);
    return 0;
}
