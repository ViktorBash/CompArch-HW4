#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "merge.h"
#include "parallel_merge.h"

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
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Can't open file, error");
        return 1;
    }

    int size = 0;
    int capacity = 1000;
    uint32_t *sorted_arr = malloc(capacity * sizeof(uint32_t));
    if (sorted_arr == NULL) {
        perror("Failed to allocate initial memory");
        fclose(file);
        return 1;
    }

    while (fscanf(file, "%u", &sorted_arr[size]) == 1) {
        size++;
        if (size >= capacity) {
            capacity *= 2;
            uint32_t *temp = realloc(sorted_arr, capacity * sizeof(uint32_t));
            if (temp == NULL) {
                perror("Failed to reallocate memory");
                free(sorted_arr);
                fclose(file);
                return 1;
            }
            sorted_arr = temp;
        }
    }
    fclose(file);
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

    char output_filepath[256];
    char *input_basename = strrchr(filename, '/');
    if (input_basename == NULL) {
        input_basename = strrchr(filename, '\\');
    }
    if (input_basename == NULL) {
        input_basename = filename;
    } else {
        input_basename++;
    }
    snprintf(output_filepath, sizeof(output_filepath), "sorted_data/%s", input_basename);

    start = clock();
    FILE *output_file = fopen(output_filepath, "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        free(sorted_arr);
        return 1;
    }

    for (int i = 0; i < size; ++i) {
        fprintf(output_file, "%u\n", sorted_arr[i]);
    }
    fclose(output_file);

    // Cleanup time
    free(sorted_arr);

    end = clock();
    write_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Now print out the timings for everything
    printf("Time to read data: %f seconds\n", read_time);
    printf("Time to sort data: %f seconds\n", sort_time);
    printf("Time to write data: %f seconds\n", write_time);

    clock_t program_end = clock();
    double total_program_time = ((double) (program_end - program_start)) / CLOCKS_PER_SEC;
    printf("Total program time: %f seconds\n", total_program_time);

    return 0;
}
