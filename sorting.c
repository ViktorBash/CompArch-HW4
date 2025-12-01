#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define DBG_UINT(x) printf(#x " = %u\n", x)
#define DBG_STR(x) printf(#x " = %s\n", x)

void merge(uint32_t *arr, size_t left, size_t mid, size_t right) {
    int i, j, k;
    int n = mid - left + 1;
    int m = right - mid;

    // TODO: Maybe there is a way to allocate all at once instead of making a bunch of mallocs, or to optimize this some other way
    uint32_t *leftArr = malloc(n * sizeof(uint32_t));
    uint32_t *rightArr = malloc(m * sizeof(uint32_t));

    for (j = 0; j < m; j++)
        rightArr[j] = arr[mid + 1 + j];

    for (i = 0; i < n; i++)
        leftArr[i] = arr[left + i];

    k = left;
    i = 0;
    j = 0;

    while (i < n && j < m) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        }
        else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    while (i < n) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    while (j < m) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }

    // Cleanup
    free(leftArr);
    free(rightArr);
}

// [left, right] (right is inclusive)
void merge_sort(uint32_t *arr, size_t left, size_t right) {

    if (left < right) {
        // Make sure we don't overflow
        int mid = left + (right - left) / 2;

        // Split right half and left half
        merge_sort(arr, left, mid);
        merge_sort(arr, mid  + 1, right);

        // Combine right half and left halfback
        merge(arr, left, mid, right);

    }
}

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {
    // Enter your logic here
    merge_sort(arr, 0, size - 1);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Can't open file, error");
        return 1;
    }

    // We alloc size=1000 and then double with realloc as we need to
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
            // Double the size of sorted_arr
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

    // Print out the input (unsorted)
    //    for (int i = 0; i < size; ++i) {
    //        printf("%u\n", sorted_arr[i]);
    //    }

    // Sort the array
    sort_array(sorted_arr, size);

    // Print out the result (sorted)
    //    for (int i = 0; i < size; ++i) {
    //        printf("%u\n", sorted_arr[i]);
    //    }

    // Construct output file path
    char output_filepath[256]; // Max 256 filename length
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

    FILE *output_file = fopen(output_filepath, "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        free(sorted_arr);
        return 1;
    }

    // Populate the output file
    for (int i = 0; i < size; ++i) {
        fprintf(output_file, "%u\n", sorted_arr[i]);
    }

    // Cleanup time
    fclose(output_file);
    free(sorted_arr);
    return 0;
}

       
