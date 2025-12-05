#include "merge.h"
#include <stdlib.h>

// Helper function to merge two sorted subarrays using a temporary buffer
/*
 * Temp = our temp buf, while arr is our input (and both our input)
 */
void merge_helper(uint32_t *arr, uint32_t *temp, size_t left, size_t mid, size_t right) {
    size_t i = left;
    size_t j = mid + 1;
    size_t k = left;

    // Copy data to temp arrays
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    // Copy the remaining elements of left subarray
    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    // Copy the remaining elements of right subarray
    while (j <= right) {
        temp[k++] = arr[j++];
    }

    // Copy the temp array to original array
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}


// [left, right] (right is inclusive)
void merge_sort_recursive(uint32_t *arr, uint32_t *temp, size_t left, size_t right) {
    if (left < right) {
        // Make sure we don't overflow
        int mid = left + (right - left) / 2;

        // Split right half and left half
        merge_sort_recursive(arr, temp, left, mid);
        merge_sort_recursive(arr, temp, mid + 1, right);

        // Combine right half and left halfback
        merge_helper(arr, temp, left, mid, right);
    }
}

void merge_sort_array(uint32_t *arr, size_t size) {
    if (size == 0) {
        return;
    }
    uint32_t *temp = (uint32_t *)malloc(size * sizeof(uint32_t));
    if (temp == NULL) {
        return;
    }
    merge_sort_recursive(arr, temp, 0, size - 1);
    free(temp);
}
