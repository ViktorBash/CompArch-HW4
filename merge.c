#include "merge.h"
#include <stdlib.h>
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

void merge_sort_array(uint32_t *arr, size_t size) {
    merge_sort(arr, 0, size - 1);
}
