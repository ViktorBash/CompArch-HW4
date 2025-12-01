#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void merge(uint32_t *arr, size_t left, size_t mid, size_t right) {
    int i, j, k;
    int n = mid - left + 1;
    int m = right - mid;

    int leftArr[n], rightArr[m];

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

int main() {
    //Initialise the array
    int size = 100;

    uint32_t *sorted_arr = malloc(size * sizeof(uint32_t)); // Allocate memory for the sorted array
    
    // Sort the copied array
    sort_array(sorted_arr, size);

    // Print the sorted array
    for (int i = 0; i < size; ++i) {
        printf("%d\n", sorted_arr[i]);
    }

    return 0;
}

       
