#include "parallel_merge.h"
#include "merge.h"
#include <pthread.h>
#include <stdio.h>

#define NUM_THREADS 12

// Struct to hold data for each thread
typedef struct {
    uint32_t *arr;
    uint32_t *temp;
    size_t left;
    size_t right;
} thread_data;

void *threaded_merge_sort(void *arg) {
    thread_data *data = (thread_data *)arg;
    if (data->left < data->right) {
        merge_sort(data->arr, data->temp, data->left, data->right);
    }
    return NULL;
}

// TODO: What if we just made 1 thread per 2048 chunk of data instead of deciding arbitrarily 12 threads for example

void parallel_merge_sort_array(uint32_t *arr, size_t size) {
    if (size < NUM_THREADS) {
        merge_sort_array(arr, size);
        return;
    }
    pthread_t threads[NUM_THREADS];
    thread_data data[NUM_THREADS];
    size_t section_size = size / NUM_THREADS;

    uint32_t *temp = malloc(size * sizeof(uint32_t));
    if (temp == NULL) {
        // Handle allocation failure
        return;
    }


    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].arr = arr;
        data[i].temp = temp;
        data[i].left = i * section_size;
        data[i].right = (i == NUM_THREADS - 1) ? size - 1 : (i + 1) * section_size - 1;
        pthread_create(&threads[i], NULL, threaded_merge_sort, &data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }


    // Merge the sorted sections
    for (int i = 1; i < NUM_THREADS; i++) {
        size_t mid = i * section_size - 1;
        size_t right = (i + 1) * section_size - 1;
        if (i == NUM_THREADS - 1) {
            right = size - 1;
        }
        merge(arr, temp, 0, mid, right);
    }

    free(temp);
}
