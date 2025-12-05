#include "parallel_merge.h"
#include "merge.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Struct to hold data for each sorting thread
typedef struct {
    uint32_t *arr;
    uint32_t *temp;
    size_t left;
    size_t right;
} thread_data;

// Struct to hold data for each merging thread
typedef struct {
    uint32_t *arr;
    uint32_t *temp;
    size_t left;
    size_t mid;
    size_t right;
} merge_data;

void *threaded_merge_sort(void *arg) {
    thread_data *data = (thread_data *)arg;
    if (data->left < data->right) {
        merge_sort_recursive(data->arr, data->temp, data->left, data->right);
    }
    return NULL;
}

void *threaded_merge(void *arg) {
    merge_data *data = (merge_data *)arg;
    merge_helper(data->arr, data->temp, data->left, data->mid, data->right);
    return NULL;
}

void parallel_merge_sort_array(uint32_t *arr, size_t size) {
    long num_threads_long = sysconf(_SC_NPROCESSORS_ONLN);
    int NUM_THREADS;
    if (num_threads_long == -1) {
        NUM_THREADS = 12;
    } else {
        NUM_THREADS = (int)num_threads_long;
    }

    if (size < NUM_THREADS) {
        merge_sort_array(arr, size);
        return;
    }

    uint32_t *temp = (uint32_t *)malloc(size * sizeof(uint32_t));
    if (temp == NULL) {
        return;
    }

    pthread_t threads[NUM_THREADS];
    thread_data data[NUM_THREADS];
    size_t section_size = size / NUM_THREADS;


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

    // Parallel merge
    for (int num_merged = 1; num_merged < NUM_THREADS; num_merged *= 2) {
        int num_threads_for_merge = (NUM_THREADS / (2 * num_merged));
        pthread_t merge_threads[num_threads_for_merge];
        merge_data m_data[num_threads_for_merge];
        for (int i = 0; i < num_threads_for_merge; i++) {
            size_t left = i * 2 * num_merged * section_size;
            size_t mid = left + num_merged * section_size - 1;
            size_t right = left + 2 * num_merged * section_size - 1;

            if (right >= size) {
                right = size - 1;
            }
            m_data[i] = (merge_data){.arr = arr, .temp = temp, .left = left, .mid = mid, .right = right};
            pthread_create(&merge_threads[i], NULL, threaded_merge, &m_data[i]);
        }
        for (int i = 0; i < num_threads_for_merge; i++) {
            pthread_join(merge_threads[i], NULL);
        }
    }
    free(temp);
}
