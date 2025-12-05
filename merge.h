#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stdlib.h>

void merge_sort_array(uint32_t *arr, size_t size);
void merge_sort_recursive(uint32_t *arr, uint32_t *temp, size_t left, size_t right);
void merge_helper(uint32_t *arr, uint32_t *temp, size_t left, size_t mid, size_t right);

#endif //MERGE_H
