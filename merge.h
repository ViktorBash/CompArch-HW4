#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stdlib.h>

void merge(uint32_t *arr, uint32_t *temp, size_t left, size_t mid, size_t right);
void merge_sort(uint32_t *arr, uint32_t *temp, size_t left, size_t right);
void merge_sort_array(uint32_t *arr, size_t size);

#endif //MERGE_H
