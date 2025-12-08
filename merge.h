#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stdlib.h>

typedef struct node {
	uint32_t val;
	struct node *next;
} Node;

void merge_sort_array(uint32_t *arr, size_t size);
void merge_sort_recursive(uint32_t *arr, uint32_t *temp, size_t left, size_t right);
void merge_helper(uint32_t *arr, uint32_t *temp, size_t left, size_t mid, size_t right);
Node* get_middle(Node *head);
Node* merge_helper_ll(Node* a, Node* b);
Node* merge_sort_ll(Node *head);
Node* array_to_list(uint32_t *arr, size_t n);
void free_list(Node *head);

#endif //MERGE_H
