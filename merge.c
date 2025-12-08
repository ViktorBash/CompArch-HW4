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
    size_t n = right - left + 1;          // number of elements in this segment
    size_t base = left;

#if defined(__AVX2__)
    // Process 8 uint32_t elements per iteration (256 bits)
    size_t vec_elems = (n / 8) * 8;
    size_t idx = 0;

    for (; idx < vec_elems; idx += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(temp + base + idx));
        _mm256_storeu_si256((__m256i *)(arr + base + idx), v);
    }

    // Handle the remaining 0–7 elements
    for (; idx < n; ++idx) {
        arr[base + idx] = temp[base + idx];
    }
#else
    // Pure scalar fallback
    for (size_t idx = 0; idx < n; ++idx) {
        arr[base + idx] = temp[base + idx];
    }
#endif
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

// Helper to find the middle of any linked list
Node* get_middle(Node *head) {
    if (head == NULL)
        return head;

    Node *slow = head;
    Node *fast = head;

    // When fast reaches the end, slow will be at the middle (for even length, "first middle")
    while (fast->next != NULL && fast->next->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

// Merge subroutine to merge two sorted lists
Node* merge_helper_ll(Node* a, Node* b) {
    // Using a stack-allocated dummy node is simpler
    Node dummy;
    Node *current = &dummy;

    dummy.next = NULL;

    while (a != NULL && b != NULL) {
        if (a->val <= b->val) {
            current->next = a;
            a = a->next;
        } else {
            current->next = b;
            b = b->next;
        }
        current = current->next;
    }

    // Attach the remaining nodes
    current->next = (a != NULL) ? a : b;

    return dummy.next;
}

// Main recursive LL merge sort function
Node* merge_sort_ll(Node *head) {
    if (head == NULL || head->next == NULL)
        return head;

    // Split the list into two halves
    Node *middle = get_middle(head);
    Node *right_head = middle->next;
    middle->next = NULL;  // break the list

    Node *left_sorted = merge_sort_ll(head);
    Node *right_sorted = merge_sort_ll(right_head);

    return merge_helper_ll(left_sorted, right_sorted);
}

void free_list(Node *head) {
    Node *cur = head;
    while (cur != NULL) {
        Node *next = cur->next;
        free(cur);
        cur = next;
    }
}

Node* array_to_list(uint32_t *arr, size_t n) {
    if (n == 0)
        return NULL;

    Node *head = NULL;
    Node *tail = NULL;

    for (size_t i = 0; i < n; ++i) {
        Node *node = malloc(sizeof(Node));
        if (!node) {
            // On malloc failure, clean up and return NULL
            free_list(head);
            return NULL;
        }
        node->val = arr[i];
        node->next = NULL;

        if (head == NULL) {
            head = tail = node;
        } else {
            tail->next = node;
            tail = node;
        }
    }

    return head;
}


