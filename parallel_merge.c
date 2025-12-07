#include "parallel_merge.h"
#include "merge.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    Node *head;
    Node *result;
    int   depth;
} parallel_args;

static void* parallel_worker(void *arg);

// Recursive parallel merge sort on a linked list.
// max_depth controls how many levels of parallel recursion you allow.
// When max_depth <= 0, it falls back to sequential merge_sort_ll.
Node* parallel_merge_sort_ll(Node *head, int max_depth) {
    if (head == NULL || head->next == NULL) {
        return head;
    }

    if (max_depth <= 0) {
        return merge_sort_ll(head);  // sequential fallback
    }

    Node *middle = get_middle(head);
    Node *right_head = middle->next;
    middle->next = NULL;  // split

    // Prepare arguments for right-half thread
    parallel_args right_args;
    right_args.head   = right_head;
    right_args.result = NULL;
    right_args.depth  = max_depth - 1;

    pthread_t t;
    int err = pthread_create(&t, NULL, parallel_worker, &right_args);
    Node *left_sorted;

    if (err == 0) {
        // Sort left half in current thread (still parallel)
        left_sorted = parallel_merge_sort_ll(head, max_depth - 1);

        // Wait for right half
        pthread_join(t, NULL);
    } else {
        // If thread creation fails, just do both sequentially
        fprintf(stderr, "pthread_create failed, falling back to sequential\n");
        left_sorted  = merge_sort_ll(head);
        right_args.result = merge_sort_ll(right_head);
    }

    Node *right_sorted = right_args.result;
    return merge_helper_ll(left_sorted, right_sorted);
}

static void* parallel_worker(void *arg) {
    parallel_args *p = (parallel_args*)arg;
    p->result = parallel_merge_sort_ll(p->head, p->depth);
    return NULL;
}
