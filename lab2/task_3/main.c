#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "timer.h"
#include "my_rand.h"
#include "rwlock.h"

#define USE_MY_RWLOCK

#ifdef USE_MY_RWLOCK
typedef my_rwlock_t rwlock_t;
#define RW_INIT(l) my_rwlock_init(l, NULL)
#define RW_DESTROY(l) my_rwlock_destroy(l)
#define RW_RDLOCK(l) my_rwlock_rdlock(l)
#define RW_WRLOCK(l) my_rwlock_wrlock(l)
#define RW_UNLOCK(l) my_rwlock_unlock(l)
const char* LOCK_TYPE = "CUSTOM Implementation (Task)";
#else
typedef pthread_rwlock_t rwlock_t;
#define RW_INIT(l) pthread_rwlock_init(l, NULL)
#define RW_DESTROY(l) pthread_rwlock_destroy(l)
#define RW_RDLOCK(l) pthread_rwlock_rdlock(l)
#define RW_WRLOCK(l) pthread_rwlock_wrlock(l)
#define RW_UNLOCK(l) pthread_rwlock_unlock(l)
const char* LOCK_TYPE = "LIBRARY (Pthreads)";
#endif

const int MAX_KEY = 100000000;

struct list_node_s {
   int data;
   struct list_node_s* next;
};

struct list_node_s* head = NULL;
int thread_count;
int total_ops;
double insert_percent;
double search_percent;
double delete_percent;
rwlock_t rwlock;
pthread_mutex_t count_mutex;
int member_count = 0, insert_count = 0, delete_count = 0;

void Usage(char* prog_name);
void Get_input(int* inserts_in_main_p);
void* Thread_work(void* rank);
int Insert(int value);
void Print(void);
int Member(int value);
int Delete(int value);
void Free_list(void);
int Is_empty(void);

int main(int argc, char* argv[]) {
    long i;
    int key, success, attempts;
    pthread_t* thread_handles;
    int inserts_in_main;
    unsigned seed = 1;
    double start, finish;

    if (argc != 2) {
       Usage(argv[0]);
    }
    thread_count = strtol(argv[1], NULL, 10);

    Get_input(&inserts_in_main);

    i = attempts = 0;
    while (i < inserts_in_main && attempts < 2 * inserts_in_main) {
        key = my_rand(&seed) % MAX_KEY;
        success = Insert(key);
        attempts++;
        if (success) {
            ++i;
        }
    }
    printf("Inserted %ld keys in empty list\n", i);
    printf("Using lock type: %s\n", LOCK_TYPE);

    thread_handles = malloc(thread_count * sizeof(pthread_t));
    pthread_mutex_init(&count_mutex, NULL);
    RW_INIT(&rwlock);

    GET_TIME(start);

    for (i = 0; i < thread_count; i++) {
        pthread_create(&thread_handles[i], NULL, Thread_work, (void*)i);
    }

    for (i = 0; i < thread_count; i++) {
        pthread_join(thread_handles[i], NULL);
    }

    GET_TIME(finish);

    printf("Elapsed time = %e seconds\n", finish - start);
    printf("Total ops = %d\n", total_ops);
    printf("member ops = %d\n", member_count);
    printf("insert ops = %d\n", insert_count);
    printf("delete ops = %d\n", delete_count);

    Free_list();
    RW_DESTROY(&rwlock);
    pthread_mutex_destroy(&count_mutex);
    free(thread_handles);

    return 0;
}

void Usage(char* prog_name) {
    fprintf(stderr, "usage: %s <thread_count>\n", prog_name);
    exit(0);
}

void Get_input(int* inserts_in_main_p) {
    printf("How many keys should be inserted in the main thread?\n");
    scanf("%d", inserts_in_main_p);
    printf("How many ops total should be executed?\n");
    scanf("%d", &total_ops);
    printf("Percent of ops that should be searches? (between 0 and 1)\n");
    scanf("%lf", &search_percent);
    printf("Percent of ops that should be inserts? (between 0 and 1)\n");
    scanf("%lf", &insert_percent);
    delete_percent = 1.0 - (search_percent + insert_percent);
}

int Insert(int value) {
    struct list_node_s* curr = head;
    struct list_node_s* pred = NULL;
    struct list_node_s* temp;
    int rv = 1;

    while (curr != NULL && curr->data < value) {
        pred = curr;
        curr = curr->next;
    }

    if (curr == NULL || curr->data > value) {
        temp = malloc(sizeof(struct list_node_s));
        temp->data = value;
        temp->next = curr;
        if (pred == NULL) {
            head = temp;
        } else {
            pred->next = temp;
        }
    } else {
        rv = 0;
    }
    return rv;
}

void Print(void) {
    struct list_node_s* temp;
    printf("list = ");
    temp = head;
    while (temp != (struct list_node_s*)NULL) {
        printf("%d ", temp->data);
        temp = temp->next;
    }
    printf("\n");
}

int Member(int value) {
    struct list_node_s* temp;
    temp = head;
    while (temp != NULL && temp->data < value)
        temp = temp->next;
    if (temp == NULL || temp->data > value) {
        return 0;
    } else {
        return 1;
    }
}

int Delete(int value) {
    struct list_node_s* curr = head;
    struct list_node_s* pred = NULL;
    int rv = 1;

    while (curr != NULL && curr->data < value) {
        pred = curr;
        curr = curr->next;
    }

    if (curr != NULL && curr->data == value) {
        if (pred == NULL) {
            head = curr->next;
            free(curr);
        } else {
            pred->next = curr->next;
            free(curr);
        }
    } else {
        rv = 0;
    }
    return rv;
}

void Free_list(void) {
    struct list_node_s* current;
    struct list_node_s* following;
    if (Is_empty()) {
        return;
    }
    current = head;
    following = current->next;
    while (following != NULL) {
        free(current);
        current = following;
        following = current->next;
    }
    free(current);
}

int Is_empty(void) {
    if (head == NULL) {
        return 1;
    } else {
        return 0;
    }
}

void* Thread_work(void* rank) {
    long my_rank = (long)rank;
    int i, val;
    double which_op;
    unsigned seed = my_rank + 1;
    int my_member_count = 0, my_insert_count = 0, my_delete_count = 0;
    int ops_per_thread = total_ops / thread_count;

    for (i = 0; i < ops_per_thread; i++) {
        which_op = my_drand(&seed);
        val = my_rand(&seed) % MAX_KEY;
        if (which_op < search_percent) {
            RW_RDLOCK(&rwlock);
            Member(val);
            RW_UNLOCK(&rwlock);
            my_member_count++;
        } else if (which_op < search_percent + insert_percent) {
            RW_WRLOCK(&rwlock);
            Insert(val);
            RW_UNLOCK(&rwlock);
            my_insert_count++;
        } else {
            RW_WRLOCK(&rwlock);
            Delete(val);
            RW_UNLOCK(&rwlock);
            my_delete_count++;
        }
    }

    pthread_mutex_lock(&count_mutex);
    member_count += my_member_count;
    insert_count += my_insert_count;
    delete_count += my_delete_count;
    pthread_mutex_unlock(&count_mutex);

    return NULL;
}
