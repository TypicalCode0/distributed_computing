#ifndef _RWLOCK_H_
#define _RWLOCK_H_

#include <pthread.h>

typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t read_cond;
    pthread_cond_t write_cond;
    int active_readers;
    int waiting_readers;
    int waiting_writers;
    int writer_active;
} my_rwlock_t;

int my_rwlock_init(my_rwlock_t* rw, void* attr) {
    pthread_mutex_init(&rw->mtx, NULL);
    pthread_cond_init(&rw->read_cond, NULL);
    pthread_cond_init(&rw->write_cond, NULL);
    rw->active_readers = 0;
    rw->waiting_readers = 0;
    rw->waiting_writers = 0;
    rw->writer_active = 0;
    return 0;
}

int my_rwlock_destroy(my_rwlock_t* rw) {
    pthread_mutex_destroy(&rw->mtx);
    pthread_cond_destroy(&rw->read_cond);
    pthread_cond_destroy(&rw->write_cond);
    return 0;
}

int my_rwlock_rdlock(my_rwlock_t* rw) {
    pthread_mutex_lock(&rw->mtx);
    
    ++rw->waiting_readers;
    while (rw->writer_active || rw->waiting_writers > 0) {
        pthread_cond_wait(&rw->read_cond, &rw->mtx);
    }
    --rw->waiting_readers;
    ++rw->active_readers;
    
    pthread_mutex_unlock(&rw->mtx);
    return 0;
}

int my_rwlock_wrlock(my_rwlock_t* rw) {
    pthread_mutex_lock(&rw->mtx);
    
    ++rw->waiting_writers;
    while (rw->writer_active || rw->active_readers > 0) {
        pthread_cond_wait(&rw->write_cond, &rw->mtx);
    }
    --rw->waiting_writers;
    rw->writer_active = 1;
    
    pthread_mutex_unlock(&rw->mtx);
    return 0;
}

int my_rwlock_unlock(my_rwlock_t* rw) {
    pthread_mutex_lock(&rw->mtx);
    
    if (rw->writer_active) {
        rw->writer_active = 0;
        if (rw->waiting_writers > 0) {
            pthread_cond_signal(&rw->write_cond);
        } else {
            pthread_cond_broadcast(&rw->read_cond);
        }
    } else {
        --rw->active_readers;
        if (rw->active_readers == 0 && rw->waiting_writers > 0) {
            pthread_cond_signal(&rw->write_cond);
        }
    }
    
    pthread_mutex_unlock(&rw->mtx);
    return 0;
}

#endif
