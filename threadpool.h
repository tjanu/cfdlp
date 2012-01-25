#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <pthread.h>

struct thread_info {
    pthread_t thread_id;
    void* (*threadfunc)(void* arg);
    void* thread_arg;
};

#define cfdlp_mutex_t pthread_mutex_t
#define CFDLP_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define lock_mutex pthread_mutex_lock
#define unlock_mutex pthread_mutex_unlock
#define destroy_mutex pthread_mutex_destroy

extern int max_num_threads;
void run_threads(struct thread_info* threads, int numthreads);

#endif/*THREADPOOL_H*/
