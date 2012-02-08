#ifndef SINGLETHREAD_H
#define SINGLETHREAD_H

#include <stdio.h>

struct thread_info {
    void* (*threadfunc)(void* arg);
    void* thread_arg;
};

#define cfdlp_mutex_t int
#define CFDLP_MUTEX_INITIALIZER 0
#define lock_mutex(x)
#define unlock_mutex(x)
#define destroy_mutex(x)

extern int max_num_threads;

void run_threads(struct thread_info* threads, int numthreads);

#endif/*SINGLETHREAD_H*/
