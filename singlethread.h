#ifndef SINGLETHREAD_H
#define SINGLETHREAD_H

struct thread_info {
    void* (*threadfunc)(void* arg);
    void* thread_arg;
};

#define cfdlp_mutex_t int
#define CFDLP_MUTEX_INITIALIZER 0
void lock_mutex(cfdlp_mutex_t* mutex) {}
void unlock_mutex(cfdlp_mutex_t* mutex) {}
void destroy_mutex(cfdlp_mutex_t* mutex) {}

extern int max_num_threads;

void run_threads(struct thread_info* threads, int numthreads);

#endif/*SINGLETHREAD_H*/
