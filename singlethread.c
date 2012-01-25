#include "singlethread.h"

void run_threads(struct thread_info* threads, int numthreads)
{
    int i = 0;
    int curr_num_threads = 0;
    struct running_thread* running_threads = NULL;
    struct timespec ts;
    while (i < numthreads)
    {
	threads[i].threadfunc(threads[i].thread_arg);
	i++;
    }
}
