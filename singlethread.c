#include "singlethread.h"

void run_threads(struct thread_info* threads, int numthreads)
{
    int i = 0;
    while (i < numthreads)
    {
	threads[i].threadfunc(threads[i].thread_arg);
	i++;
    }
}
