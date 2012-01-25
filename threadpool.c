#define _POSIX_C_SOURCE 199309L /* for clock_gettime */
#define _GNU_SOURCE /* for pthread_tryjoin_np */
#include "threadpool.h"

#include "util.h"
#include <time.h>

struct running_thread {
    pthread_t* thread_id;
    struct running_thread* next;
};

pthread_mutex_t tpool_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t tpool_cond = PTHREAD_COND_INITIALIZER;

void* run_function(void* arg)
{
    struct thread_info* t = (struct thread_info *)arg;
    void* retval = t->threadfunc(t->thread_arg);
    pthread_cond_signal(&tpool_cond);
    return retval;
}

void run_threads(struct thread_info* threads, int numthreads)
{
    int i = 0;
    int curr_num_threads = 0;
    struct running_thread* running_threads = NULL;
    struct timespec ts;
    while (i < numthreads)
    {
	if (max_num_threads < 0 || curr_num_threads < max_num_threads)
	{
	    pthread_create(&threads[i].thread_id, NULL, run_function, &(threads[i]));
	    struct running_thread* new_runner = (struct running_thread*) MALLOC(sizeof(struct running_thread));
	    new_runner->thread_id = &threads[i].thread_id;
	    new_runner->next = NULL;
	    struct running_thread* list_end = running_threads;
	    if (list_end == NULL)
	    {
		running_threads = new_runner;
	    }
	    else
	    {
		while (list_end != NULL && list_end->next != NULL)
		{
		    list_end = list_end->next;
		}
		list_end->next = new_runner;
	    }
	    curr_num_threads++;
	    i++;
	}
	// clean up running thread list in case anything is already finished
	struct running_thread* last = NULL;
	//fprintf(stderr, "Cleaning up running jobs...\n");
	pthread_mutex_lock(&tpool_mutex);
	if (max_num_threads > 0 && curr_num_threads >= max_num_threads)
	{
	    clock_gettime(CLOCK_REALTIME, &ts);
	    ts.tv_nsec += 5000000; // half a second
	    if (ts.tv_nsec >= 1000000000)
	    {
		int nsecs = ts.tv_nsec / 1000000000;
		ts.tv_sec += nsecs;
		ts.tv_nsec -= nsecs * 1000000000;
	    }
	    pthread_cond_timedwait(&tpool_cond, &tpool_mutex, &ts);
	}
	for (struct running_thread* current = running_threads; current != NULL; )
	{
	    int errcode = pthread_tryjoin_np(*(current->thread_id), NULL);
	    if (errcode == 0)
	    {
		// get it out of the list
		if (last == NULL)
		{
		    // first in the list
		    running_threads = current->next;
		    FREE(current);
		    current = running_threads;
		}
		else
		{
		    // have to get it out of the list...
		    struct running_thread* tmp = current;
		    last->next = current->next;
		    current = current->next;
		    FREE(tmp);
		}
		curr_num_threads--;
	    }
	    else
	    {
		last = current;
		current = current->next;
	    }
	}
	pthread_mutex_unlock(&tpool_mutex);
    }
    // wait for all threads to finish
    while (running_threads != NULL)
    {
	pthread_join(*(running_threads->thread_id), NULL);
	struct running_thread* tmp = running_threads;
	running_threads = running_threads->next;
	FREE(tmp);
    }
}

