// File: burst_threads.c
// Author: Paul David Harris
// Purpose: Make general parallel function with fwd and back calculations in separate functions
// Date created : 16 Oct 2022
// Date modified: 31 Oct 2022

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif
#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

int64_t get_next_burst(brst_mutex *burst)
{
	// assign variables
	int64_t cur_burst = -1;
#if defined(__linux__) || defined(__APPLE__)
	if (!pthread_mutex_lock(burst->burst_mutex)){
		cur_burst = burst->cur_burst++;
		pthread_mutex_unlock(burst->burst_mutex);
	}
#elif _WIN32
	if (WaitForSingleObject(burst->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		cur_burst = burst->cur_burst++;
		ReleaseMutex(burst->burst_mutex);
	}
#endif
	return cur_burst;
}

#if defined(__linux__) || defined(__APPLE__)
void* fwd_bck_no_gamma(void* burst)
#elif _WIN32
DWORD WINAPI fwd_bck_no_gamma(void* burst)
#endif
{
	fback_vals* D = (fback_vals*) burst;
	int64_t cur_burst, recursion_size, recursion_stride;
	double* gamma = D->gamma[0];
	D->llerror = FALSE;
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		recursion_size = D->sk * D->phot[cur_burst].nphot;
		recursion_stride = recursion_size - D->sk;
		fwd_calc(D, cur_burst, recursion_size, recursion_stride);
		bck_calc(D, cur_burst, recursion_size, recursion_stride, gamma);
	}
	thread_update_h2mm_loglik(D);
	thread_update_h2mm_arrays(D);
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	//printf("fwd_back_PhotonByPhoton(): BOTTOM: threadId: %8x  nthreads: %d\n", GetCurrentThreadId(),nt);
	ExitThread(0);
#endif
}

#if defined(__linux__) || defined(__APPLE__)
void* fwd_bck_gamma(void* burst)
#elif _WIN32
DWORD WINAPI fwd_bck_gamma(void* burst)
#endif
{
	fback_vals* D = (fback_vals*) burst;
	int64_t cur_burst, recursion_size, recursion_stride;
	double* gamma;
	D->llerror = FALSE;
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		recursion_size = D->sk * D->phot[cur_burst].nphot;
		recursion_stride = recursion_size - D->sk;
		gamma = D->gamma[cur_burst];
		fwd_calc(D, cur_burst, recursion_size, recursion_stride);
		bck_calc(D, cur_burst, recursion_size, recursion_stride, gamma);
	}
	thread_update_h2mm_loglik(D);
	thread_update_h2mm_arrays(D);
#if defined(__linux__) || defined(__APPLE__)
	//~ pthread_exit(NULL);
	return NULL;
#elif _WIN32
	//printf("fwd_back_PhotonByPhoton(): BOTTOM: threadId: %8x  nthreads: %d\n", GetCurrentThreadId(),nt);
	ExitThread(0);
#endif
}

#if defined(__linux__) || defined(__APPLE__)
void* fwd_only(void* burst)
#elif _WIN32
DWORD WINAPI fwd_only(void* burst)
#endif
{
	
	fback_vals* D = (fback_vals*) burst;
	int64_t cur_burst, recursion_size, recursion_stride;
	D->llerror = FALSE;
	while((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		recursion_size = D->sk * D->phot[cur_burst].nphot;
		recursion_stride = recursion_size - D->sk;
		fwd_calc(D, cur_burst, recursion_size, recursion_stride);
	}
	thread_update_h2mm_loglik(D);
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	//printf("fwd_back_PhotonByPhoton(): BOTTOM: threadId: %8x  nthreads: %d\n", GetCurrentThreadId(),nt);
	ExitThread(0);
#endif
}
