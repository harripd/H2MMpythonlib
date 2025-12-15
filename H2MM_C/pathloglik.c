// File: pathloglik.c
// Purpose: Calculate the log likelihood of a path
// Author: Paul David Harris
// Date created: 13 Nov 2022
// Date modified: 13 Nov 2022

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0


int pathloglik(int64_t num_burst, int64_t *len_burst, int32_t **deltas, uint8_t ** dets, uint8_t **states, h2mm_mod *model, double *loglik, int64_t num_cores)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int64_t i;
	int32_t max_delta = get_max_delta(num_burst, len_burst, deltas, dets, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t nphot = check_det(num_burst, bursts, model); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	// allocate threads/mutexes
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(num_cores, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	// setup burst_lock for managing threads
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	
	trpow *powers = transpow(model->nstate, max_delta, model->trans); // get powers of transition probability matrix
	for (i = 0; i < max_delta * model->nstate * model->nstate; i++)
		powers->A[i] = log(powers->A[i]);
	h2mm_mod *log_mod = h2mm_model_calc_log(model);		
	if (num_burst < num_cores)
		num_cores = num_burst;
	pll_vals *burst_submit = (pll_vals*) malloc(num_cores * sizeof(pll_vals));
	for (i=0; i < num_cores; i++)
	{
		burst_submit[i].ll = loglik;
		burst_submit[i].state = states;
		burst_submit[i].model = log_mod;
		burst_submit[i].A = powers;
		burst_submit[i].b = bursts;
		burst_submit[i].burst_lock = burst_lock;
	}
#if defined(__linux__) || defined(__APPLE__)
	for ( i = 0; i < num_cores; i++)
		pthread_create(&tid[i],NULL,path_ll,(void*) &burst_submit[i]);
	// wait for all threads to finish
	for ( i = 0; i < num_cores; i++)
		pthread_join(tid[i],NULL);
#elif _WIN32
	for ( i = 0; i < num_cores; i++)
	{
		tid[i] = CreateThread(NULL, 0, path_ll, (LPVOID) &burst_submit[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for ( i = 0; i < num_cores; i++)
	{
		if (tid[i] != 0)
		{
			CloseHandle(tid[i]);
		}
	}
#endif
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	if (tid != NULL) 
		free(tid);
#elif _WIN32
	free((void*)tid);
	if( h2mm_lock ) CloseHandle(h2mm_lock);
#endif
	free_models(1, log_mod);
	free_trpow(powers);
	if (burst_submit != NULL)
	{
		free(burst_submit);
		burst_submit = NULL;
	}
	if (bursts != NULL)
	{
		free(bursts);
		bursts = NULL;
	}
	if (burst_lock != NULL)
	{
		free(burst_lock);
		burst_lock = NULL;
	}
	return 0;
}


#if defined(__linux__) || defined(__APPLE__)
void* path_ll(void *in)
#elif _WIN32
DWORD WINAPI path_ll(void* in)
#endif
{
	pll_vals *D = (pll_vals*) in;
	int64_t cur_burst, i , j;
	while ((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		D->ll[cur_burst] = D->model->prior[D->state[cur_burst][0]];
		D->ll[cur_burst] += D->model->obs[D->state[cur_burst][0] + D->model->nstate * D->b[cur_burst].det[0]];
		for (i=1, j=0; i < D->b[cur_burst].nphot; i++, j++)
		{
			D->ll[cur_burst] += D->model->obs[D->state[cur_burst][i] + D->model->nstate * D->b[cur_burst].det[i]];
			D->ll[cur_burst] += D->A->A[D->A->sj * D->b[cur_burst].delta[i] + D->model->nstate * D->state[cur_burst][j] + D->state[cur_burst][i]];
		}
	}
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}


int pathloglik_path(int64_t num_burst, int64_t *len_burst, int32_t **deltas, uint8_t ** dets, uint8_t **states, h2mm_mod *model, double **loglik, int64_t num_cores)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int64_t i;
	int32_t max_delta = get_max_delta(num_burst, len_burst, deltas, dets, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t nphot = check_det(num_burst, bursts, model); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	// allocate threads/mutexes
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(num_cores, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	// setup burst_lock for managing threads
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	
	trpow *powers = transpow(model->nstate, max_delta, model->trans); // get powers of transition probability matrix
	for (i = 0; i < max_delta * model->nstate * model->nstate; i++)
		powers->A[i] = log(powers->A[i]);
	h2mm_mod *log_mod = h2mm_model_calc_log(model);		
	if (num_burst < num_cores)
		num_cores = num_burst;
	pllp_vals *burst_submit = (pllp_vals*) malloc(num_cores * sizeof(pllp_vals));
	for (i=0; i < num_cores; i++)
	{
		burst_submit[i].ll = loglik;
		burst_submit[i].state = states;
		burst_submit[i].model = log_mod;
		burst_submit[i].A = powers;
		burst_submit[i].b = bursts;
		burst_submit[i].burst_lock = burst_lock;
	}
#if defined(__linux__) || defined(__APPLE__)
	for ( i = 0; i < num_cores; i++)
		pthread_create(&tid[i],NULL,path_ll_path,(void*) &burst_submit[i]);
	// wait for all threads to finish
	for ( i = 0; i < num_cores; i++)
		pthread_join(tid[i],NULL);
#elif _WIN32
	for ( i = 0; i < num_cores; i++)
	{
		tid[i] = CreateThread(NULL, 0, path_ll_path, (LPVOID) &burst_submit[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for ( i = 0; i < num_cores; i++)
	{
		if (tid[i] != 0)
		{
			CloseHandle(tid[i]);
		}
	}
#endif
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	if (tid != NULL) 
		free(tid);
#elif _WIN32
	free((void*)tid);
	if( h2mm_lock ) CloseHandle(h2mm_lock);
#endif
	free_models(1, log_mod);
	free_trpow(powers);
	if (burst_submit != NULL)
	{
		free(burst_submit);
		burst_submit = NULL;
	}
	if (bursts != NULL)
	{
		free(bursts);
		bursts = NULL;
	}
	if (burst_lock != NULL)
	{
		free(burst_lock);
		burst_lock = NULL;
	}
	return 0;
}


#if defined(__linux__) || defined(__APPLE__)
void* path_ll_path(void *in)
#elif _WIN32
DWORD WINAPI path_ll_path(void* in)
#endif
{
	pllp_vals *D = (pllp_vals*) in;
	int64_t cur_burst, i , j;
	while ((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		D->ll[cur_burst][0] = D->model->prior[D->state[cur_burst][0]];
		D->ll[cur_burst][0] += D->model->obs[D->state[cur_burst][0] + D->model->nstate * D->b[cur_burst].det[0]];
		for (i=1, j=0; i < D->b[cur_burst].nphot; i++, j++)
		{
			D->ll[cur_burst][i] = D->model->obs[D->state[cur_burst][i] + D->model->nstate * D->b[cur_burst].det[i]];
			D->ll[cur_burst][i] += D->A->A[D->A->sj * D->b[cur_burst].delta[i] + D->model->nstate * D->state[cur_burst][j] + D->state[cur_burst][i]];
		}
	}
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}
