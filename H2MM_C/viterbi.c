// File: viterbi.c
// Purpose: Compute Viterbi path and loglik of that path from posterior probability
// Author: Paul David Harris
// Date created: 1 April 2021
// Date modified: 06 Nov 2022

#include <stdio.h>
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


#if defined(__linux__) || defined(__APPLE__)
void* viterbi_burst(void* in_vals)
#elif _WIN32
DWORD WINAPI viterbi_burst(void* in_vals)
#endif
{
	// initial pointer and variable assignment
	vit_vals *D = (vit_vals*) in_vals;
	uint8_t i, j, k;
	int64_t t;
	int64_t omegaT, omegaTp, transT, transTshift, bT; // variables to store offsets (strides) for values indexed in nested for loops
	int64_t cur_burst;
	uint8_t *psi = (uint8_t*) calloc(D->si*D->max_phot,sizeof(uint8_t)); // forward state assignment
	double runsum = 0.0;
	double *omega = (double*) calloc(D->si*D->max_phot,sizeof(double));
	double omegamax;
	while ((cur_burst = get_next_burst(D->burst_lock)) < D->burst_lock->num_burst)
	{
		// initiation
		for ( i = 0; i < D->si; i++)
		{
			omega[i] = D->model->prior[i] * D->model->obs[D->phot[cur_burst].det[0] * D->si + i];
			runsum += omega[i];
		}
		D->path[cur_burst].scale[0] = runsum;
		D->path[cur_burst].loglik = log(runsum);
		for ( i = 0; i < D->si; i++) // for loop to divide by runsum, thus normalizing omega
			omega[i] /= runsum;
		// recursion
		for ( t = 1; t < D->phot[cur_burst].nphot; t++)
		{
			//printf("Calculating photon %ld\n",t);
			omegaTp = D->si * (t - 1); // precalculate the offset for the previous omega vector
			omegaT = omegaTp + D->si; // precaclualate offset for the next omega vector
			transT = D->sT * D->phot[cur_burst].delta[t]; // precalculate offset for trans matrix A 
			bT = D->si * D->phot[cur_burst].det[t]; // offset for the current detector in the obs matrix
			runsum = 0.0;
			for ( j = 0; j < D->si; j++)
			{
				transTshift = transT + j;
				omegamax = 0.0;
				for ( k = 0; k < D->si; k++)
				{
					if ( (D->A[transTshift + (D->si * k)] * omega[omegaTp + k]) > omegamax )
					{
						omegamax = D->A[transTshift + (D->si * k)] * omega[omegaTp + k];
						psi[omegaT + j] = k;
					}
				}
				omega[omegaT + j] = omegamax * D->model->obs[bT + j];
				runsum += omega[omegaT + j];
			}
			D->path[cur_burst].scale[t] = runsum;
			D->path[cur_burst].loglik += log(runsum);
			for ( j = 0; j < D->si; j++)
				omega[omegaT + j] /= runsum;
		}
		// termination- find the state to begin the state backtracing with
		t--; // decrement t because for loop executes last increment, leaving t at 1 beyond the proper index
		omegaT = D->si * t;
		omegamax = 0.0;
		for ( i = 0; i < D->si; i++)
		{
			if ( omega[omegaT + i] > omegamax )
			{
				omegamax = omega[omegaT + i];
				D->path[cur_burst].path[t] = i;
			}
		}
		// backtracing through the psi variable
		do
		{
			t--;
			D->path[cur_burst].path[t] = psi[D->si * (t + 1) + D->path[cur_burst].path[t+1]];
		} while(t != 0);
		
	}
	if (omega != NULL)
		free(omega);
	if ( psi != NULL)
		free(psi);
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(0);
#elif _WIN32
	ExitThread(0);
#endif
}

int viterbi(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *model, ph_path *path_array, int64_t num_cores)
{
	int64_t i, j;
	if ( num_cores > num_burst ) 
		num_cores = num_burst;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = calloc(num_cores,sizeof(pthread_t));
	pthread_mutex_t *vit_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(vit_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(num_cores, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
	HANDLE vit_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	// alocate variables
	phstream *b = (phstream*) calloc(num_burst,sizeof(phstream));
	// process burst arrays
	//~ printf("Getting deltas\n");
	int32_t max_delta = get_max_delta(num_burst,burst_sizes,burst_deltas,burst_det,b); 
	//~ printf("Got max delta\n");
	if (max_delta == 0)
	{
		//~ printf("You have a NULL pointer on one or more of  your photon arrays\n");
		if (b != NULL)
			free(b);
		return -1;
	}
	for ( i = 0; i < num_burst; i++)
	{
		for ( j = 0; j < b[i].nphot; j++)
		{
			if ( b[i].det[j] >= model->ndet)
			{
				//~ printf("Your data has more photon streams than your h2mm model\n");
				if (b != NULL)
					free(b);
				return -2;
			}
		}
	}
	int64_t max_phot = get_max_phot(num_burst, b);
	brst_mutex *burst_lock = malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = vit_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	trpow* powers = transpow(model->nstate, max_delta, model->trans);
	vit_vals *vit_submit = (vit_vals*) calloc(num_burst,sizeof(vit_vals)); // allocate structures to give to viterbi_burst function
	
	for ( i = 0; i < num_cores; i++)
	{
		vit_submit[i].si = powers->sk;
		vit_submit[i].sT = powers->sj;
		vit_submit[i].burst_lock = burst_lock;
		vit_submit[i].A = powers->A;
		vit_submit[i].phot = b;
		vit_submit[i].path = path_array;
		vit_submit[i].model = model;
		vit_submit[i].max_phot = max_phot;
		vit_submit[i].burst_lock = burst_lock;
	}
	//printf("Spinning up the threads\n"); // Spin up the threads
#if defined(__linux__) || defined(__APPLE__)
	for ( i = 0; i < num_cores; i++)
		pthread_create(&tid[i],NULL,viterbi_burst,(void*) &vit_submit[i]);
	// wait for all threads to finish
	for ( i = 0; i < num_cores; i++)
		pthread_join(tid[i],NULL);
#elif _WIN32
	for ( i = 0; i < num_cores; i++)
	{
		tid[i] = CreateThread(NULL, 0, viterbi_burst, (LPVOID) &vit_submit[i], 0, (LPDWORD)&windowsThreadId);
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
	// free allocated memory
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(vit_lock);
	if (vit_lock != NULL)
		free(vit_lock);
	if (tid != NULL) 
		free(tid);
#elif _WIN32
	free((void*)tid);
	if( vit_lock ) CloseHandle(vit_lock);
#endif
	if (burst_lock != NULL)
	{
		free(burst_lock);
		burst_lock = NULL;
	}
	if (b != NULL)
	{
		free(b);
		b = NULL;
	}
	free_trpow(powers);
		if (vit_submit != NULL)
	{
		free(vit_submit);
		vit_submit = NULL;
	}
	return 0;
}
