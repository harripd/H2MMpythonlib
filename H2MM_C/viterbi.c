// File: viterbi.c
// Purpose: Compute Viterbi path and loglik of that path from posterior probability
// Author: Paul David Harris
// Date created: 1 April 2021
// Date modified: 14 Oct 2022

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "C_H2MM.h"


#define TRUE 1
#define FALSE 0


#if defined(__linux__) || defined(__APPLE__)
void* viterbi_burst(void* in_vals)
#elif _WIN32
HANDLE vit_lock = 0;

DWORD WINAPI viterbi_burst(void* in_vals)
#endif
{
	// initial pointer and variable assignment
	vit_vals *D = (vit_vals*) in_vals;
	unsigned long i, j, k, t;
	unsigned long omegaT, omegaTp, transT, transTshift, bT; // variables to store offsets (strides) for values indexed in nested for loops
	unsigned long cur_burst;
	unsigned long cont;
	unsigned long *psi = (unsigned long*) calloc(D->si*D->max_phot,sizeof(unsigned long));
	double runsum = 0.0;
	double *omega = (double*) calloc(D->si*D->max_phot,sizeof(double));
	double omegamax;
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_lock(D->vit_lock);
	if ( D->cur_burst[0] < D->num_burst ) 
	{
		cur_burst = D->cur_burst[0]++;
		cont = TRUE;
	}
	else
	{
		cont = FALSE;
	}
	pthread_mutex_unlock(D->vit_lock);
#elif _WIN32
	if (WaitForSingleObject(vit_lock, INFINITE) == WAIT_OBJECT_0)
	{
		if ( D->cur_burst[0] < D->num_burst ) 
		{
			cur_burst = D->cur_burst[0]++;
			cont = TRUE;
		}
		else
		{
			cont = FALSE;
		}
		ReleaseMutex(vit_lock);
	}
#endif
	while (cont )
	{
		D->path[cur_burst].nphot = D->phot[cur_burst].nphot;
		D->path[cur_burst].nstate = D->model->nstate;
		// allocate the arrays for the results of viterbi
		D->path[cur_burst].path = (unsigned long*) calloc(D->phot[cur_burst].nphot,sizeof(unsigned long));
		D->path[cur_burst].scale = (double*) calloc(D->phot[cur_burst].nphot,sizeof(double));
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
#if defined(__linux__) || defined(__APPLE__)
		pthread_mutex_lock(D->vit_lock);
		if ( D->cur_burst[0] < D->num_burst ) 
		{
			cur_burst = D->cur_burst[0]++;
			cont = TRUE;
		}
		else
		{
			cont = FALSE;
		}
		pthread_mutex_unlock(D->vit_lock);
#elif _WIN32
		if (WaitForSingleObject(vit_lock, INFINITE) == WAIT_OBJECT_0)
		{
			if ( D->cur_burst[0] < D->num_burst )
			{
				cur_burst = D->cur_burst[0]++;
				cont = TRUE;
			}
			else
			{
				cont = FALSE;
			}
			ReleaseMutex(vit_lock);
		}
		else
		{
			cont = FALSE;
		}
#endif
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

int viterbi(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array, unsigned long num_cores)
{
	unsigned long i, j;
	if ( num_cores > num_burst ) 
		num_cores = num_burst;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = calloc(num_cores,sizeof(pthread_t));
	pthread_mutex_t *vit_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(vit_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(num_cores, sizeof(HANDLE));
	vit_lock = CreateMutex(NULL, FALSE, NULL);
	DWORD  windowsThreadId = 0;
#endif
	// alocate variables
	phstream *b = (phstream*) calloc(num_burst,sizeof(phstream)); // remember to free all b[n]->delta to prevent memory leak
	// process burst arrays
	unsigned long *cur_burst = (unsigned long*) calloc(1,sizeof(unsigned long));
	//~ printf("Getting deltas\n");
	unsigned long max_delta = get_max_delta(num_burst,burst_sizes,burst_deltas,burst_det,b); 
	//~ printf("Got max delta\n");
	if (max_delta == NULL)
	{
		//~ printf("You have a NULL pointer on one or more of  your photon arrays\n");
		if (b != NULL)
			free(b);
		return 1;
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
				return 2;
			}
		}
	}
	trpow* powers = transpow(model->nstate, max_delta, model->trans);
	vit_vals *vit_submit = (vit_vals*) calloc(num_burst,sizeof(vit_vals)); // allocate structures to give to viterbi_burst function
	for ( i = 0; i < num_burst; i++)
	{
		if ( vit_submit[0].max_phot < b[i].nphot )
			vit_submit[0].max_phot = b[i].nphot;
	}
	for ( i = 0; i < num_cores; i++)
	{
		vit_submit[i].si = powers->sk;
		vit_submit[i].sT = powers->sj;
		vit_submit[i].max_phot = vit_submit[0].max_phot;
		vit_submit[i].cur_burst = cur_burst;
		vit_submit[i].num_burst = num_burst;
		vit_submit[i].A = powers->A;
		vit_submit[i].phot = b;
		vit_submit[i].path = path_array;
		vit_submit[i].model = model;
#if defined(__linux__) || defined(__APPLE__)
		vit_submit[i].vit_lock = vit_lock;
#endif
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
	free(b);
	free(powers->A);
	free(powers);
	return 0;
}
