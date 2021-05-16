// File: viterbi.c
// Purpose: Computer viterbi path and loglik of that path from posterior probability
// Date created: 1 April 2021
// Author: Paul David Harris

#ifdef linux
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


#ifdef linux
void* viterbi_burst(void* in_vals)
#elif _WIN32
HANDLE vit_lock = 0;

DWORD WINAPI viterbi_burst(void* in_vals)
#endif
{
	// initial pointer and variable assignment
	vit_vals *D = (vit_vals*) in_vals;
	size_t i, j, k, t;
	size_t omegaT, omegaTp, transT, transTshift, bT; // variables to store offsets (strides) for values indexed in nested for loops
	size_t cur_burst;
	size_t cont;
	size_t *psi = (size_t*) calloc(D->si*D->max_phot,sizeof(size_t));
	double runsum = 0.0;
	double *omega = (double*) calloc(D->si*D->max_phot,sizeof(double));
	double omegamax;
#ifdef linux
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
		D->path[cur_burst].path = (size_t*) calloc(D->phot[cur_burst].nphot,sizeof(size_t));
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
#ifdef linux
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
#ifdef linux
	pthread_exit(0);
#elif _WIN32
	ExitThread(0);
#endif
}

int viterbi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array, unsigned long num_cores)
{
	size_t i, j;
	if ( num_cores > num_bursts ) 
		num_cores = num_bursts;
#ifdef linux
	pthread_t *tid = calloc(num_cores,sizeof(pthread_t));
	pthread_mutex_t *vit_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(vit_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(num_cores, sizeof(HANDLE));
	vit_lock = CreateMutex(NULL, FALSE, NULL);
	DWORD  windowsThreadId = 0;
#endif
	// alocate variables
	phstream *b = (phstream*) calloc(num_bursts,sizeof(phstream)); // remember to free all b[n]->delta to prevent memory leak
	// process burst arrays
	size_t num_burst = (size_t) num_bursts;
	size_t *cur_burst = (size_t*) calloc(1,sizeof(size_t));
	//~ printf("Getting deltas\n");
	pwrs *powers = get_deltas(num_burst,burst_sizes,burst_times,burst_det,b); // note: allocates the powers->pow_list array, remember to free powers->pow_list before free powers or b, also, the stride lengths and td/tv/tq are not assigned (should be 0 because of calloc)
	//~ printf("Got powers\n");
	if (powers == NULL)
	{
		printf("You have an out of order photon\n");
		return 1;
	}
	for ( i = 0; i < num_bursts; i++)
	{
		for ( j = 0; j < b[i].nphot; j++)
		{
			if ( b[i].det[j] >= model->ndet)
			{
				printf("Your data has more photon streams than your h2mm model\n");
				return 2;
			}
		}
	}
	powers->sk = model->nstate;
	powers->sj = powers->sk * model->nstate; // set strides, since these never change, we keep them the same 
	powers->si = powers->sj * model->nstate;
	powers->sT = powers->si * model->nstate;
	powers->A = (double*) calloc(powers->sj * powers->max_pow,sizeof(double));
	powers->Rho = (double*) calloc(powers->sT * powers->max_pow,sizeof(double));
	rho_all(model->nstate,model->trans,powers);
	vit_vals *vit_submit = (vit_vals*) calloc(num_bursts,sizeof(vit_vals)); // allocate structures to give to viterbi_burst function
	for ( i = 0; i < num_bursts; i++)
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
		vit_submit[i].num_burst = num_bursts;
		vit_submit[i].A = powers->A;
		vit_submit[i].phot = b;
		vit_submit[i].path = path_array;
		vit_submit[i].model = model;
#ifdef linux
		vit_submit[i].vit_lock = vit_lock;
#endif
	}
	//printf("Spinning up the threads\n"); // Spin up the threads
	
#ifdef linux
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
	for ( i = 0; i < num_bursts; i++)
	{
		free(b[i].delta);
		free(b[i].det);
	}
#ifdef linux
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
	free(powers->pow_list);
	free(powers->Rho);
	free(powers->A);
	free(powers);
	return 0;
}
