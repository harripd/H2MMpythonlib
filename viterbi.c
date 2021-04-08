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
DWORD WINAPI viterbi_burst(void* in_vals)
#endif
{
	// initial pointer and variable assignment
	vit_vals *D = (vit_vals*) in_vals;
	size_t i, j, k, t;
	size_t omegaT, omegaTp, transT, transTshift, bT; // variables to store offsets (strides) for values indexed in nested for loops
	double runsum = 0.0;
	D->path->nphot = D->phot->nphot;
	D->path->nstate = D->model->nstate;
	// allocate the arrays for the results of viterbi
	D->path->path = (size_t*) calloc(D->phot->nphot,sizeof(size_t));
	D->path->scale = (double*) calloc(D->phot->nphot,sizeof(double));
	// allocate the delta, and psi arrays
	double *omega = (double*) calloc(D->si*D->phot->nphot,sizeof(double));
	double omegamax;
	size_t *psi = (size_t*) calloc(D->si*D->phot->nphot,sizeof(size_t));
	// initiation
	for ( i = 0; i < D->si; i++)
	{
		omega[i] = D->model->prior[i] * D->model->obs[D->phot->det[0] * D->si + i];
		runsum += omega[i];
	}
	D->path->scale[0] = runsum;
	D->path->loglik = log(runsum);
	for ( i = 0; i < D->si; i++) // for loop to divide by runsum, thus normalizing omega
		omega[i] /= runsum;
	// recursion
	for ( t = 1; t < D->phot->nphot; t++)
	{
		//printf("Calculating photon %ld\n",t);
		omegaTp = D->si * (t - 1); // precalculate the offset for the previous omega vector
		omegaT = omegaTp + D->si; // precaclualate offset for the next omega vector
		transT = D->sT * D->phot->delta[t]; // precalculate offset for trans matrix A 
		bT = D->si * D->phot->det[t]; // offset for the current 
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
		D->path->scale[t] = runsum;
		D->path->loglik += log(runsum);
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
			D->path->path[t] = i;
		}
	}
	// backtracing through the psi variable
	do
	{
		t--;
		D->path->path[t] = psi[D->si * (t + 1) + D->path->path[t+1]];
	} while(t != 0);
	if (omega != NULL)
	{
		free(omega);
	}
#ifdef linux
	pthread_exit(0);
#elif _WIN32
	ExitThread(0);
#endif
}

int viterbi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array)
{
	size_t i, j;
#ifdef linux
	pthread_t *tid = calloc(num_bursts,sizeof(pthread_t));
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(num_bursts, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
#endif
	// alocate variables
	phstream *b = (phstream*) calloc(num_bursts,sizeof(phstream)); // remember to free all b[n]->delta to prevent memory leak
	// process burst arrays
	size_t num_burst = (size_t) num_bursts;
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
		vit_submit[i].si = powers->sk;
		vit_submit[i].sT = powers->sj;
		vit_submit[i].A = powers->A;
		vit_submit[i].phot = &b[i];
		vit_submit[i].path = &path_array[i];
		vit_submit[i].model = model;
	}
	// Start the threads, 1 for each burst
#ifdef linux
	for ( i = 0; i < num_bursts; i++)
		pthread_create(&tid[i],NULL,viterbi_burst,(void*) &vit_submit[i]);
	// wait for all threads to finish
	for ( i = 0; i < num_bursts; i++)
		pthread_join(tid[i],NULL);
#elif _WIN32
	for ( i = 0; i < num_bursts; i++)
	{
		tid[i] = CreateThread(NULL, 0, viterbi_burst, (LPVOID) &vit_submit[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)num_bursts, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for ( i = 0; i < num_bursts; i++)
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
	free(tid);
	free(b);
	free(powers->pow_list);
	free(powers->Rho);
	free(powers->A);
	free(powers);
	return 0;
}
