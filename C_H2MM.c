// File: C_H2MM.c
// Author: Paul David Harris
// Purpose: main wrapping functions to take burst data and submit to central H2MM algorithm
// Date Created: 13 Feb 2021
// Date Modified: 03 March, 2021

#ifdef linux
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

// data pre-processor, takes the raw photon times and turns them into deltas
// allocating new memory for the deltas, and attaching the delta to the appropriate phstream
pwrs* get_deltas(unsigned long num_burst, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, phstream *b)
{
	size_t i, j; // basic iterator variables
	size_t max_delta = 1; // stores the maximum delta between succesive photons found
	size_t *deltas_temp; // pointer to currently allocated array of deltas (changes for each burst) 
	size_t *inds_temp; // pointer to current array of photon indexes
	if ((burst_sizes == NULL) || (burst_times == NULL) || (burst_det == NULL) || (b == NULL))
	{
		printf("get_deltas(): One or more of the pointer arguments is NULL\n");
		return NULL;
	}
	pwrs *powers = (pwrs*) calloc(1,sizeof(pwrs));
	for ( i = 0; i < num_burst; i++) // for loop building delta and index arrays, and linking to burst array
	{
		// allocating arrays for deltas and index arrays
		deltas_temp = (size_t*) calloc(burst_sizes[i],sizeof(size_t));
		inds_temp = (size_t*) calloc(burst_sizes[i],sizeof(size_t));
		deltas_temp[0] = 0;
		inds_temp[0] = burst_det[i][0];
		for ( j = 1; j < burst_sizes[i]; j++) // for loop calculates delta, and places in delta array, and copies index
		{
			if (burst_times[i][j] < burst_times[i][j-1]) // check if delta is negative (out of order photon
			{
				printf("Photon %ld in burst %ld arrives out of order, please sanitize your data\n",j,i);
				free(powers);
				return NULL;
			}
			else if ( burst_times[i][j] == burst_times[i][j-1] ) deltas_temp[j] = 0; // special case, if 2 photons arrive at the same time, delta must still be 1 time unit, but with C indexing, this is the 0th element in the A and Rho arrays
			else
			{
				deltas_temp[j] = (size_t) (burst_times[i][j] - burst_times[i][j-1]) - 1;
				if ( deltas_temp[j] > max_delta) max_delta = deltas_temp[j];
			}
			inds_temp[j] = burst_det[i][j];
		}
		// add the current burst to burst array
		b[i].delta = deltas_temp;
		b[i].det = inds_temp;
		b[i].nphot = burst_sizes[i];
		//~ printf("burst_sizes[%d]: %d\n",i,burst_sizes[i]);
	}
	// final preparations for return values
	powers->max_pow = max_delta + 1;
	powers->pow_list = (size_t*) calloc(powers->max_pow,sizeof(size_t)); // allocate the memory for an unfilled powers list, index reduce will handle filling it in
	return powers;
}

void baseprint(size_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func)
{
	printf("Iteration %ld, Current loglik %f, improvement: %e, iter time: %f, total: %f\n", niter, old->loglik, current->loglik - old->loglik, t_iter, t_total);
}

h2mm_mod* C_H2MM(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *in_model, lm *limits, void (*model_limits_func)(h2mm_mod*,h2mm_mod*,h2mm_mod*,void*), void *model_limits, void (*print_func)(size_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	size_t i, j; // basic iterator variables
	// alocate variables
	phstream *b = (phstream*) calloc(num_bursts,sizeof(phstream)); // allocate burst array, to be filled out by get_deltas function
	// process burst arrays
	size_t num_burst = (size_t) num_bursts;
	//~ printf("Getting deltas\n");
	pwrs *powers = get_deltas(num_burst,burst_sizes,burst_times,burst_det,b); // note: allocates the powers->pow_list array, remember to free powers->pow_list before free powers or b, also, the stride lengths and td/tv/tq are not assigned (should be 0 because of calloc)
	//~ printf("Got powers\n");
	//~ in_model->nphot = 0;
	//~ for ( i = 0; i < num_bursts; i++) in_model->nphot += burst_sizes[i]; // determine total number of photons in dataset, used when calculating BIC/ICL
	// check for errors
	if (powers == NULL) // in case of an out of order photon, return null to indicate error
	{
		printf("You have an out of order photon\n");
		return NULL;
	}
	// check if model and data have matching detectors, return in_model to indicate error
	for ( i = 0; i < num_bursts; i++)
	{
		for ( j = 0; j < b[i].nphot; j++)
		{
			if ( b[i].det[j] >= in_model->ndet)
			{
				printf("Your data has more photon streams than your h2mm model\n");
				return in_model;
			}
		}
	}
	// fill out powers structure with strides appropriate to in_model size
	powers->sk = in_model->nstate;
	powers->sj = powers->sk * in_model->nstate; // set strides, since these never change, we keep them the same 
	powers->si = powers->sj * in_model->nstate;
	powers->sT = powers->si * in_model->nstate;
	powers->A = (double*) calloc(powers->sj * powers->max_pow,sizeof(double));
	powers->Rho = (double*) calloc(powers->sT * powers->max_pow,sizeof(double));
	// run main routine
	//~ printf("Entering main routine\n");
	h2mm_mod *out_model = compute_ess_dhmm(num_burst, b, powers, in_model, limits, model_limits_func, model_limits, print_func, print_call);
	// free memory
	for ( i = 0; i < num_bursts; i++){
		free(b[i].delta);
		free(b[i].det);
	}
	free(b);
	free(powers->pow_list);
	free(powers->Rho);
	free(powers->A);
	free(powers);
	return out_model;
}
