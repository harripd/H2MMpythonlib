// File: C_H2MM.c
// Author: Paul David Harris
// Purpose: main wrapping functions to take burst data and submit to central H2MM algorithm
// Date created: 13 Feb 2021
// Date modified: 29 April 2022

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

// data pre-processor, finds the maximum delta, used for setting size of Rho tensor
pwrs* get_max_delta(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, phstream *b)
{
	unsigned long i, j; // basic iterator variables
	unsigned long max_delta = 1; // stores the maximum delta between succesive photons found
	if ((burst_sizes == NULL) || (burst_deltas == NULL) || (burst_det == NULL) || (b == NULL))
	{
		//~ printf("get_deltas(): One or more of the pointer arguments is NULL\n");
		return NULL;
	}
	pwrs *powers = (pwrs*) calloc(1,sizeof(pwrs));
	for ( i = 0; i < num_burst; i++) // for loop checks the max delta
	{
		for ( j = 1; j < burst_sizes[i]; j++) // for loop calculates delta, and places in delta array, and copies index
		{
			if ( burst_deltas[i][j] > max_delta)
				max_delta = burst_deltas[i][j];
		}
		// add the current burst to burst array
		b[i].delta = burst_deltas[i];
		b[i].det = burst_det[i];
		b[i].nphot = burst_sizes[i];
	}
	// final preparations for return values
	powers->max_pow = max_delta + 1;
	powers->pow_list = (unsigned long*) calloc(powers->max_pow,sizeof(unsigned long)); // allocate the memory for an unfilled powers list, index reduce will handle filling it in
	return powers;
}

void baseprint(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func)
{
	printf("Iteration %ld, Current loglik %f, improvement: %e, iter time: %f, total: %f\n", niter, old->loglik, current->loglik - old->loglik, t_iter, t_total);
}

h2mm_mod* C_H2MM(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, lm *limits, void (*model_limits_func)(h2mm_mod*,h2mm_mod*,h2mm_mod*,void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	unsigned long i, j; // basic iterator variables
	// alocate variables
	phstream *b = (phstream*) calloc(num_burst,sizeof(phstream)); // allocate burst array, to be filled out by get_deltas function
	// process burst arrays
	pwrs *powers = get_max_delta(num_burst,burst_sizes,burst_deltas,burst_det,b); // note: allocates the powers->pow_list array, remember to free powers->pow_list before free powers or b, also, the stride lengths and td/tv/tq are not assigned (should be 0 because of calloc)
	//~ printf("Got powers\n");
	//~ in_model->nphot = 0;
	//~ for ( i = 0; i < num_bursts; i++) in_model->nphot += burst_sizes[i]; // determine total number of photons in dataset, used when calculating BIC/ICL
	// check for errors
	if (powers == NULL) // in case of an out of order photon, return null to indicate error
	{
		//~ printf("You have NULL pointer in one or more of your photon arrays\n");
		return NULL;
	}
	// check if model and data have matching detectors, return in_model to indicate error
	for ( i = 0; i < num_burst; i++)
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
	// free memory not managed by cython
	free(b);
	free(powers->pow_list);
	free(powers->Rho);
	free(powers->A);
	free(powers);
	return out_model;
}
