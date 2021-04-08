// File: C_H2MM.c
// Author: Paul David Harris
// Purpose: main wrapping functions to take burst data and submit to central H2MM algorithm
// Date: 13 Feb 2021

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
	size_t i, j;
	size_t max_delta = 1;
	size_t *deltas_temp;
	size_t *inds_temp;
	if ((burst_sizes == NULL) || (burst_times == NULL) || (burst_det == NULL) || (b == NULL))
	{
		printf("get_deltas(): One or more of the pointer arguments is NULL\n");
		return NULL;
	}
	pwrs *powers = (pwrs*) calloc(1,sizeof(pwrs));
	for ( i = 0; i < num_burst; i++)
	{
		deltas_temp = (size_t*) calloc(burst_sizes[i],sizeof(size_t));
		inds_temp = (size_t*) calloc(burst_sizes[i],sizeof(size_t));
		deltas_temp[0] = 0;
		inds_temp[0] = burst_det[i][0];
		for ( j = 1; j < burst_sizes[i]; j++)
		{
			if (burst_times[i][j] < burst_times[i][j-1])
			{
				printf("Photon %ld in burst %ld arrives out of order, please sanitize your data\n",j,i);
				free(powers);
				return NULL;
			}
			else if ( burst_times[i][j] == burst_times[i][j-1] ) deltas_temp[j] = 0;
			else
			{
				deltas_temp[j] = (size_t) (burst_times[i][j] - burst_times[i][j-1]) - 1;
				if ( deltas_temp[j] > max_delta) max_delta = deltas_temp[j];
			}
			inds_temp[j] = burst_det[i][j];
		}
		b[i].delta = deltas_temp;
		b[i].det = inds_temp;
		b[i].nphot = burst_sizes[i];
		//~ printf("burst_sizes[%d]: %d\n",i,burst_sizes[i]);
	}
	powers->max_pow = max_delta + 1;
	powers->pow_list = (size_t*) calloc(powers->max_pow,sizeof(size_t)); // allocate the memory for an unfilled powers list, index reduce will handle filling it in
	return powers;
}

h2mm_mod* C_H2MM(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *in_model, lm *limits)
{
	size_t i, j;
	// alocate variables
	phstream *b = (phstream*) calloc(num_bursts,sizeof(phstream)); // remember to free all b[n]->delta to prevent memory leak
	// process burst arrays
	size_t num_burst = (size_t) num_bursts;
	//~ printf("Getting deltas\n");
	pwrs *powers = get_deltas(num_burst,burst_sizes,burst_times,burst_det,b); // note: allocates the powers->pow_list array, remember to free powers->pow_list before free powers or b, also, the stride lengths and td/tv/tq are not assigned (should be 0 because of calloc)
	//~ printf("Got powers\n");
	in_model->nphot = 0;
	for ( i = 0; i < num_bursts; i++) in_model->nphot += burst_sizes[i];
	if (powers == NULL)
	{
		printf("You have an out of order photon\n");
		return NULL;
	}
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
	powers->sk = in_model->nstate;
	powers->sj = powers->sk * in_model->nstate; // set strides, since these never change, we keep them the same 
	powers->si = powers->sj * in_model->nstate;
	powers->sT = powers->si * in_model->nstate;
	powers->A = (double*) calloc(powers->sj * powers->max_pow,sizeof(double));
	powers->Rho = (double*) calloc(powers->sT * powers->max_pow,sizeof(double));
	// reduce indexes
	//~ index_reduce_end(num_burst, b, powers);
	// run main routine
	//~ printf("Entering main routine\n");
	h2mm_mod *out_model = compute_ess_dhmm(num_burst, b, powers, in_model, limits);
	// copy out to out_model
	//~ printf("\nPrior after copy is: \n");
	//~ for ( i = 0; i < out->nstate; i++) printf("%f ",out_model->prior[i]);
	//~ printf("\n");
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
