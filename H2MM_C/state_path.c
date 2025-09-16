// File: state_path.c
// Author: Paul David Harris
// Purpose: Create a state path through data given a prior and trans array
// Created: 18 Sept 2021
// Modified: 14 Oct 2022

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

time_t tm = 0;
unsigned int randcalled = 0;

// function generates the cumulative sum of an array along dimenstion 1
void cumsum(int64_t len, double* arr, double* dest)
{
	int64_t i;
	dest[0] = arr[0];
	for ( i = 1; i < len; i++)
		dest[i] = dest[i-1] + arr[i];
}

// chooses the state and/or photon from a cumulative sum array
uint8_t randchoice(int64_t len, double* arr)
{
	uint8_t i = 0;
	double r = (double)rand() / (double)RAND_MAX;
	len = len - 1;
	while((r >= arr[i]) && (i < len) && (i < UINT8_MAX)) i ++;
	return i;
}

// generates dense (even time distribution) statepath of length len based on H2MM model
int statepath(h2mm_mod* model, int64_t lent, uint8_t* path, unsigned int seed)
{
	if ((seed != 0))
	{
		srand(seed);
		randcalled = TRUE;
	}
	else if (!randcalled)
	{
		tm = time(NULL);
		srand((unsigned) tm);
		randcalled = TRUE;
	}
	int64_t i;
	double* priorsum = (double*) malloc(model->nstate*sizeof(double));
	cumsum(model->nstate,model->prior,priorsum);
	double* transsum = (double*) malloc(model->nstate*model->nstate*sizeof(double));
	for( i = 0; i < model->nstate; i++)
		cumsum(model->nstate,model->trans+(i*model->nstate),transsum+(i*model->nstate));
	path[0] = randchoice(model->nstate,priorsum);
	for( i = 1; i < lent; i++)
		path[i] = randchoice(model->nstate,transsum+(model->nstate*path[i-1]));
	if (priorsum != NULL)
		free(priorsum);
	if (transsum != NULL)
		free(transsum);
	return 0;
}


// generate a set of states based on a set of arrival times, sparsely distributed
int sparsestatepath(h2mm_mod* model, int64_t lent, int64_t* times, uint8_t* path, unsigned int seed)
{
	if ((seed != 0))
	{
		srand(seed);
		randcalled = TRUE;
	}
	else if (!randcalled)
	{
		tm = time(NULL);
		srand((unsigned) tm);
		randcalled = TRUE;
	}
	int64_t t, i, tstride, tistride;
	int32_t* dif = (int32_t*) malloc(lent * sizeof(int32_t));
	double* priorsum = (double*) malloc(model->nstate * sizeof(double));
	cumsum(model->nstate,model->prior,priorsum);
	int64_t diftemp = 0;
	int32_t maxdif = 0;
	dif[0] = 0;
	// find the maximum difference between 
	for( i = 1; i < lent; i++ )
	{
		diftemp = (times[i] - times[i-1]);
		if( diftemp < 0 )
		{
			free(priorsum);
			free(dif);
			return 1;
		}
		if( diftemp > maxdif )
			maxdif = diftemp;
		if( diftemp == 0 )
			dif[i] = 0;
		else
			dif[i] = (int32_t) diftemp - 1;
	}
	// build the powers of trans matrix, and built the cumulative sum
	trpow* powers = transpow(model->nstate ,maxdif, model->trans);
	double* transsum = (double*) malloc(maxdif*powers->sj*sizeof(double));
	for ( t = 0; t < maxdif; t++)
	{
		tstride = t * powers->sj;
		for ( i = 0; i < model->nstate; i++)
		{
			tistride = tstride + (i * model->nstate);
			cumsum(model->nstate, powers->A + tistride , transsum + tistride); 
		}
	}
	// initiate the path
	path[0] = randchoice(model->nstate,priorsum);
	// make the array
	for ( t = 1; t < lent; t++)
		path[t] = randchoice(model->nstate, transsum + (powers->sj  * dif[t]) + (powers->sk * path[t-1]));
	if (dif != NULL) 
		free(dif);
	if (powers->A != NULL)
		free(powers->A);
	if (powers != NULL)
		free(powers);
	if (priorsum != NULL)
		free(priorsum);
	if (transsum != NULL)
		free(transsum);
	return 0;
}

// take a set of states, and assign detectors to them
int phpathgen(h2mm_mod* model, int64_t lent, uint8_t* path, uint8_t* traj, unsigned int seed)
{
	if ((seed != 0))
	{
		srand(seed);
		randcalled = TRUE;
	}
	else if (!randcalled)
	{
		tm = time(NULL);
		srand((unsigned int) tm);
		randcalled = TRUE;
	}
	int64_t i, j;
	double* obssum = (double*) malloc(model->nstate*model->ndet*sizeof(double));
	for( i = 0; i < model->nstate; i++)
	{
		for (j = 0; j < model->ndet; j++)
			obssum[model->ndet*i + j] = model->obs[model->nstate*j + i];
	}
	for (i = 0; i < model->nstate; i++)
		cumsum(model->ndet, obssum + (i*model->ndet), obssum + (i*model->ndet));
	for ( i = 0; i < lent; i++)
		traj[i] = randchoice(model->ndet, obssum + (path[i]*model->ndet));
	return 0;
}
