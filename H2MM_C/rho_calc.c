// File: rho_calc.c
// Author: Paul David Harris
// Purpose: Calcualte the A and Rho matrices, no multithreading at this point
// Date created: 13 Feb 2021
// Date modified: 14 Oct 2022

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
//#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

// calculates just the powers of the trans matrix up to maxdif
trpow* transpow(unsigned long nstate, unsigned long maxdif, double* trans)
{
	// initialize variables
	unsigned long i, j, k, t, istride, tstride, tstride_r, tistride, Ad;
	double runsum;
	trpow *power = (trpow*) malloc(sizeof(trpow));
	
	power->max_pow = maxdif;
	power->sk = nstate;
	power->sj = nstate * nstate;
	power->A = (double*) calloc(maxdif*nstate*nstate,sizeof(double));
	for (i = 0; i < nstate; i++)
	{
		istride = i * nstate;
		for ( j = 0; j < nstate; j++)
			power->A[istride + j] =  trans[istride + j];
	}
	// loop over every power
	for (t = 1; t < maxdif; t++)
	{
		tstride = power->sj * t;
		tstride_r = tstride - power->sj;
		// outer matrix multiplication loop
		for (i = 0; i < nstate; i++)
		{
			istride = i * nstate;
			tistride = istride + tstride;
			runsum = 0.0;
			// inner matrix multiplication loop
			for (j = 0; j < nstate; j++)
			{
				power->A[tstride + istride + j] = 0.0;
				Ad = tistride + j;
				for (k = 0; k < nstate; k++)
					power->A[Ad] += power->A[istride + k] * power->A[tstride_r + k * power->sk + j];
				runsum += power->A[Ad];
			}
			// for normalization
			for (j = 0; j < nstate; j++)
				power->A[tistride + j] /= runsum;
		}
	}
	return power;
}

void* rhoulate(void *vals)
{
	// initialize variables
	pwrs *D = (pwrs*) vals;
	unsigned long i, j, k, m, z;
	unsigned long Ad, Av, Rhod, Rhov;
	unsigned long sdr = D->sT * D->td;
	unsigned long sda = D->sj * D->td;
	unsigned long svr = D->sT * D->tv;
	unsigned long sva = D->sj * D->tv;
	unsigned long sqr = D->sT * D->tq;
	unsigned long sqa = D->sj * D->tq;
	unsigned long si, sj, sk;
	double normsum;
	// calculate power of trans matrix
	for ( k = 0; k < D->sk; k++)
	{
		sk = D->sk * k;
		normsum = 0.0;
		for ( m = 0; m < D->sk; m++)
		{
			D->A[sda + sk + m] = 0.0;
			Ad = sda + sk + m;
			for ( z = 0; z < D->sk; z++)
			{
				D->A[Ad] += D->A[sva + sk + z] * D->A[sqa + D->sk * z + m];
			}
			normsum += D->A[Ad];
		}
		for(m = 0 ; m < D->sk; m++) D->A[sda + sk + m] /= normsum; // this ensures matrices are row stochastic, correcting for floating point errors
	}
	// calculate power of Rho
	for ( i = 0; i < D->sk; i++)
	{
		si = D->si * i;
		for ( j = 0; j < D->sk; j++)
		{
			sj = D->sj * j;
			for ( k = 0; k < D->sk; k++)
			{
				sk = D->sk * k;
				Rhov = svr + si + sj + sk;
				Av = sva + sk;
				for ( m = 0; m < D->sk; m++)
				{
					D->Rho[sdr + si + sj + sk + m] = 0.0;
					Rhod = sdr + si + sj + sk + m;
					for ( z = 0; z < D->sk; z++)
					{
						D->Rho[Rhod] += D->Rho[Rhov + z] * D->A[sqa + D->sk * z + m] + D->Rho[sqr + si + sj + D->sk * z + m] * D->A[Av + z];
					}
				}
			}
		}
	}
	// some error checking code: prints out the A that was just calculated, and what power it should be
	//~ printf("Matrix for power %d\n",D->td);
	//~ for ( k = 0; k < D->sk; k++)
	//~ {
		//~ sk = D->sk * k;
		//~ for ( m = 0; m < D->sk; m++)
		//~ {
			//~ Ad = sda + sk + m;
			//~ printf("%f  ",D->A[Ad]);
		//~ }
		//~ printf("\n");
	//~ }
	return NULL;
	//pthread_exit(0);
}

void* rho_all(unsigned long nstate, double* transmat, pwrs *powers)
{
	unsigned long i, j;
	// free A and Rho just in case
	for ( i = 0; i < powers->max_pow * powers->sj; i++) powers->A[i] = 0.0;
	for ( i = 0; i < powers->max_pow * powers->sT; i++) powers->Rho[i] = 0.0;
	// set values for first A matrix
	for ( i = 0; i < powers->sj; i++) powers->A[i] = transmat[i];
	// set values for first Rho matrix
	for ( i = 0; i < powers->sk; i++)
	{
		for ( j = 0; j < powers->sk; j++) powers->Rho[powers->si * i + powers->sj * j + powers->sk * i + j] = transmat[powers->sk * i +j];
	}
	for ( i = 1; i < powers->max_pow; i++) // note, this is not really a mid calculation, and there are no smarts for skipping unused values of Rho, future iterations may change this for minor speed boosts
	{
		powers->td = i;
		powers->tq = i - 1;
		powers->tv = 0;
		rhoulate((void*) powers);
	}
	return NULL;
}
