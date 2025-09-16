// File: fwd_back.c
// Author: Paul David Harris
// Purpose: Calculate forward and backward probabilities
// Date Created: 20 Oct 2022
// Date Modified: 29 Oct 2022
#include <stdlib.h>
#include <stdio.h>
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

void fwd_calc(fback_vals* D, int64_t cur_burst, int64_t recursion_size, int64_t recursion_stride)
{
	int64_t i, k, t, salphad, salphadi, salphao, sA, sAi, sobs;
	double runsumalpha = 0.0;
	double* alpha = D->alpha;
	for ( i = 0; i < D->sk; i++)
	{
		//~ printf("D->phot->det[0] = %f\n",D->phot->det[0]);
		alpha[i] = D->current->prior[i] * D->current->obs[D->phot[cur_burst].det[0] * D->sk + i];
		runsumalpha += alpha[i];
		//~ printf("%f  ",alpha[i]);
	}
	//~ printf("\nrunsumalpha: %f\n",runsumalpha);
	for ( i = 0; i < D->sk; i++) 
		alpha[i] /= runsumalpha;
	D->loglik += log(runsumalpha);
	// alpha recursion
	//~ printf("Alpha recursion\n");
	for ( t = 1; t < D->phot[cur_burst].nphot; t++)
	{
		salphad = D->sk * t;
		salphao = D->sk * (t-1);
		sA = D->sj * D->phot[cur_burst].delta[t];
		sobs = D->sk * D->phot[cur_burst].det[t];
		runsumalpha = 0.0;
		for ( i = 0; i < D->sk; i++) // main propogation loop, A^dt_n * alpha(t_n-1) * b
		{
			salphadi = salphad + i;
			sAi = sA + i;
			alpha[salphadi] = 0.0;
			for ( k = 0; k < D->sk; k++)
			{
				alpha[salphadi] += D->A[sAi + D->sk * k] * alpha[salphao + k];
				//~ printf("Alpha: %f, A^t: %f, alpha %f\n",alpha[salphadi],D->A[sAi + D->sk * k],alpha[salphao + k]);
			}
			//~ printf("%f  ",alpha[salphadi]);
			alpha[salphadi] = D->current->obs[sobs + i] * alpha[salphadi];
			//~ printf("Obslik: %f\n",D->current->obs[sobs + i]);
			runsumalpha += alpha[salphadi];
		}
		if (runsumalpha != 0.0)
		{
			//~ printf("Alpha[%d] = ",t);
			for ( i = 0; i < D->sk; i++)
			{
				alpha[salphad + i] /= runsumalpha; // normalize alpha
				//~ printf("%f  ",alpha[salphad + i]);
			}
			D->loglik += log(runsumalpha);
			//~ printf(" runsumalpha: %f\n",runsumalpha);
		}
		else 
		{
			//~ printf("Alpha is 0 at t: %ld\n",t);
			D->llerror = TRUE;
		}
	}
}
// Beginning of backward algorithm
void bck_calc(fback_vals* D, int64_t cur_burst, int64_t recursion_size, int64_t recursion_stride, double* gamma)
{
	// beta initiation
	int64_t i, k;
	int64_t sobs, sbetad, sbetadi,sbetao, sA,sAi, sRho, sRhoi,sxi;
	int64_t t = D->phot[cur_burst].nphot;
	double runsumbeta, runsumgamma, runsumxi;
	double* beta = D->beta;
	double* alpha = D->alpha;
	double* b = D->b;
	double* xi_temp = D->xi_temp;
	t--;
	for ( i = recursion_size -1; i >= recursion_stride; i--)
	{
		beta[i] = 1.0;
		gamma[i] = alpha[i];
	}
	sobs = D->sk * D->phot[cur_burst].det[t];
	//~ printf("obs_temp =");
	for( i = 0; i < D->sk; i++) 
	{
		D->obs_temp[sobs + i] += gamma[recursion_stride + i];
		//~ printf("%f  ",obs_temp[sobs + i]);
	}
	//~ printf("\n");
	// beta recursion
	do // normally we'd use a for loop, but because we are iterating down to 0, and t is an unsigned integer, we use a do-while loop so that t will decrement to 0, then we will do the final calculation, and then the loop will break
	{
		t--;
		runsumbeta = 0.0;
		runsumgamma = 0.0;
		runsumxi = 0.0;
		sbetad = D->sk * t;
		sbetao = D->sk * (t + 1);
		sA = D->sj * D->phot[cur_burst].delta[t + 1];
		sobs = D->sk * D->phot[cur_burst].det[t + 1];
		sRho = D->sT * D->phot[cur_burst].delta[t + 1];
		// main propogation loop for beta and gamma (beta(t_n-1) = A^t_n * beta(t_n) * b
		for ( i = 0; i < D->sk; i++) D->b[i] = beta[sbetao + i] * D->current->obs[sobs + i]; // calculate beta(t_n) * b, this is done before multiplying by A^t_n because the value is useful for calculating xi later in the loop
		for ( i = 0; i < D->sk; i++) // main loop to calculate beta non-normalized beta
		{
			sxi = D->sk * i;
			sAi = sA + D->sk * i;
			sbetadi = sbetad + i;
			beta[sbetadi] = 0.0;
			for ( k = 0; k < D->sk; k++) // calculate each element of beta, and calculate non normalized xi_temp
			{
				beta[sbetadi] += D->A[sAi + k] * b[k];
				xi_temp[sxi + k] = D->A[sAi + k] * b[k] * alpha[sbetadi]; // calculate xi_temp
				runsumxi += xi_temp[sxi + k];
			}
			runsumbeta += beta[sbetadi];
		}
		//~ if (runsumbeta == 0 || runsumxi == 0) llerror = TRUE;
		// normalization of beta, and gamma
		for ( i = 0; i < D->sk; i++) // normalize beta, and compute gamma (another loop will be needed for normalization)
		{
			sbetadi = sbetad + i;
			beta[sbetadi] /= runsumbeta;
			gamma[sbetadi] = beta[sbetadi] * alpha[sbetadi];
			runsumgamma += gamma[sbetadi];
		}
		sobs = D->sk * D->phot[cur_burst].det[t];
		for ( i = 0; i < D->sk; i++)
		{
			gamma[sbetad + i] /= runsumgamma; // normalize gamma
			D->obs_temp[sobs + i] += gamma[sbetad + i];
		}
		//~ printf("beta [%d] =",t);
		//~ for(i = 0; i < D->sk; i++) printf("%f  ",beta[sbetad + i]);
		//~ printf("gamma [%d] = ",t);
		//~ for(i = 0; i < D->sk; i++) printf("%f  ",gamma[sbetad + i]);
		//~ printf("\n");
		for ( i = 0; i < D->sj; i++)// normalize xi_temp and divide by A^t_n
		{
			if ( D->A[sA + i] != 0) xi_temp[i] /= (runsumxi * D->A[sA + i]);
			else xi_temp[i] /= runsumxi;
		}
		// multiplication of xi_temp and Rho, and add to xi_summed
		sRho = D->sT * D->phot[cur_burst].delta[t + 1];
		for ( i = 0; i < D->sj; i++)
		{
			runsumxi = 0.0;
			sRhoi = sRho + D->sj * i;
			//~ printf("%f  ",xi_temp[i]);
			for ( k = 0; k < D->sj; k++) runsumxi += xi_temp[k] * D->Rho[sRhoi + k];
			D->xi_summed[i] += runsumxi;
			//~ printf("%f  ",xi_summed[i]);
		}
		//~ printf("\n");
	} while(t != 0);
	//~ if(llerror) printf("We got a NaN\n");
	for (i = 0; i < D->sk; i++) D->prior[i] += gamma[i];
	//printf("fwd_back_PhotonByPhoton(): (B) cur_burst: %4u  threadId: %8x, xi_temp[1] = %f, xi_temp[2] = %f\n", (unsigned int)cur_burst, GetCurrentThreadId(),(double)xi_temp[1],(double)xi_temp[2]);
}

void thread_update_h2mm_loglik(fback_vals* D)
{
	// update the h2mm_model
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_lock(D->burst_lock->burst_mutex);
	if (!D->llerror && !isnan(D->current->loglik))
		D->current->loglik += D->loglik;
	else if ( D->llerror ){
		D->current->loglik = NAN;
		D->current->conv |= CONVCODE_ERROR;
	}
	//~ printf("\n");
	pthread_mutex_unlock(D->burst_lock->burst_mutex);
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		if (!D->llerror && !isnan(D->current->loglik)) 
			D->current->loglik += D->loglik;
		else if ( D->llerror ) 
			D->current->loglik = NAN;
		ReleaseMutex(D->burst_lock->burst_mutex);
	}
#endif
	D->loglik = 0.0;
}

void thread_update_h2mm_arrays(fback_vals* D)
{
	int64_t i;
// update the h2mm_model
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_lock(D->burst_lock->burst_mutex);
	for ( i = 0; i < D->sk; i++) D->new->prior[i] += D->prior[i];
	//~ printf("obs_temp is:");
	for ( i = 0; i < D->current->nstate * D->current->ndet; i++)
	{
		D->new->obs[i] += D->obs_temp[i];
		//~ printf("%f  ",obs_temp[i]);
	}
	//~ printf("xi_summed is:\n");
	for ( i = 0; i < D->sj; i++)
	{
		D->new->trans[i] += D->xi_summed[i];
		//~ printf("%f  ",D->new->trans[i]);
		//~ if ( (i % D->new->nstate) == 1) printf("\n");
	}
	//~ printf("\n");
	pthread_mutex_unlock(D->burst_lock->burst_mutex);
#elif _WIN32
	if (WaitForSingleObject(D->burst_lock->burst_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		//printf("fwd_back_PhotonByPhoton(): within mutex: loglik: %13.5e\n", D->current->loglik);
		for ( i = 0; i < D->sk; i++) D->new->prior[i] += D->prior[i];
		//~ printf("obs_temp is:");
		for ( i = 0; i < D->current->nstate * D->current->ndet; i++)
		{
			D->new->obs[i] += D->obs_temp[i];
			//~ printf("%f  ",obs_temp[i]);
		}
		//~ printf("xi_summed is:\n");
		for ( i = 0; i < D->sj; i++)
		{
			D->new->trans[i] += D->xi_summed[i];
			//~ printf("%f  ",D->new->trans[i]);
			//~ if ( (i % D->new->nstate) == 1) printf("\n");
		}
		//~ printf("\n");
		ReleaseMutex(D->burst_lock->burst_mutex);
		D->loglik = 0.0;
		for (i=0; i < D->sk; i++)
			D->prior[i] = 0.0;
		for (i=0; i < D->sj; i++)
			D->xi_summed[i] = 0.0;
		for (i=0; i < D->current->nstate * D->current->ndet; i++)
			D->obs_temp[i] = 0.0;
	}
#endif
	for (i=0; i < D->sk; i++)
		D->prior[i] = 0.0;
	for (i=0; i < D->sj; i++)
		D->xi_summed[i] = 0.0;
	for (i=0; i < D->current->nstate * D->current->ndet; i++)
		D->obs_temp[i] = 0.0;
}
