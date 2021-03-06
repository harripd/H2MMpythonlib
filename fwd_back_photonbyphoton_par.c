// File: fwd_back_photonbyphoton.c
// Author: Paul David Harris
// Purpose: Central H2MM algorithms for calculating the logliklihood of data given the model, and creating an updated model
// Date: 13 Feb 2021

#ifdef linux
#include <unistd.h>
#include <pthread.h>
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

#ifdef linux
void* fwd_back_PhotonByPhoton(void* burst)
#elif _WIN32
HANDLE h2mm_lock = 0;

DWORD WINAPI fwd_back_PhotonByPhoton(void* burst)
#endif
{
	fback_vals *D = (fback_vals*) burst;
	size_t i, k, t; // basic iterator variables
	size_t salphad, salphadi, salphao, sbetad, sbetadi, sbetao, sxi, sA, sAi, sRho, sRhoi, sobs;
	size_t recursion_size = D->sk * D->max_phot;
	size_t recursion_stride;
	size_t cont;
	size_t cur_burst;
	int llerror = FALSE;
	double *alpha = (double*) calloc(recursion_size,sizeof(double));
	double *beta = (double*) calloc(recursion_size,sizeof(double));
	double *gamma = (double*) calloc(recursion_size,sizeof(double));
	double *b  = (double*) calloc(D->sk,sizeof(double));
	double runsumalpha = 0.0;
	double runsumbeta = 0.0;
	double runsumgamma = 0.0;
	double runsumxi = 0.0;
	double loglik = 0.0;
	double *prior = (double*) calloc(D->sk,sizeof(double));
	double *xi_temp = (double*) calloc(D->sj,sizeof(double));
	double *xi_summed = (double*) calloc(D->sj,sizeof(double));
	double *obs_temp = (double*) calloc(D->current->nstate * D->current->ndet,sizeof(double));
#ifdef linux
	pthread_mutex_lock(D->h2mm_lock);
	if ( D->cur_burst[0] < D->num_burst)
	{
		cur_burst = D->cur_burst[0]++;
		cont = TRUE;
	}
	else
	{
		cont = FALSE;
	}
	pthread_mutex_unlock(D->h2mm_lock);
#elif _WIN32
	cont = TRUE;
	cur_burst = 0;
	if (WaitForSingleObject(h2mm_lock, INFINITE) == WAIT_OBJECT_0)
	{
		if (D->cur_burst[0] < D->num_burst)
		{
			cur_burst = D->cur_burst[0]++;
			cont = TRUE;
		}
		else
		{
			cont = FALSE;
		}
		ReleaseMutex(h2mm_lock);
	}
#endif
	//printf("fwd_back_PhotonByPhoton(): (A) cur_burst: %4u  threadId: %8x Start\n", (unsigned int)cur_burst, GetCurrentThreadId());
	while(cont)
	{
		//~ printf("Brst: %ld,  ",cur_burst);
		// initialize alpha
		//~ printf("Initiating alpha\n");
		recursion_size = D->sk * D->phot[cur_burst].nphot;
		recursion_stride = recursion_size - D->sk;
		runsumalpha = 0.0;
		for ( i = 0; i < D->sk; i++)
		{
			//~ printf("D->phot->det[0] = %f\n",D->phot->det[0]);
			alpha[i] = D->current->prior[i] * D->current->obs[D->phot[cur_burst].det[0] * D->sk + i];
			runsumalpha += alpha[i];
			//~ printf("%f  ",alpha[i]);
		}
		//~ printf("\nrunsumalpha: %f\n",runsumalpha);
		for ( i = 0; i < D->sk; i++) alpha[i] /= runsumalpha;
		loglik += log(runsumalpha);
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
			if (runsumalpha != 0)
			{
				//~ printf("Alpha[%d] = ",t);
				for ( i = 0; i < D->sk; i++)
				{
					alpha[salphad + i] /= runsumalpha; // normalize alpha
					//~ printf("%f  ",alpha[salphad + i]);
				}
				loglik += log(runsumalpha);
				//~ printf(" runsumalpha: %f\n",runsumalpha);
			}
			else 
			{
				//~ printf("Alpha is 0 at t: %ld\n",t);
				llerror = TRUE;
			}
		}
		// beta initiation
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
			obs_temp[sobs + i] += gamma[recursion_stride + i];
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
			for ( i = 0; i < D->sk; i++) b[i] = beta[sbetao + i] * D->current->obs[sobs + i]; // calculate beta(t_n) * b, this is done before multiplying by A^t_n because the value is useful for calculating xi later in the loop
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
				obs_temp[sobs + i] += gamma[sbetad + i];
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
				xi_summed[i] += runsumxi;
				//~ printf("%f  ",xi_summed[i]);
			}
			//~ printf("\n");
		} while(t != 0);
		if(llerror) printf("We got a NaN\n");
		for (i = 0; i < D->sk; i++) prior[i] += gamma[i];
		//printf("fwd_back_PhotonByPhoton(): (B) cur_burst: %4u  threadId: %8x, xi_temp[1] = %f, xi_temp[2] = %f\n", (unsigned int)cur_burst, GetCurrentThreadId(),(double)xi_temp[1],(double)xi_temp[2]);
#ifdef linux
		pthread_mutex_lock(D->h2mm_lock);
		if (D->cur_burst[0] < D->num_burst)
		{
			cur_burst = D->cur_burst[0]++;
			cont = TRUE;
		}
		else cont = FALSE;
		pthread_mutex_unlock(D->h2mm_lock);
#elif _WIN32
		if (WaitForSingleObject(h2mm_lock, INFINITE) == WAIT_OBJECT_0)
		{
			if (D->cur_burst[0] < D->num_burst)
			{
				cur_burst = D->cur_burst[0]++;
				cont = TRUE;
			}
			else cont = FALSE;
			ReleaseMutex(h2mm_lock);
		}
#endif
		//printf("fwd_back_PhotonByPhoton(): (C) cur_burst: %4u  threadId: %8x\n", (unsigned int)cur_burst, GetCurrentThreadId());
	}
#ifdef linux
	pthread_mutex_lock(D->h2mm_lock);
	if (!llerror && !isnan(D->new->loglik))
		D->current->loglik += loglik;
	else if ( llerror )
		D->current->loglik = NAN;
	for ( i = 0; i < D->sk; i++) D->new->prior[i] += prior[i];
	//~ printf("obs_temp is:");
	for ( i = 0; i < D->current->nstate * D->current->ndet; i++)
	{
		D->new->obs[i] += obs_temp[i];
		//~ printf("%f  ",obs_temp[i]);
	}
	//~ printf("xi_summed is:\n");
	for ( i = 0; i < D->sj; i++)
	{
		D->new->trans[i] += xi_summed[i];
		//~ printf("%f  ",D->new->trans[i]);
		//~ if ( (i % D->new->nstate) == 1) printf("\n");
	}
	//~ printf("\n");
	pthread_mutex_unlock(D->h2mm_lock);
#elif _WIN32
	if (WaitForSingleObject(h2mm_lock, INFINITE) == WAIT_OBJECT_0)
	{
		if (!llerror && !isnan(D->new->loglik)) 
			D->current->loglik += loglik;
		else if ( llerror ) 
			D->current->loglik = NAN;
		//printf("fwd_back_PhotonByPhoton(): within mutex: loglik: %13.5e\n", D->current->loglik);
		for ( i = 0; i < D->sk; i++) D->new->prior[i] += prior[i];
		//~ printf("obs_temp is:");
		for ( i = 0; i < D->current->nstate * D->current->ndet; i++)
		{
			D->new->obs[i] += obs_temp[i];
			//~ printf("%f  ",obs_temp[i]);
		}
		//~ printf("xi_summed is:\n");
		for ( i = 0; i < D->sj; i++)
		{
			D->new->trans[i] += xi_summed[i];
			//~ printf("%f  ",D->new->trans[i]);
			//~ if ( (i % D->new->nstate) == 1) printf("\n");
		}
		//~ printf("\n");
		ReleaseMutex(h2mm_lock);
	}
#endif
	free(alpha);
	free(beta);
	free(gamma);
	free(b);
	free(prior);
	free(xi_temp);
	free(xi_summed);
	free(obs_temp);
#ifdef linux
	pthread_exit(0);
#elif _WIN32
	//printf("fwd_back_PhotonByPhoton(): BOTTOM: threadId: %8x  nthreads: %d\n", GetCurrentThreadId(),nt);
	ExitThread(0);
#endif
}


// normalizes all the arrays in the h2mm model
void h2mm_normalize(h2mm_mod *model_params)
{
	double norm_factor = 0.0;
	size_t i, j;
	// normalize the prior matrix
	for( i = 0; i < model_params->nstate; i++) norm_factor += model_params->prior[i];
	for( i = 0; i < model_params->nstate; i++) model_params->prior[i] /= norm_factor;
	// normalize the trans matrix
	for( i = 0; i < model_params->nstate; i++)
	{
		norm_factor = 0.0;
		for( j = 0; j < model_params->nstate; j++) norm_factor += model_params->trans[model_params->nstate * i + j];
		for( j = 0; j < model_params->nstate; j++) model_params->trans[model_params->nstate * i + j] /= norm_factor;
	}
	// normalize the obs matrix
	for ( i = 0 ; i < model_params->nstate ; i++)
	{
		norm_factor = 0.0;
		for ( j = 0; j < model_params->ndet ; j++) norm_factor += model_params->obs[i + model_params->nstate * j];
		for ( j = 0; j < model_params->ndet ; j++) model_params->obs[i + model_params->nstate * j] /= norm_factor;
	}
}

h2mm_mod* compute_ess_dhmm(size_t num_bursts, phstream *b, pwrs *powers, h2mm_mod *in, lm *limits) 
{
	// initiate variables
	size_t cont = TRUE;
	size_t i, j;
	size_t *cur_burst = (size_t*) calloc(1,sizeof(size_t));
	size_t niter;
	niter = in->niter;
	clock_t t_start, t_current, t_new;
	double t_iter = 0;
	double t_total = 0;
	double *pcp, *pct, *pco, *pnp, *pnt, *pno, *pop, *pot, *poo;
	h2mm_mod *pcur, *pnew, *pold;
#ifdef linux
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
	//HANDLE h2mm_lock;
	h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	h2mm_mod *current = (h2mm_mod*) malloc(sizeof(h2mm_mod));
	h2mm_mod *new = (h2mm_mod*) malloc(sizeof(h2mm_mod));
	h2mm_mod *old = (h2mm_mod*) malloc(sizeof(h2mm_mod));
	h2mm_mod *mod_temp;
	pcur = current;
	pnew = new;
	pold = old;
	new->nstate = old->nstate = current->nstate = in->nstate;
	new->ndet = old->ndet = current->ndet = in->ndet;
	new->nphot = old->nphot = current->nphot = in->nphot;
	new->conv = current->conv = old->conv = 0;
	pop = old-> prior = (double*) malloc(in->nstate * sizeof(double));
	pcp = current-> prior = (double*) malloc(in->nstate * sizeof(double));
	for ( i = 0; i < in->nstate; i++) old->prior[i] = current->prior[i] = in->prior[i];
	pot = old->trans = (double*) malloc(in->nstate * current->nstate * sizeof(double));
	pct = current->trans = (double*) malloc(in->nstate * current->nstate * sizeof(double));
	for ( i = 0; i < in->nstate * in->nstate; i++) old->trans[i] = current->trans[i] = in->trans[i];
	poo = old->obs = (double*) malloc(in->nstate * current->ndet * sizeof(double));
	pco = current->obs = (double*) malloc(in->nstate * current->ndet * sizeof(double));
	for ( i = 0; i < in->nstate * in->ndet; i++) old->obs[i] = current->obs[i] = in->obs[i];
	pnp = new-> prior = (double*) calloc(current->nstate,sizeof(double));
	pnt = new->trans = (double*) calloc(current->nstate * current->nstate,sizeof(double));
	pno = new->obs = (double*) calloc(current->nstate * current->ndet, sizeof(double));
	old->loglik = -INFINITY;
	current->loglik = 0.0;
	fback_vals *burst_submit = (fback_vals*) calloc(limits->num_cores,sizeof(fback_vals));
	for ( j = 0; j < limits->num_cores; j++)
	{
		burst_submit[j].phot = b;
		burst_submit[j].cur_burst = cur_burst;
		burst_submit[j].num_burst = num_bursts;
		burst_submit[j].sk = powers->sk;
		burst_submit[j].sj = powers->sj;
		burst_submit[j].si = powers->si;
		burst_submit[j].sT = powers->sT;
		burst_submit[j].Rho = powers->Rho;
		burst_submit[j].A = powers->A;
		burst_submit[j].current = current;
		burst_submit[j].new = new;
#ifdef linux
		burst_submit[j].h2mm_lock = h2mm_lock;
#endif
		for ( i = 0; i < num_bursts; i++)
		{
			if ( burst_submit[j].max_phot < b[i].nphot) burst_submit[j].max_phot = b[i].nphot;
		}
	}
	//begin main computation
	t_start = clock();
	t_current = t_start;
	while(cont)
	{
		// calculate rho
		rho_all(current->nstate, current->trans, powers);
		// calculate forward and backward recurrsion variable, running in parallel for all bursts
#ifdef linux
		for(i = 0; i < limits->num_cores; i++) pthread_create(&tid[i],NULL,fwd_back_PhotonByPhoton,(void*) &burst_submit[i]); // create a thread for each burst
		for(i = 0; i < limits->num_cores; i++) pthread_join(tid[i],NULL); // wait for all bursts to finish
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
		{
			tid[i] = CreateThread(NULL, 0, fwd_back_PhotonByPhoton, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId);
			//printf("compute_ess_dhmm(): i: %3d  threadId: %8x\n", (int)i, windowsThreadId);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++)
		{
			if (tid[i] != 0)
			{
				CloseHandle(tid[i]);
			}
		}
#endif
		// post processing decision making
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		//printf("Iteration %ld, Current loglik %f, improvement: %e, iter time: %f, total: %f\n",niter, old->loglik, current->loglik - old->loglik, t_iter, t_total);
		printf("Iteration %ld, Current loglik %f, improvement: %e, iter time: %f, total: %f\n", niter, old->loglik, current->loglik - old->loglik, t_iter, t_total);
		if ( !isnan(current->loglik) && (current->loglik - old->loglik) > limits->min_conv) // if the model has improved, 
		{
			niter++;
			h2mm_normalize(new); // normalize the new, h2mm model
			// update the old model to the current model, and current model to new model
			mod_temp = old;
			old = current;
			current = new;
			new = mod_temp;
			// 0 the arrays in the new model (equivalent to emptying it)
			current->loglik = 0.0;
			for ( i = 0; i < new->nstate; i ++) new->prior[i] = 0.0;
			for ( i = 0; i < new->nstate * new->nstate; i++) new->trans[i] = 0.0;
			for ( i = 0; i < new->nstate * new->ndet; i++) new->obs[i] = 0.0;
			// we need to update the pointers to the models for the burst_submit structures as well
			*cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++)	
			{
				burst_submit[i].current = current;
				burst_submit[i].new = new;
			}
		}
		else if (!isnan(current->loglik) && (current->loglik - old->loglik) <= limits->min_conv) // model converged
		{
			cont = FALSE;
			current->niter = niter;
			current->conv = 1;
		}
		else // if the loglik is a nan
		{
			cont = FALSE;
			current = old;
			current->niter = niter;
			current->conv = 4;
		}
		if ( (cont == TRUE) && (niter > limits->max_iter || t_total > limits->max_time)) // maximum iterations and no prior convergence criterions
		{
			cont = FALSE;
			current = old;
			current->niter = niter - 1;
			if (niter > limits->max_iter)
				current->conv = 2;
			else
				current->conv = 3;
		}
		t_current = t_new;
	}
	h2mm_mod *out = (h2mm_mod*) malloc(sizeof(h2mm_mod));
	out->nstate = current->nstate;
	out->ndet = current->ndet;
	out->nphot = current->nphot;
	out->loglik = current->loglik;
	out->niter = current->niter;
	out->conv = current->conv;
	out-> prior = (double*) malloc(in->nstate * sizeof(double));
	for ( i = 0; i < in->nstate; i++) out->prior[i] = current->prior[i];
	out->trans = (double*) malloc(in->nstate * in->nstate *sizeof(double));
	for ( i = 0; i < in->nstate * in->nstate; i++) out->trans[i] = current->trans[i];
	out->obs = (double*) malloc(in->nstate * in->ndet * sizeof(double));
	for ( i = 0; i < in->nstate * in->ndet; i++) out->obs[i] = current->obs[i];
#ifdef linux
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	if( h2mm_lock ) CloseHandle(h2mm_lock);
#endif
	free(burst_submit);
	free(pcp);
	free(pct);
	free(pco);
	free(pnp);
	free(pnt);
	free(pno);
	free(pop);
	free(pot);
	free(poo);
	free(pcur);
	free(pnew);
	free(pold);
	return out;
}
