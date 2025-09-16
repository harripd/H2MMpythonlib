// File: h2mm_functions.c
// Author: Paul David Harris
// Purpose: main wrapping functions to take burst data and submit to central H2MM algorithm
// Date created: 20 Oct 2022
// Date modified: 13 Sep 2025
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0


int h2mm_optimize(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst )
		limits->num_cores = num_burst;
	
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = allocate_models(3, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* old = &models[0];
	h2mm_mod* current = &models[1];
	h2mm_mod* new = &models[2];
	h2mm_mod* mod_temp;
	// loop over each model, and allocate  memory for prior, trans obs arrays
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new);
	
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	// Setup mutexes
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  *windowsThreadId = (DWORD*) calloc(limits->num_cores,sizeof(DWORD));
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif

	// setup input variable for threading
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fback_vals *burst_submit = (fback_vals*) calloc(limits->num_cores,sizeof(fback_vals));
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	for ( i=0; i < limits->num_cores; i++)
	{
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		gamma_var[i] = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = &gamma_var[i];
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].loglik = 0.0;
	}
	t_start = clock();
	t_current = t_start;
	while (conv == 0)
	{
		rho_all(current->nstate, current->trans, powers);
		// spin up threads for main calculation
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_create(&tid[i],NULL, fwd_bck_no_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_no_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++)
		{
			if (tid[i] != 0)
			{
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		current->conv |= CONVCODE_LLCOMPUTED;
		new->conv |= CONVCODE_FROMOPT;
		// check if converged
		conv = model_limits_func(new, current, old, t_total, limits, model_limits);
		if (conv == 0) // did not converge, so clean up for next iteration
		{
			if (print_func != NULL)
			{
				if (print_func(current->niter, new, current, old, t_iter, t_total, print_call) == -1)
				{
					conv = -6;
				}
			}
			// cycle arrays
			mod_temp = old;
			old = current;
			current = new;
			new = mod_temp;
			// update for next iteration
			zero_model(new);
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++)	
			{
				burst_submit[i].current = current;
				burst_submit[i].new = new;
			}
		}
		t_current = t_new;
	}
	// copy optimized model to out_model
	if (conv == 1){
		copy_model(old, out_model);
	}
	else{
		copy_model(current, out_model);
	}
	// free everything
	// free burst submit
	for (i = 0; i < limits->num_cores; i++)
	{
		free(burst_submit[i].alpha);
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
		free(gamma_var[i]);
	}
	free(burst_submit);
	free(bursts);
	free(gamma_var);
	free_models(3, models);
	free_powers(powers);
	// free mutexes and thread id's
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	free((void*) windowsThreadId);
	if( h2mm_lock ) 
		CloseHandle(h2mm_lock);
#endif
	if (burst_lock != NULL)
		free(burst_lock);
	return conv;
}

int h2mm_optimize_gamma(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst )
		limits->num_cores = num_burst;
	
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = allocate_models(3, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* old = &models[0];
	h2mm_mod* current = &models[1];
	h2mm_mod* new = &models[2];
	h2mm_mod* mod_temp;
	// loop over each model, and allocate  memory for prior, trans obs arrays
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new);
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	// Setup mutexes
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  *windowsThreadId = (DWORD*) calloc(limits->num_cores,sizeof(DWORD));
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif

	// setup input variable for threading
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fback_vals *burst_submit = (fback_vals*) calloc(limits->num_cores,sizeof(fback_vals));
	double **gamma_old = (double**) malloc(num_burst * sizeof(double*));
	double **gamma_cur = (*gamma == NULL) ? (double**) malloc(num_burst * sizeof(double*)) : *gamma;
	double **gamma_temp;
	for (i=0; i < num_burst; i++)
		gamma_old[i] = (double*) malloc(burst_sizes[i] * in_model->nstate * sizeof(double));
	if (*gamma == NULL){
		for (i=0; i < num_burst; i++)
			gamma_cur[i] = (double*) malloc(burst_sizes[i] * in_model->nstate * sizeof(double));
	}
	for ( i=0; i < limits->num_cores; i++)
	{
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = gamma_cur;
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].loglik = 0.0;
	}
	t_start = clock();
	t_current = t_start;
	while (conv == 0)
	{
		rho_all(current->nstate, current->trans, powers);
		// spin up threads for main calculation
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_create(&tid[i],NULL, fwd_bck_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++)
		{
			if (tid[i] != 0)
			{
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		current->conv |= CONVCODE_LLCOMPUTED;
		new->conv |= CONVCODE_FROMOPT;
		// idea for new code:
		conv = model_limits_func(new, current, old, t_total, limits, model_limits);
		if (conv == 0)
		{
			if (print_func != NULL)
			{
				if (print_func(current->niter, new, current, old, t_iter, t_total, print_call) == -1)
				{
					conv = -6;
				}
			}
			// cycle arrays
			mod_temp = old;
			old = current;
			current = new;
			new = mod_temp;
			// update for next iteration
			zero_model(new);
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++)	
			{
				burst_submit[i].current = current;
				burst_submit[i].new = new;
				burst_submit[i].gamma = gamma_old;
			}
			gamma_temp = gamma_cur;
			gamma_cur = gamma_old;
			gamma_old = gamma_temp;
		}
		t_current = t_new;
	}
	// copy optimized model to out_model, and gamma
	if (conv == 1)
	{
		copy_model(old, out_model);
		if (*gamma == NULL)
			*gamma = gamma_old;
		else if  (*gamma != gamma_old)
			transfer_gamma(num_burst, burst_sizes, gamma_old, *gamma);
	}
	else if (conv == 2)
	{
		copy_model(current, out_model);
		if (*gamma == NULL)
			*gamma = gamma_cur;
		else if (*gamma != gamma_cur)
			transfer_gamma(num_burst, burst_sizes, gamma_cur, *gamma);
	}
	// free everything
	if (*gamma != gamma_old)
	{
		free_gamma(num_burst, gamma_old);
		gamma_old = NULL;
	}
	if (*gamma != gamma_cur)
	{
		free_gamma(num_burst, gamma_cur);
		gamma_old = NULL;
	}
	for (i = 0; i < limits->num_cores; i++)
	{
		free(burst_submit[i].alpha);
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
	}
	free(burst_submit);
	free(bursts);
	free_models(3, models);
	free_powers(powers);
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	free((void*) windowsThreadId);
	if( h2mm_lock ) 
		CloseHandle(h2mm_lock);
#endif
	if (burst_lock != NULL)
		free(burst_lock);
	return conv;
}

int h2mm_optimize_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst )
		limits->num_cores = num_burst;
	
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = (*out_models == NULL) ? allocate_models(limits->max_iter + 2 - in_model->niter, in_model->nstate, in_model->ndet, nphot) : *out_models; // initial array, makes easier to free later
	h2mm_mod* current = models;
	h2mm_mod* new = models + 1;
	h2mm_mod* old = allocate_models(1, in_model->nstate, in_model->ndet, nphot);
	old->loglik = -INFINITY;
	h2mm_mod* pold = old;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	// Setup mutexes
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  *windowsThreadId = (DWORD*) calloc(limits->num_cores,sizeof(DWORD));
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif

	// setup input variable for threading
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fback_vals *burst_submit = (fback_vals*) calloc(limits->num_cores,sizeof(fback_vals));
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	for ( i=0; i < limits->num_cores; i++)
	{
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		gamma_var[i] = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = &gamma_var[i];
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate* in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].loglik = 0.0;
	}
	t_start = clock();
	t_current = t_start;
	while (conv == 0)
	{
		zero_model(new);
		rho_all(current->nstate, current->trans, powers);
		// spin up threads for main calculation
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_create(&tid[i],NULL, fwd_bck_no_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_no_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++)
		{
			if (tid[i] != 0)
			{
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		current->conv |= CONVCODE_LLCOMPUTED;
		new->conv |= CONVCODE_FROMOPT;
		conv = model_limits_func(new, current, old, t_total, limits, model_limits);
		if (conv == 0)
		{
			if (print_func != NULL)
			{
				if (print_func(current->niter, new, current, old, t_iter, t_total, print_call) == -1)
				{
					conv = -6;
				}
			}
			// cycle arrays
			old = current;
			current++;
			new++;
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++)	
			{
				burst_submit[i].current = current;
				burst_submit[i].new = new;
			}
		}
		t_current = t_new;
	}
	// copy optimized model to out_model
	
	// free everything
	// free burst submit
	for (i = 0; i < limits->num_cores; i++)
	{
		free(burst_submit[i].alpha);
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
		free(gamma_var[i]);
	}
	*out_models = models;
	free(burst_submit);
	free(bursts);
	free(gamma_var);
	free_models(1, pold);
	free_powers(powers);
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	free((void*) windowsThreadId);
	if( h2mm_lock ) 
		CloseHandle(h2mm_lock);
#endif
	if (burst_lock != NULL)
		free(burst_lock);
	return conv;
}

int h2mm_optimize_gamma_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst )
		limits->num_cores = num_burst;
	
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = (*out_models == NULL) ? allocate_models(limits->max_iter + 2 - in_model->niter, in_model->nstate, in_model->ndet, nphot) : *out_models; // initial array, makes easier to free later
	h2mm_mod* current = models;
	h2mm_mod* new = models + 1;
	h2mm_mod* old = allocate_models(1, in_model->nstate, in_model->ndet, nphot);
	h2mm_mod* pold = old;
	// loop over each model, and allocate  memory for prior, trans obs arrays
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	// Setup mutexes
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  *windowsThreadId = (DWORD*) calloc(limits->num_cores,sizeof(DWORD));
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	
	// setup input variable for threading
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fback_vals *burst_submit = (fback_vals*) malloc(limits->num_cores * sizeof(fback_vals));
	double **gamma_old = (double**) malloc(num_burst * sizeof(double*));
	double **gamma_cur = (*gamma == NULL) ? (double**) malloc(num_burst * sizeof(double*)) : *gamma;
	double **gamma_temp;
	for (i=0; i < num_burst; i++)
		gamma_old[i] = (double*) malloc(burst_sizes[i] * in_model->nstate * sizeof(double));
	if (*gamma == NULL){
		for (i=0; i < num_burst; i++)
			gamma_cur[i] = (double*) malloc(burst_sizes[i] * in_model->nstate * sizeof(double));
	}
	for ( i=0; i < limits->num_cores; i++)
	{
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = gamma_cur;
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].loglik = 0.0;
	}
	t_start = clock();
	t_current = t_start;
	while (conv == 0)
	{
		zero_model(new);
		rho_all(current->nstate, current->trans, powers);
		// spin up threads for main calculation
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_create(&tid[i],NULL, fwd_bck_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) 
		{
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++)
		{
			if (tid[i] != 0)
			{
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		current->conv |= CONVCODE_LLCOMPUTED;
		new->conv |= CONVCODE_FROMOPT;
		conv = model_limits_func(new, current, old, t_total, limits, model_limits);
		if (conv == 0)
		{
			if (print_func != NULL)
			{
				if (print_func(current->niter, new, current, old, t_iter, t_total, print_call) == -1)
				{
					conv = -6;
				}
			}
			// cycle arrays
			old = current;
			current++;
			new++;
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++)	
			{
				burst_submit[i].current = current;
				burst_submit[i].new = new;
				burst_submit[i].gamma = gamma_old;
			}
			gamma_temp = gamma_cur;
			gamma_cur = gamma_old;
			gamma_old = gamma_temp;
		}
		t_current = t_new;
	}
	// copy optimized model to out_model
	*out_models = models;
	if (conv == 1)
	{
		if (*gamma == NULL)
			*gamma = gamma_old;
		else if (*gamma != gamma_old)
			transfer_gamma(num_burst, burst_sizes, gamma_old, *gamma);
	}
	else if (conv == 2)
	{
		if (*gamma == NULL)
			*gamma = gamma_cur;
		else if (*gamma != gamma_cur)
			transfer_gamma(num_burst, burst_sizes, gamma_cur, *gamma);
	}
	else
	{
		for (i = 0; i < num_burst; i++)
		{
			if (gamma_old[i] != NULL)
			{
				free(gamma_old[i]);
				gamma_old[i] = NULL;
			}
			if (gamma_cur[i] != NULL)
			{
				free(gamma_cur[i]);
				gamma_cur[i] = NULL;
			}
		}
	}
	// free everything
	if (*gamma != gamma_old)
	{
		free_gamma(num_burst, gamma_old);
		gamma_old = NULL;
	}
	if (*gamma != gamma_cur)
	{
		free_gamma(num_burst, gamma_cur);
		gamma_cur = NULL;
	}
	for (i = 0; i < limits->num_cores; i++)
	{
		free(burst_submit[i].alpha);
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
	}
	free(burst_submit);
	free(bursts);
	free_models(1, pold);
	free_powers(powers);
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	free((void*) windowsThreadId);
	if( h2mm_lock ) 
		CloseHandle(h2mm_lock);
#endif
	if (burst_lock != NULL)
		free(burst_lock);
	return conv;
}
