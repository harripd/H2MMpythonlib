// File: model_array.c
// Author: Paul David Harris
// Purpose: Create a state path through data given a prior and trans array
// Created: 24 Oct 2022
// Modified: 15 Nov 2022

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

int calc_multi(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, int64_t num_models, h2mm_mod *models, lm *limits)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t i, j;
	int multi_state = FALSE;
	int multi_det = FALSE;
	uint8_t max_det = get_max_det(num_burst, bursts);
	int64_t nphot = check_det(num_burst, bursts, models); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	for (i=1; i < num_models; i++)
	{
		if (models[0].ndet != models[i].ndet)
		{
			multi_det = TRUE;
			if (models[i].ndet < (int64_t) max_det)
			{
				if (bursts != NULL)
					free(bursts);
				return -2;
			}
		}
		if (models[0].nstate != models[i].nstate)
		{
			multi_state = TRUE;
			multi_det = TRUE; // technically not true, however, for purposes of reallocating memory, if it is multi-state it will also be multi-det
		}
	}
	if (num_burst < limits ->num_cores)
		limits->num_cores = num_burst;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	pwrs* powers;
	fback_vals *burst_submit = (fback_vals*) malloc(limits->num_cores * sizeof(fback_vals));
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	if (!multi_state) // when not mutli-state, can allocate arrays for alpha/beta/gamma all at once
	{
		powers = allocate_powers(&models[0], max_delta);
		for (i=0; i < limits->num_cores; i++)
		{
			burst_submit[i].phot = bursts;
			burst_submit[i].max_phot = max_phot;
			burst_submit[i].sk = powers->sk;
			burst_submit[i].sj = powers->sj;
			burst_submit[i].si = powers->si;
			burst_submit[i].sT = powers->sT;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			//~ burst_submit[i].current = models[0]; // will be set each round
			//~ burst_submit[i].new = new; // no need to have new model when only calculating loglik
			burst_submit[i].alpha = (double*) malloc(max_phot * models[0].nstate * sizeof(double));
			burst_submit[i].beta = (double*) malloc(max_phot * models[0].nstate * sizeof(double));
			gamma_var[i] = (double*) malloc(max_phot * models[0].nstate * sizeof(double));
			burst_submit[i].gamma = &gamma_var[i];
			burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
			burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
			burst_submit[i].prior = (double*) calloc(models[0].nstate, sizeof(double));
		}
		
	}
	if (!multi_det)
	{
		for (i=0; i < limits->num_cores; i++)
			burst_submit[i].obs_temp = (double*) calloc(models[0].nstate * models[0].ndet, sizeof(double));
	}
	for (i=0; i < limits->num_cores; i++)
	{
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].loglik = 0.0;
	}
	for (i = 0; i < num_models; i++)
	{
		models[i].loglik = 0.0;
		for (j=0; j < limits->num_cores; j++)
		{
			burst_submit[j].current = &models[i];
		}
		if (multi_state)
		{
			powers = allocate_powers(&models[i], max_delta);
			for (j=0; j < limits->num_cores; j++)
			{
				burst_submit[j].phot = bursts;
				burst_submit[j].max_phot = max_phot;
				burst_submit[j].sk = powers->sk;
				burst_submit[j].sj = powers->sj;
				burst_submit[j].si = powers->si;
				burst_submit[j].sT = powers->sT;
				burst_submit[j].A = powers->A;
				burst_submit[j].Rho = powers->Rho;
				//~ burst_submit[i].current = models[0]; // always set anew
				//~ burst_submit[j].new = NULL; // no need to allocate new model when just calculating loglik
				burst_submit[j].alpha = (double*) malloc(max_phot * models[i].nstate * sizeof(double));
				burst_submit[j].beta = (double*) malloc(max_phot * models[i].nstate * sizeof(double));
				gamma_var[j] = (double*) malloc(max_phot * models[i].nstate * sizeof(double));
				burst_submit[j].gamma = &gamma_var[i];
				burst_submit[j].xi_temp = (double*) malloc(powers->sj * sizeof(double));
				burst_submit[j].xi_summed = (double*) calloc(powers->sj, sizeof(double));
				burst_submit[j].prior = (double*) calloc(models[i].nstate, sizeof(double));
			}
		}
		if (multi_det)
		{
			for (j=0; j < limits->num_cores; j++)
			{
				burst_submit[j].obs_temp = (double*) calloc(models[i].nstate * models[i].ndet, sizeof(double*));
			}
		}
		burst_lock->cur_burst = 0; // reset cur_burst to 0 for next round of optimization
		rho_all(models[i].nstate, models[i].trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(j = 0; j < limits->num_cores; j++) 
		{
			pthread_create(&tid[j],NULL, fwd_only,(void*) &burst_submit[j]); // create a thread for each burst
			//~ printf("pthread[%ld] res=%d\n", j, res);
		}
		for(j = 0; j < limits->num_cores; j++) 
		{
			pthread_join(tid[j],NULL); // wait for all bursts to finish
			//~ printf("join[%ld] res=%d\n", j, res);	
		}
#elif _WIN32
		for (j = 0; j < limits->num_cores; j++)
			tid[j] = CreateThread(NULL, 0, fwd_only, (LPVOID)&burst_submit[j], 0, (LPDWORD)&windowsThreadId);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (j = 0; j < limits->num_cores; j++)
		{
			if (tid[j] != 0)
			{
				CloseHandle(tid[j]);
			}
		}
#endif
		models[i].conv |= CONVCODE_LLCOMPUTED;
		models[i].nphot = nphot;
		if (multi_state)
		{
			free_powers(powers);
			for (j = 0; j < limits->num_cores; j++)
			{
				if (burst_submit[j].alpha != NULL)
					free(burst_submit[j].alpha);
				if (burst_submit[j].beta != NULL)
					free(burst_submit[j].beta);
				if (gamma_var[j] != NULL)
					free(gamma_var[j]);
				if (burst_submit[j].xi_temp != NULL)
					free(burst_submit[j].xi_temp);
				if (burst_submit[j].xi_summed != NULL)
					free(burst_submit[j].xi_summed);
				if (burst_submit[j].prior != NULL)
					free(burst_submit[j].prior);
			}
		}
		if (multi_det)
		{
			for (j = 0; j < limits->num_cores; j++)
			{
				if (burst_submit[j].obs_temp != NULL)
					free(burst_submit[j].obs_temp);
			}
		}
	}
	if (!multi_state)
	{
		free_powers(powers);
		for (i = 0; i < limits->num_cores; i++)
		{
			if (burst_submit[i].alpha != NULL)
				free(burst_submit[i].alpha);
			if (burst_submit[i].beta != NULL)
				free(burst_submit[i].beta);
			if (gamma_var[i] != NULL)
				free(gamma_var[i]);
			if (burst_submit[i].xi_temp != NULL)
				free(burst_submit[i].xi_temp);
			if (burst_submit[i].xi_summed != NULL)
				free(burst_submit[i].xi_summed);
			if (burst_submit[i].prior != NULL)
				free(burst_submit[i].prior);
		}
	}
	if (!multi_det)
	{
		for (i = 0; i < limits->num_cores; i++)
		{
			if (burst_submit[i].obs_temp != NULL)
				free(burst_submit[i].obs_temp);
		}
	}
	if (gamma_var != NULL)
	{
		free(gamma_var);
		gamma_var = NULL;
	}
	if (bursts != NULL)
	{
		free(bursts);
		bursts = NULL;
	}
	if (burst_submit != NULL)
	{
		free(burst_submit);
		burst_submit=NULL;
	}
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	if( h2mm_lock ) 
		CloseHandle(h2mm_lock);
#endif
	if (burst_lock != NULL)
	{
		free(burst_lock);
		burst_lock = NULL;
	}
	return 0;
}


int calc_multi_gamma(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, int64_t num_models, h2mm_mod *models, double ****gamma, lm *limits)
{
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) // bad pointer in the data
		return -1;
	int64_t i, j;
	int multi_state = FALSE;
	int multi_det = FALSE;
	uint8_t max_det = get_max_det(num_burst, bursts);
	int64_t nphot = check_det(num_burst, bursts, models); // verify detectors do not exceed ndet in model
	if (nphot == 0) 
		return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	for (i=1; i < num_models; i++)
	{
		if (models[0].ndet != models[i].ndet)
		{
			multi_det = TRUE;
			if (models[i].ndet < (int64_t) max_det)
			{
				if (bursts != NULL)
					free(bursts);
				return -2;
			}
		}
		if (models[0].nstate != models[i].nstate)
		{
			multi_state = TRUE;
			multi_det = TRUE; // technically not true, however, for purposes of reallocating memory, if it is multi-state it will also be multi-det
		}
	}
	if (num_burst < limits ->num_cores)
		limits->num_cores = num_burst;
#if defined(__linux__) || defined(__APPLE__)
	pthread_t *tid = (pthread_t*) malloc(limits->num_cores * sizeof(pthread_t));
	pthread_mutex_t *h2mm_lock = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(h2mm_lock,NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(limits->num_cores, sizeof(HANDLE));
	DWORD  windowsThreadId = 0;
	HANDLE h2mm_lock = CreateMutex(NULL, FALSE, NULL);
#endif
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	pwrs* powers;
	fback_vals *burst_submit = (fback_vals*) malloc(limits->num_cores * sizeof(fback_vals));
	double ***gamma_var = (*gamma == NULL) ? (double***) malloc(num_models * sizeof(double**)) : *gamma;
	h2mm_mod *dummy_model;
	if (!multi_state) // when not mutli-state, can allocate arrays for alpha/beta/gamma all at once
	{
		powers = allocate_powers(&models[0], max_delta);
		for (i=0; i < limits->num_cores; i++)
		{
			burst_submit[i].phot = bursts;
			burst_submit[i].max_phot = max_phot;
			burst_submit[i].sk = powers->sk;
			burst_submit[i].sj = powers->sj;
			burst_submit[i].si = powers->si;
			burst_submit[i].sT = powers->sT;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			//~ burst_submit[i].current = models[0]; // will be set each round
			burst_submit[i].alpha = (double*) malloc(max_phot * models[0].nstate * sizeof(double));
			burst_submit[i].beta = (double*) malloc(max_phot * models[0].nstate * sizeof(double));
			burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
			//~ burst_submit[i].gamma = &gamma_var[i]; // do not allocate gamma, as need new gamma for each model
			burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
			burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
			burst_submit[i].prior = (double*) calloc(models[0].nstate, sizeof(double));
		}
		
	}
	if (!multi_det)
	{
		dummy_model = allocate_models(1, models[0].nstate, models[0].ndet, nphot);
		for (i=0; i < limits->num_cores; i++)
		{
			burst_submit[i].new = dummy_model;
			burst_submit[i].obs_temp = (double*) calloc(models[0].nstate * models[0].ndet, sizeof(double));
		}
	}
	for (i=0; i < limits->num_cores; i++)
	{
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].loglik = 0.0;
	}
	for (i = 0; i < num_models; i++)
	{
		// allocate gamma
		if (*gamma == NULL){
			gamma_var[i] = (double**) malloc(num_burst * sizeof(double*));
			for (j=0; j < num_burst; j++)
			{
				gamma_var[i][j] = (double*) malloc(models[i].nstate * bursts[j].nphot * sizeof(double));
			}
		}
	}
	for (i = 0; i < num_models; i++)
	{
		models[i].loglik = 0.0;
		for (j=0; j < limits->num_cores; j++)
		{
			burst_submit[j].current = &models[i];
			burst_submit[j].gamma = gamma_var[i];
		}
		if (multi_state)
		{
			powers = allocate_powers(&models[i], max_delta);
			for (j=0; j < limits->num_cores; j++)
			{
				burst_submit[j].phot = bursts;
				burst_submit[j].max_phot = max_phot;
				burst_submit[j].sk = powers->sk;
				burst_submit[j].sj = powers->sj;
				burst_submit[j].si = powers->si;
				burst_submit[j].sT = powers->sT;
				burst_submit[j].A = powers->A;
				burst_submit[j].Rho = powers->Rho;
				//~ burst_submit[i].current = models[0]; // always set anew
				burst_submit[j].burst_lock = burst_lock;
				burst_submit[j].alpha = (double*) malloc(max_phot * models[i].nstate * sizeof(double));
				burst_submit[j].beta = (double*) malloc(max_phot * models[i].nstate * sizeof(double));
				burst_submit[j].b = (double*) malloc(powers->sk * sizeof(double));
				burst_submit[j].xi_temp = (double*) malloc(powers->sj * sizeof(double));
				burst_submit[j].xi_summed = (double*) calloc(powers->sj, sizeof(double));
				burst_submit[j].prior = (double*) calloc(models[i].nstate, sizeof(double));
			}
		}
		if (multi_det)
		{
			dummy_model = allocate_models(1, models[i].nstate, models[i].ndet, nphot);
			for (j=0; j < limits->num_cores; j++)
			{
				burst_submit[j].new = dummy_model;
				burst_submit[j].obs_temp = (double*) calloc(models[i].nstate * models[i].ndet, sizeof(double*));
			}
		}
		burst_lock->cur_burst = 0;
		rho_all(models[i].nstate, models[i].trans, powers);
		//~ fwd_bck_gamma((void*) burst_submit);
#if defined(__linux__) || defined(__APPLE__)
		for(j = 0; j < limits->num_cores; j++) 
		{
			pthread_create(&tid[j], NULL, fwd_bck_gamma,(void*) &burst_submit[j]); // create a thread for each burst
		}
		for(j = 0; j < limits->num_cores; j++) 
		{
			pthread_join(tid[j],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (j = 0; j < limits->num_cores; j++)
			tid[j] = CreateThread(NULL, 0, fwd_bck_gamma, (LPVOID)&burst_submit[j], 0, (LPDWORD)&windowsThreadId);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (j = 0; j < limits->num_cores; j++)
		{
			if (tid[j] != 0)
			{
				CloseHandle(tid[j]);
			}
		}
#endif
		models[i].conv |= CONVCODE_LLCOMPUTED;
		models[i].nphot = nphot;
		// free the necessary variables
		if (multi_state)
		{
			free_powers(powers);
			for (j = 0; j < limits->num_cores; j++)
			{
				if (burst_submit[j].alpha != NULL)
					free(burst_submit[j].alpha);
				if (burst_submit[j].beta != NULL)
					free(burst_submit[j].beta);
				if (burst_submit[j].b != NULL)
					free(burst_submit[j].b);
				if (burst_submit[j].xi_temp != NULL)
					free(burst_submit[j].xi_temp);
				if (burst_submit[j].xi_summed != NULL)
					free(burst_submit[j].xi_summed);
				if (burst_submit[j].prior != NULL)
					free(burst_submit[j].prior);
			}
		}
		if (multi_det)
		{
			free_models(1, dummy_model);
			for (j = 0; j < limits->num_cores; j++)
			{
				if (burst_submit[j].obs_temp != NULL)
					free(burst_submit[j].obs_temp);
			}
		}
	}
	*gamma = gamma_var;
	if (!multi_state)
	{
		if (powers != NULL)
			free_powers(powers);
		for (i=0; i < limits->num_cores; i++)
		{
			if (burst_submit[i].alpha != NULL)
				free(burst_submit[i].alpha);
			if (burst_submit[i].beta != NULL)
				free(burst_submit[i].beta);
			if (burst_submit[i].b != NULL)
				free(burst_submit[i].b);
			if (burst_submit[i].xi_temp != NULL)
				free(burst_submit[i].xi_temp);
			if (burst_submit[i].xi_summed != NULL)
				free(burst_submit[i].xi_summed);
			if (burst_submit[i].prior != NULL)
				free(burst_submit[i].prior);
		}
	}
	if (!multi_det)
	{
		free_models(1, dummy_model);
		for (i=0; i < limits->num_cores; i++)
		{
			if (burst_submit[i].obs_temp != NULL)
				free(burst_submit[i].obs_temp);
		}
	}
	if (burst_submit != NULL)
	{
		free(burst_submit);
		burst_submit = NULL;
	}
	if (bursts != NULL)
	{
		free(bursts);
		bursts = NULL;
	}
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	if( h2mm_lock ) 
		CloseHandle(h2mm_lock);
#endif
	if (burst_lock != NULL)
	{
		free(burst_lock);
		burst_lock = NULL;
	}
	return 0;
}
