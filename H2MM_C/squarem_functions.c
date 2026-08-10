// File: squarem_functions.c
// Author: Paul David Harris
// Purpose: main wrapping functions to take burst data and submit to central H2MM algorithm with SQUAREM acceleration
// Date created: 07 Aug 2026
// Date modified: 07 Aug 2026
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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


int h2mm_squarem(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
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
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst )
		limits->num_cores = num_burst;
	
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = allocate_models(7, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* old = &models[0];
	h2mm_mod* current = &models[1];
	h2mm_mod* new0 = &models[2];
	h2mm_mod* new1 = &models[3];
	h2mm_mod* newSQ = &models[4];
	h2mm_mod* r = &models[5];
	h2mm_mod* v = &models[6];
	h2mm_mod *temp_old, *temp_current;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	fbacka_vals *burst_submit = (fbacka_vals*) calloc(limits->num_cores,sizeof(fbacka_vals));
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	for ( i = 0; i < num_burst; i++) {
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
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
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
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
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		temp_old = old;
		temp_current = current;
		// *************************
		// * 1st Calcuation (new0) *
		// *************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_only,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_only, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if (conv) {
			break;
		}
		// **************************
		// * 2nd Calculation (new1) *
		// **************************
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
		}
		rho_all(new0->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_no_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_no_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if (conv) {
			// converged, or error, so prepare for exit
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		zero_model(old);
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = old;
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			new0->conv |= CONVCODE_OUTPUT | CONVCODE_ERROR;
			conv = 2;
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// project newSQ
		if ( !(convSQ = project_squarem(current, new0, new1, newSQ, v, r)) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alphaSQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if (! (newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		old = new0;
		new0 = temp_old; // remember that old was zeroed in new1 ll evaluation, so new0 is already zeroed for next itteration
		if (convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			new1 = temp_current;
			powersnext = powers;
			alphanext = alpha;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			newSQ = temp_current;
			powersnext = powersSQ;
			alphanext = alphaSQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	// copy optimized model to out_model
	if (conv == 1) {
		copy_model(old, out_model);
	}
	else {
		copy_model(current, out_model);
	}
	// free everything
	// free burst submit
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
		free(gamma_var[i]);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(alpha);
	free(alphaSQ);
	free(burst_submit);
	free(bursts);
	free(gamma_var);
	free_models(7, models);
	free_powers(powers);
	free_powers(powersSQ);
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


int h2mm_squarem_ll(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double *llarr, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
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
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = allocate_models(7, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* old = &models[0];
	h2mm_mod* current = &models[1];
	h2mm_mod* new0 = &models[2];
	h2mm_mod* new1 = &models[3];
	h2mm_mod* newSQ = &models[4];
	h2mm_mod* r = &models[5];
	h2mm_mod* v = &models[6];
	h2mm_mod *temp_old, *temp_current;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	fbackall_vals *burst_submit = (fbackall_vals*) calloc(limits->num_cores,sizeof(fbackall_vals));
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	double *llarr_n0 = (double*) malloc(num_burst * sizeof(double));
	double *llarr_SQ = (double*) malloc(num_burst * sizeof(double));
	double *llarr_next = llarr;
	double *llarr_out;
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	for ( i = 0; i < num_burst; i++) {
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
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
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		gamma_var[i] = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = &gamma_var[i];
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].llarr = llarr;
		burst_submit[i].loglik = 0.0;
	}
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		temp_old = old;
		temp_current = current;
		// *************************
		// * 1st Calcuation (new0) *
		// *************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_ll_only,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_ll_only, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		// update times
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		// evaluate for convergence
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if (conv) {
			if (conv == 1) {
				llarr_out = llarr_n0;
			}
			else {
				llarr_out = llarr_next;
			}
			break;
		}
		// **************************
		// * 2nd Calculation (new1) *
		// **************************
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
			burst_submit[i].llarr = llarr_n0;
		}
		rho_all(new0->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if (conv) {
			// converged, or error, so prepare for exit
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			llarr_out = ( conv == 1 ) ? llarr_next : llarr_n0;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		zero_model(old);
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = old;
			burst_submit[i].llarr = llarr;
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			new0->conv |= CONVCODE_OUTPUT | CONVCODE_ERROR;
			conv = 2;
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			llarr_out = llarr_n0;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// project newSQ
		if ( !(convSQ = project_squarem(current, new0, new1, newSQ, v, r)) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alphaSQ;
				burst_submit[i].llarr = llarr_SQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if (! (newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		old = new0;
		new0 = temp_old; // rememeber that old was zeroed in new1 ll evaluation, so new0 is already zeroed for next itteration
		if (convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			new1 = temp_current;
			powersnext = powers;
			alphanext = alpha;
			llarr_next = llarr;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			newSQ = temp_current;
			powersnext = powersSQ;
			alphanext = alphaSQ;
			llarr_next = llarr_SQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
			burst_submit[i].llarr = llarr_next;
		}
	}
	// copy optimized model to out_model
	if (conv == 1) {
		copy_model(old, out_model);
	}
	else {
		copy_model(current, out_model);
	}
	if (llarr != llarr_out) memcpy(llarr, llarr_out, num_burst * sizeof(double));
	// free everything
	// free burst submit
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
		free(gamma_var[i]);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(llarr_n0);
	free(llarr_SQ);
	free(alpha);
	free(alphaSQ);
	free(burst_submit);
	free(bursts);
	free(gamma_var);
	free_models(7, models);
	free_powers(powers);
	free_powers(powersSQ);
	// free mutexes and thread id's
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_destroy(h2mm_lock);
	if (h2mm_lock != NULL)
		free(h2mm_lock);
	free(tid);
#elif _WIN32
	free((void*)tid);
	free((void*) windowsThreadId);
	if( h2mm_lock ) {
		CloseHandle(h2mm_lock);
	}
#endif
	if (burst_lock != NULL) {
		free(burst_lock);
	}
	return conv;
}


int h2mm_squarem_gamma(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call) {
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) return -1; // bad pointer in the data
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = allocate_models(7, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* old = &models[0];
	h2mm_mod* current = &models[1];
	h2mm_mod* new0 = &models[2];
	h2mm_mod* new1 = &models[3];
	h2mm_mod* newSQ = &models[4];
	h2mm_mod* r = &models[5];
	h2mm_mod* v = &models[6];
	h2mm_mod *temp_old, *temp_current;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	fbacka_vals *burst_submit = (fbacka_vals*) calloc(limits->num_cores,sizeof(fbacka_vals));
	double **gamma_n0 = (double**) malloc(num_burst*sizeof(double*));
	double **gamma_cur = (*gamma != NULL) ? *gamma : (double**) malloc(num_burst*sizeof(double*));
	double **gamma_SQ = (double**) malloc(num_burst*sizeof(double*));
	double **gamma_next = gamma_cur;
	double **gamma_out;
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	for ( i = 0; i < num_burst; i++) {
		gamma_n0[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		gamma_SQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
	}
	if ( *gamma == NULL ){
		for ( i = 0; i < num_burst; i++) {
			gamma_cur[i] = (double*) malloc(in_model->nstate*burst_sizes[i]*sizeof(double));
		}
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
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = gamma_cur;
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].loglik = 0.0;
	}
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		temp_old = old;
		temp_current = current;
		// *************************
		// * 1st Calcuation (new0) *
		// *************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if ( conv ) {
			gamma_out = (conv == 1) ? gamma_n0 : gamma_next;
			break;
		}
		// **************************
		// * 2nd Calculation (new1) *
		// **************************
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
			burst_submit[i].gamma = gamma_n0;
		}
		rho_all(new0->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if (conv) {
			// converged, or error, so prepare for exit
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			gamma_out = (conv == 1) ? gamma_next : gamma_n0;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		zero_model(old);
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = old;
			burst_submit[i].gamma = gamma_cur;
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			conv = 2;
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			gamma_out = gamma_n0;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// project newSQ
		if ( !(convSQ = project_squarem(current, new0, new1, newSQ, v, r)) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alphaSQ;
				burst_submit[i].gamma = gamma_SQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if (! (newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		old = new0;
		new0 = temp_old; // remember that old was zeroed in new1 ll evaluation, so new0 is already zeroed for next itteration
		if (convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			new1 = temp_current;
			powersnext = powers;
			alphanext = alpha;
			gamma_next = gamma_cur;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			newSQ = temp_current;
			powersnext = powersSQ;
			alphanext = alphaSQ;
			gamma_next = gamma_SQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
			burst_submit[i].gamma = gamma_next;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	// copy optimized model to out_model
	if (conv == 1) {
		copy_model(old, out_model);
	}
	else {
		copy_model(current, out_model);
	}
	if ( *gamma == NULL ){
		*gamma = gamma_out;
	}
	else if ( *gamma != gamma_out ){
		transfer_gamma(in_model->nstate, num_burst, burst_sizes, gamma_out, *gamma);
	}
	// free everything
	// free burst submit
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(alpha);
	free(alphaSQ);
	free_gamma(num_burst, gamma_n0);
	free_gamma(num_burst, gamma_SQ);
	free(burst_submit);
	free(bursts);
	free_models(7, models);
	free_powers(powers);
	free_powers(powersSQ);
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


int h2mm_squarem_ll_gamma(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double *llarr, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call) {
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) return -1; // bad pointer in the data
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) return -2;
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;
	// Allocate old, current, and new h2mm_mod
	h2mm_mod* models = allocate_models(7, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* old = &models[0];
	h2mm_mod* current = &models[1];
	h2mm_mod* new0 = &models[2];
	h2mm_mod* new1 = &models[3];
	h2mm_mod* newSQ = &models[4];
	h2mm_mod* r = &models[5];
	h2mm_mod* v = &models[6];
	h2mm_mod *temp_old, *temp_current;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	fbackall_vals *burst_submit = (fbackall_vals*) calloc(limits->num_cores,sizeof(fbackall_vals));
	double *llarr_n0 = (double*) malloc(num_burst*sizeof(double));
	double *llarr_SQ = (double*) malloc(num_burst*sizeof(double));
	double *llarr_next = llarr;
	double *llarr_out;
	double **gamma_n0 = (double**) malloc(num_burst*sizeof(double*));
	double **gamma_cur = (*gamma != NULL) ? *gamma : (double**) malloc(num_burst*sizeof(double*));
	double **gamma_SQ = (double**) malloc(num_burst*sizeof(double*));
	double **gamma_next = gamma_cur;
	double **gamma_out;
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	for ( i = 0; i < num_burst; i++) {
		gamma_n0[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		gamma_SQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
	}
	if ( *gamma == NULL ){
		for ( i = 0; i < num_burst; i++) {
			gamma_cur[i] = (double*) malloc(in_model->nstate*burst_sizes[i]*sizeof(double));
		}
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
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = gamma_cur;
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].llarr = llarr;
		burst_submit[i].loglik = 0.0;
	}
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		temp_old = old;
		temp_current = current;
		// *************************
		// * 1st Calcuation (new0) *
		// *************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_ll_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_ll_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if ( conv ) {
			if (conv == 1) {
				gamma_out = gamma_n0;
				llarr_out = llarr_n0;
			}
			else {
				gamma_out = gamma_next;
				llarr_out = llarr_next;
			}
			break;
		}
		// **************************
		// * 2nd Calculation (new1) *
		// **************************
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
			burst_submit[i].gamma = gamma_n0;
			burst_submit[i].llarr = llarr_n0;
		}
		rho_all(new0->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_ll_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_ll_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				conv = -6;
			}
		}
		if (conv) {
			// converged, or error, so prepare for exit
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			gamma_out = (conv == 1) ? gamma_next : gamma_n0;
			llarr_out = (conv == 1) ? llarr_next : llarr_n0;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		zero_model(old);
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = old;
			burst_submit[i].gamma = gamma_cur;
			burst_submit[i].llarr = llarr;
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			conv = 2;
			old = current;
			current = new0;
			new0 = new1;
			new1 = temp_old;
			gamma_out = gamma_n0;
			llarr_out = llarr_n0;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// project newSQ
		if ( !(convSQ = project_squarem(current, new0, new1, newSQ, v, r)) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alphaSQ;
				burst_submit[i].gamma = gamma_SQ;
				burst_submit[i].llarr = llarr_SQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if (! (newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		old = new0;
		new0 = temp_old; // remember that old was zeroed in new1 ll evaluation, so new0 is already zeroed for next itteration
		if (convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			new1 = temp_current;
			powersnext = powers;
			alphanext = alpha;
			gamma_next = gamma_cur;
			llarr_next = llarr;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			newSQ = temp_current;
			powersnext = powersSQ;
			alphanext = alphaSQ;
			gamma_next = gamma_SQ;
			llarr_next = llarr_SQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
			burst_submit[i].gamma = gamma_next;
			burst_submit[i].llarr = llarr_next;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	// copy optimized model to out_model
	if (conv == 1) {
		copy_model(old, out_model);
	}
	else {
		copy_model(current, out_model);
	}
	if ( *gamma == NULL ){
		*gamma = gamma_out;
	}
	else if ( *gamma != gamma_out ){
		transfer_gamma(in_model->nstate, num_burst, burst_sizes, gamma_out, *gamma);
	}
	if (llarr != llarr_out) memcpy((void*) llarr, (void*) llarr_out, num_burst*sizeof(double));
	// free everything
	// free burst submit
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(alpha);
	free(alphaSQ);
	free_gamma(num_burst, gamma_n0);
	free_gamma(num_burst, gamma_SQ);
	free(llarr_n0);
	free(llarr_SQ);
	free(burst_submit);
	free(bursts);
	free_models(7, models);
	free_powers(powers);
	free_powers(powersSQ);
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


int h2mm_squarem_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	if (limits->max_iter < 1) return -6;
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) return -1; // bad pointer in the data
		
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) {
		free(bursts);
		return -2;
	}
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;

	// Allocate old, current, and new h2mm_mod
	h2mm_mod* modelsrv = allocate_models(2, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* r = &modelsrv[0];
	h2mm_mod* v = &modelsrv[1];
	int64_t model_pos = 0;
	h2mm_mod* models = (*out_models == NULL) ? allocate_models(limits->max_iter+1, in_model->nstate, in_model->ndet, nphot): *out_models;
	h2mm_mod* old = r;
	h2mm_mod* current = &models[model_pos];
	h2mm_mod* new0 = &models[++model_pos];
	h2mm_mod* new1; // assigned in loop later
	h2mm_mod* newSQ;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	// **************************************
	// * setup input variable for threading *
	// **************************************
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fbacka_vals *burst_submit = (fbacka_vals*) calloc(limits->num_cores,sizeof(fbacka_vals));
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	for ( i = 0; i < num_burst; i++) {
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
	}
	for ( i=0; i < limits->num_cores; i++) {
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
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
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		// *************************
		// * 1st Calcuation (new0) *
		// *************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_only,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_only, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				current->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				new0->conv |= CONVCODE_ERROR | CONVCODE_POSTMODEL;
				conv = -6;
			}
		}
		if (conv) {
			break;
		}
		// **************************
		// * 2nd Calculation (new1) *
		// **************************
		// updated new1 for calculation (next step
		if ( ++model_pos > limits->max_iter ){
			conv = 2;
			current->conv |= CONVCODE_OUTPUT_MAXITER;
			new0->conv |= CONVCODE_POSTMODEL | CONVCODE_MAXITER;
			break;
		}
		new1 = &models[model_pos];
		old = new0; // iteration moved past old, so update for next while loop
		// zero values of next model and set burst threads
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
		}
		// compute Rho
		rho_all(new0->trans, powers);
		// spin up threads
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_no_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_no_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				conv = -6;
			}
		}
		if (conv) {
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = r; // dummy assignment, prevents conflicting pointers
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			conv = 2;
			new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
			new1->conv |= CONVCODE_POSTMODEL;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// update newSQ for next step
		if ( ++model_pos > limits->max_iter ) {
			conv = 2;
			new1->conv |= CONVCODE_OUTPUT_MAXITER;
			break;
		}
		newSQ = &models[model_pos];
		// project newSQ
		if ( !(convSQ = project_squarem(current, new0, new1, newSQ, v, r)) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			zero_model(r);
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alphaSQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if ( !(newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		else model_pos--;
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		if ( ++model_pos > limits->max_iter ){
			conv = 2;
			if ( convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
				new1->conv |= CONVCODE_OUTPUT_MAXITER;
			}
			else {
				newSQ->conv |= CONVCODE_OUTPUT_MAXITER;
			}
			break;
		}
		new0 = &models[model_pos];
		zero_model(new0);
		if ( convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			powersnext = powers;
			alphanext = alpha;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			powersnext = powersSQ;
			alphanext = alphaSQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	*out_models = models;
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
		free(gamma_var[i]);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(alpha);
	free(alphaSQ);
	free(burst_submit);
	free(bursts);
	free(gamma_var);
	free_models(2, modelsrv);
	free_powers(powers);
	free_powers(powersSQ);
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


int h2mm_squarem_ll_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double *llarr, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	if (limits->max_iter < 1) return -6;
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) return -1; // bad pointer in the data
		
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) {
		free(bursts);
		return -2;
	}
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;

	// Allocate old, current, and new h2mm_mod
	h2mm_mod* modelsrv = allocate_models(2, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* r = &modelsrv[0];
	h2mm_mod* v = &modelsrv[1];
	int64_t model_pos = 0;
	h2mm_mod* models = (*out_models == NULL) ? allocate_models(limits->max_iter+1, in_model->nstate, in_model->ndet, nphot): *out_models;
	h2mm_mod* old = r;
	h2mm_mod* current = &models[model_pos];
	h2mm_mod* new0 = &models[++model_pos];
	h2mm_mod* new1; // assigned in loop later
	h2mm_mod* newSQ;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	// **************************************
	// * setup input variable for threading *
	// **************************************
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fbackall_vals *burst_submit = (fbackall_vals*) calloc(limits->num_cores,sizeof(fbackall_vals));
	double *llarr_n0 = (double*) malloc(num_burst*sizeof(double));
	double *llarr_SQ = (double*) malloc(num_burst*sizeof(double));
	double *llarr_next = llarr;
	double *llarr_out;
	double **gamma_var = (double**) malloc(limits->num_cores * sizeof(double*));
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	for ( i = 0; i < num_burst; i++) {
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
	}
	for ( i=0; i < limits->num_cores; i++) {
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		gamma_var[i] = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = &gamma_var[i];
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].llarr = llarr_next;
		burst_submit[i].loglik = 0.0;
	}
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		// *************************
		// * 1st Calcuation (new0) *
		// *************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_only,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_only, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				current->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				new0->conv |= CONVCODE_ERROR | CONVCODE_POSTMODEL;
				conv = -6;
			}
		}
		if (conv) {
			if (conv == 1) {
				// special case, conv should not be 1 for this evaluation
				conv = -7;
				llarr_out = llarr_n0;
			}
			else {
				llarr_out = llarr_next;
			}
			break;
		}
		// **************************
		// * 2nd Calculation (new1) *
		// **************************
		// updated new1 for calculation (next step
		if ( ++model_pos > limits->max_iter ){
			conv = 2;
			current->conv |= CONVCODE_OUTPUT_MAXITER;
			new0->conv |= CONVCODE_POSTMODEL | CONVCODE_MAXITER;
			llarr_out = llarr_next;
			break;
		}
		new1 = &models[model_pos];
		old = new0; // iteration moved past old, so update for next while loop
		// zero values of next model and set burst threads
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
			burst_submit[i].llarr = llarr_n0;
		}
		// compute Rho
		rho_all(new0->trans, powers);
		// spin up threads
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				conv = -6;
			}
		}
		if (conv) {
			llarr_out = ( conv == 1 ) ? llarr_next : llarr_n0;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = r; // dummy assignment, prevents conflicting pointers
			burst_submit[i].llarr = llarr;
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			conv = 2;
			new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
			new1->conv |= CONVCODE_POSTMODEL;
			llarr_out = llarr_n0;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// update newSQ for next step
		if ( ++model_pos > limits->max_iter ) {
			conv = 2;
			new1->conv |= CONVCODE_OUTPUT_MAXITER;
			llarr_out = llarr;
			break;
		}
		newSQ = &models[model_pos];
		// project newSQ
		if ( !(convSQ = project_squarem(current, new0, new1, newSQ, v, r)) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			zero_model(r);
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alphaSQ;
				burst_submit[i].llarr = llarr_SQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if ( !(newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		else model_pos--;
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		if ( ++model_pos > limits->max_iter ){
			conv = 2;
			if ( convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
				new1->conv |= CONVCODE_OUTPUT_MAXITER;
				llarr_out = llarr;
			}
			else {
				newSQ->conv |= CONVCODE_OUTPUT_MAXITER;
				llarr_out = llarr_SQ;
			}
			break;
		}
		new0 = &models[model_pos];
		zero_model(new0);
		if ( convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			powersnext = powers;
			alphanext = alpha;
			llarr_next = llarr;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			powersnext = powersSQ;
			alphanext = alphaSQ;
			llarr_next = llarr_SQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
			burst_submit[i].llarr = llarr_next;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	*out_models = models;
	if ( llarr != llarr_out ) memcpy((void*) llarr, (void*) llarr_out, num_burst*sizeof(double));
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
		free(gamma_var[i]);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(llarr_n0);
	free(llarr_SQ);
	free(alpha);
	free(alphaSQ);
	free(burst_submit);
	free(bursts);
	free(gamma_var);
	free_models(2, modelsrv);
	free_powers(powers);
	free_powers(powersSQ);
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


int h2mm_squarem_gamma_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	if (limits->max_iter < 1) return -6;
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) return -1; // bad pointer in the data
		
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) {
		free(bursts);
		return -2;
	}
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;

	// Allocate old, current, and new h2mm_mod
	h2mm_mod* modelsrv = allocate_models(2, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* r = &modelsrv[0];
	h2mm_mod* v = &modelsrv[1];
	int64_t model_pos = 0;
	h2mm_mod* models = (*out_models == NULL) ? allocate_models(limits->max_iter+1, in_model->nstate, in_model->ndet, nphot): *out_models;
	h2mm_mod* old = r;
	h2mm_mod* current = &models[model_pos];
	h2mm_mod* new0 = &models[++model_pos];
	h2mm_mod* new1; // assigned in loop later
	h2mm_mod* newSQ;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	// **************************************
	// * setup input variable for threading *
	// **************************************
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fbacka_vals *burst_submit = (fbacka_vals*) calloc(limits->num_cores,sizeof(fbacka_vals));
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	double **gamma_n0 = (double**) malloc(num_burst * sizeof(double*));
	double **gamma_SQ = (double**) malloc(num_burst * sizeof(double*));
	double **gamma_cur = (*gamma != NULL) ? *gamma : (double**) malloc(num_burst*sizeof(double*));
	double **gamma_next = gamma_cur;
	double **gamma_out;
	for ( i = 0; i < num_burst; i++) {
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		gamma_n0[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		gamma_SQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		if ( *gamma == NULL ){
			gamma_cur[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		}
	}
	for ( i=0; i < limits->num_cores; i++) {
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = gamma_cur;
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].loglik = 0.0;
	}
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		// ************************************
		// * 1st Calcuation (current -> new0) *
		// ************************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				current->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				new0->conv |= CONVCODE_ERROR | CONVCODE_POSTMODEL;
				conv = -6;
			}
		}
		if ( conv ) {
			gamma_out = ( conv == 1 ) ? gamma_n0 : gamma_next;
			break;
		}
		// **********************************
		// * 2nd Calculation (new0 -> new1) *
		// **********************************
		// updated new1 for calculation (next step
		if ( ++model_pos > limits->max_iter ){
			conv = 2;
			current->conv |= CONVCODE_OUTPUT_MAXITER;
			new0->conv |= CONVCODE_POSTMODEL | CONVCODE_MAXITER;
			gamma_out = gamma_next;
			break;
		}
		new1 = &models[model_pos];
		old = new0; // iteration moved past old, so update for next while loop
		// zero values of next model and set burst threads
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
			burst_submit[i].gamma = gamma_n0;
		}
		// compute Rho
		rho_all(new0->trans, powers);
		// spin up threads
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				conv = -6;
			}
		}
		if ( conv ) {
			gamma_out = ( conv == 1 ) ? gamma_next : gamma_n0;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = r; // dummy assignment, prevents conflicting pointers
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			conv = 2;
			new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
			new1->conv |= CONVCODE_POSTMODEL;
			gamma_out = gamma_n0;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// update newSQ for next step
		if ( ++model_pos > limits->max_iter ) {
			convSQ = 1;
		}
		newSQ = &models[model_pos];
		// project newSQ
		if ( ( !convSQ ) && (!(convSQ = project_squarem(current, new0, new1, newSQ, v, r))) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			zero_model(r);
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alpha;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if ( !(newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		else model_pos--;
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		new0 = &models[++model_pos];
		zero_model(new0);
		if ( convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			powersnext = powers;
			alphanext = alpha;
			gamma_next = gamma_cur;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			powersnext = powersSQ;
			alphanext = alphaSQ;
			gamma_next = gamma_SQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
			burst_submit[i].gamma = gamma_next;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	*out_models = models;
	if ( *gamma == NULL ) *gamma = gamma_out;
	else if ( *gamma != gamma_out ) {
		transfer_gamma(in_model->nstate, num_burst, burst_sizes, gamma_out, *gamma);
	}
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(alpha);
	free(alphaSQ);
	free_gamma(num_burst, gamma_n0);
	free_gamma(num_burst, gamma_SQ);
	free(burst_submit);
	free(bursts);
	free_models(2, modelsrv);
	free_powers(powers);
	free_powers(powersSQ);
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


int h2mm_squarem_ll_gamma_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double *llarr, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call)
{
	if (limits->max_iter < 1) return -6;
	phstream* bursts = (phstream*) malloc(num_burst*sizeof(phstream));
	int32_t max_delta = get_max_delta(num_burst, burst_sizes, burst_deltas, burst_det, bursts);
	if ( max_delta == 0) return -1; // bad pointer in the data
		
	int64_t i;
	int64_t nphot = check_det(num_burst, bursts, in_model); // verify detectors do not exceed ndet in model
	if (nphot == 0) {
		free(bursts);
		return -2;
	}
	int64_t max_phot = get_max_phot(num_burst, bursts); // deterermine size of largest burst
	int conv = 0, convSQ = 0;
	// initiate varaibles
	clock_t t_start, t_current, t_new;
	double t_iter = 0.0;
	double t_total = 0.0;
	// prevents spinning up unnecessary threads if fewer bursts than cores
	if ( limits->num_cores > num_burst ) limits->num_cores = num_burst;

	// Allocate old, current, and new h2mm_mod
	h2mm_mod* modelsrv = allocate_models(2, in_model->nstate, in_model->ndet, nphot); // initial array, makes easier to free later
	h2mm_mod* r = &modelsrv[0];
	h2mm_mod* v = &modelsrv[1];
	int64_t model_pos = 0;
	h2mm_mod* models = (*out_models == NULL) ? allocate_models(limits->max_iter+1, in_model->nstate, in_model->ndet, nphot): *out_models;
	h2mm_mod* old = r;
	h2mm_mod* current = &models[model_pos];
	h2mm_mod* new0 = &models[++model_pos];
	h2mm_mod* new1; // assigned in loop later
	h2mm_mod* newSQ;
	// allocate A and Rho arrays
	pwrs* powers = allocate_powers(in_model, max_delta);
	pwrs* powersSQ = allocate_powers(in_model, max_delta);
	pwrs* powersnext;
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
	// **************************************
	// * setup input variable for threading *
	// **************************************
	brst_mutex *burst_lock = (brst_mutex*) malloc(sizeof(brst_mutex));
	burst_lock->burst_mutex = h2mm_lock;
	burst_lock->cur_burst = 0;
	burst_lock->num_burst = num_burst;
	fbackall_vals *burst_submit = (fbackall_vals*) calloc(limits->num_cores,sizeof(fbackall_vals));
	double **alpha = (double**) malloc(num_burst * sizeof(double*));
	double **alphaSQ = (double**) malloc(num_burst * sizeof(double*));
	double **alphanext;
	double **gamma_n0 = (double**) malloc(num_burst * sizeof(double*));
	double **gamma_SQ = (double**) malloc(num_burst * sizeof(double*));
	double **gamma_cur = (*gamma != NULL) ? *gamma : (double**) malloc(num_burst*sizeof(double*));
	double **gamma_next = gamma_cur;
	double **gamma_out;
	double *llarr_n0 = (double*) malloc(num_burst * sizeof(double));
	double *llarr_SQ = (double*) malloc(num_burst * sizeof(double));
	double *llarr_next = llarr;
	double *llarr_out;
	for ( i = 0; i < num_burst; i++) {
		alpha[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		alphaSQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		gamma_n0[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		gamma_SQ[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		if ( *gamma == NULL ){
			gamma_cur[i] = (double*) malloc(in_model->nstate * burst_sizes[i] * sizeof(double));
		}
	}
	for ( i=0; i < limits->num_cores; i++) {
		burst_submit[i].phot = bursts;
		burst_submit[i].max_phot = max_phot;
		burst_submit[i].sk = powers->sk;
		burst_submit[i].sj = powers->sj;
		burst_submit[i].si = powers->si;
		burst_submit[i].sT = powers->sT;
		burst_submit[i].A = powers->A;
		burst_submit[i].Rho = powers->Rho;
		burst_submit[i].current = current;
		burst_submit[i].new = new0;
		burst_submit[i].burst_lock = burst_lock;
		burst_submit[i].alpha = alpha;
		burst_submit[i].beta = (double*) malloc(max_phot * in_model->nstate * sizeof(double));
		burst_submit[i].gamma = gamma_cur;
		burst_submit[i].b = (double*) malloc(powers->sk * sizeof(double));
		burst_submit[i].xi_temp = (double*) malloc(powers->sj * sizeof(double));
		burst_submit[i].xi_summed = (double*) calloc(powers->sj, sizeof(double));
		burst_submit[i].obs_temp = (double*) calloc(in_model->nstate * in_model->ndet, sizeof(double));
		burst_submit[i].prior = (double*) calloc(in_model->nstate, sizeof(double));
		burst_submit[i].llarr = llarr_next;
		burst_submit[i].loglik = 0.0;
	}
	// **********************************************************
	// * Start Main Calculation: Initialization fwd calculation *
	// **********************************************************
	// initialize values of newly allocated models
	old->loglik = -INFINITY;
	copy_model_vals(in_model, current);
	current->niter = in_model->niter;
	zero_model(new0);
	t_start = clock();
	t_current = t_start;
	// start calculation
	rho_all(current->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
	for(i = 0; i < limits->num_cores; i++) {
		pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
	}
	for(i = 0; i < limits->num_cores; i++) {
		pthread_join(tid[i],NULL); // wait for all bursts to finish
	}
#elif _WIN32
	for (i = 0; i < limits->num_cores; i++)
		tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
	WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
	for (i = 0; i < limits->num_cores; i++){
		if (tid[i] != 0){
			CloseHandle(tid[i]);
		}
	}
#endif
	current->conv |= CONVCODE_LLCOMPUTED;
	while (conv == 0){
		// save old and current in temp so can assign later when cycling models;
		// ************************************
		// * 1st Calcuation (current -> new0) *
		// ************************************
		burst_lock->cur_burst = 0;
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, bck_ll_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, bck_ll_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_FROMOPT;
		current->conv |= CONVCODE_LLCOMPUTED;
		conv = model_limits_func(new0, current, old, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(current->niter, new0, current, old, t_iter, t_total, print_call) == -1) {
				current->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				new0->conv |= CONVCODE_ERROR | CONVCODE_POSTMODEL;
				conv = -6;
			}
		}
		if ( conv ) {
			gamma_out = ( conv == 1 ) ? gamma_n0 : gamma_next;
			llarr_out = ( conv == 1 ) ? llarr_n0 : llarr_next;
			break;
		}
		if ( ++model_pos > limits->max_iter ){
			conv = 2;
			current->conv |= CONVCODE_OUTPUT_MAXITER;
			new0->conv |= CONVCODE_POSTMODEL | CONVCODE_MAXITER;
			gamma_out = gamma_next;
			llarr_out = llarr_next;
			break;
		}
		// **********************************
		// * 2nd Calculation (new0 -> new1) *
		// **********************************
		// updated new1 for calculation (next step
		new1 = &models[model_pos];
		old = new0; // iteration moved past old, so update for next while loop
		// zero values of next model and set burst threads
		burst_lock->cur_burst = 0;
		zero_model(new1);
		for ( i = 0; i < limits->num_cores; i++) {
			burst_submit[i].current = new0;
			burst_submit[i].new = new1;
			burst_submit[i].A = powers->A;
			burst_submit[i].Rho = powers->Rho;
			burst_submit[i].alpha = alpha;
			burst_submit[i].gamma = gamma_n0;
			burst_submit[i].llarr = llarr_n0;
		}
		// compute Rho
		rho_all(new0->trans, powers);
		// spin up threads
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_bck_alpha_ll_gamma,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++)
			tid[i] = CreateThread(NULL, 0, fwd_bck_alpha_ll_gamma, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++){
			if (tid[i] != 0){
				CloseHandle(tid[i]);
			}
		}
#endif
		t_new = clock();
		t_iter = (double) (t_new - t_current) / CLOCKS_PER_SEC;
		t_total =  (double) (t_new - t_start) / CLOCKS_PER_SEC;
		t_current = t_new;
		new0->conv |= CONVCODE_LLCOMPUTED;
		new1->conv |= CONVCODE_FROMOPT;
		// Evaluate for convergence
		conv = model_limits_func(new1, new0, current, t_total, limits, model_limits);
		if ((! conv)&&(print_func != NULL)) {
			if (print_func(new0->niter, new1, new0, current, t_iter, t_total, print_call) == -1) {
				new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
				conv = -6;
			}
		}
		if ( conv ) {
			gamma_out = ( conv == 1 ) ? gamma_next : gamma_n0;
			llarr_out = ( conv == 1 ) ? llarr_next : llarr_n0;
			break;
		}
		// ***************************
		// * Evaluate loglik of new1 *
		// ***************************
		burst_lock->cur_burst = 0;
		for ( i = 0; i < limits->num_cores; i++){
			burst_submit[i].current = new1;
			burst_submit[i].new = r; // dummy assignment, prevents conflicting pointers
			burst_submit[i].llarr = llarr;
		}
		rho_all(new1->trans, powers);
#if defined(__linux__) || defined(__APPLE__)
		for(i = 0; i < limits->num_cores; i++) {
			pthread_create(&tid[i],NULL, fwd_alpha_ll,(void*) &burst_submit[i]); // create a thread for each burst
		}
		for(i = 0; i < limits->num_cores; i++) {
			pthread_join(tid[i],NULL); // wait for all bursts to finish
		}
#elif _WIN32
		for (i = 0; i < limits->num_cores; i++) {
			tid[i] = CreateThread(NULL, 0, fwd_alpha_ll, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
		}
		WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
		for (i = 0; i < limits->num_cores; i++) {
			if (tid[i] != 0) {
				CloseHandle(tid[i]);
			}
		}
#endif
		if (new1->conv & CONVCODE_ERROR) {
			conv = 2;
			new0->conv |= CONVCODE_ERROR | CONVCODE_OUTPUT;
			new1->conv |= CONVCODE_POSTMODEL;
			gamma_out = gamma_n0;
			break;
		}
		new1->conv |= CONVCODE_LLCOMPUTED;
		// *****************************************
		// * Evaluation of Projected Model (newSQ) *
		// *****************************************
		// update newSQ for next step
		if ( ++model_pos > limits->max_iter ) {
			convSQ = 1;
		}
		newSQ = &models[model_pos];
		// project newSQ
		if ( ( !convSQ ) && (!(convSQ = project_squarem(current, new0, new1, newSQ, v, r))) ) {
			convSQ = model_limits_func(newSQ, new0, current, t_total, limits, model_limits);
		}
		if (! convSQ ) {
			burst_lock->cur_burst = 0;
			zero_model(r);
			for ( i = 0; i < limits->num_cores; i++){
				burst_submit[i].current = newSQ;
				burst_submit[i].A = powersSQ->A;
				burst_submit[i].Rho = powersSQ->Rho;
				burst_submit[i].alpha = alpha;
				burst_submit[i].llarr = llarr_SQ;
			}
			rho_all(newSQ->trans, powersSQ);
#if defined(__linux__) || defined(__APPLE__)
			for(i = 0; i < limits->num_cores; i++) {
				pthread_create(&tid[i],NULL, fwd_alpha,(void*) &burst_submit[i]); // create a thread for each burst
			}
			for(i = 0; i < limits->num_cores; i++) {
				pthread_join(tid[i],NULL); // wait for all bursts to finish
			}
#elif _WIN32
			for (i = 0; i < limits->num_cores; i++) {
				tid[i] = CreateThread(NULL, 0, fwd_alpha, (LPVOID)&burst_submit[i], 0, (LPDWORD)&windowsThreadId[i]);
			}
			WaitForMultipleObjects((DWORD)limits->num_cores, tid, TRUE, INFINITE); // Wait for all of the threads to finish
			for (i = 0; i < limits->num_cores; i++) {
				if (tid[i] != 0) {
					CloseHandle(tid[i]);
				}
			}
#endif
			if ( !(newSQ->conv & CONVCODE_ERROR) ) newSQ->conv |= CONVCODE_LLCOMPUTED;
		}
		else model_pos--;
		// *********************************
		// * Finalizing for next iteration *
		// *********************************
		new0 = &models[++model_pos];
		zero_model(new0);
		if ( convSQ ||  (newSQ->conv & CONVCODE_ERROR) || (newSQ->loglik < new1->loglik) ) {
			// new1 is better or error in newSQ, cycle models with new1
			current = new1;
			powersnext = powers;
			alphanext = alpha;
			gamma_next = gamma_cur;
			llarr_next = llarr;
		}
		else {
			// newSQ is better, cycle models with newSQ
			current = newSQ;
			powersnext = powersSQ;
			alphanext = alphaSQ;
			gamma_next = gamma_SQ;
			llarr_next = llarr_SQ;
		}
		// note this is after the arrays have been cycled
		for ( i = 0; i < limits->num_cores; i++ ) {
			burst_submit[i].current = current;
			burst_submit[i].new = new0;
			burst_submit[i].A = powersnext->A;
			burst_submit[i].Rho = powersnext->Rho;
			burst_submit[i].alpha = alphanext;
			burst_submit[i].gamma = gamma_next;
			burst_submit[i].llarr = llarr_next;
		}
	}
	// ******************************
	// * Finalization/cleanup/frees *
	// ******************************
	*out_models = models;
	if ( llarr != llarr_out) memcpy((void*) llarr, (void*) llarr_out, num_burst*sizeof(double));
	if ( *gamma == NULL ) *gamma = gamma_out;
	else if ( *gamma != gamma_out ) {
		transfer_gamma(in_model->nstate, num_burst, burst_sizes, gamma_out, *gamma);
	}
	for (i = 0; i < limits->num_cores; i++) {
		free(burst_submit[i].beta);
		free(burst_submit[i].b);
		free(burst_submit[i].xi_temp);
		free(burst_submit[i].xi_summed);
		free(burst_submit[i].obs_temp);
		free(burst_submit[i].prior);
	}
	for ( i = 0; i < num_burst; i++) {
		free(alpha[i]);
		free(alphaSQ[i]);
	}
	free(alpha);
	free(alphaSQ);
	free(llarr_n0);
	free(llarr_SQ);
	free_gamma(num_burst, gamma_n0);
	free_gamma(num_burst, gamma_SQ);
	free(burst_submit);
	free(bursts);
	free_models(2, modelsrv);
	free_powers(powers);
	free_powers(powersSQ);
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
