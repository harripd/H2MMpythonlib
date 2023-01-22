// File: utils.c
// Author: Paul David Harris
// Purpose: Small utility functions
// Created: 8 June 2021
// Modified: 13 Nov 2022

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "C_H2MM.h"

unsigned long get_max_delta(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, phstream *b)
{
	unsigned long i, j; // basic iterator variables
	unsigned long max_delta = 1; // stores the maximum delta between succesive photons found
	if ((burst_sizes == NULL) || (burst_deltas == NULL) || (burst_det == NULL) || (b == NULL))
	{
		//~ printf("get_deltas(): One or more of the pointer arguments is NULL\n");
		if (b != NULL)
			free(b);
		return 0;
	}
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
	max_delta++;
	return max_delta;
}

void baseprint(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func)
{
	printf("Iteration %ld, Current loglik %f, improvement: %e, iter time: %f, total: %f\n", niter, old->loglik, current->loglik - old->loglik, t_iter, t_total);
}

void h2mm_normalize(h2mm_mod *model_params)
{
	double norm_factor = 0.0;
	unsigned long i, j;
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

unsigned long get_max_det(unsigned long num_burst, phstream *bursts)
{
	unsigned long max_det = 0;
	unsigned long i, j;
	for (i = 0; i < num_burst; i++)
	{
		for (j=0; j < bursts[i].nphot; j++)
		{
			if (bursts[i].det[j] > max_det)
				max_det = bursts[i].det[j];
		}
	}
	return max_det;
}

unsigned long check_det(unsigned long num_burst, phstream *bursts, h2mm_mod *in_model)
{
	unsigned long i, j;
	unsigned long nphot = 0;
	// Check that no detector exceeds the model specification
	for ( i = 0; i < num_burst; i++)
	{
		nphot += bursts[i].nphot;
		for ( j = 0; j < bursts[i].nphot; j++)
		{
			if ( bursts[i].det[j] >= in_model->ndet)
			{
				//~ printf("Photon detector index exceeds model\n");
				if (bursts != NULL)
					free(bursts);
				return 0;
			}
		}
	}
	return nphot;
}

int duplicate_toempty_model(h2mm_mod *source, h2mm_mod *dest)
{
	unsigned long i;
	if (dest == NULL)
		return 0;
	dest->loglik = source->loglik;
	dest->conv = source->conv;
	dest->nphot = source->nphot;
	dest->niter = source->niter;
	dest->prior = (double*) malloc(source->nstate * sizeof(double));
	dest->trans = (double*) malloc(source->nstate * source->nstate * sizeof(double));
	dest->obs = (double*) malloc(source->nstate * source->ndet * sizeof(double));
	dest->nstate = source->nstate;
	dest->ndet = source->ndet;
		for (i=0; i < source->nstate; i++)
		dest->prior[i] = source->prior[i];
	for (i=0; i < source->nstate * source->nstate; i++)
		dest->trans[i] = source->trans[i];
	for (i=0; i < source->nstate * source->ndet; i++)
		dest->obs[i] = source->obs[i];
	return 1;
}

int duplicate_toempty_models(unsigned long num_model, h2mm_mod **source, h2mm_mod **dest)
{
	unsigned long i;
	int ret;
	h2mm_mod *models = (h2mm_mod*) malloc(num_model * sizeof(h2mm_mod));
	for (i = 0; i < num_model; i++)
	{
		ret = duplicate_toempty_model(source[i], &models[i]);
		if (ret != 1)
		{
			free_models(i, models);
			return ret;
		}
	}
	*dest = models;
	return ret;
}

h2mm_mod* h2mm_model_calc_log(h2mm_mod *source)
{
	h2mm_mod *dest = (h2mm_mod*) malloc(sizeof(h2mm_mod));
	unsigned long i;
	dest->prior = (double*) malloc(source->nstate * sizeof(double));
	dest->trans = (double*) malloc(source->nstate * source->nstate * sizeof(double));
	dest->obs = (double*) malloc(source->nstate * source->ndet * sizeof(double));
	dest->nstate = source->nstate;
	dest->ndet = source->ndet;
		for (i=0; i < source->nstate; i++)
		dest->prior[i] = log(source->prior[i]);
	for (i=0; i < source->nstate * source->nstate; i++)
		dest->trans[i] = log(source->trans[i]);
	for (i=0; i < source->nstate * source->ndet; i++)
		dest->obs[i] = log(source->obs[i]);
	return dest;
}

int copy_model(h2mm_mod *source, h2mm_mod *dest)
{
	if (source->ndet != dest->ndet)
		return 0;
	if (source->nstate != dest->nstate)
		return 0;
	if (dest->prior == NULL)
		return 0;
	if (dest->trans == NULL)
		return 0;
	if (dest->obs == NULL)
		return 0;
	unsigned long i;
	for ( i=0; i < source->nstate; i++)
		dest->prior[i] = source->prior[i];
	for ( i=0; i < source->nstate * source->nstate; i++)
		dest->trans[i] = source->trans[i];
	for ( i=0; i < source->nstate * source->ndet; i++)
		dest->obs[i] = source->obs[i];
	dest->nphot = source->nphot;
	dest->niter = source->niter;
	dest->conv = source->conv;
	dest->loglik = source->loglik;
	return 1;
}

int copy_model_vals(h2mm_mod *source, h2mm_mod *dest)
{
	if (source->ndet != dest->ndet)
		return 0;
	if (source->nstate != dest->nstate)
		return 0;
	if (dest->prior == NULL)
		return 0;
	if (dest->trans == NULL)
		return 0;
	if (dest->obs == NULL)
		return 0;
	unsigned long i;
	for ( i=0; i < source->nstate; i++)
		dest->prior[i] = source->prior[i];
	for ( i=0; i < source->nstate * source->nstate; i++)
		dest->trans[i] = source->trans[i];
	for ( i=0; i < source->nstate * source->ndet; i++)
		dest->obs[i] = source->obs[i];
	return 1;
}

h2mm_mod* allocate_models(const unsigned long n, const unsigned long nstate, const unsigned long ndet, const unsigned long nphot)
{
	unsigned long i;
	h2mm_mod *models = (h2mm_mod*) malloc(n * sizeof(h2mm_mod));
	for (i=0; i < n; i++)
	{
		models[i].nstate = nstate;
		models[i].ndet = ndet;
		models[i].loglik = 0.0;
		models[i].niter = 0;
		models[i].nphot = nphot;
		models[i].conv = 1;
		models[i].prior = (double*) malloc(nstate * sizeof(double));
		models[i].trans = (double*) malloc(nstate * nstate * sizeof(double));
		models[i].obs = (double*) malloc(nstate * ndet * sizeof(double));
	}
	return models;
}

int free_model(h2mm_mod *model)
{
	if (model != NULL)
	{
		if (model->prior != NULL)
			free(model->prior);
		if (model->trans != NULL)
			free(model->trans);
		if (model->obs != NULL)
			free(model->obs);
		free(model);
	}
	return 0;
}

int free_models(const unsigned long n, h2mm_mod *model)
{
	unsigned long i;
	for (i=0; i < n; i++)
	{
		if (model[i].prior != NULL)
			free(model[i].prior);
		if (model[i].trans != NULL)
			free(model[i].trans);
		if (model[i].obs != NULL)
			free(model[i].obs);
	}
	free(model);
	return 0;
}

int zero_model(h2mm_mod *model)
{
	unsigned long i;
	model->loglik = 0.0;
	for (i=0; i < model->nstate; i++)
		model->prior[i] = 0.0;
	for (i=0; i < model->nstate * model->nstate; i++)
		model->trans[i] = 0.0;
	for (i=0; i < model->ndet * model->nstate; i++)
		model->obs[i] = 0.0;
	return 0;
}

unsigned long get_max_phot(unsigned long num_burst, phstream *bursts)
{
	unsigned long i;
	unsigned long max_phot = 0;
	for (i = 0; i < num_burst; i++)
	{
		if (bursts[i].nphot > max_phot)
			max_phot = bursts[i].nphot;
	}
	return max_phot;
}

pwrs* allocate_powers(h2mm_mod *in_model, unsigned long max_delta)
{
	pwrs* powers = (pwrs*) malloc(sizeof(pwrs));
	powers->max_pow = max_delta;
	powers->sk = in_model->nstate;
	powers->sj = powers->sk * in_model->nstate; // set strides, since these never change, we keep them the same 
	powers->si = powers->sj * in_model->nstate;
	powers->sT = powers->si * in_model->nstate;
	powers->A = (double*) calloc(powers->sj * powers->max_pow,sizeof(double));
	powers->Rho = (double*) calloc(powers->sT * powers->max_pow,sizeof(double));
	return powers;
}

int free_powers(pwrs *power)
{
	if (power->A != NULL)
		free(power->A);
	if (power->Rho != NULL)
		free(power->Rho);
	free(power);
	return 0;
}

int free_trpow(trpow *power)
{
	if (power->A != NULL)
	{
		free(power->A);
		power->A = NULL;
	}
	if (power != NULL)
	{
		free(power);
		return 0;
	}
	return 1;
}

int allocate_path(unsigned long nphot, unsigned long nstate, ph_path* path)
{
	path->nphot = nphot;
	path->nstate = nstate;
	path->path = (unsigned long*) calloc(nphot, sizeof(unsigned long));
	path->scale = (double*) calloc(nphot, sizeof(double));
	return 0;
}

int free_path_arrs(ph_path* path)
{
	if (path->path != NULL)
	{
		free(path->path);
		path->path = NULL;
	}
	else
		return 1;
	if (path->scale != NULL)
	{
		free(path->scale);
		path->scale = NULL;
	}
	else
		return 2;
	return 0;
}

ph_path* allocate_paths(unsigned long num_burst, unsigned long* len_burst, unsigned long nstate)
{
	unsigned long i;
	ph_path* paths = (ph_path*) malloc(num_burst * sizeof(ph_path));
	for (i=0; i < num_burst; i++)
	{
		allocate_path(len_burst[i], nstate, &paths[i]);
	}
	return paths;
}

int free_paths(unsigned long num_burst, ph_path* paths)
{
	unsigned long i;
	if (paths != NULL)
	{
		for (i=0; i < num_burst; i++)
			free_path_arrs(&paths[i]);
		//~ free(paths);
		return 0;
	}
	else
		return 1;
}

int print_model(h2mm_mod* model)
{
	unsigned long i, j;
	printf("nstate=%ld, ndet=%ld, nphot=%ld, niter=%ld, conv=%ld, loglik=%f\nPrior:\n", model->nstate, model->ndet, model->nphot, model->niter, model->conv, model->loglik);
	for (i=0; i < model->nstate; i++) printf("%f ", model->prior[i]);
	printf("\nTrans:\n");
	for (i=0; i < model->nstate; i++)
	{
		for (j=0; j < model->nstate; j++)
			printf("%f ", model->trans[i*model->nstate + j]);
		printf("\n");
	}
	printf("Obs:\n");
	for (i=0; i < model->nstate; i++)
	{
		for (j=0; j < model->ndet; j++)
			printf("%f ", model->obs[i+j*model->nstate]);
		printf("\n");
	}
	return 0;
}
