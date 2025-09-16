// File: C_H2MM_interface.c
// Author: Paul David Harris
// Purpose: Wrapper function for commandline interface with C_H2MM
// Date created: Apr 2021
// Date modified: 06 Nov 2022
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "C_H2MM.h"

int main(int argc, char **argv)
{
	clock_t start = clock();
	unsigned long **times;
	unsigned long **detectors;
	unsigned long *len_bursts;
	unsigned long i, j;
	unsigned long num_burst = 0;
	long funid = 0; 
	char *eptr;
	temps *head;
	//temps *next;                                                                                   // unreferenced??
	temps *tmp;
	h2mm_mod *in_model;
	int (*lim_func[]) (h2mm_mod*,h2mm_mod*,h2mm_mod*,double,lm*,void*) = {&limit_check_only, &limit_revert, &limit_revert_old, &limit_minmax};
	lm *limits = (lm*) calloc(1,sizeof(lm));
	h2mm_minmax *minmaxlimit = (h2mm_minmax*) calloc(1,sizeof(h2mm_minmax));
	limits->max_iter = 20;
	limits->max_time = INFINITY;
	limits->min_conv = 1e-14;
	limits->num_cores = 4;
	int n = 0;
	if (argc > 2)
	{
		if (argc > 3 && argc < 5)
		{
			fprintf(stderr,"Need to specify all limits arguments");
			exit(EXIT_FAILURE);
		}
		printf("Read bursts\n");
		
		head = burst_read(argv[1],&num_burst);
		printf("Read model\n");
		in_model = h2mm_read(argv[2]);
		printf("Reading limits\n");
		if (argc > 5)
		{
			printf("Read funid\n");
			funid = strtol(argv[3],&eptr,10);
			printf("Reading minimums\n");
			minmaxlimit->mins = h2mm_read(argv[4]);
			printf("Reading maximums\n");
			minmaxlimit->maxs = h2mm_read(argv[5]);
		}
		printf("Reading optional arguments\n");
		if (argc > 6) limits->num_cores = (unsigned long) strtol(argv[6],&eptr,10);
		if (argc > 7) limits->max_iter = (unsigned long) strtol(argv[7],&eptr,10);
		if (argc > 8) limits->max_time = strtod(argv[8],&eptr);
		if (argc > 9) limits->min_conv = strtod(argv[9],&eptr);
	}
	else
	{
		fprintf(stderr,"Too few arguments\n");
		exit(EXIT_FAILURE);
	}
	// recast linked list of burst_read into
	// first step is to allocate the necessary arrays of pointers and lengths 
	printf("Allocating memory\n");
	times = (unsigned long**)calloc(num_burst, sizeof(unsigned long*));
	detectors = (unsigned long**) calloc(num_burst, sizeof(unsigned long*));
	len_bursts = (unsigned long*) calloc(num_burst, sizeof(unsigned long));
	// now follow the linked list, and put the pointers and lengths in the arrays
	printf("builing burst arrays\n");
	for( i = 0; i < num_burst; i++)
	{
		len_bursts[i] = head->len_burst;
		times[i] = (unsigned long*) malloc(head->len_burst * sizeof(unsigned long));
		for (j = head->len_burst - 1; j > 0; j--)
		{
			if (head->times[j-1] < head->times[j])
			{
				times[i][j] = head->times[j] - head->times[j-1] - 1;
			}
			else if (head->times[j-1] == head->times[j])
			{
				times[i][j] = 0;
			}
			else
			{
				printf("You have an out of order photon\n");
				return 0;
			}
		}
		times[i][0] = 0;
		detectors[i] = head->detectors;
		tmp = head;
		head = head->next;
		free(tmp); // don't keep the linked list structure around
		//~ printf("Freed burst %d\n",i);
	}
	//~ printf("There are %d bursts\n",num_burst);
	printf("main(): num_burst: %d\n", (int)num_burst);
	printf("main(): in_model->ndet: %d\n", (int)in_model->ndet);
/*	h2mm_mod* out_model = allocate_models(1,in_model->nstate, in_model->ndet, in_model->nphot);
	printf("h2mm_optimize\n");
	int res = h2mm_optimize(num_burst,len_bursts,times,detectors,in_model,out_model,limits,lim_func[funid],(void*)minmaxlimit, &baseprint, NULL);
	printf("Done h2mm_optimize\n");
	if (res < 0)
	{
		printf("Error in h2mm_optimize\n");
	}
	else
	{
		if (out_model->conv == 0)
			printf("Unoptimized model returned\n");
		else if (out_model->conv ==1)
			printf("Mid-optiization model returned at %ld iterations\n", out_model->niter);
		else if (out_model->conv == 2)
			printf("Non-optimized model %ld iterations\n", out_model->niter);
		else if (out_model->conv == 3)
			printf("Model converged at %ld iterations\n", out_model->niter);
		else if (out_model->conv == 4)
			printf("Max iterations reached (%ld)\n", out_model->niter);
		else if (out_model->conv == 5)
			printf("Max time reached at %ld iterations\n", out_model->niter);
		else if (out_model->conv == 6)
			printf("Error at %ld iterations\n", out_model->niter);
		else if (out_model->conv == 7)
			printf("Model beyond converged at %ld iterations\n", out_model->niter);
		printf("The final model is:\n");
		print_model(out_model);
	}
	free_model(out_model); */
	h2mm_mod* out_model;
	printf("h2mm_optimize_array\n");
	clock_t start_optimize_array = clock();
	int res = h2mm_optimize_array(num_burst,len_bursts,times,detectors,in_model, &out_model, limits,lim_func[funid],(void*)minmaxlimit, &baseprint, NULL);
	clock_t stop_optimize_array = clock();
	printf("Done h2mm_optimize_array, out_model loc %p\n", out_model);
	for (i=0; (i < limits->max_iter)&&(out_model[i].conv != 1); i++);
		//printf("&out_model[i]=%p, out_model[i].prior=%p, out_model[i].conv=%ld\n", &out_model[i], out_model[i].prior, out_model[i].conv);
	printf("i=%ld\n", i);
	unsigned long id_mod = i-1;
	if (out_model[id_mod].conv == 0)
		printf("Unoptimized model returned\n");
	else if (out_model[id_mod].conv ==1)
		printf("Mid-optiization model returned at %ld iterations\n", out_model[id_mod].niter);
	else if (out_model[id_mod].conv == 2)
		printf("Non-optimized model %ld iterations\n", out_model[id_mod].niter);
	else if (out_model[id_mod].conv == 3)
		printf("Model converged at %ld iterations\n", out_model[id_mod].niter);
	else if (out_model[id_mod].conv == 4)
		printf("Max iterations reached (%ld)\n", out_model[id_mod].niter);
	else if (out_model[id_mod].conv == 5)
		printf("Max time reached at %ld iterations\n", out_model[id_mod].niter);
	else if (out_model[id_mod].conv == 6)
		printf("Error at %ld iterations\n", out_model[id_mod].niter);
	else if (out_model[id_mod].conv == 7)
		printf("Model beyond converged at %ld iterations\n", out_model[id_mod].niter);
	else
		printf("Conv code %ld\n", out_model[id_mod].conv);
	printf("The final model is:\n");
	print_model(&out_model[id_mod]);
	
	
	
	//~ h2mm_mod* multi_mod = allocate_models(id_mod, in_model->nstate, in_model->ndet, in_model->nphot);
	//~ multi_mod = allocate_models(id_mod, in_model->nstate, in_model->ndet, in_model->nphot);
	//~ for (i = 0; i < id_mod; i++)
		//~ copy_model_vals(&out_model[i], &multi_mod[i]);
	//~ printf("calc_multi()\n");
	//~ res = calc_multi(num_burst, len_bursts, times, detectors, id_mod, multi_mod, limits);
	//~ printf("calc_mutli() done\n");
	
	free_models(limits->max_iter, out_model);
	
	
	out_model = allocate_models(1, in_model->nstate, in_model->ndet, in_model->nphot);
	double **gamma;
	printf("h2mm_optimize_gamma\n");
	clock_t start_optimize_gamma = clock();
	res = h2mm_optimize_gamma(num_burst,len_bursts,times,detectors,in_model, out_model, &gamma,limits, lim_func[funid],(void*)minmaxlimit, &baseprint, NULL);
	clock_t stop_optimize_gamma = clock();
	printf("Done h2mm_optimize_gamma\n");
	print_model(out_model);
	printf("gamma ptr %p\n", gamma);
	for(i=0; i < len_bursts[0]; i++)
	{
		for (j = 0; j < out_model->nstate; j++)
		{
			printf("%f ", gamma[0][i*out_model->nstate + j]);
		}
		printf("\n");
	}
	for (i=0; i< num_burst; i++)
	{
		if (gamma[i] != NULL)
			free(gamma[i]);
	}
	free(gamma);
	free_model(1, out_model);
	printf("h2mm_optimize_gamma_array\n");
	clock_t start_optimize_gamma_array = clock();
	res = h2mm_optimize_gamma_array(num_burst,len_bursts,times,detectors,in_model, &out_model, &gamma, limits, lim_func[funid],(void*)minmaxlimit, &baseprint, NULL);
	clock_t stop_optimize_gamma_array = clock();
	printf("Done h2mm_optimize_gamma_array\n");
	unsigned long max_mod;
	for(i = 0; (out_model[i].conv != 1) && (i < limits->max_iter); i++)
	{
		print_model(&out_model[i]);
	}
	if ( res != 1 )
	{
		print_model(&out_model[i]);
		max_mod = ++i;
		print_model(&out_model[i]);
	}
	for (i = 0; i < num_burst; i++)
		free(gamma[i]);
	free(gamma);
	printf("allocate paths\n");
	ph_path* paths = allocate_paths(num_burst, len_bursts, out_model->nstate);
	printf("viterbi\n");
	clock_t start_viterbi = clock();
	res = viterbi(num_burst, len_bursts, times, detectors, &out_model[1], paths, limits->num_cores);
	clock_t stop_viterbi = clock();
	if (res == 0)
	{
		printf("Viterbi finished succesfully\n");
		for (i=0; (i < num_burst) && (i < 5); i++)
		{
			for (j=0; j < paths[i].nphot; j++)
				printf("%ld ", paths[i].path[j]);
			printf("\n\n");
		}
	}
	else
	{
		printf("Error in viterbi\n");
	}
	free_paths(num_burst, paths);
	printf("num_burst=%ld: ", num_burst);
	for (i=num_burst -10; i < num_burst; i++)
		printf("i=%ld, len=%ld; ", i, len_bursts[i]);
	printf("\n");
	//~ h2mm_mod* multi_mod = allocate_models(10, in_model->nstate, in_model->ndet, in_model->nphot);
	//~ for (i = 0; i < 10; i++)
	//~ {
		//~ copy_model_vals(&out_model[max_mod-10+i], &multi_mod[i]);
	//~ }
	double ****gamma_arr;
	//~ printf("calc_multi_gamma()\n");
	//~ res = calc_multi_gamma(num_burst, len_bursts, times, detectors, 10, multi_mod, &gamma_arr, limits);
	//~ printf("calc_multi_gamma() done\n");
	//~ for (i = 0; i < 10; i++)
		//~ printf("model[%ld] loglik: original=%f, calc_multi=%f, diff=%f\n", i, out_model[max_mod-10+i].loglik, multi_mod[i].loglik, out_model[max_mod-10+i].loglik - multi_mod[i].loglik);
	//~ free_models(10, multi_mod);	
	
	h2mm_mod* multi_mod = allocate_models(max_mod+1, in_model->nstate, in_model->ndet, in_model->nphot);
	for (i = 0; i < max_mod+1; i++)
		copy_model_vals(&out_model[i], &multi_mod[i]);
	gamma_arr = (double****) malloc(10 * sizeof(double***));
	printf("calc_multi_gamma()\n");
	clock_t start_multigamma = clock();
	res = calc_multi_gamma(num_burst, len_bursts, times, detectors, 10, multi_mod, gamma_arr, limits);
	clock_t stop_multigamma = clock();
	printf("calc_multi_gamma() done\n");
	for (i = 0; i < 10; i++)
		print_model(&multi_mod[i]);
	printf("calc_multi()\n");
	clock_t start_multi = clock();
	res = calc_multi(num_burst, len_bursts, times, detectors, max_mod + 1, multi_mod, limits);
	clock_t stop_multi = clock();
	printf("calc_multi done\n");
	//~ printf("num_burst=%ld: ", num_burst);
	//~ for (i=num_burst -10; i < num_burst; i++)
		//~ printf("i=%ld, len=%ld; ", i, len_bursts[i]);
	//~ printf("\n");
	//~ for (i = 0; i < max_mod + 1; i++)
		//~ printf("model[%ld] loglik: original=%f, calc_multi=%f, diff=%f\n", i, out_model[i].loglik, multi_mod[i].loglik, out_model[i].loglik - multi_mod[i].loglik);
	free_models(max_mod+1, multi_mod);
	multi_mod = allocate_models(10, in_model->nstate, in_model->ndet, in_model->nphot);
	for (i = 0; i < 10; i++)
	{
		copy_model_vals(&out_model[max_mod-10+i], &multi_mod[i]);
	}
	clock_t stop = clock();
	printf("Total time: %fs, optimize array: %fs, optimize gamma: %fs, optimize gamma array: %fs, viterbi: %fs, calc multi gamma: %fs, calc multi: %fs\n", (double)(stop-start)/CLOCKS_PER_SEC, (double)(stop_optimize_array-start_optimize_array)/CLOCKS_PER_SEC, (double)(stop_optimize_gamma-start_optimize_gamma)/CLOCKS_PER_SEC, (double)(stop_optimize_gamma_array-start_optimize_gamma_array)/CLOCKS_PER_SEC, (double)(stop_viterbi-start_viterbi)/CLOCKS_PER_SEC, (double)(stop_multigamma-start_multigamma)/CLOCKS_PER_SEC, (double)(start_multi-start_multi)/CLOCKS_PER_SEC);
	free_models(10, multi_mod);
	for (i = 0; i < num_burst; i++){
		free(detectors[i]);
		free(times[i]);
	}
	free(detectors);
	free(times);
	return 1;
}
