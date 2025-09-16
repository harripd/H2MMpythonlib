// File: C_H2MM_interface.c
// Author: Paul David Harris
// Purpose: Wrapper function for commandline interface with C_H2MM
// Date created: 04 Nov 2022
// Date modified: 06 Nov 2022

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "C_H2MM.h"
#include <unistd.h>

int duplicate_to_empty(h2mm_mod *inmod, h2mm_mod *outmod){
	outmod->nphot = inmod->nphot;
	outmod->nstate = inmod->nstate;
	outmod->ndet = inmod->ndet;
	outmod->conv = inmod->conv;
	outmod->loglik = inmod->loglik;
	outmod->niter = inmod->niter;
	outmod->prior = (double*) malloc(outmod->nstate*sizeof(double));
	outmod->trans = (double*) malloc(outmod->nstate*outmod->nstate*sizeof(double));
	outmod->obs = (double*) malloc(outmod->ndet*outmod->nstate*sizeof(double));
	int64_t i;
	for (i = 0; i < outmod->nstate; i++){
		outmod->prior[i] = inmod->prior[i];
	}
	for (i = 0; i < (outmod->nstate*outmod->nstate); i++){
		outmod->trans[i] = inmod->trans[i];
	}
	for (i = 0; i < (outmod->ndet*outmod->nstate); i++){
		outmod->obs[i] = inmod->obs[i];
	}
	return 0;
}

int main(int argc, char **argv)
{
	int32_t **times;
	uint8_t **detectors;
	int64_t *len_bursts;
	int64_t i, j, k;
	int64_t num_burst = 0;
	long funid = 0; 
	char *eptr;
	temps *head;
	//temps *next;                                                                                   // unreferenced??
	temps *tmp;
	h2mm_mod *model;
	h2mm_mod *models;
	int (*lim_func[]) (h2mm_mod*,h2mm_mod*,h2mm_mod*,double,lm*,void*) = {&limit_check_only, &limit_revert, &limit_revert_old, &limit_minmax};
	lm *limits = (lm*) calloc(1,sizeof(lm));
	//~ h2mm_minmax *minmaxlimit = (h2mm_minmax*) calloc(1,sizeof(h2mm_minmax));
	limits->max_iter = 20;
	limits->max_time = INFINITY;
	limits->min_conv = 1e-14;
	limits->num_cores = 4;
	int n = 0;
	if (argc > 2)
	{
		printf("Read bursts\n");
		
		head = burst_read(argv[1],&num_burst);
		printf("Read model\n");
		models = (h2mm_mod*) malloc((argc-2) * sizeof(h2mm_mod));
		for (i = 0; i < (argc-2); i++)
		{
			model = h2mm_read(argv[i+2]);
			duplicate_to_empty(model, &models[i]);
			free_models(1, model);
		}
	}
	else
	{
		fprintf(stderr,"Too few arguments\n");
		exit(EXIT_FAILURE);
	}
	// recast linked list of burst_read into
	// first step is to allocate the necessary arrays of pointers and lengths 
	printf("Allocating memory\n");
	times = (int32_t**)calloc(num_burst, sizeof(int32_t*));
	detectors = (uint8_t**) calloc(num_burst, sizeof(uint8_t*));
	len_bursts = (int64_t*) calloc(num_burst, sizeof(int64_t));
	// now follow the linked list, and put the pointers and lengths in the arrays
	printf("builing burst arrays\n");
	for( i = 0; i < num_burst; i++)
	{
		len_bursts[i] = head->len_burst;
		times[i] = (int32_t*) malloc(head->len_burst * sizeof(int32_t));
		for (j = head->len_burst - 1; j > 0; j--)
		{
			if (head->times[j-1] < head->times[j])
			{
				times[i][j] = (int32_t) (head->times[j] - head->times[j-1] - 1);
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
		free(head->times);
		detectors[i] = head->detectors;
		tmp = head;
		head = head->next;
		free(tmp); // don't keep the linked list structure around
		//~ printf("Freed burst %d\n",i);
	}
	printf("num models %d\n", argc -2);
	calc_multi(num_burst, len_bursts, times, detectors, argc - 2 , models, limits);
	for (i = 0; i < (argc - 2); i++){
		print_model(&models[i]);
		models[i].conv = 0;
	}
	printf("Done calc\n");
	//~ calc_multi(num_burst, len_bursts, times, detectors, argc -2, models, limits); 
	//~ free_models((unsigned long)(argc-2),models); 
	for(i=0; i < num_burst; i++)
	{
		free(times[i]);
		free(detectors[i]);
	}
	free_models(argc - 2, models);
	free(len_bursts);
	free(times);
	free(detectors);
	free(limits);
}
