// File: C_H2MM_interface.c
// Author: Paul David Harris
// Purpose: Wrapper function for commandline interface with C_H2MM
// Date created: 03 Apr 2021
// Date modified: 13 May 2021

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "C_H2MM.h"

int main(int argc, char **argv)
{
	unsigned long long **times;
	unsigned long **detectors;
	long *len_bursts;
	size_t i, j;
	size_t num_burst = 0;
	char *eptr;
	temps *head;
	//temps *next;                                                                                   // unreferenced??
	temps *tmp;
	phstream *b;
	h2mm_mod *in_model;
	lm *limits = (lm*) calloc(1,sizeof(lm));
	limits->max_iter = 3600;
	limits->max_time = INFINITY;
	limits->min_conv = 1e-14;
	limits->num_cores = 4;
	int n = 0;
	if (argc > 2)
	{
		//~ printf("Read bursts\n");	
		head = burst_read(argv[1],&num_burst);
		//~ printf("Read model\n");
		in_model = h2mm_read(argv[2]);
		if (argc > 3) limits->num_cores = (size_t) strtol(argv[3],&eptr,10);
		if (argc > 4) limits->max_iter = (size_t) strtol(argv[4],&eptr,10);
		if (argc > 5) limits->max_time = strtod(argv[4],&eptr);
		if (argc > 6) limits->min_conv = strtod(argv[5],&eptr);
	}
	else
	{
		fprintf(stderr,"Too few arguments");
		exit(EXIT_FAILURE);
	}
	// recast linked list of burst_read into
	// first step is to allocate the necessary arrays of pointers and lengths 
	times = (unsigned long long**)calloc(num_burst, sizeof(unsigned long long*));
	detectors = (unsigned long**) calloc(num_burst, sizeof(unsigned long long*));
	len_bursts = (long*) calloc(num_burst, sizeof(long));
	// now follow the linked list, and put the pointers and lengths in the arrays
	for( i = 0; i < num_burst; i++)
	{
		len_bursts[i] = head->len_burst;
		times[i] = head->times;
		detectors[i] = head->detectors;
		tmp = head;
		head = head->next;
		free(tmp); // don't keep the linked list structure around
		//~ printf("Freed burst %d\n",i);
	}
	//~ printf("There are %d bursts\n",num_burst);
	b = (phstream*) calloc(num_burst,sizeof(phstream));
	printf("main(): num_burst: %d\n", (int)num_burst);
	printf("main(): in_model->ndet: %d\n", (int)in_model->ndet);
	printf("Entering main algorithm\n");
	h2mm_mod *out_model = C_H2MM(num_burst,len_bursts,times,detectors,in_model,limits);
	if (out_model == NULL) printf("You have an out of order photon\n");
	else if (out_model == in_model) printf("You have too many detectors in your data than allowed in your model\n");
	else
	{
		if (out_model->conv == 1)
			printf("Model converged after %ld iterations\n",out_model->niter);
		else if (out_model->conv ==2)
			printf("Maxiumum iterations reached\n");
		else if (out_model->conv == 3)
			printf("Maximum time reached\n");
		else
			printf("NAN error, returning last successful model\n");
		printf("The final model is:\n");
		printf("Prior:\n");
		for ( i = 0; i < out_model->nstate; i++) printf("%f ",out_model->prior[i]);
		printf("\nTrans:\n");
		for ( i = 0; i < out_model->nstate; i++)
		{
			for ( j = 0; j < out_model->nstate; j++) printf("%e ",out_model->trans[out_model->nstate * i + j]);
			printf("\n");
		}
		printf("Obs\n");
		for ( i = 0; i < out_model->ndet; i++)
		{
			for ( j = 0; j < out_model->nstate; j++) printf("%f ",out_model->obs[out_model->nstate * i + j]);
			printf("\n");
		}
	}
}
