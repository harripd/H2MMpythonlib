// File: viterbi_interface.c
// Author: Paul David Harris
// Purpose: Wrapper function for commandline interface with viterbi
// Date created: __linux__ Apr 2021
// Date modified: 13 May 2021

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "C_H2MM.h"

#define BFRSIZ 1000000                        // Buffer size for buffer passed to fgets()

int main(int argc, char **argv)
{
	unsigned long long **times;
	unsigned long **detectors;
	long *len_bursts;
	unsigned long i, j;
	unsigned long num_burst = 0;
	temps *head;
	//temps *next;                                                                                   // unreferenced??
	temps *tmp;
	phstream *b;
	h2mm_mod *in_model;
	int n = 0;
	if (argc < 3)
	{
		fprintf(stderr,"Too few arguments\n");
		exit(EXIT_FAILURE);
	}
	else if (argc != 3 )
	{
		printf("Skipping additional arguments, not implemented\n");
	}
	head = burst_read(argv[1],&num_burst);
	//~ printf("Read model\n");
	in_model = h2mm_read(argv[2]);
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
	ph_path *path_ret = (ph_path*) malloc(num_burst * sizeof(ph_path));
	int e_val = viterbi(num_burst,len_bursts,times,detectors,in_model,path_ret,4);
	if (e_val == 1)
		printf("You have an out of order photon\n");
	else if 
		(e_val == 2) printf("You have too many detectors in your data than allowed in your model\n");
	else
	{
		for ( i = 0; i < num_burst; i++)
		{
			printf("The loglikelihood is %f\n",path_ret[i].loglik);
			for (j = 0; j < path_ret[i].nphot; j++)
			{
				printf("%ld ",path_ret[i].path[j]);
			}
			printf("\n");
			for (j = 0; j < path_ret[i].nphot; j++)
			{
				printf("%f ",path_ret[i].scale[j]);
			}
			printf("\n");
		}
	}
}
