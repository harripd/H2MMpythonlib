// File: C_H2MM_txtread.c
// Author: Paul David Harris
// Purpose: provide the file parsers for the command line C based H2MM functions
// Date created: April 2021
// Date modified: 29 April 2022

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "C_H2MM.h"

#define BFRSIZ 1000000                        // Buffer size for buffer passed to fgets()

temps* burst_read(char *fname, int64_t *n)
{
	FILE *fid;
	int64_t i; // iterator
	int64_t num_burst = 0;
	char *lptr_t = NULL; // pointer to string containing a line of arrival times
	char *lptr_d = NULL; // pointer to a string containt a line of detector
	char *np_t, *np_d;
	char *curptr_t, *curptr_d;
	int64_t *t_temp;
	uint8_t *d_temp;
	int64_t n_t = 0;
	int64_t n_d = 0;
	int64_t len_burst = 0;
	int64_t num_t;
	uint8_t num_d;
	//
#if defined(__linux__) || defined(__APPLE__)
	int64_t len_t, len_d;
#elif _WIN32
	int len_t, len_d;
	lptr_t = (char*)malloc((unsigned long)BFRSIZ);
	if (lptr_t == NULL)
	{
		perror("malloc");
		exit(EXIT_FAILURE);
	}
	lptr_d = (char*)malloc((unsigned long)BFRSIZ);
	if (lptr_d == NULL)
	{
		perror("malloc");
		exit(EXIT_FAILURE);
	}
#endif
	//
	n_d = n_t = 0;
	fid = fopen(fname,"r");
	//~ printf("%p\n",fid);
	temps *inter = (temps*) calloc(1,sizeof(temps));
	temps *head = inter;
	if( fid == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}
#if defined(__linux__) || defined(__APPLE__)
	while ((len_t = getline(&lptr_t, &n_t, fid)) != -1)
	{
		len_d = getline(&lptr_d, &n_d, fid);
#elif _WIN32
	while (fgets(lptr_t, (int)BFRSIZ, fid) != NULL)
	{ 
		len_t = strlen(lptr_t);
		if (fgets(lptr_d, (int)BFRSIZ, fid) == NULL)
			break;
		len_d = strlen(lptr_d);
#endif
		//~ printf("%s\n",lptr_d);
		if ( len_d != -1 )
		{
			len_burst = 0;
			num_burst++;
			for( curptr_t = lptr_t, curptr_d = lptr_d; *curptr_t != '\n' && curptr_t - lptr_t < len_t; curptr_t++, curptr_d++)
			{
				num_t = (int64_t) strtoll(curptr_t,&np_t,10);
				num_d = (uint8_t) strtol(curptr_d,&np_d,10);
				if (curptr_t != np_t && curptr_d != np_t)
				{
					len_burst++;
					curptr_t = np_t;
					curptr_d = np_d;
				}
			}
			t_temp = (int64_t*) malloc(len_burst*sizeof(int64_t));
			d_temp = (uint8_t*) malloc(len_burst*sizeof(uint8_t));
			curptr_t = lptr_t;
			curptr_d = lptr_d;
			for(i = 0 ; i < len_burst; i++)
			{
				t_temp[i] = (int64_t) strtoll(curptr_t,&np_t,10);
				d_temp[i] = (uint8_t) strtol (curptr_d,&np_d,10);
				curptr_t = np_t;
				curptr_d = np_d;
				//~ printf("%ld  %d\n",t_temp[i], d_temp[i]);
				curptr_t++;
				curptr_d++;
			}
			//~ printf("Storing the burst\n");
			if( num_burst == 1)
			{
				inter->len_burst = len_burst;
				inter->detectors = d_temp;
				inter->times = t_temp;
			}
			else
			{
				//~printf("Storing subsequent burst\n");
				inter->next = (temps*) calloc(1,sizeof(temps));
				inter = inter->next;
				inter->len_burst = len_burst;
				inter->detectors = d_temp;
				inter->times = t_temp;
				//~printf("Finished storing subsequent burst\n");
			}
		}
		else
		{
			fprintf(stderr,"Odd number of lines in file, you are missing either a line of detector indeces or arrival times");
			exit(EXIT_FAILURE);
		}
#if defined(__linux__) || defined(__APPLE__)
		free(lptr_t);
		lptr_t = NULL;
		free(lptr_d);
		lptr_d = NULL;
#endif
		//~ printf("Got the burst\n");
	}
	free(lptr_t);
#ifdef _WIN32
	free(lptr_d);
#endif
	*n = num_burst;
	fclose(fid);
	printf("End of file_read\n");
	return head;
}

#if defined(__linux__) || defined(__APPLE__)
h2mm_mod* h2mm_read(char *fname)
{
	FILE *fid;
	fid = fopen(fname,"r");
	if (fid == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}
	//~ printf("About to calloc\n");
	h2mm_mod *mod = (h2mm_mod*) calloc(1,sizeof(h2mm_mod));
	char *lptr = NULL;
	char *cptr, *eptr;
	int64_t n = 0;
	int64_t len, i, j;
	//unsigned long ndet;                                                                                                     // unreferenced??
	//double *trans;                                                                                                 // unreferenced??
	//double *obs;                                                                                                   // unreferenced??
	//double *prior;                                                                                                 // unreferenced??
	len = getline(&lptr,&n,fid);
	cptr = lptr;
	mod->nstate = (int64_t) strtoll(cptr,&eptr,10);
	cptr = eptr + 1;
	mod->ndet = (int64_t) strtoll(cptr,&eptr,10);
	free(lptr);
	n = 0;
	len = getline(&lptr,&n,fid);
	mod->trans = (double*) calloc(mod->nstate * mod->nstate,sizeof(double));
	mod->obs = (double*) calloc(mod->nstate * mod->ndet,sizeof(double));
	mod->prior = (double*) calloc(mod->nstate, sizeof(double));
	cptr = lptr;
	// grab the prior matrix
	//~ printf("Loading prior matrix\n");
	for(i = 0; i < mod->nstate; i++)
	{
		mod->prior[i] = strtod(cptr,&eptr);
		cptr = eptr + 1;
	}
	free(lptr);
	n = 0;
	//~ printf("Loading trans matrix\n");
	for ( i = 0; i < mod->nstate; i++)
	{
		len = getline(&lptr,&n,fid);
		cptr = lptr;
		for ( j = 0; j < mod->nstate; j++)
		{
			mod->trans[mod->nstate*i + j] = strtod(cptr,&eptr);
			cptr = eptr + 1;
		}
		free(lptr);
		n = 0;
	}
	// get obs matrix
	//~ printf("Loading obs matrix\n");
	for ( i = 0; i < mod->ndet; i++)
	{
		len = getline(&lptr,&n,fid);
		cptr = lptr;
		for ( j = 0; j < mod->nstate; j++)
		{
			mod->obs[mod->nstate*i + j] = strtod(cptr,&eptr);
			cptr = eptr + 1;
		}
		free(lptr);
		n = 0;
	}
//~ printf("Done\n");
	fclose(fid);
	return mod;
}
#elif _WIN32
h2mm_mod* h2mm_read(char* fname)
{
	FILE* fid;
	fid = fopen(fname, "r");
	if (fid == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}
	//~ printf("About to calloc\n");
	h2mm_mod* mod = (h2mm_mod*)calloc(1, sizeof(h2mm_mod));
	char* lptr = NULL;
	char* cptr, * eptr;
	int64_t n = 0;
	int64_t len, i, j;
	lptr = (char*)malloc((unsigned long)BFRSIZ);
	if (lptr == NULL)
	{
		perror("h2mm_read: malloc");
		exit(EXIT_FAILURE);
	}
	//len = getline(&lptr, &n, fid);
	fgets(lptr, (int)BFRSIZ, fid);
	len = strlen(lptr);
	cptr = lptr;
	mod->nstate = (int64_t)strtol(cptr, &eptr, 10);
	cptr = eptr + 1;
	mod->ndet = (int64_t)strtol(cptr, &eptr, 10);
	//free(lptr);
	n = 0;
	//len = getline(&lptr, &n, fid);
	fgets(lptr, (int)BFRSIZ, fid);
	len = strlen(lptr);
	mod->trans = (double*)calloc(mod->nstate * mod->nstate, sizeof(double));
	mod->obs = (double*)calloc(mod->nstate * mod->ndet, sizeof(double));
	mod->prior = (double*)calloc(mod->nstate, sizeof(double));
	cptr = lptr;
	// grab the prior matrix
	//~ printf("Loading prior matrix\n");
	for (i = 0; i < mod->nstate; i++)
	{
		mod->prior[i] = strtod(cptr, &eptr);
		cptr = eptr + 1;
	}
	//free(lptr);
	n = 0;
	//~ printf("Loading trans matrix\n");
	for (i = 0; i < mod->nstate; i++)
	{
		//len = getline(&lptr, &n, fid);
		fgets(lptr, (int)BFRSIZ, fid);
		len = strlen(lptr);
		cptr = lptr;
		for (j = 0; j < mod->nstate; j++)
		{
			mod->trans[mod->nstate * i + j] = strtod(cptr, &eptr);
			cptr = eptr + 1;
		}
		//free(lptr);
		n = 0;
	}
	// get obs matrix
	//~ printf("Loading obs matrix\n");
	for (i = 0; i < mod->ndet; i++)
	{
		//len = getline(&lptr, &n, fid);
		fgets(lptr, (int)BFRSIZ, fid);
		len = strlen(lptr);
		cptr = lptr;
		for (j = 0; j < mod->nstate; j++)
		{
			mod->obs[mod->nstate * i + j] = strtod(cptr, &eptr);
			cptr = eptr + 1;
		}
		//free(lptr);
		n = 0;
	}
	free(lptr);
	//~ printf("Done\n");
	return mod;
}
#endif
