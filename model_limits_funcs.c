// File: model_limits_funcs.c
// Author: Paul David Harris
// Purpose: Provide functions to set limits on the h2mm_model parameters
// Created: 8 June 2021
// Modified: 8 June 2021

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "C_H2MM.h"

#define TRUE 1
#define FALSE 0

void limit_revert(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims)
{
	h2mm_minmax *limits = (h2mm_minmax*) lims;
	size_t i, j; // basic iterator variables
	size_t ind; // for storing the pre-calculated index
	size_t out_count = 0; // counts the number of values that are out of the range
	int var_correct = FALSE; // boolean for whether a 
	int *nstate_bounds = malloc(current->nstate*sizeof(int)); // boolean for which prior or trans elements are out of range
	int *ndet_bounds = malloc(current->nstate * current->ndet * sizeof(int)); // boolean array for which obs elements are out of range
	double off_diff = 0.0;
	// set the various boolean an other variables for the next loop
	var_correct = FALSE;
	out_count = new->nstate;
	off_diff = 0.0;
	// section correcting the prior array
	for (i = 0; i < new->nstate; i++)
		nstate_bounds[i] = TRUE;
	for ( i = 0; i < new->nstate; i++)
	{
		if (new->prior[i] < limits->mins->prior[i] || new->prior[i] > limits->maxs->prior[i])
		{
			// update the main iterator variables
			out_count--;
			nstate_bounds[i] = FALSE;
			var_correct = TRUE;
			// add the adjustment to the sum
			off_diff += new->prior[i] - current->prior[i];
			new->prior[i] = current->prior[i];
		}
	}
	// correct the prior array
	if (var_correct)
	{
		off_diff /= out_count;
		for ( i = 0; i < new->nstate; i++)
		{
			if ( nstate_bounds[i])
				new->prior[i] += off_diff;
		}
	}
	// section correcting the trans array
	for ( i = 0; i < new->nstate; i++)
	{
		// flush the nstate bounds array
		off_diff = 0.0;
		var_correct = FALSE;
		// iterate over row of trans matrix
		for (j = 0; j < new->nstate; j++)
		{
			ind = i * new->nstate + j;
			if (i != j && (new->trans[ind] < limits->mins->trans[ind] || new->trans[ind] > limits->maxs->trans[ind]))
			{
				var_correct = TRUE;
				// add the adjustment to the sum
				off_diff += new->trans[ind] - current->trans[ind];
				new->trans[ind] = current->trans[ind];
			}
		}
		// re-normalize the trans matrix, by adjusting only the diagonal
		if (var_correct)
			new->trans[i * new->nstate + i] += off_diff;
	}
	// section for correcting the obs array
	for ( i = 0; i < new->nstate; i++)
	{
		for ( j = 0; j < new->ndet; j++)
			ndet_bounds[j] = TRUE;
		out_count = new->ndet;
		off_diff = 0.0;
		var_correct = FALSE;
		for ( j = 0; j < new->ndet; j++)
		{
			ind = j * new->nstate + i;
			if ( new->obs[ind] < limits->mins->obs[ind] || new->obs[ind] > limits->maxs->obs[ind])
			{
				out_count--;
				ndet_bounds[j] = FALSE;
				var_correct = TRUE;
				// add the adjustment to the sum
				off_diff += new->obs[ind] - current->obs[ind];
				new->obs[ind] = current->obs[ind];
			}
		}
		if(var_correct)
		{
			off_diff /= out_count;
			for ( j = 0; j < new->ndet; j++)
			{
				ind = j * new->nstate + i;
				if (ndet_bounds[j])
					new->obs[ind] += off_diff;
			} 
		}
	} 
	if (nstate_bounds != NULL)
		free(nstate_bounds);
	if (ndet_bounds != NULL)
		free(ndet_bounds);
}

void limit_revert_old(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims)
{
	h2mm_minmax *limits = (h2mm_minmax*) lims;
	size_t i, j; // basic iterator variables
	size_t ind; // for storing the pre-calculated index
	size_t out_count = 0; // counts the number of values that are out of the range
	int var_correct = FALSE; // boolean for whether a 
	int *nstate_bounds = malloc(current->nstate*sizeof(int)); // boolean for which prior or trans elements are out of range
	int *ndet_bounds = malloc(current->nstate * current->ndet * sizeof(int)); // boolean array for which obs elements are out of range
	double off_diff = 0.0;
	// set the various boolean an other variables for the next loop
	var_correct = FALSE;
	out_count = new->nstate;
	off_diff = 0.0;
	// section correcting the prior array
	for (i = 0; i < new->nstate; i++)
		nstate_bounds[i] = TRUE;
	for ( i = 0; i < new->nstate; i++)
	{
		if (new->prior[i] < limits->mins->prior[i] || new->prior[i] > limits->maxs->prior[i])
		{
			// update the main iterator variables
			out_count--;
			nstate_bounds[i] = FALSE;
			var_correct = TRUE;
			// add the adjustment to the sum
			off_diff += new->prior[i] - current->prior[i];
			new->prior[i] = old->prior[i];
		}
	}
	// correct the prior array
	if (var_correct)
	{
		off_diff /= out_count;
		for ( i = 0; i < new->nstate; i++)
		{
			if ( nstate_bounds[i])
				new->prior[i] += off_diff;
		}
	}
	// section correcting the trans array
	for ( i = 0; i < new->nstate; i++)
	{
		// flush the nstate bounds array
		off_diff = 0.0;
		var_correct = FALSE;
		// iterate over row of trans matrix
		for (j = 0; j < new->nstate; j++)
		{
			ind = i * new->nstate + j;
			if (i != j && (new->trans[ind] < limits->mins->trans[ind] || new->trans[ind] > limits->maxs->trans[ind]))
			{
				var_correct = TRUE;
				// add the adjustment to the sum
				off_diff += new->trans[ind] - current->trans[ind];
				new->trans[ind] = old->trans[ind];
			}
		}
		// re-normalize the trans matrix, by adjusting only the diagonal
		if (var_correct)
			new->trans[i * new->nstate + i] += off_diff;
	}
	// section for correcting the obs array
	for ( i = 0; i < new->nstate; i++)
	{
		for ( j = 0; j < new->ndet; j++)
			ndet_bounds[j] = TRUE;
		out_count = new->ndet;
		off_diff = 0.0;
		var_correct = FALSE;
		for ( j = 0; j < new->ndet; j++)
		{
			ind = j * new->nstate + i;
			if ( new->obs[ind] < limits->mins->obs[ind] || new->obs[ind] > limits->maxs->obs[ind])
			{
				out_count--;
				ndet_bounds[j] = FALSE;
				var_correct = TRUE;
				// add the adjustment to the sum
				off_diff += new->obs[ind] - current->obs[ind];
				new->obs[ind] = old->obs[ind];
			}
		}
		if(var_correct)
		{
			off_diff /= out_count;
			for ( j = 0; j < new->ndet; j++)
			{
				ind = j * new->nstate + i;
				if (ndet_bounds[j])
					new->obs[ind] += off_diff;
			} 
		}
	} 
	if (nstate_bounds != NULL)
		free(nstate_bounds);
	if (ndet_bounds != NULL)
		free(ndet_bounds);
}

void limit_minmax(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims)
{
	h2mm_minmax *limits = (h2mm_minmax*) lims;
	size_t i, j; // basic iterator variables
	size_t ind; // for storing the pre-calculated index
	size_t out_count = 0; // counts the number of values that are out of the range
	int var_correct = FALSE; // boolean for whether a 
	int *nstate_bounds = malloc(current->nstate*sizeof(int)); // boolean for which prior or trans elements are out of range
	int *ndet_bounds = malloc(current->nstate * current->ndet * sizeof(int)); // boolean array for which obs elements are out of range
	double off_diff = 0.0;
	// set the various boolean an other variables for the next loop
	var_correct = FALSE;
	out_count = new->nstate;
	off_diff = 0.0;
	// section correcting the prior array
	for (i = 0; i < new->nstate; i++)
		nstate_bounds[i] = TRUE;
	for ( i = 0; i < new->nstate; i++)
	{
		if (new->prior[i] < limits->mins->prior[i] || new->prior[i] > limits->maxs->prior[i])
		{
			// update the main iterator variables
			out_count--;
			nstate_bounds[i] = FALSE;
			var_correct = TRUE;
			// add the adjustment to the sum
			/*off_diff += new->prior[i] - current->prior[i];
			new->prior[i] = current->prior[i];*/
			if (new->prior[i] < limits->mins->prior[i])
			{
				off_diff += new->prior[i] - limits->mins->prior[i];
				new->prior[i] = limits->mins->prior[i];
			}
			else
			{
				off_diff += new->prior[i] - limits->maxs->prior[i];
				new->prior[i] = limits->maxs->prior[i];
			}
		}
	}
	// correct the prior array
	if (var_correct)
	{
		off_diff /= out_count;
		for ( i = 0; i < new->nstate; i++)
		{
			if ( nstate_bounds[i])
				new->prior[i] += off_diff;
		}
	}
	// section correcting the trans array
	for ( i = 0; i < new->nstate; i++)
	{
		// flush the nstate bounds array
		off_diff = 0.0;
		var_correct = FALSE;
		// iterate over row of trans matrix
		for (j = 0; j < new->nstate; j++)
		{
			ind = i * new->nstate + j;
			if (i != j && (new->trans[ind] < limits->mins->trans[ind] || new->trans[ind] > limits->maxs->trans[ind]))
			{
				var_correct = TRUE;
				// add the adjustment to the sum
				/*off_diff += new->trans[ind] - current->trans[ind];
				new->trans[ind] = current->trans[ind];*/
				if (new->trans[ind] < limits->mins->trans[ind])
				{
					off_diff += new->trans[ind] - limits->mins->trans[ind];
					new->trans[ind] = limits->mins->trans[ind];
				}
				else
				{
					off_diff += new->trans[ind] - limits->maxs->trans[ind];
					new->trans[ind] = limits->maxs->trans[ind];
				}
			}
		}
		// re-normalize the trans matrix, by adjusting only the diagonal
		if (var_correct)
			new->trans[i * new->nstate + i] += off_diff;
	}
	// section for correcting the obs array
	for ( i = 0; i < new->nstate; i++)
	{
		for ( j = 0; j < new->ndet; j++)
			ndet_bounds[j] = TRUE;
		out_count = new->ndet;
		off_diff = 0.0;
		var_correct = FALSE;
		for ( j = 0; j < new->ndet; j++)
		{
			ind = j * new->nstate + i;
			if ( new->obs[ind] < limits->mins->obs[ind] || new->obs[ind] > limits->maxs->obs[ind])
			{
				out_count--;
				ndet_bounds[j] = FALSE;
				var_correct = TRUE;
				// add the adjustment to the sum
				/*off_diff += new->obs[ind] - current->obs[ind];
				new->obs[ind] = current->obs[ind];*/
				if (new->obs[ind] < limits->mins->obs[ind])
				{
					off_diff += new->obs[ind] - limits->mins->obs[ind];
					new->obs[ind] = limits->mins->obs[ind];
				}
				else
				{
					off_diff += new->obs[ind] - limits->maxs->obs[ind];
					new->obs[ind] = limits->maxs->obs[ind];
				}
			}
		}
		if(var_correct)
		{
			off_diff /= out_count;
			for ( j = 0; j < new->ndet; j++)
			{
				ind = j * new->nstate + i;
				if (ndet_bounds[j])
					new->obs[ind] += off_diff;
			} 
		}
	} 
	if (nstate_bounds != NULL)
		free(nstate_bounds);
	if (ndet_bounds != NULL)
		free(ndet_bounds);
}
