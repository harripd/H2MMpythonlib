#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:49:24 2021

@author: Paul David Harris
"""

import os
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# cdef extern from "rho_calc.c":
#      pass
# cdef extern from "fwd_back_photonbyphoton_par.c":
#      pass
# cdef extern from "C_H2MM.c":
#      pass
cdef extern from "C_H2MM.h":
    ctypedef struct lm:
        size_t max_iter
        size_t num_cores
        double max_time
        double min_conv
    ctypedef struct h2mm_mod:
        size_t nstate
        size_t ndet
        size_t nphot
        size_t niter
        size_t conv
        double *prior
        double *trans
        double *obs
        double loglik
    ctypedef struct ph_path:
        size_t nphot
        size_t nstate
        double loglik
        size_t *path
        double *scale
        double *omega
    void h2mm_normalize(h2mm_mod *model_params)
    h2mm_mod* C_H2MM(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *in_model, lm *limits) nogil
    int viterbi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array) nogil

cdef unsigned long long* get_ptr_ull(np.ndarray[unsigned long long, ndim=1] arr):
    cdef unsigned long long[::1] arr_view = arr
    return &arr_view[0]

cdef unsigned long* get_ptr_l(np.ndarray[unsigned long, ndim=1] arr):
    cdef unsigned long[::1] arr_view = arr
    return &arr_view[0]

cdef class h2mm_model:
    cdef:
        h2mm_mod model
    def __cinit__(self, prior, trans, obs, loglik=-np.inf, niter = 0, nphot = 0):
        # assert statements to confirm first the correct dimensinality of input matrices, then that their shapes match
        cdef size_t i, j
        assert prior.ndim == 1, "Prior matrix must have ndim=1, too many dimensions in prior"
        assert trans.ndim == 2, "Trans matrix must ahve ndim=2, wrong dimensionallity of trans matrix"
        assert obs.ndim == 2, "Obs matrix must ahve ndim=2, wrong dimensionallity of obs matrix"
        assert prior.shape[0] == trans.shape[0] == obs.shape[0], "Dim 0 of one of the matrices does not match the others, these represent the number of states, so input matrices do not represent a single model"
        assert trans.shape[0] == trans.shape[1], "Trans matrix is not square, and connot be used for a model"
        assert niter >= 0, "niter must be positive"
        assert nphot >= 0, "nphot must be positive"
        # coerce the matricies into c type double
        prior = prior.astype('double')
        trans = trans.astype('double')
        obs = obs.astype('double')
        self.model.nstate = <size_t> obs.shape[0]
        self.model.ndet = <size_t> obs.shape[1]
        self.model.nphot = nphot
        self.model.conv = 0
        self.model.loglik = <double> loglik
        self.model.trans = <double*> PyMem_Malloc(self.model.nstate**2 * sizeof(double))
        for i in range(self.model.nstate):
            for j in range(self.model.nstate):
                self.model.trans[self.model.nstate*i + j] = trans[i,j]
        self.model.prior = <double*> PyMem_Malloc(self.model.nstate * sizeof(double))
        for i in range(self.model.nstate):
            self.model.prior[i] = prior[i]
        self.model.obs =  <double*> PyMem_Malloc(self.model.ndet * self.model.nstate * sizeof(double))
        for i in range(self.model.ndet):
            for j in range(self.model.nstate):
                self.model.obs[self.model.nstate * i + j] = obs[j,i]
        self.normalize()
    # a number of propert defs so that the values are accesible from python
    @property
    def prior(self):
        return np.asarray(<double[:self.model.nstate]>self.model.prior)
    @prior.setter
    def prior(self,prior):
        assert prior.ndim == 1, "Prior must be 1D numpy floating point array"
        assert prior.shape[0] == self.model.nstate, "Cannot change the number of states"
        print('Model no longer considered convreged, and loglik reset')
        self.model.loglik = -np.inf
        self.model.conv = 0
        prior = prior.astype('double')
        for i in range(self.model.nstate):
            self.model.prior[i] = prior[i]
    @property
    def trans(self):
        return np.asarray(<double[:self.model.nstate,:self.model.nstate]>self.model.trans)
    @trans.setter
    def trans(self,trans):
        assert trans.ndim == 2, "Trans must be a 2D numpy floating point array"
        assert trans.shape[0] == trans.shape[1], "Trans must be a square array"
        assert trans.shape[0] == self.model.nstate, "Cannot change the number of states in a model"
        print('Model no longer considered convreged, and loglik reset')
        self.model.loglik = -np.inf
        self.model.conv = 0
        trans = trans.astype('double')
        for i in range(self.model.nstate):
            for j in range(self.model.nstate):
                self.model.trans[self.model.nstate*i + j] = trans[i,j]
    @property
    def obs(self):
        return np.asarray(<double[:self.model.ndet,:self.model.nstate]>self.model.obs).T
    @obs.setter
    def obs(self,obs):
        assert obs.ndim == 2, "Obs must be a 2D numpy floating point array"
        assert obs.shape[0] == self.model.nstate, "Cannot change the number of states in a model"
        assert obs.shape[1] == self.model.ndet, "Cannot change the number of streams in the model"
        print('Model no longer considered convreged, and loglik reset')
        self.model.loglik = -np.inf
        self.model.conv = 0
        obs = obs.astype('double')
        for i in range(self.model.ndet):
            for j in range(self.model.nstate):
                self.model.obs[self.model.nstate * i + j] = obs[j,i]
    @property
    def loglik(self):
        assert self.model.nphot > 0, "Must run through H2MM_C first, likelihood not known"
        return self.model.loglik
    @property
    def k(self):
        return self.model.nstate**2 + ((self.model.ndet - 1)*self.model.nstate) - 1
    @property
    def bic(self):
        assert self.model.nphot > 0, "Must run through H2MM_C first, likelihood not known"
        return -2*self.model.loglik + np.log(self.model.nphot)*(self.model.nstate**2 + ((self.model.ndet - 1)*self.model.nstate) - 1)
    @property
    def nstate(self):
        return self.model.nstate
    @property
    def ndet(self):
        return self.model.ndet
    @property
    def nphot(self):
        return self.model.nphot
    @nphot.setter
    def nphot(self,nphot):
        assert nphot > 0, "nphot must be greater than 0"
        self.model.nphot = <size_t> nphot
    @property
    def converged(self):
        if self.model.conv == 0:
            return True
        else:
            return False
    @property
    def conv_crit(self):
        if self.model.conv == 0:
            return "Model unoptimized"
        elif self.model.conv ==1:
            return f'Model converged after {self.model.niter} iterations'
        elif self.model.conv == 2:
            return f'Maxiumum of {self.model.niter} iterations reached'
        elif self.model.conv == 3:
            return f'After {self.model.niter} iterations the optimization reached the time limit'
        else:
            return f'Optimization terminated because of reaching floating point NAN on iteration {self.model.niter}, returned the last viable model'
    # all matrices in the model shoudl be row stochastic
    # def normalize(self):
    #     h2mm_normalize(&self.model)
    @property
    def niter(self):
        if self.model.niter > 0:
            return self.model.niter
    @niter.setter
    def niter(self,niter):
        assert niter > 0, "Cannot have negative iterations"
        self.model.niter = niter
    def set_converged(self,converged):
        assert converged == True or converged == False, "Input must be True or False"
        if converged:
            self.model.conv = 1
        elif not converged:
            self.model.niter = 0
    def normalize(self):
        h2mm_normalize(&self.model)
    def EM_H2MM_C(self, list burst_colors, list burst_times, max_iter=3600, max_time=np.inf, converged_min=1e-14, num_cores= os.cpu_count()//2):
        """
        Calculate the most likely state path through a set of data given a H2MM model
    
        Parameters
        ----------
        
        indexes : list of NUMPY 1D int arrays
            A list of the arrival indexes for each photon in each burst.
            Each element of the list (a numpy array) cooresponds to a burst, and
            each element of the array is a singular photon.
            The indexes list must maintain  1to1 coorespondence to the times list
        times : list of NUMPY 1D int arrays
            A list of the arrival times for each photon in each burst
            Each element of the list (a numpy array) cooresponds to a burst, and
            each element of the array is a singular photon.
            The times list must maintain  1to1 coorespondence to the indexes list
        
        Optional Keyword Parameters
        ___________________
        max_iter=3600 : int
            the maximum number of iterations to conduct before returning the current
            h2mm_model
        max_time=np.inf : float
            The maximum time (in seconds) before retunring current model
            NOTE: this uses the C clock, which has issues, often the time assesed by
            C, which is usually longer than the actual time
        converged_min=1e-14 : float
            The difference between new and current h2mm_models to consider the model
            converged, the default setting is close to floating point error, it is
            recomended to only increase the size
        num_cores = os.cpu_count()//2 : int
            the number of C threads (which ignore the gil, thus functioning more
            like python processes), to use when calculating iterations. The default
            is to take half of what python reports as the cpu count, because most
            cpus have multithreading enabled, so the os.cpu_count() will return
            twice the number of physical cores. Consider setting this parameter
            manually if either you want to optimize the speed of the computation,
            or if you want to reduce the number of cores used so your computer has
            more cpu time to spend on other tasks
            
        Returns
        -------
        out : h2mm_model
            The optimized h2mm_model. will return after one of the follwing conditions
            are met: model has converged (according to converged_min, defaule 1e-14),
            maximum iterations reached, maximum time has passed, or an error has occured
            (usually the result of a nan from a floating point precision error)
        """
        # assert statements to verify that the data is valid, ie matched lengths and dimensions for burst_times and burst_colors in all bursts
        assert len(burst_colors) == len(burst_times), "Mismatch in burst_colors and burst_times lengths, burst_colors and burst_times must be of the same length"
        cdef size_t i
        cdef size_t num_burst = len(burst_colors)
        for i in range(num_burst):
            assert burst_colors[i].ndim == 1, f"burst_colors[{i}] must be a 1D array"
            assert burst_times[i].ndim == 1 , f"burst_times[{i}] must be a 1D array"
            assert burst_times[i].shape[0] == burst_colors[i].shape[0], f"Mismatch in lengths between burst_times[{i}] and burst_colors[{i}], cannot create burst"
        # set up the limits function
        cdef lm limits
        limits.max_iter = <size_t> max_iter
        limits.num_cores = <size_t> num_cores if num_cores > 0 else 1
        limits.max_time = <double> max_time
        limits.min_conv = <double> converged_min
        # allocate the memory for the pointer arrays to be submitted to the C function
        cdef unsigned long *burst_sizes = <unsigned long*> PyMem_Malloc(num_burst * sizeof(unsigned long))
        cdef unsigned long long **b_time = <unsigned long long**> PyMem_Malloc(num_burst * sizeof(unsigned long long*))
        cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
        # for loop casts the values to the right datat type, then makes sure the data is contiguous, but don't copy the pointers just yet, that is in a separate for loop to make sure no numpy shenanigans 
        for i in range(num_burst):
            burst_sizes[i] = burst_colors[i].shape[0]
            if burst_sizes[i] < 3:
                raise Exception(f'Bursts must have at least 3 photons, burst {i} has only {burst_sizes[i]}')
            burst_colors[i] = burst_colors[i].astype('L')
            burst_times[i] = burst_times[i].astype('Q')
            if not burst_colors[i].flags['C_CONTIGUOUS']:
                burst_colors[i] = np.ascontiguousarray(burst_colors[i])
            if not burst_times[i].flags['C_CONTIGUOUS']:
                burst_times[i] = np.ascontiguousarray(burst_times[i])
        # now make the list of pointers
        for i in range(num_burst):
            b_det[i] = get_ptr_l(burst_colors[i])
            b_time[i] = get_ptr_ull(burst_times[i])
        # set up the in and out h2mm_mod variables
        cdef h2mm_mod* out_model = C_H2MM(num_burst,burst_sizes,b_time,b_det,&self.model,&limits)
        if out_model is NULL:
            PyMem_Free(b_det)
            PyMem_Free(b_time)
            PyMem_Free(burst_sizes)
            raise Exception('Bursts photons are out of order, please check your data')
        elif out_model == &self.model:
            PyMem_Free(b_det)
            PyMem_Free(b_time)
            PyMem_Free(burst_sizes)
            raise Exception('Too many photon streams in data for H2MM model')
        PyMem_Free(self.model.prior)
        PyMem_Free(self.model.trans)
        PyMem_Free(self.model.obs)
        self.model = out_model[0]
        if self.model.conv == 1:
            print(f'The model converged after {self.model.niter} iterations')
        elif self.model.conv == 2:
            print('Optimization reached maximum number of iterations')
        elif self.model.conv == 3:
            print('Optimization reached maxiumum time')
        else:
            print(f'An error occured on iteration {self.model.niter}, returning previous model')
        PyMem_Free(b_det);
        PyMem_Free(b_time);
        PyMem_Free(burst_sizes)
    def __dealloc__(self):
        if self.model.prior is not NULL:
            PyMem_Free(self.model.prior)
        if self.model.trans is not NULL:
            PyMem_Free(self.model.trans)
        if self.model.obs is not NULL:
            PyMem_Free(self.model.obs)

cpdef EM_H2MM_C(h2mm_model h_mod, list burst_colors, list burst_times, max_iter=3600, max_time=np.inf, converged_min=1e-14, num_cores= os.cpu_count()//2):
    """
    Calculate the most likely state path through a set of data given a H2MM model

    Parameters
    ----------
    h_model : h2mm_model
        An initial guess for the H2MM model, just give a general guess, the algorithm
        will optimize, and generally the algorithm will converge even when the
        initial guess is very far off
    indexes : list of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) cooresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1to1 coorespondence to the times list
    times : list of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) cooresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1to1 coorespondence to the indexes list
    
    Optional Keyword Parameters
    ___________________
    max_iter=3600 : int
        the maximum number of iterations to conduct before returning the current
        h2mm_model
    max_time=np.inf : float
        The maximum time (in seconds) before retunring current model
        NOTE: this uses the C clock, which has issues, often the time assesed by
        C, which is usually longer than the actual time
    converged_min=1e-14 : float
        The difference between new and current h2mm_models to consider the model
        converged, the default setting is close to floating point error, it is
        recomended to only increase the size
    num_cores = os.cpu_count()//2 : int
        the number of C threads (which ignore the gil, thus functioning more
        like python processes), to use when calculating iterations. The default
        is to take half of what python reports as the cpu count, because most
        cpus have multithreading enabled, so the os.cpu_count() will return
        twice the number of physical cores. Consider setting this parameter
        manually if either you want to optimize the speed of the computation,
        or if you want to reduce the number of cores used so your computer has
        more cpu time to spend on other tasks
        
    Returns
    -------
    out : h2mm_model
        The optimized h2mm_model. will return after one of the follwing conditions
        are met: model has converged (according to converged_min, defaule 1e-14),
        maximum iterations reached, maximum time has passed, or an error has occured
        (usually the result of a nan from a floating point precision error)
    """
    # assert statements to verify that the data is valid, ie matched lengths and dimensions for burst_times and burst_colors in all bursts
    assert len(burst_colors) == len(burst_times), "Mismatch in burst_colors and burst_times lengths, burst_colors and burst_times must be of the same length"
    cdef size_t i
    cdef size_t num_burst = len(burst_colors)
    for i in range(num_burst):
        assert burst_colors[i].ndim == 1, f"burst_colors[{i}] must be a 1D array"
        assert burst_times[i].ndim == 1 , f"burst_times[{i}] must be a 1D array"
        assert burst_times[i].shape[0] == burst_colors[i].shape[0], f"Mismatch in lengths between burst_times[{i}] and burst_colors[{i}], cannot create burst"
    # set up the limits function
    cdef lm limits
    limits.max_iter = <size_t> max_iter
    limits.num_cores = <size_t> num_cores if num_cores > 0 else 1
    limits.max_time = <double> max_time
    limits.min_conv = <double> converged_min
    # allocate the memory for the pointer arrays to be submitted to the C function
    cdef unsigned long *burst_sizes = <unsigned long*> PyMem_Malloc(num_burst * sizeof(unsigned long))
    cdef unsigned long long **b_time = <unsigned long long**> PyMem_Malloc(num_burst * sizeof(unsigned long long*))
    cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    # for loop casts the values to the right datat type, then makes sure the data is contiguous, but don't copy the pointers just yet, that is in a separate for loop to make sure no numpy shenanigans 
    for i in range(num_burst):
        burst_sizes[i] = burst_colors[i].shape[0]
        if burst_sizes[i] < 3:
            raise Exception(f'Bursts must have at least 3 photons, burst {i} has only {burst_sizes[i]}')
        burst_colors[i] = burst_colors[i].astype('L')
        burst_times[i] = burst_times[i].astype('Q')
        if not burst_colors[i].flags['C_CONTIGUOUS']:
            burst_colors[i] = np.ascontiguousarray(burst_colors[i])
        if not burst_times[i].flags['C_CONTIGUOUS']:
            burst_times[i] = np.ascontiguousarray(burst_times[i])
    # now make the list of pointers
    for i in range(num_burst):
        b_det[i] = get_ptr_l(burst_colors[i])
        b_time[i] = get_ptr_ull(burst_times[i])
    # set up the in and out h2mm_mod variables
    cdef h2mm_mod* out_model = C_H2MM(num_burst,burst_sizes,b_time,b_det,&h_mod.model,&limits)
    if out_model is NULL:
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise Exception('Bursts photons are out of order, please check your data')
    elif out_model == &h_mod.model:
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise Exception('Too many photon streams in data for H2MM model')
    cdef h2mm_model out = h2mm_model(np.zeros(h_mod.nstate),np.zeros((h_mod.model.nstate,h_mod.model.nstate)),np.zeros((h_mod.nstate,h_mod.ndet)))
    PyMem_Free(out.model.prior)
    PyMem_Free(out.model.trans)
    PyMem_Free(out.model.obs)
    out.model = out_model[0]
    if out.model.conv == 1:
        print(f'The model converged after {out.model.niter} iterations')
    elif out.model.conv == 2:
        print('Optimization reached maximum number of iterations')
    elif out.model.conv == 3:
        print('Optimization reached maxiumum time')
    else:
        print(f'An error occured on iteration {out.model.niter}, returning previous model')
    PyMem_Free(b_det);
    PyMem_Free(b_time);
    PyMem_Free(burst_sizes)
    return out

def viterbi_path(h2mm_model h_mod, list burst_colors, list burst_times):
    """
    Calculate the most likely state path through a set of data given a H2MM model

    Parameters
    ----------
    h_model : h2mm_model
        An H2MM model, should be optimized for the given data set
        (result of EM_H2MM_C) to ensure results coorespond to give the most likely
        path
    indexes : list of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) cooresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1to1 coorespondence to the times list
    times : list of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) cooresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1to1 coorespondence to the indexes list
    
    Returns
    -------
    path : list of NUMPY 1D int arrays
        The most likely state path for each photon
    scale : list of NUMPY 1D float arrays
        The posterior probability for each photon
    ll : NUMPY 1D float array
        loglikelihood of each burst
    icl : float
        Integrated complete likelihood, essentially the BIC for the viterbi path
    """
    # assert statements to verify that the data is valid, ie matched lengths and dimensions for burst_times and burst_colors in all bursts
    assert len(burst_colors) == len(burst_times), "Mismatch in burst_colors and burst_times lengths, burst_colors and burst_times must be of the same length"
    cdef size_t i
    cdef size_t nphot = 0
    cdef size_t num_burst = len(burst_colors)
    for i in range(num_burst):
        assert burst_colors[i].ndim == 1, f"burst_colors[{i}] must be a 1D array"
        assert burst_times[i].ndim == 1 , f"burst_times[{i}] must be a 1D array"
        assert burst_times[i].shape[0] == burst_colors[i].shape[0], f"Mismatch in lengths between burst_times[{i}] and burst_colors[{i}], cannot create burst"
    # set up the limits function
    # allocate the memory for the pointer arrays to be submitted to the C function
    cdef unsigned long *burst_sizes = <unsigned long*> PyMem_Malloc(num_burst * sizeof(unsigned long))
    cdef unsigned long long **b_time = <unsigned long long**> PyMem_Malloc(num_burst * sizeof(unsigned long long*))
    cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef ph_path *path_ret = <ph_path*> PyMem_Malloc(num_burst * sizeof(ph_path))
    # for loop casts the values to the right datat type, then makes sure the data is contiguous, but don't copy the pointers just yet, that is in a separate for loop to make sure no numpy shenanigans 
    for i in range(num_burst):
        burst_sizes[i] = burst_colors[i].shape[0]
        nphot += burst_sizes[i]
        if burst_sizes[i] < 3:
            raise Exception(f'Bursts must have at least 3 photons, burst {i} has only {burst_sizes[i]}')
        burst_colors[i] = burst_colors[i].astype('L')
        burst_times[i] = burst_times[i].astype('Q')
        if not burst_colors[i].flags['C_CONTIGUOUS']:
            burst_colors[i] = np.ascontiguousarray(burst_colors[i])
        if not burst_times[i].flags['C_CONTIGUOUS']:
            burst_times[i] = np.ascontiguousarray(burst_times[i])
    # now make the list of pointers
    for i in range(num_burst):
        b_det[i] = get_ptr_l(burst_colors[i])
        b_time[i] = get_ptr_ull(burst_times[i])
    # set up the in and out h2mm_mod variables
    cdef int e_val = viterbi(num_burst,burst_sizes,b_time,b_det,&h_mod.model,path_ret)
    if e_val == 1:
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise Exception('Bursts photons are out of order, please check your data')
    elif e_val == 2:
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise Exception('Too many photon streams in data for H2MM model')
    cdef list path = []
    cdef list scale = []
    cdef double loglik = 0
    cdef np.ndarray[double,ndim=1] ll = np.zeros(num_burst)
    for i in range(num_burst):
        loglik += path_ret[i].loglik
        ll[i] = path_ret[i].loglik
        path.append(np.copy(np.asarray(<size_t[:path_ret[i].nphot]> path_ret[i].path)))
        scale.append(np.copy(np.asarray(<double[:path_ret[i].nphot]> path_ret[i].scale)))
        PyMem_Free(path_ret[i].path)
        PyMem_Free(path_ret[i].scale)
    cdef double icl = ((h_mod.nstate**2 + ((h_mod.ndet - 1) * h_mod.nstate) - 1) * np.log(nphot)) - 2 * loglik
    PyMem_Free(path_ret)
    PyMem_Free(b_det)
    PyMem_Free(b_time)
    PyMem_Free(burst_sizes)
    return path, scale, ll, icl


def viterbi_sort(h2mm_model hmod, list indexes, list times):
    """
    An all inclusive viterbi processing algorithm. Returns the ICL, the most likely
    state path, posterior probabilities, and a host of information sorted by
    bursts and dwells

    Parameters
    ----------
    h_model : h2mm_model
        An H2MM model, should be optimized for the given data set
        (result of EM_H2MM_C) to ensure results coorespond to give the most likely
        path
    indexes : list of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) cooresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1to1 coorespondence to the times list
    times : list of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) cooresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1to1 coorespondence to the indexes list

    Returns
    -------
    icl : float
        Integrated complete likelihood, essentially the BIC for the viterbi path
    path : list of NUMPY 1D int arrays
        The most likely state path for each photon
    scale : list of NUMPY 1D float arrays
        The posterior probability for each photon
    ll : NUMPY 1D float array
        loglikelihood of each burst
    burst_type : NUMPY 1D int array
        Identifies which states are present in the burst, identified in a binary
        format, ie if states 0 and 2 are present in a burst, that element will
        be 0b101 = 5
    dwell_mid : N square list of lists of NUMPY 1D int arrays
        Gives the dwell times of all dwells with full residence time in the data
        set, sorted by the state of the dwell (Top level list), and the successor
        state (Lower level list), each element of the numpy array is the duration
        of the dwell in the clock rate of the data set
    dwell_beg : N square list of lists of NUMPY 1D int arrays
        Gives the dwell times of all dwells that start at the beginning of a burst,
        sorted by the state of the dwell (Top level list), and the successor
        state (Lower level list), each element of the numpy array is the duration
        of the dwell in the clock rate of the data set
    dwell_end : N square list of lists of NUMPY 1D int arrays
        Gives the dwell times of all dwells that start at the end of a burst,
        sorted by the state of the dwell (Top level list), and the preceeding
        state (Lower level list), each element of the numpy array is the duration
        of the dwell in the clock rate of the data set
    dwell_burst : N list of NUMPY 1D int arrays
        List of bursts that were only in one state, giving their durations
    ph_counts : N list of NUMPY 2D int arrays
        Counts of photons in each dwell, sorted by index and state of the dwell
        The index of the list identifies the state of the burst, the 1 index
        identifies the index of photon, and the 0 index are individual dwells
    
    """
    # use viterbi to find most likely path based on posterior probability through all bursts
    cdef Py_ssize_t i, b, e
    cdef list paths, scale
    cdef np.ndarray[double,ndim=1] ll
    cdef double icl
    paths, scale, ll, icl = viterbi_path(hmod,indexes,times)
    
    # sorting bursts based on which dwells occur in them
    cdef np.ndarray[long, ndim=1] burst_type = np.zeros(len(indexes),dtype=int)
    for i in range(len(indexes)):
        # determine the "type" of burst it is, the index represents if a state is present, using binary, minus 1 because there are no bursts with no dwells
        burst_type_temp = 0
        for st in range(hmod.nstate):
            if np.any(paths[i] == st):
                burst_type_temp += 2**st
        burst_type[i] = burst_type_temp
    # sorting dwells based on transition rates, and calculating their E and S values
    cdef list dwell_mid = [[[]for i in range(hmod.nstate)] for j in range(hmod.nstate)]
    cdef list dwell_beg = [[[]for i in range(hmod.nstate)] for j in range(hmod.nstate)]
    cdef list dwell_end = [[[]for i in range(hmod.nstate)] for j in range(hmod.nstate)]
    cdef list dwell_burst = [[] for i in range(hmod.nstate)]
    cdef list ph_mid = [[np.zeros((0,hmod.ndet),dtype=int)for i in range(hmod.nstate)] for j in range(hmod.nstate)]
    cdef list ph_beg = [[np.zeros((0,hmod.ndet),dtype=int)for i in range(hmod.nstate)] for j in range(hmod.nstate)]
    cdef list ph_end = [[np.zeros((0,hmod.ndet),dtype=int)for i in range(hmod.nstate)] for j in range(hmod.nstate)]
    cdef list ph_burst = [np.zeros((0,hmod.ndet),dtype=int) for i in range(hmod.nstate)]
    cdef list ph_counts = [np.zeros((0,hmod.ndet),dtype=int) for i in range(hmod.nstate)]
    cdef np.ndarray time, index, state
    cdef np.ndarray[long,ndim=2] ph_counts_temp = np.zeros((1,hmod.ndet),dtype=int)
    #sorts the dwells into dwell times and photon counts
    for time, index, state in zip(times, indexes, paths):
        demar = np.append(1,np.diff(state))
        begs = np.argwhere(demar!=0)[:,0] #identifies which photons have a dwell, this array marks the beginning indexes
        ends = np.append(begs[1:],time.shape[0]) # make an equivalent ending index array
        for b, e in zip(begs,ends):
            for i in range(hmod.ndet):
                ph_counts_temp[0,i] = (index[b:e] == i).sum().astype('long')
            ph_counts[state[b]] = np.concatenate((ph_counts[state[b]],ph_counts_temp))
            if b!= 0 and e != time.shape[0]:
                dwell_mid[state[b]][state[e]].append([time[e]-time[b], e - b])
                ph_mid[state[b]][state[e]] = np.concatenate((ph_mid[state[b]][state[e]],ph_counts_temp))
            elif b==0 and e != time.shape[0]:
                dwell_beg[state[b]][state[e]].append([time[e]-time[b], e - b])
                ph_beg[state[b]][state[e]] = np.concatenate((ph_beg[state[b]][state[e]],ph_counts_temp))
            elif b!=0 and e== time.shape[0]:
                dwell_end[state[b]][state[b-1]].append([time[e-1]-time[b], e - b])
                ph_end[state[b]][state[b-1]] = np.concatenate((ph_end[state[b]][state[b-1]],ph_counts_temp))
            else:
                dwell_burst[state[b]].append([time[e-1] - time[b], e - b])
                ph_burst[state[b]] = np.concatenate((ph_burst[state[b]],ph_counts_temp))
    for i in range(hmod.nstate):
        dwell_burst[i] = np.array(dwell_burst[i]).astype('long') if len(dwell_burst[i]) > 0 else np.zeros((0,2)).astype('long')
        for j in range(hmod.nstate):
            dwell_mid[i][j] = np.array(dwell_mid[i][j]).astype('long') if len(dwell_mid[i][j]) > 0 else np.zeros((0,2)).astype('long')
            dwell_beg[i][j] = np.array(dwell_beg[i][j]).astype('long') if len(dwell_beg[i][j]) > 0 else np.zeros((0,2)).astype('long')
            dwell_end[i][j] = np.array(dwell_end[i][j]).astype('long') if len(dwell_end[i][j]) > 0 else np.zeros((0,2)).astype('long')
    return icl, paths, scale, ll, burst_type, dwell_mid, dwell_beg, dwell_end, dwell_burst, ph_counts, ph_mid, ph_beg, ph_end, ph_burst
