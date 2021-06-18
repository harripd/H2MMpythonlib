#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:49:24 2021

@author: Paul David Harris
"""

import os
import numpy as np
cimport numpy as np
import warnings
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport PyObject

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
    ctypedef struct h2mm_minmax:
        h2mm_mod *mins
        h2mm_mod *maxs
    ctypedef struct ph_path:
        size_t nphot
        size_t nstate
        double loglik
        size_t *path
        double *scale
        double *omega
    void h2mm_normalize(h2mm_mod *model_params)
    h2mm_mod* C_H2MM(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *in_model, lm *limits, void (*model_limits_func)(h2mm_mod*,h2mm_mod*,h2mm_mod*,void*), void *model_limits) nogil
    int viterbi(unsigned long num_bursts, unsigned long *burst_sizes, unsigned long long **burst_times, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array, unsigned long num_cores) nogil
    void limit_revert(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims)
    void limit_revert_old(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims)
    void limit_minmax(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims)

ctypedef struct bound_struct:
    void *func
    void *limits

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
        # if statements check to confirm first the correct dimensinality of input matrices, then that their shapes match
        cdef size_t i, j
        if prior.ndim != 1:
            raise ValueError("Prior matrix must have ndim=1, too many dimensions in prior")
        if trans.ndim != 2:
            raise ValueError("Trans matrix must ahve ndim=2, wrong dimensionallity of trans matrix")
        if obs.ndim != 2:
            raise ValueError("Obs matrix must ahve ndim=2, wrong dimensionallity of obs matrix")
        if not (prior.shape[0] == trans.shape[0] == obs.shape[0]):
            raise ValueError("Dim 0 of one of the matrices does not match the others, these represent the number of states, so input matrices do not represent a single model")
        if trans.shape[0] != trans.shape[1]:
            raise ValueError("Trans matrix is not square, and connot be used for a model")
        if niter < 0:
            raise ValueError("niter must be positive")
        if nphot< 0: 
            raise ValueError("nphot must be positive")
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
        if prior.ndim != 1:
            raise ValueError("Prior must be 1D numpy floating point array")
        if prior.shape[0] != self.model.nstate:
            raise ValueError("Cannot change the number of states")
        if prior.sum() != 1.0:
            warnings.warn("Input array not stochastic, new array will be normalized")
        self.model.loglik = -np.inf
        self.model.conv = 0
        prior = prior.astype('double')
        for i in range(self.model.nstate):
            self.model.prior[i] = prior[i]
        self.normalize()
    @property
    def trans(self):
        return np.asarray(<double[:self.model.nstate,:self.model.nstate]>self.model.trans)
    @trans.setter
    def trans(self,trans):
        if trans.ndim != 2:
            raise ValueError("Trans must be a 2D numpy floating point array")
        if trans.shape[0] != trans.shape[1]:
            raise ValueError("Trans must be a square array")
        if trans.shape[0] != self.model.nstate:
            raise ValueError("Cannot change the number of states in a model")
        if np.any(trans.sum(axis=1) != 1.0):
            warnings.warn("Input matrix not row stochastic, new matrix will be normalized")
        self.model.loglik = -np.inf
        self.model.conv = 0
        trans = trans.astype('double')
        for i in range(self.model.nstate):
            for j in range(self.model.nstate):
                self.model.trans[self.model.nstate*i + j] = trans[i,j]
        self.normalize()
    @property
    def obs(self):
        return np.asarray(<double[:self.model.ndet,:self.model.nstate]>self.model.obs).T
    @obs.setter
    def obs(self,obs):
        if obs.ndim != 2:
            raise ValueError("Obs must be a 2D numpy floating point array")
        if obs.shape[0] != self.model.nstate:
            raise ValueError("Cannot change the number of states in a model")
        if obs.shape[1] != self.model.ndet: 
            raise ValueError("Cannot change the number of streams in the model")
        if np.any(obs.sum(axis=1) != 1.0):
            warnings.warn("Input matrix not row stochastic, new matrix will be normalized")
        self.model.loglik = -np.inf
        self.model.conv = 0
        obs = obs.astype('double')
        for i in range(self.model.ndet):
            for j in range(self.model.nstate):
                self.model.obs[self.model.nstate * i + j] = obs[j,i]
        self.normalize()
    @property
    def loglik(self):
        if self.model.nphot == 0:
            warnings.warn("Model not optimized, loglik will be meaningless")
        return self.model.loglik
    @property
    def k(self):
        return self.model.nstate**2 + ((self.model.ndet - 1)*self.model.nstate) - 1
    @property
    def bic(self):
        if self.model.nphot < 0:
            raise Exception("Must run through H2MM_C first, BIC not known")
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
        if nphot <= 0:
            raise ValueError("nphot must be greater than 0")
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
    @property
    def niter(self):
        return self.model.niter
    @niter.setter
    def niter(self,niter):
        if niter <= 0:
            raise ValueError("Cannot have negative iterations")
        self.model.niter = niter
    def set_converged(self,converged):
        if not isinstance(converged,bool):
            raise ValueError("Input must be True or False")
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
        cdef h2mm_mod* out_model = C_H2MM(num_burst,burst_sizes,b_time,b_det,&self.model,&limits,NULL,NULL)
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
    def __repr__(self):
        cdef size_t i, j
        msg = f"nstate: {self.model.nstate} ndet: {self.model.ndet} nphot: {self.model.nphot}, loglik: {self.model.loglik} converged state: {self.model.conv}\n"
        msg += "prior:\n"
        for i in range(self.model.nstate):
            msg += f"{self.model.prior[i]}, " if i < self.model.nstate -1 else f"{self.model.prior[i]}\n"
        msg += "trans:\n"
        for i in range(self.model.nstate):
            for j in range(self.model.nstate):
                msg += f"{self.model.trans[i*self.model.nstate + j]}"
                msg += ", " if j < self.model.nstate -1 else "\n"
        msg += "obs:\n"
        for i in range(self.model.nstate):
            for j in range(self.model.ndet):
                msg += f"{self.model.obs[j*self.model.nstate + i]}"
                msg += ", " if j < self.model.ndet - 1 else "\n"
        return msg
    def __str__(self):
        if self.model.conv == 0:
            msg = "Initial model,"
        elif self.model.conv == 1:
            msg = f"Converged model, {self.model.niter} iterations,"
        elif self.model.conv == 2:
            msg = "Max iterations {self.model.niter} iterations,"
        elif self.model.conv == 3:
            msg = "Max time {self.model.niter} iterations,"
        return msg + f'States: {self.model.nstate} Streams: {self.model.ndet}, loglik {self.model.loglik}, '
    def __dealloc__(self):
        if self.model.prior is not NULL:
            PyMem_Free(self.model.prior)
        if self.model.trans is not NULL:
            PyMem_Free(self.model.trans)
        if self.model.obs is not NULL:
            PyMem_Free(self.model.obs)

cdef class _h2mm_lims:
    cdef:
        h2mm_minmax limits
    def __cinit__(self, h2mm_model model, np.ndarray[double,ndim=1] min_prior, np.ndarray[double,ndim=1] max_prior, 
                  np.ndarray[double,ndim=2] min_trans, np.ndarray[double,ndim=2] max_trans, 
                  np.ndarray[double,ndim=2] min_obs, np.ndarray[double,ndim=2] max_obs):
        cdef size_t i, j
        cdef size_t nstate = model.model.nstate
        cdef size_t ndet = model.model.ndet
        self.limits.mins = <h2mm_mod*> PyMem_Malloc(sizeof(h2mm_mod))
        self.limits.mins.prior = NULL
        self.limits.mins.trans = NULL
        self.limits.mins.obs = NULL
        self.limits.maxs = <h2mm_mod*> PyMem_Malloc(sizeof(h2mm_mod))
        self.limits.maxs.prior = NULL
        self.limits.maxs.trans = NULL
        self.limits.maxs.obs = NULL
        self.limits.mins.nstate = nstate
        self.limits.mins.ndet = ndet
        self.limits.mins.nphot = model.model.nphot
        self.limits.mins.niter = model.model.niter
        self.limits.mins.loglik = model.model.loglik
        self.limits.mins.conv = model.model.conv
        self.limits.mins.prior = <double*> PyMem_Malloc(nstate*sizeof(double))
        self.limits.mins.trans = <double*> PyMem_Malloc(nstate*nstate*sizeof(double))
        self.limits.mins.obs = <double*> PyMem_Malloc(nstate*ndet*sizeof(double))
        self.limits.maxs.nstate = nstate
        self.limits.maxs.ndet = ndet
        self.limits.maxs.nphot = model.model.nphot
        self.limits.maxs.niter = model.model.niter
        self.limits.maxs.loglik = model.model.loglik
        self.limits.maxs.conv = model.model.conv
        self.limits.maxs.prior = <double*> PyMem_Malloc(nstate*sizeof(double))
        self.limits.maxs.trans = <double*> PyMem_Malloc(nstate*nstate*sizeof(double))
        self.limits.maxs.obs = <double*> PyMem_Malloc(nstate*ndet*sizeof(double))
        for i in range(nstate):
            self.limits.mins.prior[i] = min_prior[i]
            self.limits.maxs.prior[i] = max_prior[i]
            for j in range(nstate):
                if i != j:
                    self.limits.mins.trans[i*nstate + j] = min_trans[i,j]
                    self.limits.maxs.trans[i*nstate + j] = max_trans[i,j]
                else:
                    self.limits.mins.trans[i*nstate + j] = 0.0
                    self.limits.maxs.trans[i*nstate + j] = 1.0
            for j in range(ndet):
                self.limits.mins.obs[i + j*nstate] = min_obs[i,j]
                self.limits.maxs.obs[i + j*nstate] = max_obs[i,j]
    def __dealloc__(self):
        if self.limits.mins.prior is not NULL:
            PyMem_Free(self.limits.mins.prior)
        if self.limits.maxs.prior is not NULL:
            PyMem_Free(self.limits.maxs.prior)
        if self.limits.mins.trans is not NULL:
            PyMem_Free(self.limits.mins.trans)
        if self.limits.maxs.trans is not NULL:
            PyMem_Free(self.limits.maxs.trans)
        if self.limits.mins.obs is not NULL:
            PyMem_Free(self.limits.mins.obs)
        if self.limits.maxs.obs is not NULL:
            PyMem_Free(self.limits.maxs.obs)
        if self.limits.mins is not NULL:
            PyMem_Free(self.limits.mins)
        if self.limits.maxs is not NULL:
            PyMem_Free(self.limits.maxs)

class h2mm_limits:
    """
    Special class for setting limits on the h2mm_model, as min and max values
    """
    def __init__(self,model=None,min_prior=None,max_prior=None,min_trans=None,max_trans=None,min_obs=None,max_obs=None,nstate=0,ndet=0):
        """
        Special class for setting limits on the h2mm_model, as min and max values
        If min/max kwarg is given ad a float, the minimum will be set as universal

        Parameters
        ----------
        model : h2mm_model, optional
            An h2mm_model to base the limits off of, if None. The main purpose
            is to allow the user to check that the limts are valid for the model.
            Specifying this model will also lock the states/streams of the model,
            while if None, the limits is more flexible
            If None, none of these checks will be in place
            The default is None.
        min_prior : float or 1D numpy array, optional
            The minimum value(s) of the prior array. If float, all values are
            set the same, but unless fixed elsewhere, the number of states is
            flexible.
            If None, no limits are set (all values min is 0.0)
            The default is None.
        max_prior : float or 1D numpy array, optional
            The maximum value(s) of the prior array. If float, all values are
            set the same, but unless fixed elsewhere, the number of states is
            flexible.
            If None, no limits are set (all values min is 1.0)
            The default is None.
        min_trans : float or 2D square numpy array, optional
            The minimum value(s) of the trans array. If float, all values are
            set the same, but unless fixed elsewhere, the number of states is
            flexible. Values on diagonal set to 0.0
            If None, no limits are set (all values min is 0.0)
            The default is None.
        max_trans : float or 2D square numpy array, optional
            The maximum value(s) of the trans array. If float, all values are
            set the same, but unless fixed elsewhere, the number of states is
            flexible. Values on diagonal set to 1.0
            If None, no limits are set (all values min is 1.0)
            The default is None.
        min_obs : float or 2D numpy array, optional
            The minimum value(s) of the obs array. If float, all values are
            set the same, but unless fixed elsewhere, the number of states is
            flexible.
            If None, no limits are set (all values min is 0.0)
            The default is None.
        max_obs : float or 2D numpy array, optional
            The maximum value(s) of the obs array. If float, all values are
            set the same, but unless fixed elsewhere, the number of states is
            flexible.
            If None, no limits are set (all values min is 1.0)
            The default is None.
        nstate : int, optional
            Number of states in the model to be optimzied, if set to 0, and not
            spefied elsewhere, number of states is flexible.
            The default is 0.
        ndet : int, optional
            Number of streams in the model to be optimzied, if set to 0, and not
            spefied elsewhere, number of states is flexible.
            The default is 0.
            
        Raises
        ------
        ValueError
            Init method checks that the values are valid, and raises errors
            describing the issue.
        """
        none_kwargs = True
        if not isinstance(nstate,int) or not isinstance(ndet,int) or nstate < 0 or ndet < 0:
            raise ValueError("Cannot give negative or non-int values for nstate or ndet")
        arg_list = {"min_prior":min_prior, "max_prior":max_prior, 
                    "min_trans":min_trans, "max_trans":max_trans, 
                    "min_obs":min_obs, "max_obs":max_obs}
        for name, param in arg_list.items():
            if isinstance(param,float) or isinstance(param,np.ndarray):
                none_kwargs = False
                if isinstance(param,np.ndarray):
                    if nstate == 0:
                        nstate = param.shape[0]
                    elif nstate != param.shape[0]:
                        raise ValueError(f"Conflicting values for numbers of states encountered at {name}, check array dimensions")
                    if name == 'min_obs' or name == 'max_obs':
                        if ndet == 0:
                            ndet = param.shape[1]
                        elif ndet != param.shape[1]:
                            raise ValueError(f"Conflicting values for numbers of photon streams encountered at {name}, check array dimensions")
            elif param is not None:
                raise ValueError(f"{name} must be None, float or numpy array, got {type(param)}")
        if none_kwargs:
            warnings.warn("No limits specified, this is a non-limiting h2mm_limits object")
        # check that limist and supplied model are compatible
        if isinstance(model,h2mm_model):
            if nstate == 0:
                nstate == model.nstate
            elif nstate != model.nstate:
                raise ValueError(f"Limits and model have different number of states, got {nstate} and {model.nstate}")
            if ndet == 0:
                ndet = model.ndet
            elif ndet != model.ndet:
                raise ValueError(f"Limits and model have different number of photon streams, got {ndet} and {model.ndet}")
            if (min_prior is not None and np.any(min_prior > model.prior)) or (max_prior is not None and np.any(max_prior < model.prior)):
                raise ValueError("model prior out of range of min/max prior values")
            if isinstance(min_trans,float):
                if np.any(min_trans > model.trans[np.eye(model.nstate)==0]):
                    raise ValueError("model trans out of range of min/max trans values")
            elif isinstance(min_trans,np.ndarray):
                if np.any(min_trans[np.eye(model.nstate)==0] > model.trans[np.eye(model.nstate)==0]):
                    raise ValueError("model trans out of range of min/max trans values")
            if isinstance(max_trans,float):
                if np.any(max_trans < model.trans[np.eye(model.nstate)==0]):
                    raise ValueError("model trans out of range of min/max trans values")
            elif isinstance(max_trans,np.ndarray):
                if np.any(max_trans[np.eye(model.nstate)==0] < model.trans[np.eye(model.nstate)==0]):
                    raise ValueError("model trans out of range of min/max trans values")
            if (min_obs is not None and np.any(min_obs > model.obs)) or (max_obs is not None and np.any(max_obs < model.obs)):
                raise ValueError("model obs out of range of min/max obs values")
        elif model is not None:
            raise ValueError(f"model must be h2mm_model or None, got {type(model)}")
        self.min_prior = min_prior if min_prior is not None else 0.0
        self.max_prior = max_prior if max_prior is not None else 1.0
        if np.any(self.min_prior > self.max_prior):
            raise ValueError("min_prior cannot be greater than max_prior")
        self.min_trans = min_trans if min_trans is not None else 0.0
        self.max_trans = max_trans if max_trans is not None else 1.0
        self.min_obs = min_obs if min_obs is not None else 0.0
        self.max_obs = max_obs if max_obs is not None else 1.0
        if np.any(self.min_obs > self.max_obs):
            raise ValueError("min_obs cannot be greater than max_obs")
        self.ndet = ndet
        self.nstate = nstate
        self.model = model
    def make_model(self,model):
        """
        Method for chekcing the limits arrays generated from the input model
        
        Parameters
        ----------
        model : h2mm_model
            h2mm_model for which the limits are to be specified.
            Also, values are checked and a warning is raised if the model has
            values that are out of the range of the limits
        Raises
        ------
        Exception
            When user has reset a specific field so the field is no longer valid.
        ValueError
            Limits cannot be made for the given model.
        
        Returns
        -------
        model : h2mm_model
            Model to be optimized with specified limits.
        min_prior : 1D numpy float array
            Minimum values for each element of the prior array
        max_prior : 1D numpy float array
            Maximum values for each element of the prior array
        min_trans : 2D square numpy float array
            Minimum values of each element of the trans array
        max_trans : 2D square numpy float array
            Maximum values of each element of the trans array
        min_obs : 2D numpy float array
            Minimum values of each element of the obs array
        max_obs : 2D numpy float array
            Maximum values of each element of the obs array
        
        """
        if not isinstance(model,h2mm_model):
            raise Exception("Must be h2mm_model")
        if self.nstate != 0 and self.nstate != model.nstate:
            raise ValueError(f"Mismatch in states between model ({model.nstate}) and limits object ({self.nstate})")
        if self.ndet != 0 and self.ndet != model.ndet:
            raise ValueError(f"Mismatch in photon streams between model ({model.ndet}) and limits object ({self.ndet})")
        ndet = model.ndet
        nstate = model.nstate
        if self.min_prior is None:
            min_prior = np.zeros(nstate).astype('double')
        elif isinstance(self.min_prior,float):
            min_prior = (self.min_prior * np.ones(nstate)).astype('double')
        elif isinstance(self.min_prior,np.ndarray) and self.min_prior.ndim == 1 and self.min_prior.shape[0] == nstate:
            min_prior = self.min_prior.astype('double')
        else:
            raise Exception("Type of min_prior changed")
        if self.max_prior is None:
            max_prior = np.ones(nstate).astype('double')
        elif isinstance(self.min_prior,float):
            max_prior = (self.max_prior * np.ones(nstate)).astype('double')
        elif isinstance(self.max_prior,np.ndarray) and self.max_prior.ndim == 1 and self.max_prior.shape[0] == nstate:
            max_prior = self.max_prior.astype('double')
        else:
            raise Exception("Type of max_prior changed")
        if np.any(min_prior > max_prior):
            raise ValueError("min_prior cannot be greater than max_prior")
        if self.min_trans is None:
            min_trans = np.zeros((nstate,nstate)).astype('double')
        elif isinstance(self.min_trans,float):
            min_trans = (self.min_trans * np.ones((nstate,nstate))).astype('double')
            min_trans[np.eye(nstate)==1] = 0.0
        elif isinstance(self.min_trans,np.ndarray) and self.min_trans.ndim == 2 and self.min_trans.shape[0] == self.min_trans.shape[1] == nstate:
            min_trans = self.min_trans.astype('double')
        else:
            raise Exception("Type of min_trans changed")
        if np.any(min_trans.sum(axis=1) > 1.0):
            raise Exception("min_trans disallows row stochastic matrix")
        if self.max_trans is None:
            max_trans = np.ones((nstate,nstate)).astype('double')
        elif isinstance(self.max_trans,float):
            max_trans = (self.max_trans * np.ones((nstate,nstate))).astype('double')
            max_trans[np.eye(nstate)==1] = 1.0
        elif isinstance(self.max_trans,np.ndarray) and self.max_trans.shape[0] == self.max_trans.shape[1] == nstate:
            max_trans = self.max_trans.astype('double')
        else:
            raise Exception("Type of max_trans changed")
        if np.any(min_trans > max_trans):
            raise ValueError("min_trans cannot be greater than max_trans")
        if self.min_obs is None:
            min_obs = np.zeros((nstate,ndet)).astype('double')
        elif isinstance(self.min_obs,float):
            min_obs = (self.min_obs * np.ones((nstate,ndet))).astype('double')
        elif isinstance(self.min_obs,np.ndarray) and self.min_obs.ndim == 2 and self.min_obs.shape[0] == nstate and self.min_obs.shape[1] == ndet:
            min_obs = self.min_obs.astype('double')
        else:
            raise Exception("Type of min_obs changed")
        if np.any(min_obs.sum(axis=1) > 1.0):
            raise ValueError("min_obs disallows row stochastic matrix")
        if self.max_obs is None:
            min_obs = np.ones((nstate,ndet)).astype('double')
        elif isinstance(self.max_obs,float):
            max_obs = (self.max_obs * np.ones((nstate,ndet))).astype('double')
        elif isinstance(self.max_obs,np.ndarray) and self.max_obs.ndim == 2 and self.max_obs.shape[0] == nstate and self.max_obs.shape[1] == ndet:
            max_obs = self.max_obs.astype('double')
        else:
            raise Exception("Type of max_obs changed")
        if np.any(max_obs.sum(axis=1) < 1.0):
            raise ValueError("max_obs dissallows row stochastic matrix")
        if np.any(min_obs > max_obs):
            raise ValueError("min_obs cannot be greater than max_obs")
        if np.any(min_prior > model.prior) or np.any(max_prior < model.prior):
            warnings.warn("Initial model out of min/max prior range, will result in undefined behavior")
        if np.any(min_trans > model.trans) or np.any(max_trans < model.trans):
            warnings.warn("Initial model out of min/max trans range, will result in undefined behavior")
        if np.any(min_obs > model.obs) or np.any(max_obs < model.obs):
            warnings.warn("Initial model out of min/max obs range, will result in undefined behavior")
        return model, min_prior,max_prior,min_trans,max_trans,min_obs,max_obs
    def _make_model(self,model):
        """
        Hidden method identical to make_model, except returns the hidden class
        _h2mm_lims, which is a wrapper for the C-level h2mm_minmax structure
        that the C code uses
        Parameters
        ----------
        model : h2mm_model
            Model to be optimized with limits
        Raises
        ------
        Exception
            When there is a wrong type or other issue
        ValueError
            When mismatch between limits streams and model streams
        Returns
        -------
        _h2mm_lims
            wrapper for the C-level h2mm_minmax structure that does the limiting
            of the model
        """
        if not isinstance(model,h2mm_model):
            raise Exception("Must be h2mm_model")
        if self.nstate != 0 and self.nstate != model.nstate:
            raise ValueError(f"Mismatch in states between model ({model.nstate}) and limits object ({self.nstate})")
        if self.ndet != 0 and self.ndet != model.ndet:
            raise ValueError(f"Mismatch in photon streams between model ({model.ndet}) and limits object ({self.ndet})")
        ndet = model.ndet
        nstate = model.nstate
        if self.min_prior is None:
            min_prior = np.zeros(nstate).astype('double')
        elif isinstance(self.min_prior,float):
            min_prior = (self.min_prior * np.ones(nstate)).astype('double')
        elif isinstance(self.min_prior,np.ndarray) and self.min_prior.ndim == 1 and self.min_prior.shape[0] == nstate:
            min_prior = self.min_prior.astype('double')
        else:
            raise Exception("Type of min_prior changed")
        if self.max_prior is None:
            max_prior = np.ones(nstate).astype('double')
        elif isinstance(self.min_prior,float):
            max_prior = (self.max_prior * np.ones(nstate)).astype('double')
        elif isinstance(self.max_prior,np.ndarray) and self.max_prior.ndim == 1 and self.max_prior.shape[0] == nstate:
            max_prior = self.max_prior.astype('double')
        else:
            raise Exception("Type of max_prior changed")
        if np.any(min_prior > max_prior):
            raise ValueError("min_prior cannot be greater than max_prior")
        if self.min_trans is None:
            min_trans = np.zeros((nstate,nstate)).astype('double')
        elif isinstance(self.min_trans,float):
            min_trans = (self.min_trans * np.ones((nstate,nstate))).astype('double')
            min_trans[np.eye(nstate)==1] = 0.0
        elif isinstance(self.min_trans,np.ndarray) and self.min_trans.ndim == 2 and self.min_trans.shape[0] == self.min_trans.shape[1] == nstate:
            min_trans = self.min_trans.astype('double')
        else:
            raise Exception("Type of min_trans changed")
        if np.any(min_trans.sum(axis=1) > 1.0):
            raise Exception("min_trans disallows row stochastic matrix")
        if self.max_trans is None:
            max_trans = np.ones((nstate,nstate)).astype('double')
        elif isinstance(self.max_trans,float):
            max_trans = (self.max_trans * np.ones((nstate,nstate))).astype('double')
            max_trans[np.eye(nstate)==1] = 1.0
        elif isinstance(self.max_trans,np.ndarray) and self.max_trans.shape[0] == self.max_trans.shape[1] == nstate:
            max_trans = self.max_trans.astype('double')
        else:
            raise Exception("Type of max_trans changed")
        if np.any(min_trans > max_trans):
            raise ValueError("min_trans cannot be greater than max_trans")
        if self.min_obs is None:
            min_obs = np.zeros((nstate,ndet)).astype('double')
        elif isinstance(self.min_obs,float):
            min_obs = (self.min_obs * np.ones((nstate,ndet))).astype('double')
        elif isinstance(self.min_obs,np.ndarray) and self.min_obs.ndim == 2 and self.min_obs.shape[0] == nstate and self.min_obs.shape[1] == ndet:
            min_obs = self.min_obs.astype('double')
        else:
            raise Exception("Type of min_obs changed")
        if np.any(min_obs.sum(axis=1) > 1.0):
            raise ValueError("min_obs disallows row stochastic matrix")
        if self.max_obs is None:
            min_obs = np.ones((nstate,ndet)).astype('double')
        elif isinstance(self.max_obs,float):
            max_obs = (self.max_obs * np.ones((nstate,ndet))).astype('double')
        elif isinstance(self.max_obs,np.ndarray) and self.max_obs.ndim == 2 and self.max_obs.shape[0] == nstate and self.max_obs.shape[1] == ndet:
            max_obs = self.max_obs.astype('double')
        else:
            raise Exception("Type of max_obs changed")
        if np.any(max_obs.sum(axis=1) < 1.0):
            raise ValueError("max_obs dissallows row stochastic matrix")
        if np.any(min_obs > max_obs):
            raise ValueError("min_obs cannot be greater than max_obs")
        if np.any(min_prior > model.prior) or np.any(max_prior < model.prior):
            warnings.warn("Initial model out of min/max prior range, will result in undefined behavior")
        if np.any(min_trans > model.trans) or np.any(max_trans < model.trans):
            warnings.warn("Initial model out of min/max trans range, will result in undefined behavior")
        if np.any(min_obs > model.obs) or np.any(max_obs < model.obs):
            warnings.warn("Initial model out of min/max obs range, will result in undefined behavior")
        return _h2mm_lims(model,min_prior,max_prior,min_trans,max_trans,min_obs,max_obs)
            
cdef h2mm_model model_from_ptr(h2mm_mod *model):
    # function to make a h2mm_model object from a pointer generated by C code
    cdef h2mm_model ret_model = h2mm_model(np.zeros((model.nstate)),np.zeros((model.nstate,model.nstate)),np.zeros((model.nstate,model.ndet)))
    if ret_model.model.prior is not NULL:
        PyMem_Free(ret_model.model.prior)
    if ret_model.model.trans is not NULL:
        PyMem_Free(ret_model.model.trans)
    if ret_model.model.obs is not NULL:
        PyMem_Free(ret_model.model.obs)
    ret_model.model = model[0]
    return ret_model

cdef h2mm_model model_copy_from_ptr(h2mm_mod *model):
    # function for copying the values h2mm_mod C structure into an new h2mm_model object
    # this is slower that model_from _ptr, but the copy makes sure that changes
    # to the h2mm_model object do not change the original pointer
    # primarily used in the cy_limit function to create the model objects handed
    # to the user supplied python function
    cdef h2mm_model ret_model = h2mm_model(np.zeros((model.nstate)),np.zeros((model.nstate,model.nstate)),np.zeros((model.nstate,model.ndet)))
    cdef size_t i
    for i in range(model.nstate):
        ret_model.model.prior[i] = model.prior[i]
    for i in range(model.nstate**2):
        ret_model.model.trans[i] = model.trans[i]
    for i in range(model.nstate*model.ndet):
        ret_model.model.obs[i] = model.obs[i]
    ret_model.model.niter = model.niter
    ret_model.model.nphot = model.nphot
    ret_model.model.conv = model.conv
    ret_model.model.loglik = model.loglik
    return ret_model

cdef void model_copy_to_ptr(h2mm_model model, h2mm_mod *mod_ptr):
    # copy a h2mm_model object prior, trans, and obs into the C structure
    # skips other non-array values
    # primarily for copying the output of the python function called by cy_limit
    # into the C structure given by the pointer
    if model.model.nstate != mod_ptr.nstate:
        raise ValueError("Mismatched number of photon streams, got %d bounded model and %d from optimization" % (model.model.nstate, mod_ptr.nstate))
    if model.model.ndet != mod_ptr.ndet:
        raise ValueError("Mismatched number of photon streams, got %d bounded model and %d from optimization" % (model.model.ndet, mod_ptr.ndet))
    for i in range(mod_ptr.nstate):
        mod_ptr.prior[i] = model.model.prior[i]
    for i in range(mod_ptr.nstate**2):
        mod_ptr.trans[i] = model.model.trans[i]
    for i in range(mod_ptr.nstate*mod_ptr.ndet):
        mod_ptr.obs[i] = model.model.obs[i]

# The wrapper for the user supplied python limits function
cdef void cy_limit(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims):
    cdef bound_struct *bound = <bound_struct*> lims
    cdef object func = <object> bound.func
    cdef object limits = <object> bound.limits
    cdef h2mm_model new_model = model_copy_from_ptr(new)
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    cdef h2mm_model old_model = model_copy_from_ptr(old)
    cdef h2mm_model bounded_model = func(new_model, current_model, old_model, limits)
    model_copy_to_ptr(bounded_model,new)

# Function called by C when 'print' is given as bounds_func kwarg to EM_H2MM_C
# Prints the jupyter notebook
cdef void model_print(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, void *lims):
    cdef object style = <object> lims
    cdef h2mm_model new_model = model_copy_from_ptr(new)
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    cdef h2mm_model old_model = model_copy_from_ptr(old)
    if isinstance(style,str):
        if style == 'all':
            print(current_model.__repr__())
        elif style == 'diff':
            print(f'Iteration:{current.niter}, loglik {current.loglik}, improvement {current.loglik - old.loglik}')
        elif style == 'comp':
            print(f'Iteration:{current.niter}, new loglik: {current.loglik}, old loglik: {old.loglik}')

def EM_H2MM_C(h2mm_model h_mod, list burst_colors, list burst_times, max_iter=3600, 
                bounds=None, bounds_func=None, max_time=np.inf, converged_min=1e-14, 
                num_cores= os.cpu_count()//2):
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
    ---------------------------
    bounds_func = None: str or callable
        function to be evaluated after every iteration of the H2MM algorithm
        its primary function is to place bounds on h2mm_model, secondary is to
        allow printing of iterations to python output like Jupyter Notebooks
        
        Note: bounding the h2mm_model causes the guarantee of improvement on each
        iteration until convergence to no longer apply, therefore the results are
        no longer guaranteed to be the optimal model within bounds
        -----------------
        Acceptable inputs
        -----------------
            C level limits: 'minmax' 'revert' 'revert_old'
                prevents the model from having values outside of those defined
                by h2mm_limits class given to bounds, if an iteration produces
                a new model for loglik calculation in the next iteration with 
                values that are out of those bounds, the 3 function differ in
                how they correct the model when a new model has a value out of
                bounds, they are as follows:
                'minmax': sets the out of bounds value to the min or max value
                    closer to the original value
                'revert': sets the out of bounds value to the value from the last
                    model for which the loglik was calculated
                'revert_old': similar to revert, but instead reverts to the value
                    from the model one before the last calculated model
            Print only function: 'print'
                prints each iteration to python output, format is specified in 
                the string passed to bounds, optional strings are:
                'all': (very verbose) prints the __repr__ for each model iteration
                'diff': gives the iteration number, the current loglik, and the 
                    difference (improvement) between the loglik of the current
                    and previous models
                'comp': prints the iteration number, the current and previous
                    logliks
            Callable: python function that takes 4 inputs and returns h2mm_model object
                WARNING: the user takes full responsibility for errors in the 
                    results
                must be a python function that takes 4 arguments, the first 3 are
                the h2mm_model objects, the third is the argument supplied to the
                bounds keyword argument, the function must return a single h2mm_limits
                object, the prior, trans and obs fields of which will be optimized
                next, other values are ignored.
                This can also be used to create your own printing function....
                just make sure to return the first argument passed to the function
                unchanged--- have fun with that...
    bounds = None: h2mm_limits, str, or 4th input to callable
        The argument to be passed to the bounds_func function, see entry for bounds_func
        to see valid valued based on bounds_func input, see the options for bounds_func
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
    # if statements verify that the data is valid, ie matched lengths and dimensions for burst_times and burst_colors in all bursts
    if len(burst_colors) != len(burst_times):
        raise ValueError("Mismatch in burst_colors and burst_times lengths, burst_colors and burst_times must be of the same length")
    cdef size_t i
    cdef size_t num_burst = len(burst_colors)
    for i in range(num_burst):
        if burst_colors[i].ndim != 1:
            raise ValueError(f"burst_colors[{i}] must be a 1D array")
        if burst_times[i].ndim != 1: 
            raise ValueError(f"burst_times[{i}] must be a 1D array")
        if burst_times[i].shape[0] != burst_colors[i].shape[0]:
            (f"Mismatch in lengths between burst_times[{i}] and burst_colors[{i}], cannot create burst")
    # check the bounds func
    if bounds_func is not None and bounds_func not in ['minmax','revert', 'revert_old', 'print'] and not callable(bounds_func):
        raise ValueError("Invalid bounds_func input, must be 'minmax', 'revert', 'revert_old' 'print' or function")
    elif bounds_func in ['minmax','revert','revert_old'] and not isinstance(bounds,h2mm_limits):
        raise ValueError("If bounds_func is 'minmax', 'revert' or 'revert_old', bounds must be a h2mm_limits object")
    elif bounds_func == 'print' and bounds not in ['all','diff','comp']:
        if bounds is None:
            bounds = 'diff'
        else:
            raise ValueError("must specify 'all', 'diff', or 'comp' if bounds_func is 'print'")
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
    # if the bounds_func is not None, then setup the bounds arrays
    cdef void (*bound_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, void*)
    cdef void *b_ptr
    # cdef h2mm_minmax *bound = <h2mm_minmax*> PyMem_Malloc(sizeof(h2mm_minmax))
    cdef _h2mm_lims bound
    cdef bound_struct *b_struct = <bound_struct*> PyMem_Malloc(sizeof(bound_struct))
    if bounds_func is None:
        bound_func = NULL
        b_ptr = NULL
    elif bounds_func == 'minmax' or bounds_func == 'revert' or bounds_func == 'revert_old':
        if isinstance(bounds, h2mm_limits):
            bound = bounds._make_model(h_mod)
            b_ptr = <void*> &bound.limits
            if bounds_func == 'minmax':
                bound_func = limit_minmax
            elif bounds_func == 'revert':
                bound_func = limit_revert
            elif bounds_func == 'revert_old':
                bound_func = limit_revert_old
        else:
            PyMem_Free(b_struct)
            PyMem_Free(b_det)
            PyMem_Free(b_time)
            PyMem_Free(burst_sizes)
            raise ValueError("bounds keword argument must is not of class h2mm_limits")
    elif bounds_func == 'print':
        bound_func = model_print
        b_ptr = <void*> bounds
    elif callable(bounds_func):
        b_struct.func = <void*> bounds_func
        b_struct.limits = <void*> bounds
        b_ptr = b_struct
        bound_func = cy_limit
    else:
        PyMem_Free(b_struct)
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise ValueError("bounds_func must be 'minmax', 'revert', 'print' or a function")
    # set up the in and out h2mm_mod variables
    cdef h2mm_mod* out_model = C_H2MM(num_burst,burst_sizes,b_time,b_det,&h_mod.model,&limits, bound_func, b_ptr)
    # free the limts arrays
    if out_model is NULL:
        PyMem_Free(b_struct)
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise Exception('Bursts photons are out of order, please check your data')
    elif out_model == &h_mod.model:
        PyMem_Free(b_struct)
        PyMem_Free(b_det)
        PyMem_Free(b_time)
        PyMem_Free(burst_sizes)
        raise ValueError('Too many photon streams in data for H2MM model')
    cdef h2mm_model out = model_from_ptr(out_model)
    if out.model.conv == 1:
        print(f'The model converged after {out.model.niter} iterations')
    elif out.model.conv == 2:
        print('Optimization reached maximum number of iterations')
    elif out.model.conv == 3:
        print('Optimization reached maxiumum time')
    else:
        print(f'An error occured on iteration {out.model.niter}, returning previous model')
    PyMem_Free(b_struct)
    PyMem_Free(b_det);
    PyMem_Free(b_time);
    PyMem_Free(burst_sizes)
    return out

def viterbi_path(h2mm_model h_mod, list burst_colors, list burst_times, num_cores = os.cpu_count()//2):
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
    Optional Keyword Parameters
    ---------------------------
    num_cores = os.cpu_count()//2 : int
        the number of C threads (which ignore the gil, thus functioning more
        like python processes) used to calculate the viterbi path. The default
        is to take half of what python reports as the cpu count, because most
        cpus have multithreading enabled, so the os.cpu_count() will return
        twice the number of physical cores. Consider setting this parameter
        manually if either you want to optimize the speed of the computation,
        or if you want to reduce the number of cores used so your computer has
        more cpu time to spend on other tasks
    
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
    if len(burst_colors) != len(burst_times):
        raise ValueError("Mismatch in burst_colors and burst_times lengths, burst_colors and burst_times must be of the same length")
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
    cdef n_core = <unsigned long> num_cores if num_cores > 0 else 1
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
    cdef int e_val = viterbi(num_burst,burst_sizes,b_time,b_det,&h_mod.model,path_ret,n_core)
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


def viterbi_sort(h2mm_model hmod, list indexes, list times, num_cores = os.cpu_count()//2):
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
    Optional Keyword Parameters
    ---------------------------
    num_cores = os.cpu_count()//2 : int
        the number of C threads (which ignore the gil, thus functioning more
        like python processes) used to calculate the viterbi path. The default
        is to take half of what python reports as the cpu count, because most
        cpus have multithreading enabled, so the os.cpu_count() will return
        twice the number of physical cores. Consider setting this parameter
        manually if either you want to optimize the speed of the computation,
        or if you want to reduce the number of cores used so your computer has
        more cpu time to spend on other tasks

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
    cdef Py_ssize_t i, b, e, st
    cdef list paths, scale
    cdef np.ndarray[double,ndim=1] ll
    cdef double icl
    paths, scale, ll, icl = viterbi_path(hmod,indexes,times,num_cores=num_cores)
    
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
