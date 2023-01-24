# -*- coding: utf-8 -*-
#cython: language_level=3
# Created 20 Feb 2021
# Modified: 25 Oct 2022
#Author: Paul David Harris
"""
Module for analyzing data with photon by photon hidden Markov moddeling
"""

import os
import numpy as np
cimport numpy as np
from IPython.display import DisplayHandle, Pretty
import IPython
import warnings
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport PyObject

# cdef extern from "rho_calc.c":
#      pass
# cdef extern from "fwd_back_photonbyphoton_par.c":
#      pass
# cdef extern from "C_H2MM.c":
#      pass

cdef extern from "math.h":
    bint isnan(double x)

cdef extern from "C_H2MM.h":
    ctypedef struct lm:
        unsigned long max_iter
        unsigned long num_cores
        double max_time
        double min_conv
    ctypedef struct h2mm_mod:
        unsigned long nstate
        unsigned long ndet
        unsigned long nphot
        unsigned long niter
        unsigned long conv
        double *prior
        double *trans
        double *obs
        double loglik
    ctypedef struct h2mm_minmax:
        h2mm_mod *mins
        h2mm_mod *maxs
    ctypedef struct ph_path:
        unsigned long nphot
        unsigned long nstate
        double loglik
        unsigned long *path
        double *scale
    int duplicate_toempty_model(h2mm_mod *source, h2mm_mod *dest)
    int duplicate_toempty_models(unsigned long num_model, h2mm_mod **source, h2mm_mod **dest)
    int copy_model(h2mm_mod *source, h2mm_mod *dest)
    int copy_model_vals(h2mm_mod *source, h2mm_mod *dest)
    h2mm_mod* allocate_models(const unsigned long n, const unsigned long nstate, const unsigned long ndet, const unsigned long nphot)
    ph_path* allocate_paths(unsigned long num_burst, unsigned long* len_burst, unsigned long nstate)
    int free_paths(unsigned long num_burst, ph_path* paths)
    int free_model(h2mm_mod *model)
    int free_models(const unsigned long n, h2mm_mod *model)
    void h2mm_normalize(h2mm_mod *model_params)
    int h2mm_optimize(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call) nogil
    int h2mm_optimize_array(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call) nogil
    int h2mm_optimize_gamma(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call) nogil
    int h2mm_optimize_gamma_array(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*), void *model_limits, void (*print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*),void *print_call) nogil
    unsigned long viterbi(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, h2mm_mod *model, ph_path *path_array, unsigned long num_cores) nogil
    void baseprint(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func)
    int calc_multi(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, unsigned long num_models, h2mm_mod *models, lm *limits) nogil
    int calc_multi_gamma(unsigned long num_burst, unsigned long *burst_sizes, unsigned long **burst_deltas, unsigned long **burst_det, unsigned long num_models, h2mm_mod *models, double ****gamma, lm *limits) nogil
    int h2mm_check_converged(h2mm_mod * new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limits)
    int limit_check_only(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int limit_revert(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int limit_revert_old(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int limit_minmax(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int statepath(h2mm_mod* model, unsigned long lent, unsigned long* path, unsigned int seed)
    int sparsestatepath(h2mm_mod* model, unsigned long lent, unsigned long long* times, unsigned long* path, unsigned int seed)
    int phpathgen(h2mm_mod* model, unsigned long lent, unsigned long* path, unsigned long* traj, unsigned int seed)
    int pathloglik(unsigned long num_burst, unsigned long *len_burst, unsigned long **deltas, unsigned long ** dets, unsigned long **states, h2mm_mod *model, double *loglik, unsigned long num_cores) nogil

ctypedef struct bound_struct:
    void *func
    void *limits

ctypedef struct print_struct:
    void *func
    void *args

ctypedef struct print_args_struct:
    void *txt
    void *handle
    unsigned long disp_freq
    unsigned long keep
    unsigned long max_iter

#: Version string
__version__ = '1.0.2'

# copy data from a numpy array into an unsigned long array, and return the pointer
cdef unsigned long* np_copy_ul(np.ndarray arr):
    cdef unsigned long* out = <unsigned long*> PyMem_Malloc(arr.shape[0]*sizeof(unsigned long))
    for i in range(arr.shape[0]):
        out[i] = <unsigned long> arr[i]
    return out

# copy data from a numpy array into an unsigned long long array, and return the pointer
cdef unsigned long long* np_copy_ull(np.ndarray arr):
    cdef unsigned long long* out = <unsigned long long*> PyMem_Malloc(arr.shape[0]*sizeof(unsigned long long))
    for i in range(arr.shape[0]):
        out[i] = <unsigned long long> arr[i]
    return out

# make the times differences array for h2mm from times array
cdef unsigned long* time_diff(np.ndarray time):
    cdef unsigned long* delta = <unsigned long*> PyMem_Malloc(time.shape[0] * sizeof(unsigned long))
    cdef unsigned long i
    cdef unsigned long df
    delta[0] = 0
    for i in range(1,time.shape[0]):
        if time[i-1] > time[i]:
            PyMem_Free(delta)
            return NULL
        df = <unsigned long> (time[i] - time[i-1])
        if df == 0:
            delta[i] = 0
        else:
            delta[i] = df - 1
    return delta

cdef burst_free(unsigned long n, unsigned long** dets, unsigned long** deltas, unsigned long* sizes):
    cdef unsigned long i
    for i in range(n):
        if dets[i] is not NULL: PyMem_Free(dets[i])
        if deltas[i] is not NULL: PyMem_Free(deltas[i])
    if dets is not NULL: PyMem_Free(dets)
    if deltas is not NULL: PyMem_Free(deltas)
    if sizes is not NULL: PyMem_Free(sizes)

# make burst differences and burst times c arrays from lists
cdef unsigned long* burst_convert(unsigned long num_burst, color, times, unsigned long** b_det, unsigned long** b_delta):
    cdef unsigned long i = 0
    cdef unsigned long* b_size = <unsigned long*> PyMem_Malloc(num_burst * sizeof(unsigned long))
    for i, (col, tim) in enumerate(zip(enum_arrays(color), enum_arrays(times))):
        b_size[i] = col.shape[0]
        b_det[i] = np_copy_ul(col)
        b_delta[i] = time_diff(tim)
        if b_delta[i] is NULL:
            burst_free(i+1, b_det, b_delta, b_size)
            return NULL
        i += 1
    return b_size

# copy array into numpy
cdef np.ndarray copy_to_np_ul(unsigned long lenp, unsigned long* arr):
    cdef np.ndarray out = np.empty(lenp, dtype=np.uint32)
    cdef i
    for i in range(lenp):
        out[i] = arr[i]
    PyMem_Free(arr)
    return out

cdef np.ndarray copy_to_np_d(unsigned long lenp, double* arr):
    cdef np.ndarray out = np.empty(lenp, dtype=np.float64)
    cdef i
    for i in range(lenp):
        out[i] = arr[i]
    PyMem_Free(arr)
    return out

cdef enum_arrays(model_array):
    if isinstance(model_array, (list, tuple)):
        return iter(model_array)
    elif isinstance(model_array, np.ndarray):
        return iter(model_array.ravel())
    else:
        return iter((model_array, ))

def verify_input(indexes, times, ndet):
    """
    Check if indexes and times contain any issues making them incompatible for 
    processing with H2MM

    Parameters
    ----------
    indexes : list of numpy arrays
        The indeces of photons in bursts, 1 array per burst
    times : list of numpy arrays
        The tims of photons in bursts, 1 array per burst
    ndet : int
        Number of photon streams in the data

    Returns
    -------
    int or Exception
        The maximum index in the data, unless an issue exists, in which case an
        appropriate exception is returned

    """
    cdef type indexes_t = type(indexes)
    if indexes_t not in (list, tuple, np.ndarray):
        return TypeError(f"Unusable type: indexes and times must be of list, tuple or numpy array of numpy arrays, got {indexes_t}")
    elif indexes_t != type(times):
        return TypeError(f"Mismatched types: indexes and times must be same type")
    elif array_size(indexes) != array_size(times):
        return ValueError(f"Mismatched bursts: indexes and times must be same size, got {array_size(indexes)} and {array_size(times)}")
    cdef str burst_errors = str()
    cdef unsigned long max_ind = 0
    cdef unsigned long max_temp
    cdef unsigned long n = indexes.size if indexes_t == np.ndarray else len(indexes)
    for i, (index, time) in enumerate(zip(enum_arrays(indexes), enum_arrays(times))):
        if isinstance(index, np.ndarray) and isinstance(time, np.ndarray):
            if index.ndim != 1 or time.ndim != 1:
                burst_errors += f"{i}: TypeError: bursts must be 1D numpy arrays, got {type(indexes[i])} and {type(times[i])}\n"
            elif index.shape[0] != time.shape[0]:
                burst_errors += f"{i}: Mismatched burst size: indexes has {indexes[i].shape[0]} photons while times has {times[i].shape[0]} photons\n"
            elif index.shape[0] < 3:
                burst_errors += f"{i}: Insufficient photons: bursts must be at least 3 photons\n"
            elif not np.issubdtype(index.dtype, np.integer):
                burst_errors += f"{i}: TypeError: indexes must integer arrays, got {indexes[i].dtype}\n"
            elif np.any(index < 0):
                burst_errors += f"{i}: ValueError: indexes contains negative values, indices must be non-negative\n"
            elif np.any(np.diff(time) < 0):
                burst_errors += f"{i}: ValueError: out of order photon\n"
            else:
                max_temp = <unsigned long> np.max(index)
                if max_temp >= ndet:
                    burst_errors += f"{i}: ValueError: index exceeds model definition\n"
                if max_temp > max_ind:
                    max_ind = max_temp
        else:
            burst_errors += f"{i}: TypeError: indexes and times must be 1D numpy arrays, got {type(index)} and {type(time)}\n"
    if burst_errors != str():
        return ValueError("Incorect type for bursts:\n" + burst_errors)
    else:
        return max_ind
    

cdef tuple array_size(inp):
    if isinstance(inp, (list, tuple)):
        return (len(inp), )
    elif isinstance(inp, np.ndarray):
        return inp.shape
    else:
        return (1, )
    

cdef class opt_lim_const:
    """
    Special class for setting default optimization min/max values.
    Provides access to and type protected setting of these values, 
    either as attributes or keys.
    Attributes have same names as key-word arguments of functions.
    
    The optimization_limits module variable should be of this class
    
    """
    cdef:
        unsigned long _max_iter
        double _max_time
        double _converged_min
        unsigned long _num_cores
    def __cinit__(self, max_iter=3600, max_time=np.inf, converged_min=1e-14, num_cores=os.cpu_count()//2):
        assert max_iter > 0 and np.issubdtype(type(max_iter), np.integer), ValueError("max_iter must be integer greater than 0")
        assert max_time > 0 and np.issubdtype(type(max_time), np.floating), ValueError("max_time must be float and greater than 0")
        assert converged_min > 0 and np.issubdtype(type(converged_min), np.floating), ValueError("converged_min must be float greater than 0")
        assert num_cores > 0 and np.issubdtype(type(num_cores), np.integer), ValueError("num_cores must be integer greater than 0")
        self._max_iter = <unsigned long> max_iter
        self._max_time = <double> max_time
        self._converged_min = <double> converged_min
        self._num_cores = <unsigned long> num_cores

    @property
    def max_iter(self):
        """Maximum number of iterations before ending optimization"""
        return self._max_iter
    @max_iter.setter
    def max_iter(self, max_iter):
        assert max_iter > 0 and np.issubdtype(type(max_iter), np.integer), ValueError("max_iter must be integer greater than 0")
        self._max_iter = <unsigned long> max_iter

    @property
    def max_time(self):
        """Maximum time of optimization.
        
        .. note::
            
            This uses the inaccurate C-clock, actual duration of optimization
            will be variable and generally less thatn the input value
            
        """
        return self._max_time
    @max_time.setter
    def max_time(self, max_time):
        assert max_time > 0 and np.issubdtype(type(max_time), np.floating), ValueError("max_time must be float and greater than 0")
        self._max_time = <double> max_time

    @property
    def converged_min(self):
        """Minimum difference between loglik of succesive iterations for a model to be considered converged"""
        return self._converged_min
    @converged_min.setter
    def converged_min(self, converged_min):
        assert converged_min > 0 and np.issubdtype(type(converged_min), np.floating), ValueError("converged_min must be float and greater than 0")
        self._converged_min = <double> converged_min

    @property
    def num_cores(self):
        """Number of cores to use in optimization
        
        .. note::
            
            This more specifically sets the number of threads (pthreads in Linux/MacOS)
            used in optimization
        """
        return self._num_cores
    @num_cores.setter
    def num_cores(self, num_cores):
        assert num_cores > 0 and np.issubdtype(type(num_cores), np.integer), ValueError("num_cores must be int and greater than 0")
        self._num_cores = <unsigned long> num_cores

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __repr__(self):
        return f'Optimization limits:: num_cores: {self._num_cores}, max_iter: {self._max_iter}, converged_min: {self._converged_min}, max_time: {self._max_time}'
    
    def _get_max_iter(self, max_iter):
        """
        Process the max_iter keyword argument value.

        Parameters
        ----------
        max_iter : int or None
            Maximum number of iterations during optimization, the value passed to
            the max_iter keyword argument.

        Returns
        -------
        max_iter : int
            Maximum number of iterations, if keyword argument is None, the value
            stored in the default argument.

        """
        if max_iter is not None and np.issubdtype(type(max_iter),int):
            assert max_iter > 0, ValueError("max_iter must be greater than 0")
            m_iter = max_iter
        elif max_iter is None:
            m_iter = self._max_iter
        else:
            raise ValueError(f"max_iter must be int, got {type(max_iter)}")
        return m_iter
    
    def _get_max_time(self, max_time):
        """
        Process the max_iter keyword argument value.

        Parameters
        ----------
        max_time : float or None
            Maximum duration of optimization, the value passed to the max_time
            keyword argument

        Returns
        -------
        max_iter : float
            Maximum time of optimization, if keyword argument is None, the
            value stored in the default argument.

        """
        if np.issubdtype(type(max_time), np.floating):
            assert max_time > 0, ValueError("converged_min must be greater than 0")
            m_time = max_time
        elif max_time is None:
            m_time = self._max_time
        else:
            raise ValueError(f"max_time must be float, got {type(max_time)}")
        return m_time
    
    def _get_converged_min(self, converged_min):
        """
        Process the converged_min keyword argument value.

        Parameters
        ----------
        converged_min : float or None
            Minimum difference between successive iteration for optimization to be
            considered converged, the value passed to
            the max_iter keyword argument.

        Returns
        -------
        converged_min : float
            Minimum difference between successive iteration for optimization to be
            considered converged, if keyword argument is None, the value stored 
            in the default argument.

        """
        if np.issubdtype(type(converged_min), np.floating):
            assert converged_min > 0, ValueError("converged_min must be greater than 0")
            c_min = converged_min
        elif converged_min is None:
            c_min = self._converged_min
        else:
            raise ValueError(f"converged_min must be float, got {type(converged_min)}")
        return c_min
    
    def _get_num_cores(self, num_cores):
        """
        Process the num_cores keyword argument value.

        Parameters
        ----------
        max_iter : int or None
            Maximum number of iterations during optimization, the value passed to
            the max_iter keyword argument.

        Returns
        -------
        max_iter : int
            Maximum number of iterations, if keyword argument is None, the value
            stored in the default argument.

        """
        if type(num_cores) == str:
            if 'multi' in num_cores:
                num_cores = os.cpu_count()//2 
            elif 'single' in num_cores:
                num_cores = os.cpu_count()
            else:
                raise ValueError(f"Non-int options for num_cores are 'single' or 'multi', cannot process {num_cores}")
        elif num_cores is not None and np.issubdtype(type(num_cores), int):
            assert num_cores > 0, ValueError("num_cores must be greater than 0")
            n_core = <unsigned long> num_cores
        elif num_cores is None:
            n_core = self._num_cores
        else:
            raise ValueError(f"Non-int options for num_cores are 'single' or 'multi', cannot process {num_cores}")
        return n_core

#: Like rcParams, a place to set defaults for optimization limits
optimization_limits = opt_lim_const(max_iter=3600, max_time=np.inf, converged_min=1e-14, num_cores=os.cpu_count()//2)
    
cdef class h2mm_model:
    """
    The base class for storing objects representing the H2MM model, stores
    the relevant matrices, loglikelihood and information about status of
    optimization.
    
    Parameters
    ----------
    prior : numpy.ndarray
        The initial probability matrix, 1D, size nstate
    trans : numpy.ndarray
        The transition probability matrix, 2D, shape [nstate, nstate]
    obs : numpy.ndarray
        The emission probability matrix, 2D, shape [nstate, ndet]
    Optional Parameters
    -------------------
    .. note::
        
        These are generally only used when trying to re-create a model from, for
        instance a previous optimization of a closed notebook.
    
    loglik : float (optional)
        The loglikelihood, always set with nphot otherwise BIC will not be calculated
        correctly. Usually should be set by evaluation against data. Default is -inf
    niter : int
        The number of iterations, used for recreating previously optimized model.
        Usually should be set by evaluation against data. Default is 0
    nphot : int (optional)
        The number of photons in data set used for optimization, should only be set
        when recreating model, Usually should be set by evaluation against data.
        Default is 0
    is_conv : bool (optional)
        Whether or not the model has met a convergence criterion, should only be 
        set when recreating model, otherwise allow evaluation functions to handle
        this operation. Default is False
    """
    cdef:
        h2mm_mod *model
    
    def __cinit__(self, *args, **kwargs):
        self.model = NULL
    
    def __init__(self, prior, trans, obs, loglik=-np.inf, niter = 0, nphot = 0, is_conv=False):
        # if statements check to confirm first the correct dimensinality of input matrices, then that their shapes match
        cdef unsigned long i, j
        if prior.ndim != 1:
            raise ValueError("Prior matrix must have ndim=1, too many dimensions in prior")
        if trans.ndim != 2:
            raise ValueError("Trans matrix must have ndim=2, wrong dimensionallity of trans matrix")
        if obs.ndim != 2:
            raise ValueError("Obs matrix must have ndim=2, wrong dimensionallity of obs matrix")
        if not (prior.shape[0] == trans.shape[0] == obs.shape[0]):
            raise ValueError("Dim 0 of one of the matrices does not match the others, these represent the number of states, so input matrices do not represent a single model")
        if trans.shape[0] != trans.shape[1]:
            raise ValueError("Trans matrix is not square, and connot be used for a model")
        if niter < 0:
            raise ValueError("niter must be positive")
        if nphot< 0: 
            raise ValueError("nphot must be positive")
        if type(is_conv) != bool:
            raise TypeError("is_conv must be boolean")
        if loglik > 0.0:
            raise ValueError("loglik must be negative")
        elif loglik != -np.inf:
            if nphot == 0:
                raise ValueError("Must specify number of photons if loglik is specified")
        else: 
            if is_conv == True:
                raise ValueError("Converged Model cannot have -inf loglik")
        if np.any(prior < 0.0):
            raise ValueError("prior array can have no negative values")
        if np.any(trans < 0.0):
            raise ValueError("trans array can have no negative values")
        if np.any(obs < 0.0):
            raise ValueError("obs array can have no negative values")
        # coerce the matricies into c type double
        prior = prior.astype('double')
        trans = trans.astype('double')
        obs = obs.astype('double')
        # allocate and copy information over to h2mm_mod pointer
        self.model = allocate_models(1, obs.shape[0], obs.shape[1], <unsigned long> nphot)
        if loglik == -np.inf:
            self.model.conv = 0
            self.model.loglik = <double> loglik
        else:
            self.model.conv = 3 if is_conv else 2
            self.model.loglik = <double> loglik
        # allocate and copy array values
        for i in range(self.model.nstate):
            for j in range(self.model.nstate):
                self.model.trans[self.model.nstate*i + j] = trans[i,j]
        for i in range(self.model.nstate):
            self.model.prior[i] = prior[i]
        for i in range(self.model.ndet):
            for j in range(self.model.nstate):
                self.model.obs[self.model.nstate * i + j] = obs[j,i]
        self.normalize()
        
    def __dealloc__(self):
        if self.model is not NULL:
            free_models(1, self.model)
        
    @staticmethod
    cdef h2mm_model from_ptr(h2mm_mod *model):
        cdef h2mm_model new = h2mm_model.__new__(h2mm_model)
        new.model = model
        return new
        
    def __repr__(self):
        cdef unsigned long i, j
        msg = f"nstate: {self.model.nstate}, ndet: {self.model.ndet}, nphot: {self.model.nphot}, niter: {self.model.niter}, loglik: {self.model.loglik} converged state: {self.model.conv}\n"
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
            msg = "Initial model, "
            ll = ', loglik unknown'
        elif self.model.conv == 1:
            if self.model.loglik == 0.0:
                msg = f"Non-calculated optimization model, {self.model.niter} iterations, "
                ll = f'nphot={self.model.nphot}'
            else:
                msg = f"Mid-optimization, {self.model.niter} iterations, "
                ll = f'nphot={self.model.nphot}, loglik={self.model.loglik}'
        elif self.model.conv == 2:
            msg = "Non-optimzed model, "
            ll = f'nphot={self.model.nphot}, loglik={self.model.loglik}'
        elif self.model.conv == 3:
            msg = f"Converged model, {self.model.niter} iterations, "
            ll = f'nphot={self.model.nphot}, loglik={self.model.loglik}'
        elif self.model.conv == 4:
            msg = f"Max iterations {self.model.niter} iterations, "
            ll = f'nphot={self.model.nphot}, loglik={self.model.loglik}'
        elif self.model.conv == 5:
            msg = f"Max time {self.model.niter} iterations, "
            ll = f'nphot={self.model.nphot}, loglik={self.model.loglik}'
        elif self.model.conv == 6:
            msg = "Optimization stopped after error, "
            ll = f'nphot={self.model.nphot}, loglik={self.model.loglik}'
        return msg + f'States={self.model.nstate}, Streams={self.model.ndet}, ' + ll
        
    # a number of property defs so that the values are accesible from python
    @property
    def prior(self):
        """Prior probability matrix, 1D, size nstate"""
        return np.asarray(<double[:self.model.nstate]>self.model.prior).copy()
    
    @prior.setter
    def prior(self,prior):
        if prior.ndim != 1:
            raise ValueError("Prior must be 1D numpy floating point array")
        if prior.shape[0] != self.model.nstate:
            raise ValueError("Cannot change the number of states")
        if (prior.sum() - 1.0) > 1e-14:
            warnings.warn("Input array not stochastic, new array will be normalized")
        prior = prior.astype('double')
        self.model.loglik = -np.inf
        self.model.conv = 0
        self.model.niter = 0
        for i in range(self.model.nstate):
            self.model.prior[i] = prior[i]
        self.normalize()
    
    @property
    def trans(self):
        """Transition probability matrix, square, dimenstions nstate x nstate"""
        return np.asarray(<double[:self.model.nstate,:self.model.nstate]>self.model.trans).copy()
    
    @trans.setter
    def trans(self,trans):
        if trans.ndim != 2:
            raise ValueError("Trans must be a 2D numpy floating point array")
        if trans.shape[0] != trans.shape[1]:
            raise ValueError("Trans must be a square array")
        if trans.shape[0] != self.model.nstate:
            raise ValueError("Cannot change the number of states in a model")
        if np.any((trans.sum(axis=1) - 1.0) > 1e-14):
            warnings.warn("Input matrix not row stochastic, new matrix will be normalized")
        trans = trans.astype('double')
        self.model.loglik = -np.inf
        self.model.conv = 0
        self.model.niter = 0
        for i in range(self.model.nstate):
            for j in range(self.model.nstate):
                self.model.trans[self.model.nstate*i + j] = trans[i,j]
        self.normalize()
    
    @property
    def obs(self):
        """Emission probability matrix, 2D shape nstate x ndet"""
        return np.asarray(<double[:self.model.ndet,:self.model.nstate]>self.model.obs).T.copy()
    
    @obs.setter
    def obs(self,obs):
        if obs.ndim != 2:
            raise ValueError("Obs must be a 2D numpy floating point array")
        if obs.shape[0] != self.model.nstate:
            raise ValueError("Cannot change the number of states in a model")
        if obs.shape[1] != self.model.ndet: 
            raise ValueError("Cannot change the number of streams in the model")
        if np.any((obs.sum(axis=1) -  1.0) > 1e-14):
            warnings.warn("Input matrix not row stochastic, new matrix will be normalized")
        obs = obs.astype('double')
        self.model.loglik = -np.inf
        self.model.conv = 0
        self.model.niter = 0
        for i in range(self.model.ndet):
            for j in range(self.model.nstate):
                self.model.obs[self.model.nstate * i + j] = obs[j,i]
        self.normalize()
    
    @property
    def loglik(self):
        """Loglikelihood of model"""
        if self.model.nphot == 0:
            warnings.warn("loglik not calculated against data, will be meaningless -inf")
        return self.model.loglik
    
    @property
    def k(self):
        """Number of free parameters in model"""
        return self.model.nstate**2 + ((self.model.ndet - 1)*self.model.nstate) - 1
    
    @property
    def bic(self):
        """Bayes Information Criterion of model"""
        if self.model.nphot == 0:
            raise Exception("Init model, no data linked, therefore BIC not known.")
        if self.model.conv == 0:
            warnings.warn("loglik not calculated against data, BIC meaningless -inf")
        if self.model.conv == 1 and self.model.loglik == 0.0:
            return 0.0
        else:
            return -2*self.model.loglik + np.log(self.model.nphot)*(self.model.nstate**2 + ((self.model.ndet - 1)*self.model.nstate) - 1)
    
    @property
    def nstate(self):
        """Number of states in model"""
        return self.model.nstate
    
    @property
    def ndet(self):
        """Nubmer of detectors/photon streams in model"""
        return self.model.ndet
    
    @property
    def nphot(self):
        """Number of photons in data set used to optimize model"""
        return self.model.nphot
    
    @property
    def is_conv(self):
        """Whether or not the optimization reached convergence rather than exceeding the maximum iteration/time of optimization"""
        if self.model.conv == 3:
            return True
        else:
            return False
    
    @property
    def is_opt(self):
        """Whether or not the model has undergone optimization, as opposed to evaluation or being an initial model"""
        if self.model.conv >= 3:
            return True
        else:
            return False
    
    @property
    def is_calc(self):
        """If model has been optimized/eveluated, vs being an initial model"""
        if np.isnan(self.model.loglik) or self.model.loglik == -np.inf or self.model.loglik == 0 or self.model.nphot == 0:
            return False
        else:
            return True
    
    @property
    def conv_code(self):
        """The convergence code of model, an int"""
        if self.model.conv == 1 and self.model.loglik == 0:
            return -1
        else:
            return self.model.conv
    
    @property
    def conv_str(self):
        """String description of how model optimization/calculation ended"""
        if self.model.conv == 0:
            return 'Model unoptimized'
        elif self.model.conv == 1:
            if self.model.loglik == 0.0:
                return f'Non-calculated model in optimization, {self.model.niter} iterations'
            else:
                return f'Mid-optimization model, {self.model.niter} iterations'
        elif self.model.conv == 2:
            return 'Loglik of model calculated without optimization'
        elif self.model.conv == 3:
            return f'Model converged after {self.model.niter} iterations'
        elif self.model.conv == 4:
            return f'Maxiumum of {self.model.niter} iterations reached'
        elif self.model.conv == 5:
            return f'After {self.model.niter} iterations the optimization reached the time limit'
        elif self.model.conv == 6:
            return f'Optimization terminated because of reaching floating point NAN on iteration {self.model.niter}, returned the last viable model'
        elif self.model.conv == 7:
            return f'Model after optimal model found {self.model.niter}'
    
    @property
    def niter(self):
        """Number of iterations optimization took"""
        return self.model.niter
    
    @niter.setter
    def niter(self,niter):
        if niter <= 0:
            raise ValueError("Cannot have negative iterations")
        self.model.niter = niter
    
    def set_converged(self,converged):
        if not isinstance(converged,bool):
            raise ValueError("Input must be True or False")
        if self.model.nphot == 0 or self.model.loglik == 0 or self.model.loglik == np.inf or np.isnan(self.model.loglik):
            raise Exception("Model uninitialized with data, cannot set converged")
        if converged:
            self.model.conv = 3
        elif self.model.conv == 3:
            self.model.conv = 1
    
    def normalize(self):
        """For internal use, ensures all model arrays are row stochastic"""
        h2mm_normalize(self.model)
    
    def optimize(self, indexes, times, max_iter=None, print_func='iter', 
              print_args = None, bounds=None, bounds_func=None,
              max_time=None, converged_min=None, num_cores=None, 
              reset_niter=False, gamma = False, opt_array=False, inplace=True):
        """
        Optimize the H2MM model for the given set of data.
        
        .. note::
            
            This method calls the :func:`EM_H2MM_C` function.
    
        Parameters
        ----------
        indexes : list of NUMPY 1D int arrays
            A list of the arrival indexes for each photon in each burst.
            Each element of the list (a numpy array) corresponds to a burst, and
            each element of the array is a singular photon.
            The indexes list must maintain  1-to-1 correspondence to the times list
        times : list of NUMPY 1D int arrays
            A list of the arrival times for each photon in each burst
            Each element of the list (a numpy array) corresponds to a burst, and
            each element of the array is a singular photon.
            The times list must maintain  1-to-1 correspondence to the indexes list
        
        print_func : None, str or callable, optional
            Specifies how the results of each iteration will be displayed, several 
            strings specify built-in functions. Default is 'iter'
            Acceptable inputs: str, Callable or None
                
                **None**\:
                
                    causes no printout anywhere of the results of each iteration
                    
                Str: 'console', 'all', 'diff', 'comp', 'iter'
                
                    **'console'**\: 
                        
                        the results will be printed to the terminal/console 
                        window this is useful to still be able to see the results, 
                        but not clutter up the output of your Jupyter Notebook, this 
                        is the default
                    
                    **'all'**\: 
                        
                        prints out the full h2mm_model just evaluated, this is very
                        verbose, and not generally recommended unless you really want
                        to know every step of the optimization
                    
                    **'diff'**\: 
                        
                        prints the loglik of the iteration, and the difference 
                        between the current and old loglik, this is the same format
                        that the 'console' option used, the difference is whether the
                        print function used is the C printf, or Cython print, which 
                        changes the destination from the console to the Jupyter notebook
                        
                    **'diff_time'**\: 
                        
                        same as 'diff' but adds the time of the last iteration
                        and the total optimization time
                        
                    **'comp'**\: 
                        
                        prints out the current and old loglik
                    
                    **'comp_time'**\: 
                        
                        same as 'comp'  but with times, like in 'diff_time'
                    
                    **'iter'**\: 
                        
                        prints out only the current iteration
                
                Callable: user defined function
                    A python function for printing out a custom output, the function must
                    accept the input of 
                    (int, :class:`h2mm_model`, :class:`h2mm_model`, :class:`h2mm_model`,float, float)
                    as the function will be handed (niter, new, current, old, t_iter, t_total)
                    from a special Cython wrapper. t_iter and t_total are the times
                    of the iteration and total time respectively, in seconds, based
                    on the C level clock function, which is notably inaccurate,
                    often reporting larger than actual values.
        
        print_args : 2-tuple/list (int, bool) or None, optional
            Arguments to further customize the printing options. The format is
            (int bool) where int is how many iterations before updating the display
            and the bool is True if the printout will concatenate, and False if the
            display will be kept to one line, The default is None, which is changed
            into (1, False). If only an int or a bool is specified, then the default
            value for the other will be used. If a custom printing function is given
            then this argument will be passed to the function as \*args.
            Default is None
        bounds_func : str, callable or None, optional
            function to be evaluated after every iteration of the H2MM algorithm
            its primary function is to place bounds on :class:`h2mm_model`
            
            .. note:: 
                
                bounding the :class:`h2mm_model` causes the guarantee of improvement on each
                iteration until convergence to no longer apply, therefore the results are
                no longer guaranteed to be the optimal model within bounds.
            
            Default is None
            Acceptable inputs\:
                
                C level limits: 'minmax' 'revert' 'revert_old'
                    prevents the model from having values outside of those defined
                    by :class:`h2mm_limits` class given to bounds, if an iteration produces
                    a new model for loglik calculation in the next iteration with 
                    values that are out of those bounds, the 3 function differ in
                    how they correct the model when a new model has a value out of
                    bounds, they are as follows:
                    
                    **'minmax'**\: 
                        
                        sets the out of bounds value to the min or max value
                        closer to the original value
                    
                    **'revert'**\: 
                        
                        sets the out of bounds value to the value from the last
                        model for which the loglik was calculated
                    
                    **'revert_old'**\: 
                        
                        similar to revert, but instead reverts to the value
                        from the model one before the last calculated model
                
                Callable: python function that takes 4 inputs and returns :class:`h2mm_model` object
                    .. warning:: 
                    
                        The user takes full responsibility for errors in the results.
                    
                    must be a python function that takes 4 arguments, the first 3 are
                    the :class:`h2mm_model` objects, in this order: new, current, old, 
                    the fourth is the argument supplied to the bounds keyword argument,
                    the function must return a single :class:`h2mm_limits` object, the prior, 
                    trans and obs fields of which will be optimized next, other values 
                    are ignored.
        bounds : h2mm_limits, str, 4th input to callable, or None, optional
            The argument to be passed to the bounds_func function. If bounds_func is
            None, then bounds will be ignored. If bounds_func is 'minmax', 'revert'
            or 'revert_old' (calling C-level bounding functions), then bounds must be
            a :class:`h2mm_limits` object, if bounds_func is a callable, bounds must be an
            acceptable input as the fourth argument to the function passed to bounds_func
            Default is None
        max_iter : int or None, optional
            the maximum number of iterations to conduct before returning the current
            :class:`h2mm_model`, if None (default) use value from optimization_limits. 
            Default is None
        max_time : float or None, optional
            The maximum time (in seconds) before returning current model
            
            .. note:: 
                
                this uses the C clock, which has issues, often the time assessed by
                C, which is usually longer than the actual time. 
                
            If None (default) use value from optimization_limits. 
            Default is None
        converged_min : float or None, optional
            The difference between new and current :class:`h2mm_model` to consider the model
            converged, the default setting is close to floating point error, it is
            recommended to only increase the size. If None (default) use value from 
            optimization_limits. Default is None
        num_cores : int or None, optional
            the number of C threads (which ignore the gil, thus functioning more
            like python processes), to use when calculating iterations.
            
            .. note:: 
                
                optimization_limtis sets this to be `os.cpu_count() // 2`, as most machines
                are multithreaded. Because os.cpu_count() returns the number of threads,
                not necessarily the number of cpus. For most machines, being multithreaded,
                this actually returns twice the number of physical cores. Hence the default
                to set at `os.cpu_count() // 2`. If your machine is not multithreaded, it
                is best to set `optimization_limits.num_cores = os.cpu_count()`
                
            If None (default) use value from optimization_limits. 
            Default is None
        reset_niter : bool, optional
            Tells the algorithm whether or not to reset the iteration counter of the
            model, True means that even if the model was previously optimized, the 
            iteration counter will start at 0, so calling :func:`EM_H2MM_C` on a model that
            reached its maximum iteration number will proceed another max_iter 
            iterations. On the other hand, if set to False, then the new iterations
            will be added to the old. If set to False, you likely will have to increase
            max_iter, as the optimization will count the previous iterations towards
            the max_iter threshold.
            Default is False
        gamma : bool, optional
            Whether or not to return the gamma array, which gives the probabilities
            of each photon being in a given state.
            
            .. note::
                
                If opt_array is True, then only the gamma arrays for the ideal
                model are returned.
                
            Default is False
        opt_array : bool
            Defines how the result is returned, if False (default) then return only
            the ideal model, if True, then a numpy array is returned containing all
            models during the optimization.
            
            .. note::
                
                The last model in the array is the model after the convergence criterion
                have been met. Therefore check the :attr:`h2mm_model.conv_code`, if it
                is 7, then the second to last model is the ideal model. Still, usually
                the last model will be so close to ideal that it will not make any 
                significant difference.
                
            Default is False
        inplace : bool, optional
            Whether or not to store the optimized model in the current model object.
            
            .. note::
                
                When opt_array is True, then the ideal model is copied into the
                current model object, while the return value contains all models
                
            Default is True
        
        Returns
        -------
        out : h2mm_model
            The optimized :class:`h2mm_model`. will return after one of the following conditions
            are met: model has converged (according to converged_min, default 1e-14),
            maximum iterations reached, maximum time has passed, or an error has occurred
            (usually the result of a nan from a floating point precision error)
        gamma : list, tuple or np.ndarray (optional)
            If gamma = True, this variable is returned, which contains (as numpy arrays)
            the probabilities of each photon to be in each state, for each burst.
            The arrays are returned in the same order and type as the input indexes,
            individual data points organized as [photon, state] within each data
            sequence.
            
        """
        cdef h2mm_model out_model
        if self.model.conv == 4 and self.model.niter >= max_iter:
            max_iter = self.model.niter + max_iter
        out = EM_H2MM_C(self, indexes, times, max_iter=max_iter, 
                        print_func=print_func, print_args = print_args, 
                        bounds=bounds, bounds_func=bounds_func,  
                        max_time=max_time, converged_min=converged_min, 
                        num_cores=num_cores,reset_niter=reset_niter, 
                        gamma=gamma, opt_array=opt_array)
        if inplace:
            # separate the models from gamma
            if gamma:
                out_arr, gamma = out
            else:
                out_arr = out
            # find the ideal model
            if opt_array:
                for i in range(out_arr.size-1, -1, -1):
                    if out_arr[i] != 7:
                        out_model = out_arr[i]
                        break
            else:
                out_model = out_arr
            copy_model(out_model.model, self.model)
        return out
    
    def evaluate(self, indexes, times, gamma=False, num_cores=None, 
                 inplace=True):
        """
        Calculate the loglikelihood of the model given a set of data. The 
        loglikelihood is stored with the model.
        NOTE: this method calls the H2MM_arr function, and has essentially the 
        same syntax
    
        Parameters
        ----------
        indexes : list or tuple of NUMPY 1D int arrays
            A list of the arrival indexes for each photon in each burst.
            Each element of the list (a numpy array) corresponds to a burst, and
            each element of the array is a singular photon.
            The indexes list must maintain  1-to11 correspondence to the times list
        times : list or tuple of NUMPY 1D int arrays
            A list of the arrival times for each photon in each burst
            Each element of the list (a numpy array) corresponds to a burst, and
            each element of the array is a singular photon.
            The times list must maintain  1-to-1 correspondence to the indexes list
        gamma : bool (optional)
            Whether or not to return the gamma array. (The default is False)
        num_cores : int or None, optional
            the number of C threads (which ignore the gil, thus functioning more
            like python processes), to use when calculating iterations.
            
            .. note:: 
                
                optimization_limtis sets this to be `os.cpu_count() // 2`, as most machines
                are multithreaded. Because os.cpu_count() returns the number of threads,
                not necessarily the number of cpus. For most machines, being multithreaded,
                this actually returns twice the number of physical cores. Hence the default
                to set at `os.cpu_count() // 2`. If your machine is not multithreaded, it
                is best to set `optimization_limits.num_cores = os.cpu_count()`
                
            If None (default) use value from optimization_limits. 
            Default is None
        inplace: bool, optional
            Whether or not to store the evaluated model in the current model object.
            Default is True
        
        Returns
        -------
        out: h2mm_model
            The evaluated model
        gamma_arr : list, tuple or np.ndarray (optional)
            If gamma = True, this variable is returned, which contains (as numpy arrays)
            the probabilities of each photon to be in each state, for each burst and model.
            The arrays are returned in the same order and type as the indexes, 
            individual data points organized as [photon, state] within each data
            sequence.
        """
        cdef h2mm_model out_model
        out = H2MM_arr(self, indexes, times, gamma=gamma, num_cores=num_cores)
        if gamma:
            out_model, gamma = out
        else:
            out_model = out
        if inplace:
            copy_model(out_model.model, self.model)
        
        return out
    
    def copy(self):
        """ Make a duplicate model in new memory"""
        return model_copy_from_ptr(self.model)


cdef class _h2mm_lims:
    # hidden type for making a C-level h2mm_limits object
    cdef:
        h2mm_minmax limits
    def __cinit__(self, h2mm_model model, np.ndarray[double,ndim=1] min_prior, np.ndarray[double,ndim=1] max_prior, 
                  np.ndarray[double,ndim=2] min_trans, np.ndarray[double,ndim=2] max_trans, 
                  np.ndarray[double,ndim=2] min_obs, np.ndarray[double,ndim=2] max_obs):
        cdef unsigned long i, j
        cdef unsigned long nstate = model.model.nstate
        cdef unsigned long ndet = model.model.ndet
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
    
    Parameters
    ----------
    model : h2mm_model, optional
        An :class:`h2mm_model` to base the limits off of, if None. The main purpose
        is to allow the user to check that the limits are valid for the model.
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
        Number of states in the model to be optimized, if set to 0, and not
        specified elsewhere, number of states is flexible.
        The default is 0.
    ndet : int, optional
        Number of streams in the model to be optimized, if set to 0, and not
        specified elsewhere, number of states is flexible.
        The default is 0.
        
    """
    def __init__(self,model=None,min_prior=None,max_prior=None,min_trans=None,max_trans=None,min_obs=None,max_obs=None,nstate=0,ndet=0):
        none_kwargs = True
        if not isinstance(nstate,int) or not isinstance(ndet,int) or nstate < 0 or ndet < 0:
            raise TypeError("Cannot give negative or non-int values for nstate or ndet")
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
    
    def make_model(self,model,warning=True):
        """
        Method for checking the limits arrays generated from the input model
        
        Parameters
        ----------
        model : h2mm_model
            :class:`h2mm_model` for which the limits are to be specified.
            Also, values are checked and a warning is raised if the model has
            values that are out of the range of the limits
        warning: bool optional
            Whether or not to check of the initial model is within the bounds
            generated by make_model
            Default is True
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
        # chekcing model bounds
        if warning:
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
        return min_prior,max_prior,min_trans,max_trans,min_obs,max_obs
    
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


def factory_h2mm_model(nstate, ndet, bounds=None, trans_scale=1e-5, 
                  prior_dist='equal', trans_dist='equal',obs_dist='equal'):
    """
    Function for generating a model of an arbitrary number of states/streams

    Parameters
    ----------
    nstate : int
        Number of states in the return model
    ndet : int, optional
        Number of photons streams in the return model
    bounds : h2mm_limits or None, optional
        The limits expected to be used when optimizing the model. For limits
        are given, the model will always be within those limits, distribution of
        states within the model will depend on how 
        The default is None.
    trans_scale: float
        Only used when no :class:`h2mm_limits` object is supplied to bounds, sets the
        minimum (slowest) transition rates in the trans array
    prior_dist : 'even' or 'random', optional
        Define how to distribute prior states. options are 'equal' and 'random'
        The default is 'equal'.
    trans_dist : 'even' or 'random', optional
        Define how to distribute trans states. options are 'equal' and 'random'
        The default is 'equal'.
    obs_dist : 'even' or 'random', optional
        Define how to distribute obs states. options are 'equal' and 'random'
        The default is 'equal'.

    Returns
    -------
    model : h2MM_model
        An :class:`h2mm_model` object ready for optimization

    """
    # check all values are useful
    cdef np.ndarray[double,ndim=1] prior, min_prior, max_prior
    cdef np.ndarray[double,ndim=2] trans, min_trans, max_trans
    cdef np.ndarray[double,ndim=2] obs, min_obs, max_obs
    cdef Py_ssize_t i
    if prior_dist not in ['equal', 'random']:
        raise TypeError("prior_dist must be 'equal' or 'random'")
    if trans_dist not in ['equal', 'random']:
        raise TypeError("trans_dist must be 'equal' or 'random'")
    if obs_dist not in ['equal', 'random']:
        raise TypeError("obs_dist must be 'equal' or 'random'")
    if type(bounds) == h2mm_limits:
        if bounds.ndet != 0 and bounds.ndet != bounds.ndet:
            raise ValueError("bounds streams incompatible with model")
        if bounds.nstate != 0 and bounds.nstate != bounds.nstate:
            raise ValueError("bounds states incompatible with model")
        # setup min/max arrays from limits
        min_prior = np.ones(nstate) * bounds.min_prior
        max_prior = np.ones(nstate) * bounds.max_prior
        min_trans = np.ones((nstate,nstate)) * bounds.min_trans
        min_trans[min_trans==0.0] = trans_scale
        max_trans = np.ones((nstate,nstate)) * bounds.max_trans
        max_trans[max_trans==1.0] = trans_scale
        min_obs = np.ones((nstate,ndet)) * bounds.min_obs
        max_obs = np.ones((nstate,ndet)) * bounds.max_obs
        const_obs = type(bounds.min_obs) == float and type(bounds.max_obs) == float
    elif bounds is not None:
        raise TypeError("bounds must be None or h2mm_limits object")
    else:
        # setup min/max arrays with 0/1 as bounds
        min_prior = np.zeros(nstate)
        max_prior = np.ones(nstate)
        min_trans = np.ones((nstate,nstate)) * trans_scale
        max_trans = np.ones((nstate,nstate)) * trans_scale
        min_obs = np.zeros((nstate,ndet))
        max_obs = np.ones((nstate,ndet))
        const_obs = True
    # check min/max values are valid
    if min_prior.sum() > 1.0 or max_prior.sum() < 1.0:
        raise ValueError("prior bounds not valid for the number of states")
    if np.any(min_obs.sum(axis=1) > 1.0) or np.any(max_obs.sum(axis=1) < 1.0):
        raise ValueError("obs bounds not valid for this number of streams")
    if np.any(min_trans.sum(axis=1) > 1.0):
        raise ValueError("trans bounds not valid for this number of states")
    # setup actual arrays based on even/random selections
    if prior_dist == 'equal':
        prior_alpha, prior_beta, prior_sigma = max_prior.sum(), min_prior.sum(), max_prior - min_prior
        prior = min_prior + ((1-prior_beta)/(prior_alpha-prior_beta)*prior_sigma)
    else:
        prior = np.random.uniform(min_prior,max_prior)
        prior /= prior.sum()
        while np.any(prior < min_prior) or np.any(prior > max_prior):
            prior = np.random.uniform(min_prior,max_prior)
            prior /= prior.sum()
    if trans_dist == 'equal':
        min_trans[min_trans==0.0] = np.min(max_trans)*trans_scale
        trans = np.exp((np.log(min_trans)+np.log(max_trans))/2)
    else:
        trans = np.random.uniform(min_trans,max_trans)
    trans[np.eye(nstate)==1] = 0.0
    trans[np.eye(nstate)==1] = 1 - trans.sum(axis=1)
    if obs_dist == 'equal':
        if const_obs:
            obs = np.zeros((nstate,ndet))
            for i in range(ndet):
                if i % 2 == 0:
                    obs[:,i] = np.linspace(min_obs[0,i],max_obs[0,i],num=nstate+2)[1:nstate+1]
                else :
                    obs[:,i] = np.linspace(max_obs[0,i],min_obs[0,i],num=nstate+2)[1:nstate+1]
            obs = obs / obs.sum(axis=1).reshape(nstate,1)
        else:
            obs_alpha, obs_beta, obs_sigma = max_obs.sum(axis=1), min_obs.sum(axis=1), max_obs - min_obs
            obs = min_obs + (((1-obs_beta)/(obs_alpha-obs_beta)).reshape(nstate,1)*obs_sigma)
    else:
        # setup for random obs
        obs = np.random.uniform(min_obs,max_obs)
        obs = obs / obs.sum(axis=1).reshape(nstate,1)
        while np.any(obs < min_obs) or np.any(obs > max_obs):
            obs = np.random.uniform(min_obs,max_obs)
            obs = obs / obs.sum(axis=1).reshape(nstate,1)
        if const_obs:
            sort = np.argsort(obs[:,0])
            obs = obs[sort,:]
    model = h2mm_model(prior,trans,obs)
    return model


cdef void model_full_ptr_copy(h2mm_model in_model, h2mm_mod* mod_ptr):
    # c function for copying an entire Cython h2mm_model into a C level h2mm_mod
    cdef unsigned long i
    mod_ptr.ndet = in_model.model.ndet
    mod_ptr.nstate = in_model.model.nstate
    mod_ptr.conv = in_model.model.conv
    mod_ptr.niter = in_model.model.niter
    mod_ptr.loglik = in_model.model.loglik
    mod_ptr.nphot = mod_ptr.nphot
    mod_ptr.prior = <double*> PyMem_Malloc(mod_ptr.nstate * sizeof(double))
    mod_ptr.trans = <double*> PyMem_Malloc(mod_ptr.nstate**2 * sizeof(double))
    mod_ptr.obs = <double*> PyMem_Malloc(mod_ptr.nstate * mod_ptr.ndet *sizeof(double))
    for i in range(in_model.model.nstate):
        mod_ptr.prior[i] = in_model.model.prior[i]
    for i in range(in_model.model.nstate**2):
        mod_ptr.trans[i] = in_model.model.trans[i]
    for i in range(in_model.model.nstate*in_model.model.ndet):
        mod_ptr.obs[i] = in_model.model.obs[i]


# The wrapper for the user supplied python limits function

cdef h2mm_model model_copy_from_ptr(h2mm_mod *model):
    # function for copying the values h2mm_mod C structure into an new h2mm_model object
    # this is slower that model_from _ptr, but the copy makes sure that changes
    # to the h2mm_model object do not change the original pointer
    # primarily used in the cy_limit function to create the model objects handed
    # to the user supplied python function
    cdef h2mm_mod *ret_model = allocate_models(1, model.nstate, model.ndet, model.nphot)
    copy_model(model, ret_model)
    return h2mm_model.from_ptr(ret_model)

cdef int cy_limit(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double time, lm* limits, void *lims):
    # initial checks ensuring model does not optimize too long or have loglik error
    if current.niter >= limits.max_iter:
        current.conv = 4
        return 2
    if time > limits.max_time:
        current.conv = 5
        return 2
    if isnan(current.loglik):
        old.conv = 6
        return 1
    cdef int ret
    cdef h2mm_model limit_model
    cdef h2mm_model old_mod = model_copy_from_ptr(old)
    cdef h2mm_model cur_mod = model_copy_from_ptr(current)
    cdef h2mm_model new_mod = model_copy_from_ptr(new)
    cdef bound_struct *bound = <bound_struct*> lims
    cdef unsigned long niter = current.niter
    cdef object func = <object> bound.func 
    cdef object py_limits = <object> bound.limits
    h2mm_normalize(new)
    new.niter = niter + 1
    # execute the limit function
    try:
        limit_res = func(new_mod, cur_mod, old_mod, py_limits)
    except:
        current.conv = 6
        return -4
    if not isinstance(limit_res, (list, tuple, h2mm_model, np.ndarray, int)):
        current.conv = 6
        return -4
    elif isinstance(limit_res, bool):
        if limit_res:
            current.conv = 2
            new.conv = 7
            return 0
        else:
            old.conv = 3
            current.conv = 7
            return 1
    elif isinstance(limit_res, int):
        if limit_res == 0:
            current.conv = 2
            new.conv = 1
            return 0
        elif limit_res == 1:
            old.conv = 3
            current.conv = 7
            return 1
        elif limit_res == 2:
            current.conv = 3
            return 2
        else:
            old.conv = 6
            return -4
    elif isinstance(limit_res, h2mm_model): # when limit function only returns h2mm_model
        limit_model = limit_res
        # check that return model is valid
        if limit_model.model.ndet != current.ndet or limit_model.model.nstate != current.nstate:
            old.conv = 6
            return -3
        ret = h2mm_check_converged(new, current, old, time, limits)
        if ret != 0:
            return ret
        else:
            copy_model_vals(limit_model.model, new)
            return ret
    # if the function returns 2 values, run longer processing code
    elif isinstance(limit_res, (list, tuple, np.ndarray)):
        # check deeper validity of return value
        if len(limit_res) != 2 or not isinstance(limit_res[0], h2mm_model) or not isinstance(limit_res[1], (bool, int)) or limit_res[1] > 2:
            old.conv = 6
            return -4
        limit_model = limit_res[0]
        if limit_model.model.ndet != current.ndet or limit_model.model.nstate != current.nstate: # bad return model
            old.conv = 6
            return -3
        copy_model_vals(limit_model.model, new)
        if isinstance(limit_res[1], bool):
            if limit_res[1]:
                current.conv = 2
                new.conv = 1
                return 0
            else:
                old.conv = 3
                current.conv = 7
                return 1
        else:
            ret = limit_res[1]
            if ret == 0:
                current.conv = 2
                new.conv = 1
            elif ret == 1:
                old.conv = 3
                current.conv = 7
            elif ret == 2:
                current.conv = 3
            else:
                old.conv = 6
                return -3
            return ret
    else:
        old.conv = 6
        return -4



# The wrapper for the user supplied print function
cdef void model_print_call(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_struct *print_in = <print_struct*> func
    cdef object print_func = <object> print_in.func
    cdef object args = <object> print_in.args
    cdef h2mm_model new_model = model_copy_from_ptr(new)
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    cdef h2mm_model old_model = model_copy_from_ptr(old)
    new_model.normalize()
    if niter % args[2] == 0:
        if len(args) == 4:
            print_func(niter, new_model, current_model, old_model, t_iter, t_total)
        else:
            print_func(niter, new_model, current_model, old_model, t_iter, t_total,*args[4:])

# The wrapper for the user supplied print function that displays a string
cdef void model_print_call_str(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_struct *print_in = <print_struct*> func
    cdef object print_func = <object> print_in.func
    cdef object args = <object> print_in.args
    cdef h2mm_model new_model = model_copy_from_ptr(new)
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    cdef h2mm_model old_model = model_copy_from_ptr(old)
    new_model.normalize()
    if niter % args[2] == 0:
        if len(args) == 4:
            disp_str = print_func(niter, new_model, current_model, old_model, t_iter, t_total)
        else:
            disp_str = print_func(niter, new_model, current_model, old_model, t_iter, t_total,*args[4:])
        if args[3]:
            args[1].data += disp_str
        else:
            args[1].data = disp_str
        args[0].update(args[1])

# function to hand to the print_func, prints the entire h2mm_model
cdef void model_print_all(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_args_struct *print_args = <print_args_struct*> func
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    cdef object disp_txt = <object> print_args.txt
    cdef object disp_handle = <object> print_args.handle
    if niter % print_args.disp_freq == 0:
        if print_args.keep == 1:
            disp_txt.data += current_model.__repr__() + f'\nIteration time:{t_iter}, Total:{t_total}\n'
        else:
            disp_txt.data = current_model.__repr__() + f'\nIteration time:{t_iter}, Total:{t_total}\n'
        disp_handle.update(disp_txt)

# function to hand to the print_func, prints the current loglik and the improvement
cdef void model_print_diff(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_args_struct *print_args = <print_args_struct*> func
    cdef object disp_txt = <object> print_args.txt
    cdef object disp_handle = <object> print_args.handle
    if niter % print_args.disp_freq == 0:
        if print_args.keep == 1:
            disp_txt.data += f'Iteration:{niter:5d}, loglik:{current.loglik:12e}, improvement:{current.loglik - old.loglik:6e}\n'
        else:
            disp_txt.data = f'Iteration:{niter:5d}, loglik:{current.loglik:12e}, improvement:{current.loglik - old.loglik:6e}\n'
        disp_handle.update(disp_txt)

cdef void model_print_diff_time(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_args_struct *print_args = <print_args_struct*> func
    cdef object disp_txt = <object> print_args.txt
    cdef object disp_handle = <object> print_args.handle
    if niter % print_args.disp_freq == 0:
        if print_args.keep == 1:
            disp_txt.data += f'Iteration:{niter:5d}, loglik:{current.loglik:12e}, improvement:{current.loglik - old.loglik:6e} iteration time:{t_iter}, total:{t_total}\n'
        else:
            disp_txt.data = f'Iteration:{niter:5d}, loglik:{current.loglik:12e}, improvement:{current.loglik - old.loglik:6e} iteration time:{t_iter}, total:{t_total}\n'
        disp_handle.update(disp_txt)

# function to hand to the print_func, prints current and old loglik
cdef void model_print_comp(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_args_struct *print_args = <print_args_struct*> func
    cdef object disp_txt = <object> print_args.txt
    cdef object disp_handle = <object> print_args.handle
    if niter % print_args.disp_freq == 0:
        if print_args.keep == 1:
            disp_txt.data += f"Iteration:{niter:5d}, loglik:{current.loglik:12e}, previous loglik:{old.loglik:12e}\n"
        else:
            disp_txt.data = f"Iteration:{niter:5d}, loglik:{current.loglik:12e}, previous loglik:{old.loglik:12e}\n"
        disp_handle.update(disp_txt)

# function to hand to the print_func, print comparison between current and old loglik, and iteration and total time
cdef void model_print_comp_time(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_args_struct *print_args = <print_args_struct*> func
    cdef object disp_txt = <object> print_args.txt
    cdef object disp_handle = <object> print_args.handle
    if niter % print_args.disp_freq == 0:
        if print_args.keep == 1:
            disp_txt.data += f"Iteration:{niter:5d}, loglik:{current.loglik:12e}, previous loglik:{old.loglik:12e} iteration time:{t_iter}, total:{t_total}\n"
        else:
            disp_txt.data = f"Iteration:{niter:5d}, loglik:{current.loglik:12e}, previous loglik:{old.loglik:12e} iteration time:{t_iter}, total:{t_total}\n"
        disp_handle.update(disp_txt)


# function to hand to the print_func, prints current and old loglik
cdef void model_print_iter(unsigned long niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func):
    cdef print_args_struct *print_args = <print_args_struct*> func
    cdef object disp_txt = <object> print_args.txt if print_args.txt is not NULL else None
    cdef object disp_handle = <object> print_args.handle if print_args.handle is not NULL else None
    if niter % print_args.disp_freq == 0:
        if print_args.keep == 1:
            disp_txt.data += f"Iteration {niter:5d} (Max:{print_args.max_iter:5d})\n"
        else:
            disp_txt.data = f"Iteration {niter:5d} (Max:{print_args.max_iter:5d})\n"
        disp_handle.update(disp_txt)
  
# function to generate numpy array of 2d numpy double arrays (primarily for returning gamma)
cdef np.ndarray gen_gamma_numpy(tuple shape, unsigned long* dim1, unsigned long dim2, double** arrays):
    cdef unsigned long i
    cdef unsigned long size = 1
    for i in shape:
        size *= i
    cdef np.ndarray[object] out = np.empty(size, dtype=object)
    for i in range(size):
        out[i] = copy_to_np_d(dim1[i] * dim2, arrays[i]).reshape((dim1[i], dim2))
    return out.reshape(shape)

 # function to generate list or tuple of 2d numpy double arrays (primarily for returning gamma)
cdef gen_gamma_py(type tp, tuple shape, unsigned long* dim1, unsigned long dim2, double** arrays):
    return tp(copy_to_np_d(dim1[i]*dim2, arrays[i]).reshape((dim1[i], dim2)) for i in range(shape[0]))

# wrapper to choose correct gamma generating function
cdef gen_gamma(type tp, tuple shape, unsigned long* dim1, unsigned long dim2, double** gamma):
    if tp == np.ndarray:
        return gen_gamma_numpy(shape, dim1, dim2, gamma)
    elif tp in (list, tuple):
        return gen_gamma_py(tp, shape, dim1, dim2, gamma)

# function to generate numpy array of 1d numpy unsigned long arrays (primarily for returning path)
cdef np.ndarray gen_path_numpy(tuple shape, unsigned long* dim1, unsigned long** arrays):
    cdef unsigned long i
    cdef unsigned long size = 1
    for i in shape:
        size *= i
    cdef np.ndarray[object] out = np.empty(size, dtype=object)
    for i in range(size):
        out[i] = copy_to_np_ul(dim1[i], arrays[i])
    return out.reshape(shape)

 # function to generate list or tuple of 1d numpy unsinged long arrays (primarily for returning path)
cdef gen_path_py(type tp, tuple shape, unsigned long* dim1, unsigned long** arrays):
    return tp(copy_to_np_ul(dim1[i], arrays[i]) for i in range(shape[0]))


# function to generate numpy array of 1d numpy double arrays (primarily for returning scale)
cdef np.ndarray gen_scale_numpy(tuple shape, unsigned long* dim1, double** arrays):
    cdef unsigned long i
    cdef unsigned long size = 1
    for i in shape:
        size *= i
    cdef np.ndarray[object] out = np.empty(size, dtype=object)
    for i in range(size):
        out[i] = copy_to_np_d(dim1[i], arrays[i])
    return out.reshape(shape)

 # function to generate list or tuple of 1d numpy double arrays (primarily for returning scale)
cdef gen_scale_py(type tp, tuple shape, unsigned long* dim1, double** arrays):
    return tp(copy_to_np_d(dim1[i], arrays[i]) for i in range(shape[0]))


# wrapper to choose correct path generating function
cdef gen_path(type tp, tuple shape, unsigned long* dim1, ph_path* path):
    cdef unsigned long i
    cdef unsigned long size = 1
    for i in shape:
        size *= i
    cdef unsigned long **ppath = <unsigned long**> PyMem_Malloc(size*sizeof(unsigned long*))
    cdef double **pscale = <double**> PyMem_Malloc(size*sizeof(double*))
    # build pointer arrays from path and scale
    for i in range(size):
        ppath[i] = path[i].path
        pscale[i] = path[i].scale
    if tp == np.ndarray:
        rpath = gen_path_numpy(shape, dim1, ppath)
        rscale = gen_scale_numpy(shape, dim1, pscale)
    elif tp in (list, tuple):
        rpath = gen_path_py(tp, shape, dim1, ppath)
        rscale = gen_scale_py(tp, shape, dim1, pscale)
    if ppath is not NULL:
        PyMem_Free(ppath)
        ppath = NULL
    if pscale is not NULL:
        PyMem_Free(pscale)
        pscale = NULL
    return rpath, rscale

def EM_H2MM_C(h2mm_model h_mod, indexes, times, max_iter=None, 
              print_func='iter', print_args=None, bounds_func=None, 
              bounds=None, max_time=np.inf, converged_min=None, 
              num_cores=None, reset_niter=True, gamma=False, opt_array = False):
    """
    Calculate the most likely model that explains the given set of data. The 
    input model is used as the start of the optimization.

    Parameters
    ----------
    h_model : h2mm_model
        An initial guess for the H2MM model, just give a general guess, the algorithm
        will optimize, and generally the algorithm will converge even when the
        initial guess is very far off
    indexes : list of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1-to-1 correspondence to the times list
    times : list of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1-to-1 correspondence to the indexes list
    
    print_func : None, str or callable, optional
        Specifies how the results of each iteration will be displayed, several 
        strings specify built-in functions. Default is 'iter'
        Acceptable inputs: str, Callable or None
            
            **None**\:
            
                causes no printout anywhere of the results of each iteration
                
            Str: 'console', 'all', 'diff', 'comp', 'iter'
            
                **'console'**\: 
                    
                    the results will be printed to the terminal/console 
                    window this is useful to still be able to see the results, 
                    but not clutter up the output of your Jupyter Notebook, this 
                    is the default
                
                **'all'**\: 
                    
                    prints out the full :class:`h2mm_model` just evaluated, this is very
                    verbose, and not generally recommended unless you really want
                    to know every step of the optimization
                
                **'diff'**\: 
                    
                    prints the loglik of the iteration, and the difference 
                    between the current and old loglik, this is the same format
                    that the 'console' option used, the difference is whether the
                    print function used is the C printf, or Cython print, which 
                    changes the destination from the console to the Jupyter notebook
                    
                **'diff_time'**\: 
                    
                    same as 'diff' but adds the time of the last iteration
                    and the total optimization time
                    
                **'comp'**\: 
                    
                    prints out the current and old loglik
                
                **'comp_time'**\: 
                    
                    same as 'comp'  but with times, like in 'diff_time'
                
                **'iter'**\: 
                    
                    prints out only the current iteration
            
            Callable: user defined function
                A python function for printing out a custom output, the function must
                accept the input of 
                (int, :class:`h2mm_model`, :class:`h2mm_model`, :class:`h2mm_model`, float, float)
                as the function will be handed (niter, new, current, old, t_iter, t_total)
                from a special Cython wrapper. t_iter and t_total are the times
                of the iteration and total time respectively, in seconds, based
                on the C level clock function, which is notably inaccurate,
                often reporting larger than actual values.
    
    print_args : 2-tuple/list (int, bool) or None, optional
        Arguments to further customize the printing options. The format is
        (int bool) where int is how many iterations before updating the display
        and the bool is True if the printout will concatenate, and False if the
        display will be kept to one line, The default is None, which is changed
        into (1, False). If only an int or a bool is specified, then the default
        value for the other will be used. If a custom printing function is given
        then this argument will be passed to the function as \*args.
        Default is None
    bounds_func : str, callable or None, optional
        function to be evaluated after every iteration of the H2MM algorithm
        its primary function is to place bounds on :class:`h2mm_model`
        
        .. note:: 
            
            bounding the :class:`h2mm_model` causes the guarantee of improvement on each
            iteration until convergence to no longer apply, therefore the results are
            no longer guaranteed to be the optimal model within bounds.
        
        Default is None
        
        Acceptable inputs\:
            
            C level limits: 'minmax' 'revert' 'revert_old'
                prevents the model from having values outside of those defined
                by :class:`h2mm_limits` class given to bounds, if an iteration produces
                a new model for loglik calculation in the next iteration with 
                values that are out of those bounds, the 3 function differ in
                how they correct the model when a new model has a value out of
                bounds, they are as follows:
                
                **'minmax'**\: 
                    
                    sets the out of bounds value to the min or max value
                    closer to the original value
                
                **'revert'**\: 
                    
                    sets the out of bounds value to the value from the last
                    model for which the loglik was calculated
                
                **'revert_old'**\: 
                    
                    similar to revert, but instead reverts to the value
                    from the model one before the last calculated model
            
            Callable: python function that takes 4 inputs and returns :class:`h2mm_model` object
                .. warning:: 
                
                    The user takes full responsibility for errors in the results.
                
                must be a python function that takes 4 arguments, the first 3 are
                the :class:`h2mm_model` objects, in this order: new, current, old, 
                the fourth is the argument supplied to the bounds keyword argument,
                the function must return a single :class:`h2mm_limits` object, the prior, 
                trans and obs fields of which will be optimized next, other values 
                are ignored.
    bounds : h2mm_limits, str, 4th input to callable, or None, optional
        The argument to be passed to the bounds_func function. If bounds_func is
        None, then bounds will be ignored. If bounds_func is 'minmax', 'revert'
        or 'revert_old' (calling C-level bounding functions), then bounds must be
        a :class:`h2mm_limits` object, if bounds_func is a callable, bounds must be an
        acceptable input as the fourth argument to the function passed to bounds_func
        Default is None
    max_iter : int or None, optional
        the maximum number of iterations to conduct before returning the current
        :class:`h2mm_model`, if None (default) use value from optimization_limits. 
        Default is None
    max_time : float or None, optional
        The maximum time (in seconds) before returning current model
        NOTE this uses the C clock, which has issues, often the time assessed by
        C, which is usually longer than the actual time. If None (default) use
        value from optimization_limits. Default is None
    converged_min : float or None, optional
        The difference between new and current :class:`h2mm_model` to consider the model
        converged, the default setting is close to floating point error, it is
        recommended to only increase the size. If None (default) use value from 
        optimization_limits. Default is None
    num_cores : int or None, optional
        the number of C threads (which ignore the gil, thus functioning more
        like python processes), to use when calculating iterations.
        
        .. note:: 
            
            optimization_limtis sets this to be `os.cpu_count() // 2`, as most machines
            are multithreaded. Because os.cpu_count() returns the number of threads,
            not necessarily the number of cpus. For most machines, being multithreaded,
            this actually returns twice the number of physical cores. Hence the default
            to set at `os.cpu_count() // 2`. If your machine is not multithreaded, it
            is best to set `optimization_limits.num_cores = os.cpu_count()`
            
        If None (default) use value from optimization_limits. 
        Default is None
    reset_niter : bool, optional
        Tells the algorithm whether or not to reset the iteration counter of the
        model, True means that even if the model was previously optimized, the 
        iteration counter will start at 0, so calling :func:`EM_H2MM_C` on a model that
        reached its maximum iteration number will proceed another max_iter 
        iterations. On the other hand, if set to False, then the new iterations
        will be added to the old. If set to False, you likely will have to increase
        max_iter, as the optimization will count the previous iterations towards
        the max_iter threshold.
        Default is False
    gamma : bool, optional
        Whether or not to return the gamma array, which gives the probabilities
        of each photon being in a given state.
        
        .. note::
            
            If opt_array is True, then only the gamma arrays for the ideal
            model are returned.
        
        Default is False
    opt_array : bool
        Defines how the result is returned, if False (default) then return only
        the ideal model, if True, then a numpy array is returned containing all
        models during the optimization.
        
        .. note::
            
            The last model in the array is the model after the convergence criterion
            have been met. Therefore check the :attr:`h2mm_model.conv_code`, if it
            is 7, then the second to last model is the ideal model. Still, usually
            the last model will be so close to ideal that it will not make any 
            significant difference
            
        Default is False
    
    Returns
    -------
    out : h2mm_model
        The optimized :class:`h2mm_model`. will return after one of the following conditions
        are met: model has converged (according to converged_min, default 1e-14),
        maximum iterations reached, maximum time has passed, or an error has occurred
        (usually the result of a nan from a floating point precision error)
        
    gamma : list, tuple or np.ndarray (optional)
        If gamma = True, this variable is returned, which contains (as numpy arrays)
        the probabilities of each photon to be in each state, for each burst.
        The arrays are returned in the same order and type as the input indexes,
        individual data points organized as [photon, state] within each data
        sequence.
        
    """
    # check inputs are valid
    burst_check = verify_input(indexes, times, h_mod.model.ndet)
    if isinstance(burst_check, Exception):
        raise burst_check
    if h_mod.model.ndet > burst_check + 1:
        warnings.warn(f"Overdefined model:\nThe given data has fewer photon streams than initial model\nModel has {h_mod.model.ndet} streams, but data has only {burst_check + 1} streams.\nExtra photon streams in model will have 0 values in emission probability matrix")
    # check the bounds_func
    cdef unsigned long num_burst = indexes.size if isinstance(indexes, np.ndarray) else len(indexes)
    cdef tuple bounds_strings = ('minmax', 'revert', 'revert_old')
    cdef tuple print_strings = (None,'console','all','diff','diff_time','comp','comp_time','iter')
    cdef object disp_txt = Pretty("Print Func validation")
    cdef object disp_handle = DisplayHandle()
    disp_handle.display(disp_txt)
    cdef h2mm_model h_test_new = h_mod.copy()
    cdef h2mm_model h_test_current = h_mod.copy()
    cdef h2mm_model h_test_old = h_mod.copy()
    h_test_new.model.niter, h_test_current.model.niter, h_test_old.model.niter = 3, 2, 1
    h_test_new.model.conv, h_test_current.model.conv, h_test_old.model.conv = 1, 1, 1
    h_test_new.model.nphot, h_test_current.model.nphot, h_test_old.model.nphot = 1000, 1000, 1000
    h_test_new.model.loglik, h_test_current.model.loglik, h_test_old.model.loglik = -0.0, -2e-4, -3e-4
    if print_func not in print_strings and not callable(print_func):
        raise ValueError("print_func must be None, 'console', 'all', 'diff', or 'comp' or callable")
    if print_func in print_strings:
        if not isinstance(print_args, (int, bool, tuple, list)) and print_args is not None:
            raise TypeError(f"print_args must be None, int, bool, or a 2-tuple or 2-list, got {type(print_args)}")
        elif isinstance(print_args, (tuple, list)) and len(print_args) != 2:
            raise TypeError(f"If print args is a tuple or list, it must have length 2, got {len(print_args)}")
        elif type(print_args) in (tuple,list) and (not isinstance(print_args[0],int) or not isinstance(print_args[1],bool)):
            raise ValueError("For print_args as 2-tuple or 2-list, but be composed of (int, bool) types, got ({type(print_args[0])}, {type(print_args[0])})")
        elif isinstance(print_args,int) and not isinstance(print_args,bool) and print_args <= 0:
            raise ValueError("print_args int values must be positive")
        if print_args is None:
            print_args = (1, False)
        elif isinstance(print_args,int) and not isinstance(print_args,bool):
            print_args = (print_args, False)
        elif isinstance(print_args,bool):
            print_args = (1,print_args)
    elif callable(print_func):
        print_args = tuple(print_args) if isinstance(print_args, list) else print_args
        try:
            disp_txt.data += "print_func validation\n"
            disp_handle.update(disp_txt)
            if print_args is None:
                print_return = print_func(1,h_test_new,h_test_current,h_test_old,0.1,0.2)
                print_args = (disp_handle,disp_txt,1,False)
            elif isinstance(print_args, int):
                print_return = print_func(1,h_test_new,h_test_current,h_test_old,0.1,0.2)
                print_args = (disp_handle,disp_txt,1, print_args) if isinstance(print_args,bool) else (disp_handle,disp_txt,print_args, False)
            elif  isinstance(print_args, tuple) and len(print_args) >= 2 and isinstance(print_args[0],int) and isinstance(print_args[1],bool):
                if len(print_args) == 2:
                    print_return = print_func(1,h_test_new,h_test_current,h_test_old,0.1,0.2)
                else:
                    print_return = print_func(1,h_test_new,h_test_current,h_test_old,0.1,0.2,*print_args[2:])
                print_args = ((disp_handle,disp_txt) + print_args)
            else:
                print_return = print_func(1,h_test_new,h_test_current,h_test_old,0.1,0.2,*print_args)
                print_args = ((disp_handle, disp_txt, 1, False) + print_args)
            if print_return is not None and not isinstance(print_return,str):
                raise TypeError(f"print_func must either return nothing, or a str, got {type(print_return)}")
            disp_txt.data += "print_func validated\n"
            disp_handle.update(disp_txt)
        except Exception as e:
            disp_txt.data += "print_func invalid function, must take (niter,new,current,old,t_iter,t_total,*print_args)\n"
            disp_handle.update(disp_txt)
            raise e
    # set up optimzation min/max limits
    cdef lm limits
    limits.max_iter = <unsigned long> optimization_limits._get_max_iter(max_iter)
    limits.num_cores = <unsigned long> optimization_limits._get_num_cores(num_cores)
    limits.max_time = <double> optimization_limits._get_max_time(max_time)
    limits.min_conv = <double> optimization_limits._get_converged_min(converged_min)
    # setup the printing function
    cdef void (*ptr_print_func)(unsigned long,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*)
    cdef void *ptr_print_call = NULL
    cdef print_args_struct *ptr_print_args = <print_args_struct*> PyMem_Malloc(sizeof(print_args_struct))
    cdef print_struct *ptr_print_struct = <print_struct*> PyMem_Malloc(sizeof(print_struct*))
    ptr_print_args.keep = 1
    ptr_print_args.max_iter = limits.max_iter
    disp_txt.data = ""
    if print_func in print_strings:
        ptr_print_args.txt = <void*> disp_txt
        ptr_print_args.handle = <void*> disp_handle
        ptr_print_args.keep = 1 if print_args[1] else 0
        ptr_print_args.disp_freq = <unsigned long> print_args[0]
        ptr_print_call = <void*> ptr_print_args
        if print_func == 'console':
            ptr_print_func = baseprint
            ptr_print_args.keep = 0
        elif print_func == 'all':
            ptr_print_func = model_print_all
        elif print_func == 'diff':
            ptr_print_func = model_print_diff
        elif print_func == 'diff_time':
            ptr_print_func = model_print_diff_time
        elif print_func == 'comp':
            ptr_print_func = model_print_comp
        elif print_func == 'comp_time':
            ptr_print_func = model_print_comp_time
        elif print_func == 'iter':
            ptr_print_func = model_print_iter
    elif callable(print_func):
        ptr_print_args.keep = 1 if print_args[3] else 0
        ptr_print_func = model_print_call_str if isinstance(print_return,str) else model_print_call
        ptr_print_struct.func = <void*> print_func
        ptr_print_struct.args = <void*> print_args
        ptr_print_call = <void*> ptr_print_struct
    elif print_func is None:
        ptr_print_func = NULL
        ptr_print_args.keep = 0

    # allocate the memory for the pointer arrays to be submitted to the C function
    # print("Allocating C bursts data")
    disp_txt.data = "Preparing data"
    disp_handle.update(disp_txt)
    cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long **b_delta = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long *burst_sizes = burst_convert(num_burst, indexes, times, b_det, b_delta)
    if burst_sizes is NULL:
        if ptr_print_args is not NULL: 
            PyMem_Free(ptr_print_args)
            ptr_print_args = NULL
        if ptr_print_struct is not NULL:
            PyMem_Free(ptr_print_struct)
            ptr_print_struct = NULL
        raise ValueError("Photon out of order")
    # print("Done Allocating C bursts data")
    # if the bounds_func is not None, then setup the bounds arrays
    cdef int (*bound_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm* ,void*)
    cdef void *b_ptr
    # cdef h2mm_minmax *bound = <h2mm_minmax*> PyMem_Malloc(sizeof(h2mm_minmax))
    disp_txt.data = "Bounds preparation"
    disp_handle.update(disp_txt)
    cdef _h2mm_lims bound
    cdef bound_struct *b_struct = <bound_struct*> PyMem_Malloc(sizeof(bound_struct))
    if bounds_func is None:
        bound_func = limit_check_only
        b_ptr = NULL
    elif bounds_func in ['minmax', 'revert', 'revert_old']:
        bound = bounds._make_model(h_mod)
        b_ptr = <void*> &bound.limits
        if bounds_func == 'minmax':
            bound_func = limit_minmax
        elif bounds_func == 'revert':
            bound_func = limit_revert
        elif bounds_func == 'revert_old':
            bound_func = limit_revert_old
    elif callable(bounds_func):
        b_struct.func = <void*> bounds_func
        b_struct.limits = <void*> bounds
        b_ptr = b_struct
        bound_func = cy_limit
    # set up the in and out h2mm_mod variables
    cdef unsigned long old_niter = h_mod.model.niter
    if reset_niter:
        h_mod.model.niter = 0
    # disp_handle.update(disp_txt)
    cdef int res
    cdef unsigned long i, n
    cdef h2mm_mod *out_arr
    cdef double **gamma_arr = NULL
    disp_txt.data = "Bounds prepared"
    disp_handle.update(disp_txt)
    # based on gamma and opt_array kwargs, choose which h2mm optimization to run
    if gamma and opt_array:
        res =  h2mm_optimize_gamma_array(num_burst,burst_sizes,b_delta,b_det, h_mod.model, &out_arr, &gamma_arr, &limits, bound_func, b_ptr, ptr_print_func, ptr_print_call)
    elif gamma:
        out_arr = allocate_models(1, h_mod.nstate, h_mod.ndet, 0)
        res =  h2mm_optimize_gamma(num_burst,burst_sizes,b_delta,b_det, h_mod.model, out_arr, &gamma_arr,&limits, bound_func, b_ptr, ptr_print_func, ptr_print_call)
    elif opt_array:
        res =  h2mm_optimize_array(num_burst,burst_sizes,b_delta,b_det, h_mod.model, &out_arr, &limits, bound_func, b_ptr, ptr_print_func, ptr_print_call)
    else:
        out_arr = allocate_models(1, h_mod.nstate, h_mod.ndet, 0)
        res =  h2mm_optimize(num_burst,burst_sizes,b_delta,b_det, h_mod.model, out_arr, &limits, bound_func, b_ptr, ptr_print_func, ptr_print_call)
    if gamma and res >= 0:
        gamma_out = gen_gamma(type(indexes), array_size(indexes), burst_sizes, h_mod.model.nstate, gamma_arr)
        if gamma_arr is not NULL:
            PyMem_Free(gamma_arr)
            gamma_arr = NULL
    cdef unsigned long keep = ptr_print_args.keep
    # free pointers
    if b_struct is not NULL:
        PyMem_Free(b_struct)
        b_struct = NULL
    burst_free(num_burst, b_det, b_delta, burst_sizes)
    if ptr_print_args is not NULL:
        PyMem_Free(ptr_print_args)
        ptr_print_args = NULL
    if ptr_print_struct is not NULL:
        PyMem_Free(ptr_print_struct)
        ptr_print_struct = NULL
    h_mod.model.niter = old_niter # restore niter of input model (if reset_niter, this was reset to 0)
    # post processing, converting
    cdef unsigned long conv, niter
    if opt_array and res > 0:
        n = 0
        while out_arr[n].conv != 1 and n < limits.max_iter + 2:
            n += 1
        n -= 1
        out = np.array([model_copy_from_ptr(&out_arr[i]) for i in range(n+1)], dtype=object)
        free_models(limits.max_iter + 2, out_arr)
        niter = out_arr[n].niter
        conv = out_arr[n].conv
    elif res > 0:
        conv = out_arr.conv
        niter = out_arr.niter
        out = h2mm_model.from_ptr(out_arr)
    # process the return code
    if res == -1:
        raise ValueError('Bad pointer, check cython code')
    elif res == -2:
        raise ValueError('Too many photon streams in data for H2MM model')
    elif res == -3:
        raise ValueError('limits function must return model of same shape as inintial model')
    elif res == -4:
        raise TypeError('limits function must return bool, h2mm_model, (h2mm_model, bool) or (h2mm_model, int [0,2])')
    elif res < -4:
        raise ValueError(f'Unknown error, check C code- raise issue on GitHub, res={res}, conv={conv}, n={n}')
    elif conv == 3:
        out_text = f'The model converged after {niter} iterations'
    elif conv == 4:
        out_text = 'Optimization reached maximum number of iterations'
    elif conv == 5:
        out_text = 'Optimization reached maxiumum time'
    elif conv == 6:
        out_text = f'An error occured on iteration {niter}, returning previous model'
    elif conv == 7:
        out_text = f'The model converged after {niter-1} iterations'
    else:

        raise ValueError(f'Unknown error, check C code- raise issue on GitHub, res={res}, conv={conv}, n={n}')
    if keep == 1:
        disp_txt.data += out_text
    else:
        disp_txt.data = out_text
    disp_handle.update(disp_txt)
    if gamma:
        return out, gamma_out
    else:
        return out

# function generates output numpy array of h2mm objects
cdef h2mm_arr_gen(type mod_tp, tuple mod_shape, h2mm_mod* models):
    cdef unsigned long i
    cdef unsigned long mod_size = 1
    for i in mod_shape:
        mod_size *= i
    if mod_tp in (list, tuple):
        out = mod_tp(model_copy_from_ptr(&models[i]) for i in range(mod_size))
    elif mod_tp ==  np.ndarray:
        out = np.array(tuple(model_copy_from_ptr(&models[i]) for i in range(mod_size)), dtype=object).reshape(mod_shape)
    else:
        out = model_copy_from_ptr(models)
    return out



cdef all_arr_gen(type mod_tp, type burst_tp, tuple mod_shape, tuple burst_shape, unsigned long* burst_sizes, h2mm_mod* models, double*** gamma):
    cdef unsigned long mod_size = 1
    cdef Py_ssize_t i
    for i in mod_shape:
        mod_size *= i
    if mod_tp == np.ndarray:
        gamma_out = np.empty(mod_size, dtype=object)
        for i in range(mod_size):
            gamma_out[i] = gen_gamma(burst_tp, burst_shape, burst_sizes, models[i].nstate, gamma[i])
        gamma_out = gamma_out.reshape(mod_shape)
    elif mod_tp in (list, tuple):
        gamma_out = mod_tp(gen_gamma(burst_tp, burst_shape, burst_sizes, models[i].nstate, gamma[i]) for i in range(mod_size))
    else:
        gamma_out = gen_gamma(burst_tp, burst_shape, burst_sizes, models[0].nstate, gamma[0])
    h2mm_out = h2mm_arr_gen(mod_tp, mod_shape, models)
    return h2mm_out, gamma_out


def H2MM_arr(h_mod, indexes, times, num_cores=None, gamma=False):
    """
    Calculate the logliklihood of every model in a list/array given a set of
    data

    Parameters
    ----------
    h_model : list, tuple, or numpy.ndarray of h2mm_model objects
        All of the :class:`h2mm_model` object for which the loglik will be calculated
    indexes : list or tuple of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1-to-1 correspondence to the times list
    times : list or tuple of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1-to-1 correspondence to the indexes list
    gamma : bool, optional
        Whether or not to return the gamma array, which gives the probabilities
        of each photon being in a given state.
        the number of C threads (which ignore the gil, thus functioning more
        like python processes), to use when calculating iterations.
        
        .. note:: 
            
            optimization_limtis sets this to be `os.cpu_count() // 2`, as most machines
            are multithreaded. Because os.cpu_count() returns the number of threads,
            not necessarily the number of cpus. For most machines, being multithreaded,
            this actually returns twice the number of physical cores. Hence the default
            to set at `os.cpu_count() // 2`. If your machine is not multithreaded, it
            is best to set `optimization_limits.num_cores = os.cpu_count()`
            
        If None (default) use value from optimization_limits. 
        Default is None
        
    Returns
    -------
    out : list tuple, or numpy.ndarray of h2mm_model
        A list, tuple, or numpy array of the :class:`h2mm_model` with the loglik computed
        The converged state is automatically set to 0, and nphot is set
        in accordance with the number of photons in the data set.
        The datatype returned is the same as the datatype of h_model
    gamma_arr : list, tuple or np.ndarray (optional)
        If gamma = True, this variable is returned, which contains (as numpy arrays)
        the probabilities of each photon to be in each state, for each burst and model.
        The arrays are returned in the same order and type as the input models and 
        indexes, individual data points organized as [photon, state] within each data
        sequence.
    """
    cdef unsigned long i = 1
    cdef unsigned long j
    cdef type tp = type(h_mod)
    cdef type btp = type(indexes)
    cdef unsigned long num_burst = indexes.size if btp == np.ndarray else len(indexes)
    cdef unsigned long ndet
    cdef unsigned long mod_size = 1
    cdef tuple mod_shape = array_size(h_mod)
    for i in mod_shape:
        mod_size *= i
    if mod_size == 0:
        return tuple()
    if tp not in (h2mm_model, list, tuple, np.ndarray):
        raise TypeError('First argument must be list or numpy array of h2mm_model objects')
    for i, h in enumerate(enum_arrays(h_mod)):
        if not isinstance(h, h2mm_model):
            raise TypeError(f'All elements of first argument must of type h2mm_model, element {i} is not')
        if i == 0:
            ndet = h.ndet
        if h.ndet != ndet:
            raise ValueError("All models must have same number of photon streams")
    # check bursts of correct type
    burst_shape = array_size(indexes)
    # if statements verify that the data is valid, ie matched lengths and dimensions for times and indexes in all bursts
    burst_check = verify_input(indexes, times, ndet)
    if isinstance(burst_check, Exception):
        raise burst_check
    if burst_check != ndet - 1:
        warnings.warn(f"Overdefined models: models have {ndet} photons streams, while data only suggests you have {burst_check + 1} photon streams")
    # allocate the memory for the pointer arrays to be submitted to the C function
    cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long **b_delta = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long *burst_sizes = burst_convert(num_burst, indexes, times, b_det, b_delta)
    cdef int e_val
    if burst_sizes is NULL:
        raise ValueError("Photon out of order")
    cdef lm *limits = <lm*> PyMem_Malloc(sizeof(lm))
    limits.num_cores = <unsigned long> optimization_limits._get_num_cores(num_cores)
    cdef h2mm_mod **mod_ptr_array =  <h2mm_mod**> PyMem_Malloc(mod_size * sizeof(h2mm_mod*))
    cdef h2mm_model h_temp
    for i, h in enumerate(enum_arrays(h_mod)):
        h_temp = h
        mod_ptr_array[i] = h_temp.model
    cdef h2mm_mod *mod_array;
    duplicate_toempty_models(mod_size, mod_ptr_array, &mod_array)
    if mod_ptr_array is not NULL:
        PyMem_Free(mod_ptr_array)
        mod_ptr_array = NULL
    # set up the in and out h2mm_mod variables
    cdef double ***gamma_arr = NULL
    if gamma:
        e_val = calc_multi_gamma(num_burst, burst_sizes, b_delta, b_det, mod_size, mod_array, &gamma_arr, limits)
    else:
        e_val = calc_multi(num_burst, burst_sizes, b_delta, b_det, mod_size, mod_array, limits)
    if gamma and e_val==0:
        # out = h2mm_arr_gen(tp, mod_shape, mod_array)
        out = all_arr_gen(tp, btp, mod_shape, burst_shape, burst_sizes, mod_array, gamma_arr)
        for i in range(mod_size):
            if gamma_arr[i] is not NULL:
                PyMem_Free(gamma_arr[i])
                gamma_arr[i] = NULL
        if gamma_arr is not NULL:
            PyMem_Free(gamma_arr)
            gamma_arr = NULL
    elif e_val == 0:
        out = h2mm_arr_gen(tp, mod_shape, mod_array)
    # free the limts arrays
    burst_free(num_burst, b_det, b_delta, burst_sizes)
    free_models(mod_size, mod_array)
    PyMem_Free(limits)
    if e_val == 1:
        raise ValueError('Bursts photons are out of order, please check your data')
    elif e_val == 2:
        raise ValueError('Too many photon streams in data for one or more H2MM models')
    return out

# viterbi function

def viterbi_path(h2mm_model h_mod, indexes, times, num_cores=None):
    """
    Calculate the most likely state path through a set of data given a H2MM model

    Parameters
    ----------
    h_model : h2mm_model
        An :class:`h2mm_model`, should be optimized for the given data set
        (result of :func:`EM_H2MM_C` or :meth:`h2mm_model.optimize`) to ensure results correspond to give the most likely
        path
    indexes : list or tuple of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1-to-1 correspondence to the times list
    times : list or tuple of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1-to-1 correspondence to the indexes list
    num_cores : int or None, optional
        the number of C threads (which ignore the gil, thus functioning more
        like python processes), to use when calculating iterations.
        Note that os.cpu_count() returns number of threads, but it is ideal 
        to take the number of physical cores. Therefore, unless reset by user,
        optimization_limtis sets this to be os.cpu_count() // 2, as most machines
        are multithreaded. If None (default) use value from optimization_limits. 
        Default is None
    
    Returns
    -------
    path : list of NUMPY 1D int arrays
        The most likely state path for each photon
    scale : list of NUMPY 1D float arrays
        The posterior probability for each photon
    ll : NUMPY 1D float array
        loglikelihood of each burst
    icl : float
        Integrated complete likelihood, essentially the BIC for the *Viterbi* path
    """
    # assert statements to verify that the data is valid, ie matched lengths and dimensions for times and indexes in all bursts
    burst_check = verify_input(indexes, times, h_mod.model.ndet)
    if isinstance(burst_check, Exception):
        raise burst_check
    cdef unsigned long i
    cdef unsigned long num_burst = indexes.size if type(indexes) == np.ndarray else len(indexes)
    
    # allocate the memory for the pointer arrays to be submitted to the C function
    cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long **b_delta = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long *burst_sizes = burst_convert(num_burst, indexes, times, b_det, b_delta)
    if burst_sizes is NULL:
        raise ValueError("Photon out of order")
    cdef unsigned long nphot = sum(burst_sizes[i] for i in range(num_burst))
    cdef ph_path *path_ret = allocate_paths(num_burst, burst_sizes, h_mod.model.nstate)
    cdef n_core = <unsigned long> optimization_limits._get_num_cores(num_cores)
    # set up the in and out h2mm_mod variables
    cdef unsigned long e_val = viterbi(num_burst, burst_sizes, b_delta, b_det, h_mod.model, path_ret, n_core)
    if e_val == 0:
        path, scale = gen_path(type(indexes), array_size(indexes), burst_sizes, path_ret) 
    burst_free(num_burst, b_det, b_delta, burst_sizes)
    if e_val == 1:
        raise ValueError('Bursts photons are out of order, please check your data')
    elif e_val == 2:
        raise ValueError('Too many photon streams in data for H2MM model')
    cdef double loglik = 0.0
    cdef np.ndarray[double] ll = np.zeros(num_burst)
    for i in range(num_burst):
        loglik += path_ret[i].loglik
        ll[i] = path_ret[i].loglik
    if isinstance(indexes, np.ndarray):
        ll = ll.reshape(indexes.shape)
    cdef double icl = ((h_mod.nstate**2 + ((h_mod.ndet - 1) * h_mod.nstate) - 1) * np.log(nphot)) - 2 * loglik
    PyMem_Free(path_ret)
    return path, scale, ll, icl
    

def viterbi_sort(h2mm_model hmod, indexes, times, num_cores=None):
    """
    An all inclusive viterbi processing algorithm. Returns the ICL, the most likely
    state path, posterior probabilities, and a host of information sorted by
    bursts and dwells

    Parameters
    ----------
    h_model : h2mm_model
        An :class:`h2mm_model`, should be optimized for the given data set
        (result of :func:`EM_H2MM_C` or :meth:`h2mm_model.optimize`) to ensure results correspond to give the most likely
        path
    indexes : list or tuple of NUMPY 1D int arrays
        A list of the arrival indexes for each photon in each burst.
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The indexes list must maintain  1-to-1 correspondence to the times list
    times : list or tuple of NUMPY 1D int arrays
        A list of the arrival times for each photon in each burst
        Each element of the list (a numpy array) corresponds to a burst, and
        each element of the array is a singular photon.
        The times list must maintain  1-to-1 correspondence to the indexes list
    num_cores : int or None, optional
        the number of C threads (which ignore the gil, thus functioning more
        like python processes), to use when calculating iterations.
        
        .. note:: 
            
            optimization_limtis sets this to be `os.cpu_count() // 2`, as most machines
            are multithreaded. Because os.cpu_count() returns the number of threads,
            not necessarily the number of cpus. For most machines, being multithreaded,
            this actually returns twice the number of physical cores. Hence the default
            to set at `os.cpu_count() // 2`. If your machine is not multithreaded, it
            is best to set `optimization_limits.num_cores = os.cpu_count()`
            
        If None (default) use value from optimization_limits. 
        Default is None

    Returns
    -------
    icl : float
        Integrated complete likelihood, essentially the BIC for the *Viterbi* path
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
    dwell_mid : square list of lists of NUMPY 1D int arrays
        Gives the dwell times of all dwells with full residence time in the data
        set, sorted by the state of the dwell (Top level list), and the successor
        state (Lower level list), each element of the numpy array is the duration
        of the dwell in the clock rate of the data set.
    dwell_beg : square list of lists of NUMPY 1D int arrays
        Gives the dwell times of all dwells that start at the beginning of a burst,
        sorted by the state of the dwell (Top level list), and the successor
        state (Lower level list), each element of the numpy array is the duration
        of the dwell in the clock rate of the data set.
    dwell_end : square list of lists of NUMPY 1D int arrays
        Gives the dwell times of all dwells that start at the end of a burst,
        sorted by the state of the dwell (Top level list), and the preceding
        state (Lower level list), each element of the numpy array is the duration
        of the dwell in the clock rate of the data set.
    dwell_burst : list of NUMPY 1D int arrays
        List of bursts that were only in one state, giving their durations.
    ph_counts : list of NUMPY 2D int arrays
        Counts of photons in each dwell, sorted by index and state of the dwell
        The index of the list identifies the state of the burst, the 1 index
        identifies the index of photon, and the 0 index are individual dwells.
    ph_mid : square list of NUMPY 2D arrays
        Gives the photon counts of dwells in the middle of bursts, sorted
        by state of the dwell (top level of the list) and the state of the next
        dwell in the burst (lower level burst), each row in an array is a single
        dwell, the columns correspond to the counts of each photon stream.
    ph_beg : square list of NUMPY 2D arrays
        Gives the photon counts of dwells at the beginning of bursts, sorted
        by state of the dwell (top level of the list) and the state of the next
        dwell in the burst  (lower level list), each row in an array is a single
        dwell, the columns correspond to the counts of each photon stream.
    ph_end : square list of NUMPY 2D arrays
        Gives the photon counts of dwells at the ends of bursts, sorted
        by state of the dwell (top level of the list) and the state of the previous
        dwell in the burst, each row in an array is a single dwell, the columns 
        correspond to the counts of each photon stream.
    ph_burst : list of NUMPY 2D arrays
        Gives the counts of photons in bursts with only a single state. Sorted
        according to the state of the burst. Each row in an array is a single
        dwell, the columns correspond to the counts of each photon stream.
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


def path_loglik(h2mm_model model, indexes, times, state_path, num_cores=None, 
                BIC=True, total_loglik=False, loglikarray=False):
    """
    Function for calculating the statistical parameters of a specific state path 
    and data set given a model (ln(P(XY|lambda))). By default returns the BIC of
    the entire data set, can also return the base logliklihood in total and/or
    per burst. Which return values selected are defined in keyword arguments.

    Parameters
    ----------
    model : h2mm_model
        DESCRIPTION.
    indexes : list[numpy.ndarray], tuple[numpy.ndarray] or numpy.ndarray[numpy.ndarray]
        DESCRIPTION.
    times : list, tuple, or numpy.ndarray of integer numpy.ndarray
        DESCRIPTION.
    state_path : list, tuple, or numpy.ndarray of integer numpy.ndarray
        DESCRIPTION.
    num_cores : int, optional
        Number of cores to use in calculation, if None, then use value in 
        optimization_limits. The default is None.
    BIC : bool, optional
        Whether to return the Bayes Information Criterion of the calculation. 
        The default is True.
    total_loglik : bool, optional
        Whether to return the loglikelihood of the entire data set as a single number.
        The default is False.
    loglikarray : bool, optional
        Whether to return the loglikelihoods of each burst as a numpy array. 
        The default is False.

    Raises
    ------
    TypeError
        One or more inputs did not contain the proper data types.
    ValueError
        One or more values out of the acceptable ranges given the input model
    RuntimeError
        Deeper issue, raise issue on github.

    Returns
    -------
    bic : float
        Bayes information criterion of entire data set and path.
    loglik : float
        Log likelihood of entire dataset
    loglik_array : np.ndarray
        Loglikelihood of each burst separately

    """
    burst_check = verify_input(indexes, times, model.model.ndet)
    cdef unsigned long i
    cdef str burst_errors = ""
    if isinstance(burst_check, Exception):
        raise burst_check
    elif type(indexes) != type(state_path):
        raise TypeError(f"Mismatched types: indexes and state_path must be same type")
    elif array_size(indexes) != array_size(state_path):
        raise ValueError("Mismatched sizes: indexes and state_path must be same size, got {array_size(indexes)} and {array_size(state_path)}")
    else:
        for i, path in enumerate(enum_arrays(state_path)):
            if not isinstance(path, np.ndarray):
                burst_errors += f"{i}: TypeError: state_path elements must be 1D numpy arrays, got {type(path)}"
            elif path.ndim != 1:
                burst_errors += f"{i}: TypeError: state path elements must be 1D numpy arrays, got {type(path)} and {type(path)}\n"
            elif indexes[i].shape[0] != path.shape[0]:
                burst_errors += f"{i}: Mismatched burst size: indexes has {indexes[i].shape[0]} photons while state_path has {path.shape[0]} photons\n"
            elif not np.issubdtype(path.dtype, np.integer):
                burst_errors += f"{i}: TypeError: state path arrays must integer arrays, got {path.dtype}"
            elif np.any(path < 0):
                burst_errors += f"{i}: ValueError: states cannot be negative"
            elif np.any(path >= model.model.nstate):
                burst_errors += f"{i}: ValueError: too many states in path for model"
        if burst_errors != "":
            raise ValueError("Incorrect data for state_path\n" + burst_errors)
    # get burst size, could also calculate with array_size, but this is shorter
    cdef unsigned long num_burst = indexes.size if type(indexes) == np.ndarray else len(indexes)
    # allocate the memory for the pointer arrays to be submitted to the C function
    cdef unsigned long **b_det = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long **b_delta = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    cdef unsigned long *burst_sizes = burst_convert(num_burst, indexes, times, b_det, b_delta)
    if burst_sizes is NULL:
        raise ValueError("Photon out of order")
    cdef unsigned long **state = <unsigned long**> PyMem_Malloc(num_burst * sizeof(unsigned long*))
    for i, path in enumerate(enum_arrays(state_path)):
        state[i] = np_copy_ul(path)
    cdef unsigned long nphot = sum(burst_sizes[i] for i in range(num_burst))
    cdef n_core = <unsigned long> optimization_limits._get_num_cores(num_cores)
    cdef double *ll = <double*> PyMem_Malloc(num_burst * sizeof(double))
    cdef res = pathloglik(num_burst, burst_sizes, b_delta, b_det, state, model.model, ll, n_core)
    # process values
    cdef double sum_loglik, bic
    cdef np.ndarray loglik
    cdef list out = list()
    if res == 0:
        loglik = copy_to_np_d(num_burst, ll).reshape(array_size(indexes))
        sum_loglik = np.sum(loglik)
        bic = (model.k * np.log(nphot)) - (2*sum_loglik)
    else:
        if ll is not NULL:
            PyMem_Free(ll)
            ll = NULL
    for i in range(num_burst):
        if state[i] is not NULL:
            PyMem_Free(state[i])
            state[i] = NULL
    if state is not NULL:
        PyMem_Free(state)
        state = NULL
    burst_free(num_burst, b_det, b_delta, burst_sizes)
    if res == -2:
        raise ValueError("Detector indexes out of range")
    elif res == -1:
        raise ValueError("Empty or null array, raise issue on github")
    elif res < -2:
        raise RuntimeError("Uknown error, raise issue on github")
    if BIC:
        out += [bic]
    if total_loglik:
        out += [sum_loglik]
    if loglikarray:
        out += [loglik]
    if len(out) == 0:
        return None
    elif len(out) == 1:
        return out[0]
    else:
        return tuple(out)


# simulation functions

def sim_statepath(h2mm_model hmod, int lenp, seed=None):
    """
    A function to generate a dense statepath (HMM as opposed to H2MM) of length lenp

    Parameters
    ----------
    hmod : h2mm_model
        An :class:`h2mm_model` to build the state path from
    lenp : int
        The length of the path you want to generate
    seed : positive int, optional
        The initial random seed, use if you want to be able to replicate results.
        If None, then uses the current time as the seed.
        The default is None.

    Raises
    ------
    RuntimeError
        A problem with the C code, this is for future proofing
    
    Returns
    -------
    path_ret : numpy ndarray, positive integer
        The random dense state-path

    """
    if lenp < 2:
        raise ValueError("Length must be at least 3")
    cdef unsigned long* path_n = <unsigned long*> PyMem_Malloc(lenp * sizeof(unsigned long))
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = statepath(hmod.model, lenp, path_n, seedp)
    if exp != 0:
        PyMem_Free(path_n)
        raise RuntimeError("Unknown error, raise issue on github")
    cdef np.ndarray path = copy_to_np_ul(lenp, path_n)
    return path


def sim_sparsestatepath(h2mm_model hmod, np.ndarray times, seed=None):
    """
    Generate a state path from a model and a sparse set of arrival times

    Parameters
    ----------
    hmod : h2mm_model
        An :class:`h2mm_model` to build the state path from
    times : numpy ndarray 1D int
        An unsigned integer of monotonically increasing times for the burst
    seed : positive int, optional
        The initial random seed, use if you want to be able to replicate results.
        If None, then uses the current time as the seed.
        The default is None.

    Raises
    ------
    TypeError
        Your array was not 1D
    ValueError
        Your time were not monotonically increasing
    RuntimeError
        Unknown error, please raise issue on github

    Returns
    -------
    path : numpy ndarray 1D long int
        A randomly generated statepath based on the input model and times

    """
    if times.ndim != 1:
        raise TypeError("times array must be 1D")
    if times.shape[0] < 3:
        raise ValueError("Must have at least 3 times")
    cdef unsigned long long* times_n = np_copy_ull(times)
    cdef unsigned long lenp = <unsigned long> times.shape[0]
    cdef unsigned long* path_n = <unsigned long*> PyMem_Malloc(lenp * sizeof(unsigned long))
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = sparsestatepath(hmod.model, lenp, times_n, path_n, seedp)
    PyMem_Free(times_n)
    if exp == 1:
        raise ValueError("Out of order photon")
    elif exp != 0:
        raise RuntimeError("Unknown error, raise issue on github")
    cdef np.ndarray path = copy_to_np_ul(lenp, path_n)
    return path


def sim_phtraj_from_state(h2mm_model hmod, np.ndarray states, seed=None):
    """
    Generate a photon trajectory from a h2mm model and state trajectory

    Parameters
    ----------
    hmod : h2mm_model
        An :class:`h2mm_model` to build the stream trajectory from
    states : numpy ndarray 1D int
        
    seed : positive int, optional
        The initial random seed, use if you want to be able to replicate results.
        If None, then uses the current time as the seed.
        The default is None.

    Raises
    ------
    TypeError
        The state array was not 1D
    RuntimeError
        Unknown error, please raise issue on github

    Returns
    -------
    dets : numpy ndarray 1D int
        The random photon stream indexes based on model and statepath

    """
    if states.ndim != 1:
        raise TypeError("Times must be 1D")
    if states.shape[0] < 3:
        raise ValueError("Must have at least 3 time points")
    cdef unsigned long lenp = states.shape[0]
    cdef unsigned long* states_n = np_copy_ul(states)
    cdef unsigned long* dets_n =  <unsigned long*> PyMem_Malloc(lenp * sizeof(unsigned long))
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = phpathgen(hmod.model, lenp, states_n, dets_n, seedp)
    PyMem_Free(states_n)
    if exp != 0:
        PyMem_Free(dets_n)
        raise RuntimeError("Unknown error, raise issue on github")
    cdef np.ndarray dets = copy_to_np_ul(lenp, dets_n)
    return dets


def sim_phtraj_from_times(h2mm_model hmod, np.ndarray times, seed=None):
    """
    Generate a state path and photon trajectory for a given set of times

    Parameters
    ----------
    hmod : h2mm_model
        An :class:`h2mm_model` to build the path and stream trajectories from
    times : numpy.ndarray 1D int
        An unsigned integer of monotonically increasing times for the burst
    seed : positive int, optional
        The initial random seed, use if you want to be able to replicate results.
        If None, then uses the current time as the seed.
        The default is None.

    Raises
    ------
    TypeError
        tiems array must be 1D
    ValueError
        times were not monotonically increasing
    RuntimeError
        Unknown error, raise issue on github
    
    Returns
    -------
    path : numpy.ndarray 1D int
        State trajectory generated for the input times
    dets : numpy.ndarray 1D int
        stream trajectory generate for the input times, derived from path

    """
    if times.ndim != 1:
        raise TypeError("times array must be 1D")
    if times.shape[0] < 2:
        raise ValueError("Must have at least 3 times")
    cdef unsigned long lenp = times.shape[0]
    cdef unsigned long long* times_n = np_copy_ull(times)
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef unsigned long* path_n = <unsigned long*> PyMem_Malloc(lenp * sizeof(unsigned long))
    cdef int exp = sparsestatepath(hmod.model, lenp, times_n, path_n, seedp)
    if exp == 1:
        PyMem_Free(times_n)
        PyMem_Free(path_n)
        raise ValueError("Out of order photon")
    elif exp != 0:
        PyMem_Free(times_n)
        PyMem_Free(path_n)
        raise RuntimeError("Unknown error in sparsestatepath, raise issue on github")
    cdef unsigned long* dets_n = <unsigned long*> PyMem_Malloc(lenp * sizeof(unsigned long))
    exp = phpathgen(hmod.model, lenp, path_n, dets_n, seedp)
    PyMem_Free(times_n)
    if exp != 0:
        PyMem_Free(path_n)
        PyMem_Free(dets_n)
        raise RuntimeError("Unknown error in phtragen, raise issue on github")
    cdef np.ndarray path = copy_to_np_ul(lenp, path_n)
    cdef np.ndarray dets = copy_to_np_ul(lenp, dets_n)
    return path , dets
