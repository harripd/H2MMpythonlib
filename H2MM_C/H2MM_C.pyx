# -*- coding: utf-8 -*-
#cython: language_level=3
# Created 20 Feb 2021
# Modified: 25 Oct 2022
#Author: Paul David Harris
"""
Module for analyzing data with photon by photon hidden Markov moddeling
"""
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF, Py_XDECREF
from libc.stdlib cimport malloc, calloc, realloc, free
from libc.stdint cimport int64_t, int32_t, uint8_t, INT32_MAX
from libc.string cimport memcpy
from cpython cimport array
cimport numpy as cnp

import os
import numpy as np
import array

import warnings
from collections.abc import Sequence
import sys


cnp.import_array()

cdef extern from "math.h" nogil:
    bint isnan(double x)

cdef extern from "C_H2MM.h" nogil:
    ctypedef struct lm:
        int64_t max_iter
        int64_t num_cores
        double max_time
        double min_conv
    ctypedef struct h2mm_mod:
        int64_t nstate
        int64_t ndet
        int64_t nphot
        int64_t niter
        int64_t conv
        double *prior
        double *trans
        double *obs
        double loglik
    ctypedef struct h2mm_minmax:
        h2mm_mod *mins
        h2mm_mod *maxs
    ctypedef struct ph_path:
        int64_t nphot
        int64_t nstate
        double loglik
        uint8_t *path
        double *scale
    int print_model(h2mm_mod* model)
    int copy_model(h2mm_mod *source, h2mm_mod *dest)
    int copy_model_vals(h2mm_mod *source, h2mm_mod *dest)
    int move_model_ptrs(h2mm_mod *source, h2mm_mod *dest)
    void h2mm_normalize(h2mm_mod *model_params)
    int h2mm_optimize(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*) noexcept with gil, void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*) noexcept with gil,void *print_call)
    int h2mm_optimize_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*) noexcept with gil, void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*) noexcept with gil,void *print_call)
    int h2mm_optimize_gamma(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod *out_model, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*) noexcept with gil, void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*) noexcept with gil,void *print_call)
    int h2mm_optimize_gamma_array(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *in_model, h2mm_mod **out_models, double ***gamma, lm *limits, int (*model_limits_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm*, void*) noexcept with gil, void *model_limits, int (*print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*) noexcept with gil,void *print_call)
    int viterbi(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, h2mm_mod *model, ph_path *path_array, int64_t num_cores)
    int baseprint(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *func)
    int calc_multi(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, int64_t num_models, h2mm_mod *models, lm *limits)
    int calc_multi_gamma(int64_t num_burst, int64_t *burst_sizes, int32_t **burst_deltas, uint8_t **burst_det, int64_t num_models, h2mm_mod *models, double ****gamma, lm *limits)
    int h2mm_check_converged(h2mm_mod * new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limits)
    int limit_check_only(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int limit_revert(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int limit_revert_old(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int limit_minmax(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double total_time, lm *limit, void *lims)
    int statepath(h2mm_mod* model, int64_t lent, uint8_t* path, unsigned int seed)
    int sparsestatepath(h2mm_mod* model, int64_t lent, int64_t* times, uint8_t* path, unsigned int seed)
    int phpathgen(h2mm_mod* model, int64_t lent, uint8_t* path, uint8_t* traj, unsigned int seed)
    int pathloglik(int64_t num_burst, int64_t *len_burst, int32_t **deltas, uint8_t ** dets, uint8_t **states, h2mm_mod *model, double *loglik, int64_t num_cores)
    int pathloglik_path(int64_t num_burst, int64_t *len_burst, int32_t **deltas, uint8_t ** dets, uint8_t **states, h2mm_mod *model, double **loglik, int64_t num_cores)
    
    int64_t CONVCODE_FROMOPT
    int64_t CONVCODE_LLCOMPUTED
    int64_t CONVCODE_OUTPUT
    int64_t CONVCODE_POSTMODEL
    int64_t CONVCODE_ERROR
    int64_t CONVCODE_CONVERGED
    int64_t CONVCODE_MAXITER
    int64_t CONVCODE_MAXTIME
    int64_t CONVCODE_FIXEDMODEL
    
    int64_t CONVCODE_OUTPUT_CONVERGED
    int64_t CONVCODE_OUTPUT_MAXITER
    int64_t CONVCODE_OUTPUT_MAXTIME
    int64_t CONVCODE_POST_CONVERGED
    int64_t CONVCODE_POST_MAXITER
    int64_t CONVCODE_POST_MAXTIME
    int64_t CONVCODE_ANYFINAL



ctypedef struct BoundStruct:
    PyObject *func
    PyObject *args
    PyObject *kwargs
    PyObject *error


ctypedef struct PrintStruct:
    bint keep
    int64_t disp_freq
    int64_t max_iter
    double max_time
    PyObject *formatter
    PyObject *func
    PyObject *args
    PyObject *kwargs
    PyObject *error


#: Version string
from sys import version_info as python_version
if python_version.major > 3 or python_version.minor > 7:
    from importlib.metadata import version, PackageNotFoundError
else:
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version("H2MM_C")
except PackageNotFoundError:
    print("cannot find package version")
del version, PackageNotFoundError, python_version


###############################################################################
########## Defining printer classes of Optimization Progress Output  ##########
###############################################################################
from abc import ABC, abstractmethod


class Printer(ABC):
    """
    Abstract base class for formatting progress display of optimizations.
    """
    @abstractmethod
    def update(self, text):
        """
        Method called to update output each iteration. Accepts output of
        print_funcion as only argument.
        """
        raise NotImplementedError("Must implement update")
    
    @abstractmethod
    def close(self):
        """
        Method called at end of :func:`H2MM_C.EM_H2MM_C`, used to terminate the
        output, for instance adding a newline character to output.
        """
        raise NotImplementedError("Must implement close")
    
    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()


class StdPrinter(Printer):
    __slots__ = ('buffer', 'width', 'keep', 'single_line')
    def __init__(self, buffer, keep=False, single_line=True):
        self.buffer = buffer
        self.width = 0
        self.keep = bool(keep)
        self.single_line = bool(single_line)
    
    def update(self, text):
        text = str(text)
        if self.single_line:
            text = text.replace('\n', ' ')
        if self.keep:
            text = text + '\n'
        else:
            ln = len(text)
            text = '\r' + text + ' '*(self.width - ln)
            if ln > self.width:
                self.width = ln
        self.buffer.write(text)
        self.buffer.flush()
    
    def close(self):
        self.buffer.write("\n")

has_ipython = False
has_matplotlib = False
try:
    from itertools import product, chain
    from IPython.display import DisplayHandle, Pretty, clear_output
    has_ipython = True
    class IPyPrinter(Printer):
        """
        Class for prining to :code:`IPython.display.DisplayHandle` object.
        
        Parameters
        ----------
        buffer: IPython.display.DisplayHandle
            DisplayHandle where output is displayed
        keep: bool, optional
            Whether to keep old output or replace with new. The default is False
        """
        def __init__(self, buffer, keep=False):
            self.buffer = buffer
            self.keep = bool(keep)
            self.text = Pretty("")
            self.buffer.display(self.text)
        
        def update(self, text):
            """
            Update display with text in text
            
            Parameters
            ----------
            text: str
                Text to display
            
            Returns
            -------
            None
            
            """
            test = str(text)
            if self.keep:
                self.text.data =self.text.data + '\n' + text if self.text.data else text
            else:
                self.text.data = text
            self.buffer.update(self.text)
        
        def close(self):
            pass
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
        import matplotlib
        
        class IPyPlot(Printer):
            """
            **Beta Feature**
            
            .. note::
                
                This class is subject to rapid iteration and/or removal in
                future minor version increments.
            
            Class for plotting optimization progress on matplotlib axes.
            
            Parameters
            ----------
            
            stream: DiplayHandle
                DisplayHandle for where plots will be displayed
            nrow: int, optional
                number of rows of plots in output, the default is 1
            ncol: int, optional
                number of columns of plots in output, the default is 1
            xlim: tuple[tuple[tuple[int|float|str, int|float|str],...],...]
                nrow x ncol x 2 nested tuples, identifying min and max values
                for x axis of given [row][col] plot. If 'min' or 'max', take
                said values in arrays. Defautl is for 1x1 'min', 'max' values
            ylim: tuple[tuple[tuple[int|float|str, int|float|str],...],...]
                nrow x ncol x 2 nested tuples, identifying min and max values
                for y axis of given [row][col] plot. If 'min' or 'max', take
                said values in arrays. Defautl is for 1x1 'min', 'max' values
            plot_kwargs: dict, optional
                dictionary of keword argumetns given to ax.plot on first iteration
                Default is empty dictionary.
            **kwargs: dict
                kwargs handed to plt.subplots when creating initial figure and plots.
            """
            __slots__ = ('stream', 'nrows', 'ncols', 'data', 'xlim', 'ylim', 'kwargs', 'plot_type', 'plot_kwargs')
            def __init__(self, stream, nrows=1, ncols=1, 
                         xlim=((('min', 'max'),),), ylim=((('min', 'max'),),),
                         plot_kwargs=None, plot_type=(('plot',),), **kwargs):
                if len(xlim) != nrows or len(ylim) != nrows:
                    raise ValueError("xlim and ylim must be nrow x ncol array")
                if any(len(x) != ncols or len(y) != ncols for x, y in zip(xlim, ylim)):
                    raise ValueError("xlim and ylim must be nrow x ncol x 2 array")
                if any(len(x) != 2 or len(y) != 2 for x, y in chain.from_iterable(zip(xx,yy) for xx, yy in zip(xlim, ylim))):
                    raise ValueError("xlim and ylim must be nrow x ncol x 2 array")
                self.stream = stream
                self.nrows = nrows
                self.ncols = ncols
                self.data = False
                self.xlim = xlim
                self.ylim = ylim
                self.kwargs = kwargs
                self.plot_type = plot_type
                self.plot_kwargs = plot_kwargs if plot_kwargs is not None else [[[dict()] for j in range(ncols)] for i in range(nrows)]
        
            def update(self, data):
                """
                Update plot with new data points

                Parameters
                ----------
                data : list[list[np.ndarray]]
                    An array-like nrow x ncol object, with each pair of elements
                    being an x-y value pair to plot.

                Returns
                -------
                None.

                """
                if isinstance(data, str):
                    return
                if self.data is False:
                    self.data = np.empty((self.nrows, self.ncols), dtype=np.object_)
                    for i, j in product(range(self.nrows), range(self.ncols)):
                        self.data[i,j] = data[i][j][np.newaxis]
                else:
                    for i, j in product(range(self.nrows), range(self.ncols)):
                        self.data[i,j] = np.concatenate([self.data[i,j], data[i][j][np.newaxis]])
                fig, ax = plt.subplots(self.nrows, self.ncols, **self.kwargs)
                ax = np.atleast_2d(ax)
                for i, j in product(range(self.nrows), range(self.ncols)):
                    for k in range(self.data[i,j].shape[1]//2):
                        if self.plot_type[i][j] == 'plot':
                            ax[i,j].plot(self.data[i,j][:,2*i], self.data[i,j][:,2*i+1])
                        else:
                            ax[i,j].scatter(self.data[i,j][:,2*i], self.data[i,j][:,2*i+1], **self.plot_kwargs[i][j][k])
                        xlim, ylim = self._get_xylim(i,j)
                        ax[i,j].set_xlim(xlim)
                        ax[i,j].set_ylim(ylim)
                if self.data.shape[0] == 1:
                    self.stream.display(fig)
                else:
                    clear_output()
                    self.stream.update(fig)
            
            def _get_xylim(self, i, j):
                xmin, xmax = self.xlim[i][j]
                if xmin == 'min':
                    xmin = np.min(self.data[i,j][:,::2])
                if xmax == 'max':
                    xmax = np.max(self.data[i,j][:,::2])
                ymin, ymax = self.ylim[i][j]
                if ymin == 'min':
                    ymin = np.min(self.data[i,j][:,1::2])
                if ymax == 'max':
                    ymax = np.max(self.data[i,j][:,1::2])
                if xmin == xmax:
                    xmin -= 1.0
                    xmax += 1.0
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                return (xmin, xmax), (ymin, ymax)
        
            def close(self):
                """
                Clear the display handle.

                Returns
                -------
                None.

                """
                clear_output()
                
        def plot_bic(niter, new, current, old, iter_time, total_time):
            """
            Printing function for :class:`IPyPlot`, plots BIC in single plot
            
            .. note::
                
                This class is subject to rapid iteration and/or removal in
                future minor version increments.
            """
            return [[np.array([niter, current.bic])]]
        
        def plot_obs(niter, new, current, old, iter_time, total_time, dx=0, dy=1):
            """
            Printing function for :class:`IPyPlot`, plots obs values in single plot.
            Takes 2 keyword arguments, dx, dy specifing index of det to plot
            on x and y axis for each state.
            
            .. note::
                
                This function is subject to rapid iteration and/or removal in
                future minor version increments.
            
            """
            return [[np.array([[current.obs[i,dx], current.obs[i,dy]] for i in range(current.nstate)]).reshape(-1)]]
        
        def plot_bic_obs(niter, new, current, old, iter_time, total_time, dx=0, dy=1):
            """
            Printing function for :class:`IPyPlot`, plots plot_bic in first plot
            plot_obs values in second plot.
            Takes 2 keyword arguments, dx, dy specifing index of det to plot
            on x and y axis for each state.
            
            .. note::
                
                This function is subject to rapid iteration and/or removal in
                future minor version increments.
            
            """
            return [[np.array([niter, current.bic]), np.array([[current.obs[i,dx], current.obs[i,dy]] for i in range(current.nstate)]).reshape(-1)]]

    except:
        has_matplotlib = False
except:
    has_ipython = False

###############################################################################
############################### Utility Classes ###############################
###############################################################################
cdef class opt_lim_const:
    """
    Special class for setting default optimization min/max values.
    Provides access to and type protected setting of these values, 
    either as attributes or keys.
    Attributes have same names as key-word arguments of functions.
    
    The optimization_limits module variable should be of this class
    
    """
    cdef:
        int64_t _max_iter
        double _max_time
        double _converged_min
        int64_t _num_cores
        object _formatter
        object _outstream
    def __cinit__(self, max_iter=2046, max_time=np.inf, converged_min=1e-14, 
                  num_cores=os.cpu_count()//2, formatter=StdPrinter,
                  outstream=lambda: sys.stdout):
        assert max_iter > 0 and np.issubdtype(type(max_iter), np.integer), ValueError("max_iter must be integer greater than 0")
        assert max_time > 0 and np.issubdtype(type(max_time), np.floating), ValueError("max_time must be float and greater than 0")
        assert converged_min > 0 and np.issubdtype(type(converged_min), np.floating), ValueError("converged_min must be float greater than 0")
        assert num_cores > 0 and np.issubdtype(type(num_cores), np.integer), ValueError("num_cores must be integer greater than 0")
        self._max_iter = <int64_t> max_iter
        self._max_time = <double> max_time
        self._converged_min = <double> converged_min
        self._num_cores = <int64_t> num_cores
        self._formatter = formatter
        self._outstream = outstream

    @property
    def max_iter(self):
        """Maximum number of iterations before ending optimization"""
        return self._max_iter
    @max_iter.setter
    def max_iter(self, max_iter):
        assert max_iter > 0 and np.issubdtype(type(max_iter), np.integer), ValueError("max_iter must be integer greater than 0")
        self._max_iter = <int64_t> max_iter

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
        self._num_cores = <int64_t> num_cores
    
    @property
    def formatter(self):
        """A :class:`Printer` subclass, used to format result of print_func for display of optimiation progress"""
        return self._formatter
    
    @formatter.setter
    def formatter(self, formatter):
        self._formatter = formatter

    @property
    def outstream(self):
        """Callable, when called, generates an output stream for where to print displayed progress"""
        return self._outstream
    
    @outstream.setter
    def outstream(self, outstream):
        self._outstream = outstream

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __repr__(self):
        return (f'Optimization limits:: num_cores: {self._num_cores}, max_iter: {self._max_iter}, '
                f'converged_min: {self._converged_min}, max_time: {self._max_time},\n'
                f'formatter: {self._formatter.__qualname__}, outstream={self._outstream}')
    
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
            n_core = <int64_t> num_cores
        elif num_cores is None:
            n_core = self._num_cores
        else:
            raise ValueError(f"Non-int options for num_cores are 'single' or 'multi', cannot process {num_cores}")
        return n_core
    
    def _get_formatter(self, formatter):
        """
        Get the formatter based on the value form kwargs.

        Parameters
        ----------
        formatter : type | None
            Value given in keword argument for print_formatter.

        Returns
        -------
        type
            The class of :class:`Printer` to be used for display of output.

        """
        if formatter is None:
            return self._formatter
        return formatter
    
    def _get_outstream(self, outstream):
        """
        Ge the output stream based on keyword argument to print_stream

        Parameters
        ----------
        outstream : Output Stream
            Stream .

        Returns
        -------
        OutputStream
            DESCRIPTION.

        """
        if outstream is None:
            return self._outstream()
        return outstream

def get_stdout():
    """Simple callable returns sys.stdout"""
    return sys.stdout

#: Like rcParams, a place to set defaults for optimization limits
optimization_limits = opt_lim_const(max_iter=2046, max_time=np.inf, 
                                    converged_min=1e-14, num_cores=os.cpu_count()//2,
                                    formatter=StdPrinter, outstream=get_stdout)

cdef class ConvCodes:
    """Bitmask for conv_code flags"""
    @property
    def ll_computed(self):
        """Model has had loglik computed against some data (either during optimization or singly"""
        return CONVCODE_LLCOMPUTED
    @property
    def from_opt(self):
        """Model was created during optimization"""
        return CONVCODE_FROMOPT
    @property
    def output(self):
        """Model is best model during optimization, by some criterion marked by other flags"""
        return CONVCODE_OUTPUT
    @property
    def converged(self):
        """Model has best logliklihood within converged_min threshold"""
        return CONVCODE_CONVERGED
    @property
    def max_iter(self):
        """Optimization reached maximum number of iterations"""
        return CONVCODE_MAXITER
    @property
    def max_time(self):
        """Optimization reach maximum time"""
        return CONVCODE_MAXTIME
    @property
    def error(self):
        """Error occured during optimization"""
        return CONVCODE_ERROR
    @property
    def post_opt(self):
        """
        Model is either the model with poorer loglikelihood between current 
        and old, or is new and loglik is not calculated
        """
        return CONVCODE_POSTMODEL
    @property
    def frozen(self): 
        """Model cannot be modified in place"""
        return CONVCODE_FIXEDMODEL

convcode = ConvCodes()


###############################################################################
######################## Alloc/Free utility functions  ########################
###############################################################################
cdef int Py_free_model_fields(h2mm_mod *model):
    if model is NULL:
        return 0
    if model.prior is not NULL:
        PyMem_Free(model.prior)
        model.prior = NULL
    if model.trans is not NULL:
        PyMem_Free(model.trans)
        model.trans = NULL
    if model.obs is not NULL:
        PyMem_Free(model.obs)
        model.obs = NULL
    model.ndet = 0
    model.nstate = 0
    model.nphot = 0
    model.loglik = 0.0
    model.conv = 0
    return 0


cdef int Py_free_models(const int64_t nmodels, h2mm_mod *models):
    if models is NULL:
        return 0
    cdef int64_t i
    for i in range(nmodels):
        Py_free_model_fields(&models[i])
    PyMem_Free(models)
    return 0
        

cdef h2mm_mod* Py_allocate_models(const int64_t nmodel, const int64_t nstate, const int64_t ndet, const int64_t nphot):
    cdef h2mm_mod *out = <h2mm_mod*> PyMem_Malloc(nmodel*sizeof(h2mm_mod))
    if out is NULL:
        return out
    cdef int64_t i
    for i in range(nmodel):
        out[i].loglik = 0.0
        out[i].niter = 0
        out[i].conv = 0
        out[i].prior = <double*> PyMem_Malloc(nstate*sizeof(double))
        if out[i].prior is NULL:
            Py_free_models(i, out)
            out = NULL
            return out
        out[i].trans = <double*> PyMem_Malloc(nstate*nstate*sizeof(double))
        if out[i].trans is NULL:
            PyMem_Free(out[i].prior)
            out[i].prior = NULL
            Py_free_models(i, out)
            out = NULL
            return out
        out[i].obs = <double*> PyMem_Malloc(nstate*ndet*sizeof(double))
        if out[i].prior is NULL:
            PyMem_Free(out[i].prior)
            out[i].prior = NULL
            PyMem_Free(out[i].trans)
            out[i].trans = NULL
            Py_free_models(i, out)
            out = NULL
            return out
        out[i].nstate = nstate
        out[i].ndet = ndet
        out[i].nphot = nphot
        out[i].niter = 0
        out[i].loglik = 0.0
        out[i].conv = 0
    return out


cdef str get_conv_convtype(int64_t conv):
    cdef str out = str()
    if conv & CONVCODE_CONVERGED:
        out += ' converged'
    if conv & CONVCODE_MAXITER:
        out += ' maximum iterations'
    if conv & CONVCODE_MAXTIME:
        out += ' maximum time'
    if conv & CONVCODE_ERROR:
        if conv & CONVCODE_OUTPUT:
            out += ' error in next iteration' 
        else:
            out += ' error in current calculation'
    return out.strip(' ')

cdef str get_conv_str(int64_t conv):
    cdef str out = str()
    if conv & CONVCODE_FROMOPT:
        if conv & CONVCODE_OUTPUT:
            return 'final optimized model by ' + get_conv_convtype(conv)    
        if conv & CONVCODE_POSTMODEL:
            out = 'post-optimization LL'
            if not (conv & CONVCODE_LLCOMPUTED):
                out += 'un-'
            out += 'copmuted model by ' + get_conv_convtype(conv)
            return out
        out = 'mid-optimization '
        if not (conv & CONVCODE_LLCOMPUTED):
            out += 'un-'
        out += 'computed mode'
        return out
    if conv & CONVCODE_LLCOMPUTED:
        out = 'singly-computed'
        if conv & CONVCODE_ERROR:
            out += ' with error'
        return out
    return 'newly created model'


cdef str get_conv_descr(int64_t conv):
    cdef str out = get_conv_str(conv)
    if conv & CONVCODE_FIXEDMODEL:
        if conv & ~(CONVCODE_FIXEDMODEL):
            return 'hashed ' + out
        return 'hashed model'
    return out


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
        
        These are generally only specified when trying to re-create a model from, for
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

    def __init__(self, prior, trans, obs, loglik=-np.inf, niter=0, nphot = 0, is_conv=False):
        # if statements check to confirm first the correct dimensinality of input matrices, then that their shapes match
        cdef int64_t i, j
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
        if not np.issubdtype(type(niter), np.integer):
            raise TypeError("niter must be integer")
        if niter < 0:
            raise ValueError("niter must be positive")
        if nphot< 0: 
            raise ValueError("nphot must be positive")
        if not isinstance(is_conv, int):
            raise TypeError("is_conv must be boolean or int")
        elif isinstance(is_conv, bool):
            is_conv = CONVCODE_OUTPUT_CONVERGED
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
        self.model = Py_allocate_models(1, obs.shape[0], obs.shape[1], <int64_t> nphot)
        if loglik == -np.inf:
            self.model.conv = 0
            self.model.loglik = <double> loglik
        else:
            self.model.conv = is_conv
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
            Py_free_models(1, self.model)

    @staticmethod
    cdef h2mm_model from_ptr(h2mm_mod *model):
        cdef h2mm_model new = h2mm_model.__new__(h2mm_model)
        new.model = model
        return new

    @staticmethod
    cdef h2mm_model move_ptr(h2mm_mod *model):
        cdef h2mm_model new = h2mm_model.__new__(h2mm_model)
        new.model = <h2mm_mod*> PyMem_Malloc(sizeof(h2mm_mod))
        new.model.prior = NULL
        new.model.trans = NULL
        new.model.obs = NULL
        move_model_ptrs(model, new.model)
        return new

    def tobytes(self):
        """
        Serialize :class:`h2mm_model` into bytes, which can be read back into
        :class:`h2mm_model` with :meth:`h2mm_model.frombytes`.
        
        Returns
        -------
        bytes
            Serialized bytes represenation of :class:`h2mm_model`. Includes
            all information contained within model.
            
        """
        cdef array.array arr = array.array('b', [])
        array.resize(arr, 5*sizeof(int64_t)+(self.model.nstate*(1+self.model.nstate+self.model.ndet)+1)*sizeof(double))
        memcpy(arr.data.as_chars, <char*>&self.model.nstate, sizeof(int64_t))
        memcpy(arr.data.as_chars+sizeof(int64_t), <char*>&self.model.ndet, sizeof(int64_t))
        memcpy(arr.data.as_chars+sizeof(int64_t)*2, <char*>&self.model.nphot, sizeof(int64_t))
        memcpy(arr.data.as_chars+sizeof(int64_t)*3, <char*>&self.model.niter, sizeof(int64_t))
        memcpy(arr.data.as_chars+sizeof(int64_t)*4, <char*>&self.model.conv, sizeof(int64_t))
        memcpy(arr.data.as_chars+sizeof(int64_t)*5, <char*>&self.model.loglik, sizeof(double))
        memcpy(arr.data.as_chars+sizeof(int64_t)*5+sizeof(double), <char*>self.model.prior, self.model.nstate*sizeof(double))
        memcpy(arr.data.as_chars+sizeof(int64_t)*5+sizeof(double)+self.model.nstate*sizeof(double), <char*>self.model.trans, (self.model.nstate**2)*sizeof(double))
        memcpy(arr.data.as_chars+sizeof(int64_t)*5+sizeof(double)+self.model.nstate*(self.model.nstate+1)*sizeof(double), <char*>self.model.obs, self.model.ndet*self.model.nstate*sizeof(double))
        return arr.tobytes()

    @classmethod
    def frombytes(cls, buffer):
        """
        Create :class:`h2mm_model` from a bytes object generated by
        :meth:`h2mm_model.toobytes`.
        

        Parameters
        ----------
        buffer : bytes
            Bytes representation of :class:`h2mm_model` object. Should have
            originated from :meth:`h2mm_model.tobytes`.

        Raises
        ------
        ValueError
            Invalid bytes represenation of :class:`h2mm_model`, data either was
            never a :class:`h2mm_model` or was corrupted.

        Returns
        -------
        h2mm_model
            Data converted to :class:`h2mm_model`

        """
        cdef array.array buf = array.array('b', buffer)
        cdef Py_ssize_t ln = len(buf)
        if ln < (sizeof(int64_t)*5 + sizeof(double)):
            raise ValueError("bytes too small to read as h2mm_model")
        cdef int64_t nstate
        cdef int64_t ndet
        cdef int64_t nphot
        memcpy(&nstate, buf.data.as_chars, sizeof(int64_t))
        memcpy(&ndet, buf.data.as_chars+sizeof(int64_t), sizeof(int64_t))
        memcpy(&nphot, buf.data.as_chars+sizeof(int64_t)*2, sizeof(int64_t))
        cdef Py_ssize_t eln = 5*sizeof(int64_t)+sizeof(double)+(1+nstate+ndet)*nstate*sizeof(double)
        if eln != ln:
            raise ValueError(f"mallformed h2mm_model bytes array, specified {nstate} states and {ndet} dets, meaning buffer size should be {eln}, but buffer has size {ln}")
        cdef h2mm_model new = h2mm_model.__new__(h2mm_model)
        new.model = Py_allocate_models(1, nstate, ndet, nphot)
        memcpy(<void*>&new.model.niter, buf.data.as_chars+sizeof(int64_t)*3, sizeof(int64_t))
        memcpy(<void*>&new.model.conv, buf.data.as_chars+sizeof(int64_t)*4, sizeof(int64_t))
        memcpy(<void*>&new.model.loglik, buf.data.as_chars+sizeof(int64_t)*5, sizeof(double))
        memcpy(<void*>new.model.prior, buf.data.as_chars+sizeof(int64_t)*5+sizeof(double), new.model.nstate*sizeof(double))
        memcpy(<void*>new.model.trans, buf.data.as_chars+sizeof(int64_t)*5+sizeof(double)+sizeof(double)*new.model.nstate, new.model.nstate*new.model.nstate*sizeof(double))
        memcpy(<void*>new.model.obs, buf.data.as_chars+5*sizeof(int64_t)+sizeof(double)+(1+new.model.nstate)*new.model.nstate*sizeof(double), new.model.nstate*new.model.ndet*sizeof(double))
        return new

    def __repr__(self):
        cdef int64_t i, j
        msg = f"nstate: {self.model.nstate}, ndet: {self.model.ndet}, nphot: {self.model.nphot}, niter: {self.model.niter}, loglik: {self.model.loglik} converged state: {hex(self.model.conv)}\n"
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
        out = get_conv_descr(self.model.conv) + f' states={self.model.nstate}, streams={self.model.ndet}'
        if self.model.conv & CONVCODE_LLCOMPUTED:
            out += f', loglik: {self.model.loglik}'
        return out
        
    # a number of property defs so that the values are accesible from python
    @property
    def prior(self):
        """Prior probability matrix, 1D, size nstate"""
        return np.asarray(<double[:self.model.nstate]>self.model.prior).copy()
    
    @prior.setter
    def prior(self,prior):
        if self.model.conv & CONVCODE_FIXEDMODEL:
            raise AttributeError("Fixed models cannot be altered")
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
    def trans(self, trans):
        if self.model.conv & CONVCODE_FIXEDMODEL:
            raise AttributeError("Fixed models cannot be altered")
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
        if self.model.conv & CONVCODE_FIXEDMODEL:
            raise AttributeError("Fixed models cannot be altered")
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
        if self.model.conv & CONVCODE_OUTPUT:
            return True
        return False

    @property
    def is_opt(self):
        """Whether or not the model has undergone optimization, as opposed to evaluation or being an initial model"""
        if self.model.conv & CONVCODE_FROMOPT:
            return True if self.model.niter != 0 else False
        return False
    
    @property
    def is_calc(self):
        """If model has been optimized/eveluated, vs being an initial model"""
        if np.isnan(self.model.loglik) or self.model.loglik == -np.inf or self.model.loglik == 0 or self.model.nphot == 0:
            return False
        return True
    
    @property
    def conv_code(self):
        """The convergence code of model, an int"""
        return self.model.conv
    
    @property
    def conv_str(self):
        """String description of how model optimization/calculation ended"""
        return get_conv_descr(self.model.conv)
    
    @property
    def niter(self):
        """Number of iterations optimization took"""
        return self.model.niter
    
    @niter.setter
    def niter(self,niter):
        if self.model.conv & CONVCODE_FIXEDMODEL:
            raise AttributeError("Cannot set niter for this type of model")
        if niter <= 0:
            raise ValueError("Cannot have negative iterations")
        self.model.niter = niter
    
    def set_converged(self, converged=True):
        """Modify model to mark as converged or not"""
        converged = bool(converged)
        if self.model.nphot == 0 or self.model.loglik == 0 or self.model.loglik == np.inf or np.isnan(self.model.loglik):
            raise Exception("Model uninitialized with data, cannot set converged")
        if self.model.conv & CONVCODE_FIXEDMODEL:
            raise Exception("Fixed model, cannot mutate converged state")
        if converged:
            self.model.conv |= CONVCODE_OUTPUT_CONVERGED
        else:
            self.model.conv &= ~(CONVCODE_OUTPUT|CONVCODE_ANYFINAL)

    def sort_states(self):
        """Return model with states sorted by values in prior, and set as 'hashable'"""
        srt = np.argsort(self.prior)
        prior = self.prior[srt]
        trans = self.trans[srt]
        trans = trans[:,srt]
        obs = self.obs[srt]
        cdef h2mm_model out=  h2mm_model(prior, trans, obs, loglik=self.model.loglik, niter=self.model.niter, nphot=self.model.nphot)
        out.model.conv |= CONVCODE_FIXEDMODEL
        return out
    
    def __eq__(self, other):
        if not isinstance(other, h2mm_model):
            return False
        cdef h2mm_model sf = self if self.model.conv == 8 else self.sort_states()
        cdef h2mm_model ot = other if other.conv_code == 8 else other.sort_states()
        if sf.model.nstate != ot.model.nstate or sf.model.ndet != ot.model.ndet:
            return False
        cdef int i
        cdef int j
        for i in range(sf.model.nstate):
            if sf.model.prior[i] != ot.model.prior[i]:
                return False
            for j in range(sf.model.nstate):
                if sf.model.trans[i*sf.model.nstate+j] != ot.model.trans[i*sf.model.nstate+j]:
                    return False
            for j in range(sf.model.ndet):
                if sf.model.obs[j*sf.model.nstate+i] != sf.model.obs[j*sf.model.nstate+i]:
                    return False
        return True
    
    def normalize(self):
        """For internal use, ensures all model arrays are row stochastic"""
        if not self.model.conv & CONVCODE_FIXEDMODEL:
            h2mm_normalize(self.model)
    
    def copy(self):
        """ Make a duplicate model in new memory"""
        return model_copy_from_ptr(self.model)
    
    def __copy__(self):
        return self.copy()
    
    def __hash__(self):
        if not (self.model.conv & CONVCODE_FIXEDMODEL):
            raise TypeError("unhashable model, must sort_states first")
        p = tuple(self.model.prior[i] for i in range(self.model.nstate))
        t = tuple(self.model.trans[i] for i in range(self.model.nstate**2))
        o = tuple(self.model.obs[i] for i in range(self.model.nstate*self.model.ndet))
        return hash(p+t+o)
    
    def optimize(self, indexes, times, max_iter=None, 
                  bounds_func=None, bounds=None, bounds_kwargs=None,
                  print_func='iter', print_freq=1, print_args=None, print_kwargs=None,
                  print_stream=None, print_formatter=None,
                  print_fmt_args=None, print_fmt_kwargs=None,
                  max_time=np.inf, converged_min=None, num_cores=None, 
                  reset_niter=True, gamma=False, opt_array=False, inplace=False):
        """
        Optimize the H2MM model for the given set of data.
        
        .. note::
            
            This method calls the :func:`EM_H2MM_C` function.
    
        Parameters
        ----------
        model : h2mm_model
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
        
        max_iter : int or None, optional
            the maximum number of iterations to conduct before returning the current
            :class:`h2mm_model`, if None (default) use value from optimization_limits. 
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
                    
                    must be a python function that takes the signature
                    ``bounds_func(new, current, old, *bounds, **bounds_kwargs)``
                    ``new``, ``current``, and ``old`` are :class:`h2mm_model` objects, 
                    with ``new`` being the model whose loglik has yet to be calculated
                    resulting from the latest iteration, ``current`` being the model
                    whose loglik was just calculated, and ``old`` being the result
                    of the previous iteration. Be one of the following: :class:`h2mm_model`,
                    :code:`bool`, :code:`int`, or a 2-tuple of :class:`h2mm_model` and
                    either :code:`bool` or :code:`int`. In all cases, :code:`int` values
                    must 0, 1, or 2. :class:`h2mm_model` objects will be used as the
                    next model to compute the loglik of. :class:`bool` and :code:`int`
                    objects determine if optimization will continue.
                    If :code:`True`, then optimization will continue, if :code:`False`,
                    then optimization will cease, and the ``old`` model will be returned
                    as the ideal model. If :code:`0`, then optimization will continue,
                    if :code:`1`, then return ``old`` model as optimal model, if 
                    :code:`2`, then return ``current`` model as optimizal model.
        
        bounds : h2mm_limits, tuple, or None, optional
            Argument(s) pass to bounds_func function. If bounds_func is a string,
            then bounds **must** be specified, and **must** be a :class:`h2mm_limits`
            object.
            
            If ``bounds_func`` is callable, and bounds is not None or a tuple, bounds
            is passed as the 4th argument to bounds_func, ie 
            :code:`bounds_func(new, current, old, bounds, **bounds_kwargs)`, 
            if bounds is a tuple, then passed as \*args, 
            ie :code:`bounds_func(new, current, old, *bounds, **bounds_kwargs)`
            Default is None
        
        bounds_kwargs : dict, or None, optional
            Only used when ``bounds_func`` is callable, passed as \*\*kwargs to
            ``bounds_func`` ie ie :code:`bounds_func(new, current, old, *bounds, **bounds_kwargs)`.
            If :code:`None`
            The default is None
        
        print_func : None, str or callable, optional
            Specifies how the results of each iteration will be displayed, several 
            strings specify built-in functions. Default is 'iter'
            Acceptable inputs: str, Callable or None
                
                **None**\:
                
                    causes no printout anywhere of the results of each iteration
                    
                Str: 'all', 'diff', 'comp', 'iter'
                
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
                    as the function will be handed ``(niter, new, current, old, iter_time, total_time)``
                    from a special Cython wrapper. iter_time and total_time are the times
                    of the latest iteration and total time of optimization, respectively.
                    Note that iter_time and total_time are based on the fast, but
                    inaccurate C clock function.
        
        print_freq: int, optional
            Number of iterations between updating display. Default is 1
        
        print_args : tuple or None, optional
            Only used when ``print_func`` is callable. Passed as final argument to
            ``print_func`` if not :code:`None` or :code:`tuple`, ie
            ``print_func(niter, new, current, old, iter_time, total_time, print_args, **print_kwargs)``
            if :code:`tuple`, then passed as \*args, ie
            ``print_func(niter, new, current, old, iter_time, total_time, *print_args, **print_kwargs)``
            If :code:`None`, then ignored. The default is :code:`None`.
        
        print_kwargs: dict or None, optional
            Only used when ``print_func`` is callable, passed as \*\*kwargs to ``print_func``
            ie ``print_func(niter, new, current, old, iter_time, time_total, *print_args, **print_kwargs)``.
            If :code:`None`, then ignored. The default is :code:`None`.
        
        print_stream : OutStream, optional
            Typically OutStream, the stream where the output will be displayed.
            Passed as first argument to print_formatter, if :code:`None`, then use
            stream specified in optimization_limites. Default is None
        
        print_formatter : Printer, optional
            Class object (usually subclass of :class:`Printer`) that formats
            text for output to ``print_stream``. Instance of ``print_formatter`` class
            will be created with 
            ``fmtr = print_formatter(print_stream, *print_fmt_args, **print_fmt_kwargs)``
            to use for printing. Each print will call ``fmtr.update(text)`` where
            ``text`` is output of ``print_func`` call. At end of optimization,
            ``fmtr.close()`` will be called. If ``print_formatter`` is :code:`None`
            then will use the formatter specified in ``optimization_limits``.
            The default is :code:`None`.
        
        print_fmt_args : tuple, Any or None, optional
            Additional arguments passed to ``print_formatter``, if None, ignored,
            if not tuple, then treated as single argument. The default is :code:`None`.
        
        print_fmt_kwargs : dict or None, optional
            Additional  keyword arguments passed to ``print_formatter``, if None, ignored.
            The default is :code:`None`.
            
        max_time : float or None, optional
            The maximum time (in seconds) before returning current model
            **NOTE** this uses the inaccurate C clock, which is often longer than the
            actual time. If :code:`None` (default) use value from optimization_limits. 
            Default is :code:`None`.
        
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
        
        opt_array : bool | int, optional
            Defines how the result is returned, if False (default) (or 0) then return only
            the ideal model, if True (or 1), then a numpy array is returned containing all
            models during the optimization, with the last model being the optimial model.
            If 2 then return all models, including models after converged.
            Default is False
        
        inplace : bool, optional
            Whether or not to store the optimized model in the current model object.
            
            .. note::
                
                When opt_array is True, then the ideal model is copied into the
                current model object, while the return value contains all models
                
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
        if self.model.conv & CONVCODE_FIXEDMODEL and inplace:
            raise TypeError("cannot inpnlace optimize fixed hashable model")
        cdef h2mm_model out_model
        if max_iter is None:
            max_iter = optimization_limits.max_iter
        if self.model.conv == 4 and self.model.niter >= max_iter:
            max_iter = self.model.niter + max_iter
        out = EM_H2MM_C(self, indexes, times, max_iter=max_iter, 
                      bounds_func=bounds_func, bounds=bounds, bounds_kwargs=bounds_kwargs,
                      print_func=print_func, print_freq=print_freq, print_args=print_args, print_kwargs=print_kwargs,
                      print_stream=print_stream, print_formatter=print_formatter,
                      print_fmt_args=print_fmt_args, print_fmt_kwargs=print_fmt_kwargs,
                      max_time=max_time, converged_min=converged_min, num_cores=num_cores, 
                      reset_niter=reset_niter, gamma=gamma, opt_array=opt_array)
        if inplace:
            # separate the models from gamma
            if gamma:
                out_arr, gamma = out
            else:
                out_arr = out
            # find the ideal model
            if opt_array:
                for i in range(out_arr.size-1, -1, -1):
                    if out_arr[i].conv_code & CONVCODE_OUTPUT:
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
            if self.model.conv & CONVCODE_FIXEDMODEL:
                raise TypeError("cannot inplace evaluate fixed model")
            copy_model(out_model.model, self.model)
        return out
    

cdef class _h2mm_lims:
    # hidden type for making a C-level h2mm_limits object
    cdef:
        h2mm_minmax limits
    def __cinit__(self, h2mm_model model, cnp.ndarray[double,ndim=1] min_prior, cnp.ndarray[double,ndim=1] max_prior, 
                  cnp.ndarray[double,ndim=2] min_trans, cnp.ndarray[double,ndim=2] max_trans, 
                  cnp.ndarray[double,ndim=2] min_obs, cnp.ndarray[double,ndim=2] max_obs):
        cdef int64_t i, j
        cdef int64_t nstate = model.model.nstate
        cdef int64_t ndet = model.model.ndet
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
            if isinstance(param,float) or isinstance(param, cnp.ndarray):
                none_kwargs = False
                if isinstance(param, cnp.ndarray):
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
            elif isinstance(min_trans, cnp.ndarray):
                if np.any(min_trans[np.eye(model.nstate)==0] > model.trans[np.eye(model.nstate)==0]):
                    raise ValueError("model trans out of range of min/max trans values")
            if isinstance(max_trans,float):
                if np.any(max_trans < model.trans[np.eye(model.nstate)==0]):
                    raise ValueError("model trans out of range of min/max trans values")
            elif isinstance(max_trans, cnp.ndarray):
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
    
    def make_model(self, model, warning=True):
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
        elif isinstance(self.min_prior, cnp.ndarray) and self.min_prior.ndim == 1 and self.min_prior.shape[0] == nstate:
            min_prior = self.min_prior.astype('double')
        else:
            raise Exception("Type of min_prior changed")
        if self.max_prior is None:
            max_prior = np.ones(nstate).astype('double')
        elif isinstance(self.min_prior,float):
            max_prior = (self.max_prior *np.ones(nstate)).astype('double')
        elif isinstance(self.max_prior, cnp.ndarray) and self.max_prior.ndim == 1 and self.max_prior.shape[0] == nstate:
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
        elif isinstance(self.min_trans, cnp.ndarray) and self.min_trans.ndim == 2 and self.min_trans.shape[0] == self.min_trans.shape[1] == nstate:
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
        elif isinstance(self.max_trans, cnp.ndarray) and self.max_trans.shape[0] == self.max_trans.shape[1] == nstate:
            max_trans = self.max_trans.astype('double')
        else:
            raise Exception("Type of max_trans changed")
        if np.any(min_trans > max_trans):
            raise ValueError("min_trans cannot be greater than max_trans")
        if self.min_obs is None:
            min_obs = np.zeros((nstate,ndet)).astype('double')
        elif isinstance(self.min_obs,float):
            min_obs = (self.min_obs * np.ones((nstate,ndet))).astype('double')
        elif isinstance(self.min_obs, cnp.ndarray) and self.min_obs.ndim == 2 and self.min_obs.shape[0] == nstate and self.min_obs.shape[1] == ndet:
            min_obs = self.min_obs.astype('double')
        else:
            raise Exception("Type of min_obs changed")
        if np.any(min_obs.sum(axis=1) > 1.0):
            raise ValueError("min_obs disallows row stochastic matrix")
        if self.max_obs is None:
            min_obs = np.ones((nstate,ndet)).astype('double')
        elif isinstance(self.max_obs,float):
            max_obs = (self.max_obs * np.ones((nstate,ndet))).astype('double')
        elif isinstance(self.max_obs, cnp.ndarray) and self.max_obs.ndim == 2 and self.max_obs.shape[0] == nstate and self.max_obs.shape[1] == ndet:
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
        return min_prior, max_prior, min_trans, max_trans, min_obs, max_obs
    
    def _make_model(self, h2mm_model model):
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
        return _h2mm_lims(model,  *self.make_model(model))


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
    cdef cnp.ndarray[double,ndim=1] prior, min_prior, max_prior
    cdef cnp.ndarray[double,ndim=2] trans, min_trans, max_trans
    cdef cnp.ndarray[double,ndim=2] obs, min_obs, max_obs
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


cdef h2mm_model model_copy_from_ptr(h2mm_mod *model):
    # function for copying the values h2mm_mod C structure into an new h2mm_model object
    # this is slower that model_from _ptr, but the copy makes sure that changes
    # to the h2mm_model object do not change the original pointer
    # primarily used in the cy_limit function to create the model objects handed
    # to the user supplied python function
    cdef h2mm_mod *ret_model = Py_allocate_models(1, model.nstate, model.ndet, model.nphot)
    copy_model(model, ret_model)
    return h2mm_model.from_ptr(ret_model)


# The wrapper for the user supplied python limits function
cdef int cy_limit(h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double time, lm* limits, void *instruct) noexcept with gil:
    # initial checks ensuring model does not optimize too long or have loglik error
    new.niter = current.niter + 1
    h2mm_normalize(new)
    if current.conv & CONVCODE_ERROR:
        old.conv |= CONVCODE_ERROR | CONVCODE_OUTPUT
        current.conv |= CONVCODE_POSTMODEL
        new.conv |= CONVCODE_ERROR | CONVCODE_POSTMODEL
        return 1
    if current.niter >= limits.max_iter:
        current.conv |= CONVCODE_OUTPUT_MAXITER
        new.conv |= CONVCODE_POST_MAXITER
        return 2
    if time > limits.max_time:
        current.conv |= CONVCODE_OUTPUT_MAXTIME
        new.conv |= CONVCODE_POST_MAXTIME
        return 2
    cdef int ret
    cdef h2mm_model limit_model
    cdef h2mm_model old_mod = model_copy_from_ptr(old)
    cdef h2mm_model cur_mod = model_copy_from_ptr(current)
    cdef h2mm_model new_mod = model_copy_from_ptr(new)
    cdef BoundStruct *bound = <BoundStruct*> instruct
    cdef object func = <object> bound.func
    cdef object args = <object> bound.args
    cdef object kwargs = <object> bound.kwargs
    # execute the limit function
    try:
        limit_res = func(new_mod, cur_mod, old_mod, *args, **kwargs)
    except Exception as e:
        current.conv |= CONVCODE_ERROR
        Py_INCREF(e)
        bound.error = <PyObject*> e
        return -5
    if not isinstance(limit_res, (list, tuple, h2mm_model, cnp.ndarray, int)):
        current.conv |= CONVCODE_ERROR
        return -4
    elif isinstance(limit_res, bool):
        if limit_res:
            old.conv |= CONVCODE_OUTPUT_CONVERGED
            current.conv |= CONVCODE_POST_CONVERGED
            return 1
        else:
            return 0
    elif isinstance(limit_res, int):
        if limit_res == 0:
            return 0
        elif limit_res == 1:
            old.conv |= CONVCODE_OUTPUT_CONVERGED
            current.conv |= CONVCODE_POST_CONVERGED
            new.conv |= CONVCODE_POST_CONVERGED
            return 1
        elif limit_res == 2:
            current.conv |= CONVCODE_OUTPUT_CONVERGED
            new.conv |= CONVCODE_POST_CONVERGED
            return 2
        else:
            current.conv |= CONVCODE_ERROR
            return -4
    elif isinstance(limit_res, h2mm_model): # when limit function only returns h2mm_model
        limit_model = limit_res
        # check that return model is valid
        if limit_model.model.ndet != current.ndet or limit_model.model.nstate != current.nstate:
            current.conv |= CONVCODE_ERROR
            new.conv |= CONVCODE_ERROR
            return -3
        ret = h2mm_check_converged(new, current, old, time, limits)
        if ret != 0:
            return ret
        else:
            copy_model_vals(limit_model.model, new)
            return ret
    # if the function returns 2 values, run longer processing code
    elif isinstance(limit_res, (list, tuple, cnp.ndarray)):
        # check deeper validity of return value
        if len(limit_res) != 2 or not isinstance(limit_res[0], h2mm_model) or not isinstance(limit_res[1], (bool, int)) or limit_res[1] > 2:
            current.conv |= CONVCODE_ERROR
            new.conv |= CONVCODE_ERROR
            return -4
        limit_model = limit_res[0]
        if limit_model.model.ndet != current.ndet or limit_model.model.nstate != current.nstate: # bad return model
            current.conv |= CONVCODE_ERROR
            new.conv |= CONVCODE_ERROR
            return -3
        copy_model_vals(limit_model.model, new)
        if isinstance(limit_res[1], bool):
            if limit_res[1]:
                return 0
            else:
                old.conv |= CONVCODE_OUTPUT_CONVERGED
                current.conv |= CONVCODE_POST_CONVERGED
                return 1
        else:
            ret = limit_res[1]
            if ret == 0:
                pass
            elif ret == 1:
                old.conv |= CONVCODE_OUTPUT_CONVERGED
                current.conv |= CONVCODE_POST_CONVERGED
                new.conv |= CONVCODE_POST_CONVERGED
            elif ret == 2:
                current.conv |= CONVCODE_OUTPUT_CONVERGED
                new.conv |= CONVCODE_POST_CONVERGED
            else:
                current.conv |= CONVCODE_ERROR
                new.conv |= CONVCODE_ERROR
                return -3
            return ret
    # invalid return value of limits func
    current.conv |= CONVCODE_ERROR
    new.conv |= CONVCODE_ERROR
    return -4


# The wrapper for the user supplied print function that displays a string
cdef int model_print_call(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef object pfunc = <object> prnt_strct.func
    cdef object pargs = <object> prnt_strct.args
    cdef object pkwargs = <object> prnt_strct.kwargs
    cdef int ret = 0
    cdef h2mm_model new_model = model_copy_from_ptr(new)
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    cdef h2mm_model old_model = model_copy_from_ptr(old)
    new_model.normalize()
    try:
        formatter.update(pfunc(niter, new_model, current_model, old_model, t_iter, t_total, *pargs, **pkwargs))
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


cdef int model_print_all(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef int ret = 0
    cdef h2mm_model current_model = model_copy_from_ptr(current)
    try:
        formatter.update(repr(current_model)+f"\nIteration time:{t_iter}, Total:{t_total}")
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


cdef int model_print_diff(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef int ret = 0
    try:
        formatter.update(f'Iteration:{niter:5d}, loglik:{current.loglik:12e}, improvement:{current.loglik - old.loglik:6e}')
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


cdef int model_print_diff_time(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef int ret = 0
    try:
        formatter.update(f'Iteration:{niter:5d}, loglik:{current.loglik:12e}, improvement:{current.loglik - old.loglik:6e} iteration time:{t_iter}, total:{t_total}')
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


cdef int model_print_comp(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef int ret = 0
    try:
        formatter.update(f"Iteration:{niter:5d}, loglik:{current.loglik:12e}, previous loglik:{old.loglik:12e}")
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


cdef int model_print_comp_time(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef int ret = 0
    try:
        formatter.update(f"Iteration:{niter:5d}, loglik:{current.loglik:12e}, previous loglik:{old.loglik:12e} iteration time:{t_iter}, total:{t_total}")
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


cdef int model_print_iter(int64_t niter, h2mm_mod *new, h2mm_mod *current, h2mm_mod *old, double t_iter, double t_total, void *instruct) noexcept with gil:
    cdef PrintStruct *prnt_strct = <PrintStruct*> instruct
    if niter % prnt_strct.disp_freq != 0:
        return 0
    cdef object formatter = <object> prnt_strct.formatter
    cdef int ret = 0
    try:
        formatter.update(f"Iteration {niter:5d} (Max:{prnt_strct.max_iter:5d})")
    except Exception as e:
        prnt_strct.error = <PyObject*> e
        Py_INCREF(e)
        ret = -1
    return ret


###############################################################################
#################### Casting burst input arrays funcstions ####################
###############################################################################
cdef int free_deltas(int64_t nbursts, int32_t **data):
    if data is NULL:
        return 0
    cdef int64_t i
    for i in range(nbursts):
        if data[i] is not NULL:
            PyMem_Free(data[i])
            data[i] = NULL
    PyMem_Free(data)
    return 0

cdef int free_idx_diffs_arrays(int64_t i, int64_t *ilens, uint8_t **iidxs, int32_t **idiffs):
    free_deltas(i, idiffs)
    PyMem_Free(iidxs)
    PyMem_Free(ilens)
    return 0

cdef tuple reshape_burst_arrays(burst_array, str name):
    if not isinstance(burst_array, (cnp.ndarray, Sequence)):
        return tuple(), tuple(), False, True, TypeError(f"{name} must be readable as sequence or numpy array of 1D arrays")
    cdef bint single = False
    if isinstance(burst_array, cnp.ndarray):
        if burst_array.dtype != np.object_:
            single = True
            burst_array = (burst_array, )
            shape = (1, )
        else:
            shape = burst_array.shape
            burst_array = burst_array.reshape(-1)
    elif len(burst_array) != 0 and not isinstance(burst_array[0], (cnp.ndarray, Sequence)):
        single = True
        burst_array = (burst_array, )
        shape = (1, )
    else:
        shape = (len(burst_array), )
    return burst_array, shape, single, False, False


cdef object cast_burst_uint8(bursts, str name, int64_t *nbursts, int64_t **len_bursts, uint8_t ***dout):
    cdef int64_t i
    cdef bint err = False
    e = None
    #### check lens are consistent ####
    if nbursts[0] == 0:
        nbursts[0] = len(bursts)
    elif nbursts[0] != len(bursts):
        return ValueError(f"nubmer of bursts in {name} ({len(bursts)} different from others specified ({nbursts[0]})")
    cdef cnp.ndarray[object, ndim=1] out 
    try:
        out = np.empty(nbursts[0], dtype=np.object_)    
    except Exception as e:
        err = True
    if err:
        return e
    cdef bint firstarray = len_bursts[0] is NULL
    cdef object otemp
    cdef cnp.ndarray[uint8_t] temp
    cdef uint8_t **data = <uint8_t**> PyMem_Malloc(nbursts[0]*sizeof(uint8_t*))
    cdef int64_t *lbursts = <int64_t*> PyMem_Malloc(nbursts[0]*sizeof(int64_t)) if firstarray else len_bursts[0]
    if lbursts is NULL:
        return MemoryError("insufficient memory")
    if data is NULL:
        if firstarray:
            PyMem_Free(lbursts)
            lbursts = NULL
        return MemoryError("insufficient memory")
    for i in range(nbursts[0]):
        try:
            otemp = np.ascontiguousarray(bursts[i], dtype=np.uint8)
            temp = otemp
        except:
            err = True
        if err:
            if firstarray:
                PyMem_Free(lbursts)
                lbursts = NULL
            return TypeError(f"burst {i} in {name} cannot be read as 1D uint8 numpy array")
        if temp.ndim != 1:
            if firstarray:
                PyMem_Free(lbursts)
                lbursts = NULL
            return ValueError(f"burst {i} in {name} is {temp.ndim}D, must be 1D uint8 array")
        if firstarray:
            lbursts[i] = <int64_t> temp.shape[0]
            if lbursts[i] < 2:
                if firstarray:
                    PyMem_Free(lbursts)
                    lbursts = NULL
                return ValueError(f"burst {i} in {name} is too short, must have at least 2 photons, got {temp.shape[0]}")
        elif lbursts[i] != temp.shape[0]:
            if firstarray:
                PyMem_Free(lbursts)
                lbursts = NULL
            return ValueError(f"burst {i} in {name} has different size from other specified ({temp.shape[0]} vs {lbursts[i]}")
        out[i] = otemp
        data[i] = <uint8_t*> temp.data
    if firstarray:
        len_bursts[0] = lbursts
    dout[0] = data
    return out


cdef object cast_bursts_deltas(bursts, int64_t nbursts, int64_t *lbursts, int32_t ***dout):
    ### check number of bursts is correct
    if len(bursts) != nbursts:
        return ValueError(f"nubmer of bursts in times ({len(bursts)} different from others specified ({nbursts})")
    ### make out array ###
    cdef int32_t **data = <int32_t**> PyMem_Malloc(nbursts*sizeof(int32_t*))
    if data is NULL:
        return MemoryError('insufficient memory for deltas')
    cdef cnp.ndarray[int64_t] temp
    cdef int64_t i, j, jj, delta
    for i in range(nbursts):
        temp = np.ascontiguousarray(bursts[i], dtype=np.int64)
        if temp.ndim != 1 or temp.shape[0] != lbursts[i]:
            free_deltas(i, data)
            data = NULL
            if temp.ndim != 1:
                return ValueError(f"burst {i} in time is not 1D")
            else:
                return ValueError(f"burst {i} in times has different size from other bursts arrays ({temp.shape[0]}, vs {nbursts})")
        data[i] = <int32_t*> PyMem_Malloc(lbursts[i]*sizeof(int32_t))
        if data[i] is NULL:
            free_deltas(i, data)
            data = NULL
            return MemoryError("insufficient memory for time deltas")
        data[i][0] = 0
        jj = 0
        for j in range(1, lbursts[i]):
            delta = temp[j] - temp[jj]
            if delta < 0 or delta > INT32_MAX:
                free_deltas(i+1, data)
                data = NULL
                if delta < 0:
                    return ValueError(f"burst {i} photons times {jj} and {j} out of order")
                return ValueError(f"burst {i} photons times {jj} and {j} difference is too large")
            if delta != 0:
                delta -= 1
            data[i][j] = delta
            jj = j
    dout[0] = data
    return None


cdef tuple cast_indexes_times(indexes, times, bint *single, int64_t *nbursts, int64_t **len_bursts, uint8_t ***idxs, int32_t ***deltas):
    indexes, shape, sngl, haserr, err = reshape_burst_arrays(indexes, "indexes")
    if haserr:
        return tuple(), tuple(), err
    times, _, _, haserr, err = reshape_burst_arrays(times, "times")
    if haserr:
        return tuple(), tuple(), err
    indexes = cast_burst_uint8(indexes, "indexes", nbursts, len_bursts, idxs)
    if idxs[0] is NULL:
        return tuple(), tuple(), indexes
    err = cast_bursts_deltas(times, nbursts[0], len_bursts[0], deltas)
    if deltas[0] is NULL:
        PyMem_Free(idxs[0])
        idxs[0] = NULL
        return tuple(), tuple(), err
    single[0] = sngl
    return shape, indexes, None


cdef cnp.ndarray[object, ndim=1] make_gamma_arrays(int64_t nbursts, int64_t nstate, int64_t *burst_sizes, double ***gamma):
    cdef cnp.ndarray[object, ndim=1] out 
    cdef bint err = False
    e = None
    try:
        out = np.empty(nbursts, dtype=np.object_)
    except Exception as e:
        err = True
    if err:
        return np.empty(0, dtype=np.object_)
    cdef object obtemp
    cdef cnp.ndarray[double, ndim=2] temp
    cdef int64_t i
    cdef double **data = <double**> PyMem_Malloc(nbursts*sizeof(double*))
    if data is NULL:
        return out, MemoryError("insufficient memory for gamma array")
    for i in range(nbursts):
        try:
            obtemp = np.empty((burst_sizes[i], nstate), dtype=np.double)
            temp = obtemp
        except Exception as e:
            err = True
        if err:
            PyMem_Free(data)
            data = NULL
            return out
        out[i] = obtemp
        data[i] = <double*> temp.data
    gamma[0] = data
    return out


cdef cnp.ndarray[object, ndim=1] make_h2mm_out_arrays(int64_t nmodels, h2mm_mod *mods):
    cdef cnp.ndarray[object, ndim=1] models
    cdef bint err = False
    try:
        models = np.empty(nmodels, dtype=np.object_)
    except Exception as e:
        err = True
        models[0] = e
    if err:
        return models
    for i in range(nmodels):
        try:
            models[i] = h2mm_model.move_ptr(&mods[i])
        except Exception as e:
            err = True
            models[i] = e
        if err:
            return models
    return models


def EM_H2MM_C(h2mm_model model, indexes, times, max_iter=None, 
              bounds_func=None, bounds=None, bounds_kwargs=None,
              print_func='iter', print_freq=1, print_args=None, print_kwargs=None,
              print_stream=None, print_formatter=None,
              print_fmt_args=None, print_fmt_kwargs=None,
              max_time=np.inf, converged_min=None, num_cores=None, 
              reset_niter=True, gamma=False, opt_array=False):
    """
    Calculate the most likely model that explains the given set of data. The 
    input model is used as the start of the optimization.

    Parameters
    ----------
    model : h2mm_model
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
    
    max_iter : int or None, optional
        the maximum number of iterations to conduct before returning the current
        :class:`h2mm_model`, if None (default) use value from optimization_limits. 
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
                
                must be a python function that takes the signature
                ``bounds_func(new, current, old, *bounds, **bounds_kwargs)``
                ``new``, ``current``, and ``old`` are :class:`h2mm_model` objects, 
                with ``new`` being the model whose loglik has yet to be calculated
                resulting from the latest iteration, ``current`` being the model
                whose loglik was just calculated, and ``old`` being the result
                of the previous iteration. Be one of the following: :class:`h2mm_model`,
                :code:`bool`, :code:`int`, or a 2-tuple of :class:`h2mm_model` and
                either :code:`bool` or :code:`int`. In all cases, :code:`int` values
                must 0, 1, or 2. :class:`h2mm_model` objects will be used as the
                next model to compute the loglik of. :class:`bool` and :code:`int`
                objects determine if optimization will continue.
                If :code:`True`, then optimization will continue, if :code:`False`,
                then optimization will cease, and the ``old`` model will be returned
                as the ideal model. If :code:`0`, then optimization will continue,
                if :code:`1`, then return ``old`` model as optimal model, if 
                :code:`2`, then return ``current`` model as optimizal model.
    
    bounds : h2mm_limits, tuple, or None, optional
        Argument(s) pass to bounds_func function. If bounds_func is a string,
        then bounds **must** be specified, and **must** be a :class:`h2mm_limits`
        object.
        
        If ``bounds_func`` is callable, and bounds is not None or a tuple, bounds
        is passed as the 4th argument to bounds_func, ie 
        :code:`bounds_func(new, current, old, bounds, **bounds_kwargs)`, 
        if bounds is a tuple, then passed as \*args, 
        ie :code:`bounds_func(new, current, old, *bounds, **bounds_kwargs)`
        Default is None
    
    bounds_kwargs : dict, or None, optional
        Only used when ``bounds_func`` is callable, passed as \*\*kwargs to
        ``bounds_func`` ie ie :code:`bounds_func(new, current, old, *bounds, **bounds_kwargs)`.
        If :code:`None`
        The default is None
    
    print_func : None, str or callable, optional
        Specifies how the results of each iteration will be displayed, several 
        strings specify built-in functions. Default is 'iter'
        Acceptable inputs: str, Callable or None
            
            **None**\:
            
                causes no printout anywhere of the results of each iteration
                
            Str: 'all', 'diff', 'comp', 'iter'
            
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
                as the function will be handed ``(niter, new, current, old, iter_time, total_time)``
                from a special Cython wrapper. iter_time and total_time are the times
                of the latest iteration and total time of optimization, respectively.
                Note that iter_time and total_time are based on the fast, but
                inaccurate C clock function.
    
    print_freq: int, optional
        Number of iterations between updating display. Default is 1
    
    print_args : tuple or None, optional
        Only used when ``print_func`` is callable. Passed as final argument to
        ``print_func`` if not :code:`None` or :code:`tuple`, ie
        ``print_func(niter, new, current, old, iter_time, total_time, print_args, **print_kwargs)``
        if :code:`tuple`, then passed as \*args, ie
        ``print_func(niter, new, current, old, iter_time, total_time, *print_args, **print_kwargs)``
        If :code:`None`, then ignored. The default is :code:`None`.
    
    print_kwargs: dict or None, optional
        Only used when ``print_func`` is callable, passed as \*\*kwargs to ``print_func``
        ie ``print_func(niter, new, current, old, iter_time, time_total, *print_args, **print_kwargs)``.
        If :code:`None`, then ignored. The default is :code:`None`.
    
    print_stream : OutStream, optional
        Typically OutStream, the stream where the output will be displayed.
        Passed as first argument to print_formatter, if :code:`None`, then use
        stream specified in optimization_limites. Default is None
    
    print_formatter : Printer, optional
        Class object (usually subclass of :class:`Printer`) that formats
        text for output to ``print_stream``. Instance of ``print_formatter`` class
        will be created with 
        ``fmtr = print_formatter(print_stream, *print_fmt_args, **print_fmt_kwargs)``
        to use for printing. Each print will call ``fmtr.update(text)`` where
        ``text`` is output of ``print_func`` call. At end of optimization,
        ``fmtr.close()`` will be called. If ``print_formatter`` is :code:`None`
        then will use the formatter specified in ``optimization_limits``.
        The default is :code:`None`.
    
    print_fmt_args : tuple, Any or None, optional
        Additional arguments passed to ``print_formatter``, if None, ignored,
        if not tuple, then treated as single argument. The default is :code:`None`.
    
    print_fmt_kwargs : dict or None, optional
        Additional  keyword arguments passed to ``print_formatter``, if None, ignored.
        The default is :code:`None`.
        
    max_time : float or None, optional
        The maximum time (in seconds) before returning current model
        **NOTE** this uses the inaccurate C clock, which is often longer than the
        actual time. If :code:`None` (default) use value from optimization_limits. 
        Default is :code:`None`.
    
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
    
    opt_array : bool | int, optional
        Defines how the result is returned, if False (default) (or 0) then return only
        the ideal model, if True (or 1), then a numpy array is returned containing all
        models during the optimization, with the last model being the optimial model.
        If 2 then return all models, including models after converged.
        Default is False
    
    Returns
    -------
    out : h2mm_model
        The optimized :class:`h2mm_model`. will return after one of the following conditions
        are met: model has converged (according to converged_min, default 1e-14),
        maximum iterations reached, maximum time has passed, or an error has occurred
        (usually the result of a nan from a floating point precision error)
        
    gamma : numpy.ndarray (optional)
        If gamma = True, this variable is returned, which contains (as numpy arrays)
        the probabilities of each photon to be in each state, for each burst.
        The arrays are returned in the same order and type as the input indexes,
        individual data points organized as [photon, state] within each data
        sequence.
        
    """
    print_fmt_args = tuple() if print_fmt_args is None else print_fmt_args
    print_fmt_args = print_fmt_args if isinstance(print_fmt_args, tuple) else (print_fmt_args, )
    print_fmt_kwargs = dict() if print_fmt_kwargs is None else dict(print_fmt_kwargs)
    print_args = tuple() if print_args is None else print_args
    print_args = print_args if isinstance(print_args, tuple) else (print_args, )
    print_kwargs = dict() if print_kwargs is None else dict(print_kwargs)
    bounds_kwargs = dict() if bounds_kwargs is None else dict(bounds_kwargs)
    cdef int64_t i
    cdef bint gamma_ = bool(gamma)
    cdef int opt_array_ = int(opt_array)
    if opt_array_ not in (0, 1, 2):
        raise ValueError("opt_array must be True, False, 0 (False), 1 (True), or 2 (return all assigned models)")
    cdef h2mm_model mdl = model_copy_from_ptr(model.model)
    if reset_niter:
        mdl.model.niter = 0
    ######################### get optimization limits #########################
    cdef lm limits
    limits.max_iter = <int64_t> optimization_limits._get_max_iter(max_iter)
    limits.num_cores = <int64_t> optimization_limits._get_num_cores(num_cores)
    limits.max_time = <double> optimization_limits._get_max_time(max_time)
    limits.min_conv = <double> optimization_limits._get_converged_min(converged_min)
    ################# process print_func and print_func args  #################
    cdef tuple print_strings = (None,'all','diff','diff_time','comp','comp_time','iter')
    if print_func not in print_strings and not callable(print_func):
        raise ValueError("print_func must be None, 'all', 'diff', or 'comp' or callable")
    cdef PrintStruct prnt_strct
    cdef object print_fmtr = optimization_limits._get_formatter(print_formatter)(optimization_limits._get_outstream(print_stream), 
                                                                                 *print_fmt_args, **print_fmt_kwargs)
    cdef int (*ptr_print_func)(int64_t,h2mm_mod*,h2mm_mod*,h2mm_mod*,double,double,void*) noexcept with gil
    prnt_strct.disp_freq = <int64_t> int(print_freq)
    prnt_strct.max_iter = limits.max_iter
    prnt_strct.max_time = limits.max_time
    prnt_strct.formatter = <PyObject*> print_fmtr
    prnt_strct.func = NULL
    prnt_strct.args = NULL
    prnt_strct.kwargs = NULL
    prnt_strct.error = NULL
    ptr_print_func = NULL
    if print_func in print_strings:
        if print_func == 'all':
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
        prnt_strct.func = <PyObject*> print_func
        prnt_strct.args = <PyObject*> print_args
        prnt_strct.kwargs = <PyObject*> print_kwargs
        ptr_print_func = model_print_call
    ########################## process bounds inputs ##########################
    if bounds_func not in (None, 'none', 'minmax', 'revert', 'revert_old', 'revert-old') and not callable(bounds_func):
        raise ValueError("bounds_func must be None, 'minmax', 'revert', or 'revert_old', or or callable")
    cdef int (*bound_func)(h2mm_mod*, h2mm_mod*, h2mm_mod*, double, lm* ,void*) noexcept with gil
    cdef _h2mm_lims bnds
    cdef BoundStruct bound
    cdef void *bnd_ptr = NULL
    bound_func = NULL
    bound.args = NULL
    bound.kwargs = NULL
    bound.error = NULL
    if bounds_func in (None, 'none'):
        bound_func = limit_check_only
    if bounds_func in ('minmax', 'revert', 'revert_old', 'revert-old'):
        if bounds_func == 'minmax':
            bound_func = limit_minmax
        elif bounds_func == 'revert':
            bound_func = limit_revert
        elif bounds_func in ('revert_old', 'revert-old'):
            bound_func = limit_revert_old
        if not isinstance(bounds, h2mm_limits):
            raise TypeError(f"standard bounding styles requires bounds argument to be of type h2mm_limits, not {type(bounds)}")
        bnds = bounds._make_model(model)
        bnd_ptr = <void*> &bnds.limits
    elif callable(bounds_func):
        bounds = tuple() if bounds is None else bounds
        bounds = bounds if isinstance(bounds, tuple) else (bounds, )
        bound_func = cy_limit
        bound.func = <PyObject*> bounds_func
        bound.args = <PyObject*> bounds
        bound.kwargs = <PyObject*> bounds_kwargs
        bnd_ptr = <void*> &bound
    ############################### cast bursts ###############################
    # this is last because this requires allocating the most arrays 
    # (reduces number of frees on errors before this code)
    cdef bint single = False
    cdef int64_t nbursts = 0
    cdef int64_t *burst_sizes = NULL
    cdef uint8_t **idxs = NULL
    cdef int32_t **deltas = NULL
    shape, indxs, err = cast_indexes_times(indexes, times, &single, &nbursts, &burst_sizes, &idxs, &deltas)
    if err is not None:
        raise err
    cdef int64_t nphot = 0
    for i in range(nbursts):
        nphot += burst_sizes[i]
    #################### allocate gamma arrays for output  ####################
    cdef double **gamma_arr = NULL
    cdef cnp.ndarray[object, ndim=1] gamma_out
    cdef int res
    if gamma_:
        gamma_out = make_gamma_arrays(nbursts, mdl.model.nstate, burst_sizes, &gamma_arr)
        if gamma_arr == NULL:
            free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
            raise MemoryError("insufficient memory for gamma")
    ######################## make model output arrays  ########################
    cdef int64_t out_arr_size = limits.max_iter + 2 - mdl.model.niter if opt_array_ else 1
    cdef h2mm_mod *out_arr = Py_allocate_models(out_arr_size, mdl.model.nstate, mdl.model.ndet, nphot)
    if out_arr == NULL:
        free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
        PyMem_Free(gamma_arr)
        raise MemoryError("insufficient memory for h2mm_model output")
    ######################## perform h2mm optimization ########################
    with nogil:
        if gamma_:
            if opt_array_:
                res =  h2mm_optimize_gamma_array(nbursts, burst_sizes, deltas, idxs, mdl.model, &out_arr, &gamma_arr, &limits, bound_func, bnd_ptr, ptr_print_func, <void*> &prnt_strct)
            else:
                res =  h2mm_optimize_gamma(nbursts, burst_sizes, deltas, idxs, mdl.model, out_arr, &gamma_arr, &limits, bound_func, bnd_ptr, ptr_print_func, <void*> &prnt_strct)
        else:
            if opt_array_:
                res =  h2mm_optimize_array(nbursts, burst_sizes, deltas, idxs, mdl.model, &out_arr, &limits, bound_func, bnd_ptr, ptr_print_func, <void*> &prnt_strct)
            else:
                res =  h2mm_optimize(nbursts, burst_sizes, deltas, idxs, mdl.model, out_arr, &limits, bound_func, bnd_ptr, ptr_print_func, <void*> &prnt_strct)
    free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
    if gamma_arr is not NULL:
        PyMem_Free(gamma_arr)
        gamma_arr = NULL
    ############################# Process output  #############################
    cdef int64_t nout = 0 if opt_array_ else 1
    cdef int64_t conv = -1
    cdef int64_t niter
    if res > 0:
        if opt_array_ == 0:
            conv = out_arr[0].conv
            niter = out_arr[0].niter
        elif opt_array_ == 1:
            while nout < out_arr_size and not (out_arr[nout].conv & CONVCODE_OUTPUT):
                nout += 1
            if nout == out_arr_size:
                nout -= 1
            conv = out_arr[nout].conv
            niter = out_arr[nout].niter
            nout += 1
        elif opt_array_ == 2:
            while nout < out_arr_size and out_arr[nout].conv != 0:
                if out_arr[nout].conv & CONVCODE_OUTPUT:
                    conv = out_arr[nout].conv
                    niter = out_arr[nout].niter
                nout += 1
        out = make_h2mm_out_arrays(nout, out_arr) # note error stored in out, should be impossible, but most likely problem is memory error
    Py_free_models(out_arr_size, out_arr)
    out_arr = NULL
    ############################ check for errors  ############################
    cdef object print_err = None
    if prnt_strct.error is not NULL:
        print_err = <object> prnt_strct.error
        Py_DECREF(print_err)
        prnt_strct.error = NULL
    cdef object bound_err = None
    if bound.error is not NULL:
        bound_err = <object> bound.error
        Py_DECREF(bound_err)
        bound.error = NULL
    if res == -1:
        raise ValueError('Bad pointer, check cython code')
    elif res == -2:
        raise ValueError('Too many photon streams in data for H2MM model')
    elif res == -3:
        raise ValueError('limits function must return model of same shape as inintial model')
    elif res == -4:
        raise TypeError('limits function must return bool, h2mm_model, (h2mm_model, bool) or (h2mm_model, int [0,2])')
    elif res == -5:
        raise bound_err
    elif res == -6:
        raise print_err
    elif res < -6:
        raise ValueError(f'Unknown error, check C code- raise issue on GitHub, res={res}, conv={conv}')
    # next error should be essentially imposible
    for i in range(out.shape[0]):
        if isinstance(out[i], Exception):
            raise out[i]
    ############################# forming output  #############################
    if conv & CONVCODE_CONVERGED:
        out_text = f'The model converged after {niter} iterations'
    elif conv & CONVCODE_MAXITER:
        out_text = 'Optimization reached maximum number of iterations'
    elif conv & CONVCODE_MAXTIME:
        out_text = 'Optimization reached maxiumum time'
    elif conv & CONVCODE_ERROR:
        out_text = f'An error occured on iteration {niter}, returning previous model'
    elif conv == -1:
        raise ValueError(f'Unknown error, check C code- raise issue on GitHub, res={res}, conv={conv}')
    if not opt_array_:
        out = out[0]
    if ptr_print_func is not NULL:
        print_fmtr.update(out_text)
        print_fmtr.close()
    if gamma_:
        out = out, gamma_out[0] if single else gamma_out.reshape(shape)
    return out


cdef tuple make_h2mm_arr_modptr(models, bint *modelsingle, int64_t *nmodels, h2mm_mod **mod_arr):
    if isinstance(models, h2mm_model):
        modelsingle[0] = True
        models = [models, ]
    if not isinstance(models, (cnp.ndarray, Sequence)):
        return tuple(), TypeError("models must be sequence or array of h2mm_model objects")
    if isinstance(models, cnp.ndarray):
        shape = models.shape
        models = models.reshape(-1)
    else:
        shape = (len(models), )
    if any(not isinstance(model, h2mm_model) for model in models):
        return tuple(), TypeError("all elements of models must be h2mm_model objects")
    cdef int64_t nmodel = len(models)
    cdef int64_t i, j
    cdef h2mm_mod *imod_arr = <h2mm_mod*> PyMem_Malloc(nmodel*sizeof(h2mm_mod))
    if imod_arr is NULL:
        return tuple(), MemoryError(f"insufficient memory, cannot allocate {nmodel*sizeof(h2mm_mod)//1024} kb for output models array")
    cdef h2mm_model temp
    cdef h2mm_mod *mtemp
    for i in range(nmodel):
        temp = models[i]
        mtemp = temp.model
        imod_arr[i].prior = <double*> PyMem_Malloc(mtemp.nstate*sizeof(double))
        if imod_arr[i].prior is NULL:
            Py_free_models(i, imod_arr)
            return tuple(), MemoryError(f"insufficient memory, cannot allocate prior array")
        imod_arr[i].trans = <double*> PyMem_Malloc(mtemp.nstate*mtemp.nstate*sizeof(double))
        if imod_arr[i].prior is NULL:
            PyMem_Free(imod_arr[i].prior)
            imod_arr[i].prior = NULL
            Py_free_models(i, imod_arr)
            return tuple(), MemoryError(f"insufficient memory, cannot allocate trans array")
        imod_arr[i].obs = <double*> PyMem_Malloc(mtemp.ndet*mtemp.nstate*sizeof(double))
        if imod_arr[i].obs is NULL:
            PyMem_Free(imod_arr[i].prior)
            imod_arr[i].prior = NULL
            PyMem_Free(imod_arr[i].trans)
            imod_arr[i].trans = NULL
            Py_free_models(i, imod_arr)
            return tuple(), MemoryError(f"insufficient memory, cannot allocate prior array")
        for j in range(mtemp.nstate):
            imod_arr[i].prior[j] = mtemp.prior[j]
        for j in range(mtemp.nstate*mtemp.nstate):
            imod_arr[i].trans[j] = mtemp.trans[j]
        for j in range(mtemp.nstate*mtemp.ndet):
            imod_arr[i].obs[j] = mtemp.obs[j]
        imod_arr[i].nstate = mtemp.nstate
        imod_arr[i].ndet = mtemp.ndet
        imod_arr[i].nphot = 0
        imod_arr[i].niter = 0
        imod_arr[i].conv = 0
        imod_arr[i].loglik = 0.0
    mod_arr[0] = imod_arr
    nmodels[0] = nmodel
    return shape, False


cdef cnp.ndarray[object, ndim=2] make_gamma_gamma_arrays(int64_t nmodels, h2mm_mod *models, int64_t nbursts, int64_t *burst_sizes, double ****gamma):
    cdef cnp.ndarray[object, ndim=2] out
    cdef bint err = False
    e = None
    try:
        out = np.empty((nmodels, nbursts), dtype=np.object_)
    except Exception as e:
        err = True
    if err:
        return np.empty(0, dtype=np.object_)
    cdef object obtemp
    cdef cnp.ndarray[double, ndim=2] temp
    cdef int64_t i, j
    cdef double ***data = <double***> PyMem_Malloc(nbursts*sizeof(double**))
    if data is NULL:
        return out
    for i in range(nmodels):
        data[i] = <double**> PyMem_Malloc(nbursts*sizeof(double*))
        if data[i] is NULL:
            free_gammagamma(i, data)
            data = NULL
            return out
        for j in range(nbursts):
            try:
                obtemp = np.empty((burst_sizes[j], models[i].nstate), dtype=np.double)
                temp = obtemp
            except Exception as e:
                err = True
            if err:
                free_gammagamma(i, data)
                data = NULL
                return out
            
            out[i,j] = obtemp
            data[i][j] = <double*> temp.data
    gamma[0] = data
    return out


cdef int free_gammagamma(int64_t nmodels, double ***gamma):
    if gamma is NULL:
        return 0
    cdef int64_t i, j
    if gamma is not NULL:
        for i in range(nmodels):
            if gamma[i] is not NULL:
                PyMem_Free(gamma[i])
                gamma[i] = NULL
        PyMem_Free(gamma)
    return 0


def H2MM_arr(models, indexes, times, num_cores=None, gamma=False):
    """
    Calculate the logliklihood of every model in a list/array given a set of
    data

    Parameters
    ----------
    models : list, tuple, or numpy.ndarray of h2mm_model objects
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
    cdef bint gamma_ = bool(gamma)
    cdef lm limits
    limits.num_cores = <int64_t> optimization_limits._get_num_cores(num_cores)
    ########################## process input bursts  ##########################
    cdef bint burstsingle
    cdef int64_t nbursts = 0
    cdef int64_t *burst_sizes = NULL
    cdef uint8_t **idxs = NULL
    cdef int32_t **deltas = NULL
    burstshape, indxs, err = cast_indexes_times(indexes, times, &burstsingle, &nbursts, &burst_sizes, &idxs, &deltas)
    if err is not None:
        raise err
    ############################ Cast models array ############################
    cdef bint modelsingle = False
    cdef int64_t nmodels = 0
    cdef h2mm_mod *out_mod = NULL
    modelshape, err = make_h2mm_arr_modptr(models, &modelsingle, &nmodels, &out_mod)
    if not modelshape:
        free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
        raise err
    ############################# allocate for gamma ##########################
    cdef cnp.ndarray[object, ndim=2] gamma_out
    cdef double ***gamma_arr = NULL
    if gamma_:
        gamma_out = make_gamma_gamma_arrays(nmodels, out_mod, nbursts, burst_sizes, &gamma_arr)
        if gamma_arr is NULL:
            free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
            Py_free_models(nmodels, out_mod)
            raise MemoryError("insufficient memory for gamma")
    ############################ main calculation  ############################
    cdef int res
    with nogil:
        if gamma_:
            res = calc_multi_gamma(nbursts, burst_sizes, deltas, idxs, nmodels, out_mod, &gamma_arr, &limits)
        else:
            res = calc_multi(nbursts, burst_sizes, deltas, idxs, nmodels, out_mod, &limits)
    free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
    if gamma_:
        free_gammagamma(nmodels, gamma_arr)
    ############################# check for error #############################
    if res == 0:
        out = make_h2mm_out_arrays(nmodels, out_mod).reshape(modelshape)
        if modelsingle:
            out = out[0]    
    Py_free_models(nmodels, out_mod)
    out_mod = NULL
    if res == 1:
        raise ValueError('Bursts photons are out of order, please check your data')
    elif res == 2:
        raise ValueError('Too many photon streams in data for one or more H2MM models')
    if gamma_:
        if modelsingle and burstsingle:
            out = (out, gamma_out[0])
        elif modelsingle:
            out = (out, gamma_out.reshape(burstshape))
        elif burstsingle:
            out = (out, gamma_out.reshape(modelshape))
        else:
            out = (out, gamma_out.reshape(modelshape+burstshape))
    return out


# viterbi function
cdef tuple make_ph_path_arrays(int64_t nbursts, int64_t *burst_sizes, int64_t nstate, ph_path **paths):
    cdef cnp.ndarray[object, ndim=1] state_paths = np.empty(nbursts, dtype=np.object_)
    cdef cnp.ndarray[object, ndim=1] scale_paths = np.empty(nbursts, dtype=np.object_)
    cdef ph_path *cpaths = <ph_path*> PyMem_Malloc(nbursts*sizeof(ph_path))
    cdef cnp.ndarray[uint8_t, ndim=1] statetemp
    cdef cnp.ndarray[double, ndim=1] scaletemp
    cdef i
    for i in range(nbursts):
        try:
            statetemp = np.empty(burst_sizes[i], dtype=np.uint8)
        except:
            PyMem_Free(cpaths)
            return state_paths, scale_paths
        try:
            scaletemp = np.empty(burst_sizes[i], dtype=np.double)
        except:
            PyMem_Free(cpaths)
            return state_paths, scale_paths
        state_paths[i] = statetemp
        scale_paths[i] = scaletemp
        cpaths[i].path = <uint8_t*> statetemp.data
        cpaths[i].scale = <double*> scaletemp.data
        cpaths[i].nstate = nstate
        cpaths[i].nphot = burst_sizes[i]
    paths[0] = cpaths
    return state_paths, scale_paths


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
    cdef bint single = False
    cdef int64_t nbursts = 0
    cdef int64_t *burst_sizes = NULL
    cdef uint8_t **idxs = NULL
    cdef int32_t **deltas = NULL
    shape, indxs, err = cast_indexes_times(indexes, times, &single, &nbursts, &burst_sizes, &idxs, &deltas)
    if err is not None:
        raise err
    cdef int64_t i
    cdef int64_t nphot = 0
    for i in range(nbursts):
        nphot += burst_sizes[i]
    cdef int64_t n_core = <int64_t> optimization_limits._get_num_cores(num_cores)
    cdef ph_path *cpaths = NULL
    cdef cnp.ndarray[object,ndim=1] paths
    cdef cnp.ndarray[object,ndim=1] scale
    paths, scale = make_ph_path_arrays(nbursts, burst_sizes, h_mod.model.nstate, &cpaths)
    if cpaths == NULL:
        raise MemoryError("insufficent memory to allocate path arrays")
    cdef int ret
    with nogil:
        ret = viterbi(nbursts, burst_sizes, deltas, idxs, h_mod.model, cpaths, n_core)
    free_idx_diffs_arrays(nbursts, burst_sizes, idxs, deltas)
    cdef cnp.ndarray[double] ll
    try:
        ll = np.empty(nbursts, dtype=np.double)
    except:
        PyMem_Free(cpaths)
        raise MemoryError("insufficient memory to allocate loglik array")
    
    cdef double loglik = 0.0
    for i in range(nbursts):
        loglik += cpaths[i].loglik
        ll[i] = cpaths[i].loglik
    PyMem_Free(cpaths)
    cdef double icl = ((h_mod.nstate**2 + ((h_mod.ndet - 1) * h_mod.nstate) - 1) * np.log(nphot)) - 2 * loglik
    out = paths[0] if single else paths.reshape(shape), scale[0] if single else scale.reshape(shape), ll[0] if single else ll.reshape(shape), icl
    return out


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
    cdef cnp.ndarray[double,ndim=1] ll
    cdef double icl
    paths, scale, ll, icl = viterbi_path(hmod,indexes,times,num_cores=num_cores)
    
    # sorting bursts based on which dwells occur in them
    cdef cnp.ndarray[int, ndim=1] burst_type = np.zeros(len(indexes),dtype=int)
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
    cdef cnp.ndarray time, index, state
    cdef cnp.ndarray[long,ndim=2] ph_counts_temp = np.zeros((1,hmod.ndet),dtype=int)
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


#########################################
### Function specifically for path_loglik
#########################################
cdef int free_idx_diffs_path_arrays(int64_t i, int64_t *ilens, uint8_t **iidxs, int32_t **idiffs, uint8_t **ist_path):
    free_deltas(i, idiffs)
    PyMem_Free(iidxs)
    PyMem_Free(ilens)
    PyMem_Free(ist_path)
    return 0


cdef tuple cast_indexes_times_paths(indexes, times, paths, bint *single, int64_t *nbursts, int64_t **len_bursts, uint8_t ***idxs, int32_t ***deltas, uint8_t ***stpath):
    indexes, shape, sngl, haserr, err = reshape_burst_arrays(indexes, "indexes")
    if haserr:
        return tuple(), tuple(), tuple(), err
    times, _, _, haserr, err = reshape_burst_arrays(times, "times")
    if haserr:
        return tuple(), tuple(), tuple(), err
    paths, _, _, haserr, err = reshape_burst_arrays(paths, "state_path")
    if haserr:
        return tuple(), tuple(), err
    indexes = cast_burst_uint8(indexes, "indexes", nbursts, len_bursts, idxs)
    if idxs[0] is NULL:
        return tuple(), tuple(), tuple(), indexes
    paths = cast_burst_uint8(paths, "state_path", nbursts, len_bursts, stpath)
    if stpath[0] is NULL:
        PyMem_Free(idxs[0])
        idxs[0] = NULL
        return tuple(), tuple(), tuple(), paths
    err = cast_bursts_deltas(times, nbursts[0], len_bursts[0], deltas)
    if deltas[0] is NULL:
        PyMem_Free(stpath[0])
        stpath[0] = NULL
        PyMem_Free(idxs[0])
        idxs[0] = NULL
        return tuple(), tuple(), err
    single[0] = sngl
    return shape, indexes, paths, None


cdef tuple cpath_ll_full(int64_t nbursts, int64_t *burst_sizes, int32_t **cdeltas, uint8_t **cindexes, uint8_t **cstate_path, h2mm_mod *model, int64_t ncore):
    cdef cnp.ndarray[object, ndim=1] loglik = np.empty(nbursts, dtype=object)
    cdef object obtemp
    cdef cnp.ndarray[double, ndim=1] temp
    cdef int64_t i
    # allocate loglik arrays
    cdef double **ll = <double**> PyMem_Malloc(sizeof(double*)*nbursts)
    if ll is NULL:
        return MemoryError("insufficient memory"), loglik
    for i in range(nbursts):
        try:
            obtemp = np.empty(burst_sizes[i], dtype=np.double)
        except Exception as exception:
            PyMem_Free(ll)
            return exception, loglik
        temp = obtemp
        loglik[i] = obtemp
        ll[i] = <double*> temp.data
    cdef int ret = pathloglik_path(nbursts, burst_sizes, cdeltas, cindexes, cstate_path, model, ll, ncore)
    PyMem_Free(ll)
    return ret, loglik


cdef tuple cpath_ll(int64_t nbursts, int64_t *burst_sizes, int32_t **cdeltas, uint8_t **cindexes, uint8_t **cstate_path, h2mm_mod *model, int64_t ncore):
    cdef cnp.ndarray[double, ndim=1] loglik = np.empty(nbursts, dtype=np.double)
    cdef double *ll = <double*> loglik.data
    cdef int ret = pathloglik(nbursts, burst_sizes, cdeltas, cindexes, cstate_path, model, ll, ncore)
    return ret, loglik


def path_loglik(h2mm_model model, indexes, times, state_path, num_cores=None, 
                BIC=True, total_loglik=False, loglikarray=False, loglikpath=False):
    """
    Function for calculating the statistical parameters of a specific state path 
    and data set given a model (ln(P(XY|lambda))). By default returns the BIC of
    the entire data set, can also return the base logliklihood in total and/or
    per burst. Which return values selected are defined in keyword arguments.

    Parameters
    ----------
    model : h2mm_model
        :class:`h2mm_model` for which to calculate the loglik of the path.
    indexes : list[numpy.ndarray], tuple[numpy.ndarray] or numpy.ndarray[numpy.ndarray]
        Indexes of photons in bursts in data (simulated or real).
    times : list, tuple, or numpy.ndarray of integer numpy.ndarray
        Arrival times of photons in bursts in (simulated or real).
    state_path : list, tuple, or numpy.ndarray of integer numpy.ndarray
        state for each photon in bursts in data (must be infered through
        viterbi or other algorithm).
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
    loglikpath : bool, optional
        Whether to return the loglikelihoods of each photon in each burst
        as a numpy array of numpy arrays. The default is False

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
    loglik_array : numpy.ndarray
        Loglikelihood of each burst separately
    loglikpath: numpy.ndarray
        Loglikelihood of each photon of each burst array.

    """
    cdef int64_t ncore = <int64_t> optimization_limits._get_num_cores(num_cores)
    cdef bint single
    cdef int64_t nbursts = 0
    cdef int64_t *burst_sizes = NULL
    cdef int32_t **cdeltas = NULL
    cdef uint8_t **cindexes = NULL
    cdef uint8_t **cstate_path = NULL
    shape, indexes, state_path, err = cast_indexes_times_paths(indexes, times, state_path, &single, &nbursts, &burst_sizes, &cindexes, &cdeltas, &cstate_path)
    if err is not None:
        raise err
    cdef int64_t i
    cdef int64_t nphot = sum(burst_sizes[i] for i in range(nbursts))
    if loglikpath:
        res, loglikp = cpath_ll_full(nbursts, burst_sizes, cdeltas, cindexes, cstate_path, model.model, ncore)
        if not isinstance(res, Exception):
            loglik = np.array([l.sum() for l in loglikp], dtype=np.double)
    else:
        res, loglik = cpath_ll(nbursts, burst_sizes, cdeltas, cindexes, cstate_path, model.model, ncore)
    # cdef cnp.ndarray[double, ndim=1] loglik = np.empty(nbursts, dtype=np.double)
    # cdef double *ll = <double*> loglik.data
    # cdef res = pathloglik(nbursts, burst_sizes, cdeltas, cindexes, cstate_path, model.model, ll, ncore)
    free_idx_diffs_path_arrays(nbursts, burst_sizes, cindexes, cdeltas, cstate_path)
    # catch errors
    if isinstance(res, Exception):
        raise res
    elif res == -2:
        raise ValueError("Detector indexes out of range")
    elif res == -1:
        raise ValueError("Empty or null array, raise issue on github")
    elif res < -2:
        raise RuntimeError("Uknown error, raise issue on github")
    # build output
    out = list()
    cdef double sum_loglik = np.sum(loglik)
    if BIC:
        out.append((model.k * np.log(nphot)) - (2*sum_loglik))
    if total_loglik:
        out.append(sum_loglik)
    if loglikarray:
        out.append(loglik.reshape(shape))
    if loglikpath:
        out.append(loglikp.reshape(shape))
    if len(out) == 1:
        out = out[0]
    else:
        out = tuple(out)
    return out


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
    cdef cnp.ndarray[uint8_t, ndim=1] path = np.empty(lenp, dtype=np.uint8)
    cdef uint8_t* path_n = <uint8_t*> path.data
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = statepath(hmod.model, lenp, path_n, seedp)
    if exp != 0:
        raise RuntimeError("Unknown error, raise issue on github")
    return path


def sim_sparsestatepath(h2mm_model hmod, times, seed=None):
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
    times = np.ascontiguousarray(times, dtype=np.int64)
    if times.ndim != 1:
        raise TypeError("times array must be 1D")
    if times.shape[0] < 3:
        raise ValueError("Must have at least 3 times")
    cdef cnp.ndarray[int64_t, ndim=1] times_arr = times
    cdef int64_t lenp = <int64_t> times_arr.shape[0]
    cdef int64_t* times_n = <int64_t*> times_arr.data
    cdef cnp.ndarray[uint8_t, ndim=1] path = np.empty(lenp, dtype=np.uint8)
    cdef uint8_t* path_n = <uint8_t*> path.data
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = sparsestatepath(hmod.model, lenp, times_n, path_n, seedp)
    if exp == 1:
        raise ValueError("Out of order photon")
    elif exp != 0:
        raise RuntimeError("Unknown error, raise issue on github")
    return path


def sim_phtraj_from_state(h2mm_model hmod, states, seed=None):
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
    states = np.ascontiguousarray(states, dtype=np.uint8)
    if states.ndim != 1:
        raise TypeError("Times must be 1D")
    if states.shape[0] < 3:
        raise ValueError("Must have at least 3 time points")
    cdef cnp.ndarray[uint8_t, ndim=1] states_arr = states
    cdef int64_t lenp = states_arr.shape[0]
    cdef uint8_t* states_n = <uint8_t*> states_arr.data
    cdef cnp.ndarray[uint8_t, ndim=1] dets = np.empty(lenp, dtype=np.uint8)
    cdef uint8_t* dets_n =  <uint8_t*> dets.data
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = phpathgen(hmod.model, lenp, states_n, dets_n, seedp)
    if exp != 0:
        raise RuntimeError("Unknown error, raise issue on github")
    return dets


def sim_phtraj_from_times(h2mm_model hmod, times, seed=None):
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
    times = np.ascontiguousarray(times, dtype=np.int64)
    if times.ndim != 1:
        raise TypeError("times array must be 1D")
    if times.shape[0] < 2:
        raise ValueError("Must have at least 2 times")
    cdef cnp.ndarray[int64_t, ndim=1] ctimes = times
    cdef int64_t lenp = times.shape[0]
    cdef int64_t* times_n = <int64_t*> ctimes.data
    cdef cnp.ndarray[uint8_t, ndim=1] path = np.empty(lenp, dtype=np.uint8)
    cdef uint8_t* path_n = <uint8_t*> path.data
    cdef unsigned int seedp = 0
    if seed is not None:
        seedp = <unsigned int> seed
    cdef int exp = sparsestatepath(hmod.model, lenp, times_n, path_n, seedp)
    if exp == 1:
        raise ValueError("Out of order photon")
    elif exp != 0:
        raise RuntimeError("Unknown error in sparsestatepath, raise issue on github")
    cdef cnp.ndarray[uint8_t, ndim=1] dets = np.empty(lenp, dtype=np.uint8)
    cdef uint8_t* dets_n = <uint8_t*> dets.data
    exp = phpathgen(hmod.model, lenp, path_n, dets_n, seedp)
    if exp != 0:
        raise RuntimeError("Unknown error in phtragen, raise issue on github")
    return path , dets
