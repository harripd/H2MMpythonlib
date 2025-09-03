# H2MM_C

[![Build and Test](https://github.com/harripd/H2MMpythonlib/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/harripd/H2MMpythonlib/actions)
[![Documentation Status](https://readthedocs.org/projects/h2mmpythonlib/badge/?version=latest)](https://h2mmpythonlib.readthedocs.io/en/latest/?badge=latest)

## Project Desciption
H2MM_C is a python extension module that implements the H<sup>2</sup>MM algorithm originally developed by [Pirchi, Tsukanov et. al. J. Phys. Chem B. 2016, 120, 13065-12075](http://dx.doi.org/10.1371/journal.pone.0160716) in a highly efficent and multithreaded manner, along with functions for posterior analysis with the *Viterbi* algorithm. 

`H2MM_C` was designed from the ground up to handle multiparameter models, described in [Harris, P.D., Narducci, A., Gebhardt, C. et al. Multi-parameter photon-by-photon hidden Markov modeling. Nat Commun 13, 1000 (2022).]( https://doi.org/10.1038/s41467-022-28632-x), which also introduced this package.

The API is intended to be user friendly, while still allowing for great flexibility. Suggestions are welcome for ways to improve the code and interface.

## Full Documentation
Full dodumentation can be found at [h2mmpythonlib.readthedocs.io](https://h2mmpythonlib.readthedocs.io)

### See Also

This package is offers the basic function to perform H<sup>2</sup>MM, which means that it does not restrict the use of H<sup>2</sup>MM to any particular modality. However this means that much of the analysis specific to smFRET applications is left up to the user.

For those 

## Core Features
- H<sup>2</sup>MM model optimization: finding the ideal model given a set of data.
	- limit functions to bound the values that a model can take
- *Viterbi* analysis: finds the most likely state path through a set of data given a H<sup>2</sup>MM model
	- Reporting of model reliability statistics BIC and ICL
- Simulation functions: Monte Carlo type simulations based on a hidden Markov model
	- Useful for verifying results

## Installation

The easiest way to install `H2MM_C` is via pip:

```bash
$ pip install H2MM-C
```
It should be noted that the setup files require a recent version of numpy, and at least Python 3.7.
This is not because the code actually requires features introduced in these verstions, but rather because in Linux, with the numpy version change, the size of certain numpy data-types changed, making compiles with recent versions of numpy incompatible with earler versions. These recent numpy versions do not support Python 3.6.
Therefore, the intrepid programmer can download the code, edit the setup files, and compile with earlier versions of numpy and/or Python, and the code *should* still work.

### Alternative Installation Methods

If for some reason installing from PyPi doesn't work, you can try installing directly from github:

```bash
$ pip install git+https://github.com/harripd/H2MMpythonlib
```

Or, if you download the repository, and have the files stored locally, from the top directory of the project (where the `setup.py` file is):

```bash
$ python setup.py install
```

or if you only want to have it accessible from the current directory, use:

```bash
$ python setup.py build_ext --inplace
```

*Note on some systems the commands are `pip3` and `python3` instead of `pip` and `python` respectively.

### Compatibility

We are trying to provide the broadest degree of compatibility as possible, but there are limitations, and assistance is welcome to expand compatibility as much as possible. 
Currently we are using github actions and `cibuildwheel` to generate wheels that are uploaded to PyPi, as well as the sdist archive (`.tar.gz` file), however we are having trouble building certain wheels.

Working wheels:

- Windows wheels
- manylinux wheels
- MacOS X wheels

Currently we do not have wheels for:

- musllinux

For systems we do not have wheels for, it may still be possible to compile from the sdist archive (`.tar.gz` file). 
`cibuildwheel` uses the most recent version of numpy. This means that these wheels will generally not work if you have a version of numpy before 1.20.0, and therefore this is given as a minimum version requirement. 
However, we have been able to compile working versions with lesser versions of numpy. 
Therefore, if you wish to keep your version of numpy, we suggest downloading the github repository, editing the `setup.py` file to support your version of numpy, and compiling with the following commands (run from the base directory where you have your local copy).
```bash
$ python setup.py bdist_wheel sdist
$ pip install *path to wheel file*
```

## Tutorial Code
For a full tutorial on H2MM_C, please see the zenodo repository:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5566886.svg)](https://doi.org/10.5281/zenodo.5566886)

Below is a small sample of tutorial code.
~~~
	# H2MM_C accepts numpy arrays, so we mush import numpy
	import numpy as np
	import H2MM_C
	# note: it can be more convenient to use from H2MM_C import * so that it is unnecessary to type H2MM_C. repeatedly

	###Data must be defined, so here is some data *made purely for demonstration, and not meant to be realistic*

	# lets define sum fake bursts IMPORTANT: this is ENTIRELY FOR DEMONSTRATION, the fake data below is not based on any model
	# burst 1
	burst_stream1 = np.array([  0,  1,  0,  1,  0,  2,  0,  1,  2,  0,  1,  2]) 
	burst_times1 =  np.array([100,110,112,117,123,124,128,131,139,148,168,182]) # note that burst_stream1 is of the same length as burst_times1
	
	# burst 2
	burst_stream2 = np.array([  2,  1,  0,  0,  2,  1,  0,  1,  0,  0])
	burst_times2  = np.array([202,231,340,370,372,381,390,405,410,430]) # note that burst_stream2 is of the same length as burst_times2, but different from burst_stream1 and burst_stream1
	
	# burst N
	burst_streamN = np.array([  0,  2,  1,  2,  0,  2,  1,  0,  1,  2,  1,  0,  1,  0,  0])
	burst_timesN  = np.array([500,502,511,515,518,522,531,540,544,548,561,570,581,590,593]) # again burst_streamN is the same length as burst_timeN


	###The burst arrays must now be put into two lists, one for the photon streams and one for the arrival times


	# Now the bursts must be put into lists (real data should have hundreds to thousands of bursts)
	# Also, normally, you will be importing the data from a file, so the previous definition of burst_streamN and burst_timesN
	# will more likely be done with a file read, or by using your burst-analysis software to split your data into bursts
	streams = [burst_stream1, burst_stream2, burst_streamN] # each element is a numpy array of indexes identifying the stream of each photon
	times = [burst_times1, burst_times2, burst_timesN] # each element is a numpy array of arrival times, must be in in order


	###The above does not invoke H2MM_C (except the import statements), they are purely for demonstrating how to format the data that H2MM_C accepts.
	###The rest is actually using H2MM_C, first, an initial model must be defined, (an object of the 'H2MM_C.h2mm_model' class) and then initial model and data can be given to the  'H2MM_C.EM_H2MM_C' for optimization.


	# first define the initial arrays for the initial guess
	prior = np.array([0.3, 0.7]) # 1D array, the size is the number of states, here we have 2, the array will sum to 1
	trans = np.array([[0.99, 0.01],[0.01,0.99]]) # 2D square array, each dimenstion the number of states
	obs = np.array([[0.1, 0.4, 0.5],[0.3, 0.2, 0.5]]) # 2D array, number of rows is the number of states, the number of columns is the number of detectors
	
	# Now make the initial model
	initial_model = H2MM_C.h2mm_model(prior,trans,obs) 
	
	# Run the main algorithm
	optimized_model = H2MM_C.EM_H2MM_C(initial_model,streams,times)
	
	# Printing out the main results
	print(optimized_model.prior, optimized_model.trans, optimized_model.obs)
	# Print out the number of iterations it took to converge
	print(optimized_model.niter)


	###And viterbi analysis


	# doing the fitting
	fitting = H2MM_C.viterbi_sort(optimized_model,streams,times)
	
	print(fitting[0]) # print the ICL
	# the state path is in index 1
	print(fitting[1])
~~~


## Classes
1. `h2mm_model`: the core python extension type of the package: this contains the *H<sup>2</sup>MM* model, which has the core fields:
	- nstate: the number of states in the model
	- ndet: the number of photon streams in the model
 	- trans: the transition probability matrix
	- obs: the emmision probability matrix, shape nstate x ndet
	- prior: the prior probability, shape nstate
	- k: the number of free parameters in the model
	- loglik: the loglikelihood of the model
	- nphot: the number of photons in the dataset that the model is optimized against
	- bic: the Baysian Information Criterion of the model
	- converged: True if the model reached convergence criterion, False if the optimization stopped due to reaching the maximum number of iterations or if an error occured in the next iteration.

2. `h2mm_limits`: class for bounding the values a model can take, min/max values can be specified for all 3 core arrays (trans, obs, and prior) of an h2mm_model, either as single floats, or full arrays, values are specified as keyword arguments, not specifiying a value for a particular field will mean that field will be unbounded
	- min_trans: the minimum values for the trans array (ie the slowest possible transition rate(s) allowed), values on the diagonal will be ignored
	- max_trans: the maximum values for the trans array (ie the fastest possible transition rate(s) allowed), values on the diagonal will be ignored
	- min_obs: the minimum values for the obs array
	- max_obs: the maximum values for the obs array
	- min_prior: the minimum value for the prior array
	- max_prior: the maximum value for the prior array

## Functions

1. `EM_H2MM_C`: the core function of the package, used to perform model optimizations.
    
    Arguments:
    - Initial model : an `h2mm_model` object that will be optimized.
    - Streams: a set of burst photon indeces. Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s of photon indeces, must be of integer type, and positive. The indeces will be converted to `unsigned long int` when given to C-code
    - Times: a set of burst photon times (macrotimes), Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s. The macrotimes will be converted to `unsigned long long int` when given to C-code. Therefore while floating point arrays are accepted, they are **strongly discouraged**. Must be same length as streams
    
    Returns:
	- Optimized model: the `h2mm_model` optimized for the given input data.

2. `H2MM_arr`: calculate the loglik of a bunch of h2mm_model objects at once, but with no optimization. The first agruments can be an h2mm_model, of a list, tuple, or numpy array of h2mm_model objects. The second and third arguments are the same as in EM_H2MM_C
    
    Arguments:
    - Models : a `list`, `tuple` or `numpy.ndarray` of `h2mm_model` objects whose *loglikelihood* will be calculated agains the given data.
    - Streams: a set of burst photon indeces. Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s of photon indeces, must be of integer type, and positive. The indeces will be converted to `unsigned long int` when given to C-code
    - Times: a set of burst photon times (macrotimes), Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s. The macrotimes will be converted to `unsigned long long int` when given to C-code. Therefore while floating point arrays are accepted, they are **strongly discouraged**. Must be same length as streams
    
    Returns:
	- Calculated Models: a set of `h2mm_model` objects organized in the same way as Models
    
3. `viterbi_path`: takes the same inputs as EM_H2MM_C, but the 'h2mm_model' should be optimized through 'EM_H2MM_C' first, returns a tuple the:
    Arguments:
    - Model : an `h2mm_model` object that will has been optimized against the given data.
    - Streams: a set of burst photon indeces. Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s of photon indeces, must be of integer type, and positive. The indeces will be converted to `unsigned long int` when given to C-code
    - Times: a set of burst photon times (macrotimes), Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s. The macrotimes will be converted to `unsigned long long int` when given to C-code. Therefore while floating point arrays are accepted, they are **strongly discouraged**. Must be same length as streams
    
    Returns:
	- path: the most likely state path
	- scale: the posterior probability of each photon
	- ll: the loglikelihood of the path for each burst
	- icl: the Integrated Complete Likelihood (ICL) of the state path given the model and data, provides an extremum based criterion for selecting the ideal number of states

4. `viterbi_sort`: the viterbi algorithm but with additional parameters included:
    Arguments:
    - Model : an `h2mm_model` object that will has been optimized against the given data.
    - Streams: a set of burst photon indeces. Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s of photon indeces, must be of integer type, and positive. The indeces will be converted to `unsigned long int` when given to C-code
    - Times: a set of burst photon times (macrotimes), Must be given as a `list`, `tuple` or 1-D object `numpy.ndarray` of 1D `numpy.ndarray`s. The macrotimes will be converted to `unsigned long long int` when given to C-code. Therefore while floating point arrays are accepted, they are **strongly discouraged**. Must be same length as streams
    
    Returns:
	- icl: the Integrated Complete Likelihood (ICL) of the state path given the model and data, provides an extremum based criterion for selecting the ideal number of states
	- path: the most likely state path
	- scale: the posterior probability of each photon
	- ll: the loglikelihood of the path for each burst
	- burst_type: a binary classification of which states are in each burst
	- dwell_mid: returns the lengths of dwells in each state, for dwells with full residence time in the burst
	- dwell_beg: same as dwell_mid, except for dwells that begin each burst
	- dwell_end: same as dwell_beg, but for ending dwells
	- ph_counts: gives counts of photons per stream per dwell
	- ph_mid: same as ph_counts, but further sorted as in dwell_mid
	- ph_beg: same as ph_counts, but futher sorted as in dwell_beg
	- ph_end: same as ph_counts, but futher sorted as in dwell_end
	- ph_burst: same as ph_counts, but futher soreted as in dwell_burst

5. `sim_statepath`: from an model, generate a random state path of equally spaced time points
    
    Arguments:
    - Model: a `h2mm_model` object to use as the defined parameters of the simulation
    - Length: the number of time steps to simulate, defines the number of elements in the ouput array
    
    Returns:
    - Path: an array of the states of the system at each time point, based on Monte-Carlo simulation

6. `sim_sparsestatepath`: from a model and a set of sparse times, generate a random state path
    
    Arguments:
    - Model: a `h2mm_model` object to use as the defined parameters of the simulation
    - Times: a 1D `numpy.ndarray` object of times for a simulated burst
    
    Returns:
    - Path: the states of the simulated photons based on the input times

7. `sim_phtraj_from_state`: randomly select photons given a set of states and a model
    Arguments:
    - Model: a `h2mm_model` object to use as the defined parameters of the simulation.. Note: the model transition rates are ignored, only the emission probability matrix is considered
    - states: a 1D `numpy.ndarray` of positive integers, specifying the state of each photon. Note: this state path over-rides any transition probability matrix used in the model
    
    Returns:
    - Stream: the indeces (photon indeces) of the simulated photons

8. `sim_phtraj_from_times`: from a model and a set of sparse times, generate a random photon trajectory

    Arguments:
    - Model: a `h2mm_model` object to use as the defined parameters of the simulation
    - Times: a 1D `numpy.ndarray` object of times for a simulated burst
    
    Returns:
    - Path: 
    - Stream: a 1D `numpy.ndarray` of the simulated photon indeces (streams) 

## Acknowledgements
Significant advice and help in understanding C code was provided by William Harris, who was also responsible for porting the code to Windows
## License and Copyright
This work falls under the MIT open source lisence
