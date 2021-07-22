# H2MM_C

## Project Desciption
H2MM_C is a python extension module that implements the H<sup>2</sup>MM algorithm originally developed by [Pirchi et. al. J. Phys. Chem B. 2016, 120, 13065-12075](http://dx.doi.org/10.1371/journal.pone.0160716) in a highly efficent and multithreaded manner, along with functions for posterior analysis with the *Viterbi* algorithm. 

H2MM_C is also designed from the ground up to handle multiparameter models.

The API is intended to be user friendly, while still allowing for great flexibility. Suggestions are welcome for way to improve the code and interface.
 
## Core Features
- H<sup>2</sup>MM model optimization: finding the ideal model given a set of data.
	- limit functions to bound the values that a model can take
- *Viterbi* analysis: finds the most likely state path through a set of data given a H<sup>2</sup>MM model
	- Reporting of model reliability statistics BIC and ICL 

## Installation
Simply unzip the .zip file and run:
'python setup.py build_ext --inplace' to install the package in the local directory, or 'python setup.py install' if you want to install H2MM_C to the system directory, allowing access to H2MM_C from anywhere.
### System Peculiarities
#### Windows
On windows, you may need to make sure that Visual Studios is installed- including vcvarsall.bat, which can be added ticking the correct option: [https://social.msdn.microsoft.com/Forums/en-US/1071be0e-2a46-4c30-9546-ea9d7c4755fa/where-is-vcvarsallbat-file?forum=visualstudiogeneral](https://social.msdn.microsoft.com/Forums/en-US/1071be0e-2a46-4c30-9546-ea9d7c4755fa/where-is-vcvarsallbat-file?forum=visualstudiogeneral)
So far H2MM_C has not been tested using the MinGW compiler, but this may be an option for those who do not want to install VisualStudios.
#### Linux
Different Linux distributions may have slightly different synatax, make the proper adjustments, for instance, in Mint, the command 'python3' is used instead of 'python', and you often the install also requires running with 'sudo' so the proper commands are 'sudo python3 setup.py build_ext --inplace' and 'sudo python3 setup.py install'.
Therefore the correct 
## Core Items
1. h2mm_model: the core python extension type of the package: this contains the *H<sup>2</sup>MM* model, which has the core fields:
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
2. h2mm_limits: class for bounding the values a model can take, min/max values can be specified for all 3 core arrays (trans, obs, and prior) of an h2mm_model, either as single floats, or full arrays, values are specified as keyword arguments, not specifiying a value for a particular field will mean that field will be unbounded
	- min_trans: the minimum values for the trans array (ie the slowest possible transition rate(s) allowed), values on the diagonal will be ignored
	- max_trans: the maximum values for the trans array (ie the fastest possible transition rate(s) allowed), values on the diagonal will be ignored
	- min_obs: the minimum values for the obs array
	- max_obs: the maximum values for the obs array
	- min_prior: the minimum value for the prior array
	- max_prior: the maximum value for the prior array
3. EM_H2MM_C: the core function of the package, it takes an initial 'h2mm_model' object, and two lists of numpy arrays as input. The lists of numpy arrays are the data to be optimized. Each array references a burst, the first list is the index of the photon stream, the second is the arrival time. Arrays should be of an integer type, and will be converted to np.uint32 and np.uint64 types for the stream and time respectively. Returns:
	- h2mm_model: the model optimized for the given input data.
4. H2MM_arr: calculate the loglik of a bunch of h2mm_model objects at once, but with no optimization. The first agruments can be an h2mm_model, of a list, tuple, or numpy array of h2mm_model objects. The second and third arguments are the same as in EM_H2MM_C
5. viterbi_path: takes the same inputs as EM_H2MM_C, but the 'h2mm_model' should be optimized through 'EM_H2MM_C' first, returns a tuple the:
	- path: the most likely state path
	- scale: the posterior probability of each photon
	- ll: the loglikelihood of the path for each burst
	- icl: the Integrated Complete Likelihood (ICL) of the state path given the model and data, provides an extremum based criterion for selecting the ideal number of states
6. viterbi_sort: the viterbi algorithm but with additional parameters included:
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

## Tutorial Code
First the import statements

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

## Acknowledgements
Significant advice and help in understanding C code was provided by William Harris, who was also responsible for porting the code to Windows
## License and Copyright
This work falls under the MIT open source lisence
