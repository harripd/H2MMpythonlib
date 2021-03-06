# H2MM_C

## Project Desciption
H2MM_C is a python extension module that implements the H<sup>2</sup>MM algorithm originally developed by [Pirchi et. al. J. Phys. Chem B. 2016, 120, 13065-12075](http://dx.doi.org/10.1371/journal.pone.0160716) in a highly efficent and multithreaded manner, along with functions for posterior analysis with the *Viterbi* algorithm. 

H2MM_C is also designed from the ground up to handle multiparameter models.

 
## Core Features
- H<sup>2</sup>MM model optimization: finding the ideal model given a set of data.
- *Viterbi* analysis: finds the most likely state path through a set of data given a H<sup>2</sup>MM model
- Reporting of model reliability statistics BIC and ICL

## Installation
Simply unzip the .zip file and run:
'python setup.py install'
On windows, you may need to make sure that Visual Studios is installed- including vcvarsall.bat, which can be added ticking the correct option: [https://social.msdn.microsoft.com/Forums/en-US/1071be0e-2a46-4c30-9546-ea9d7c4755fa/where-is-vcvarsallbat-file?forum=visualstudiogeneral](https://social.msdn.microsoft.com/Forums/en-US/1071be0e-2a46-4c30-9546-ea9d7c4755fa/where-is-vcvarsallbat-file?forum=visualstudiogeneral)

## Core Items
1. h2mm_model the core python extension type of the package: this contains the *H<sup>2</sup>MM* model, which has the core fields:
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
2. EM_H2MM_C: the core function of the package, it takes an initial 'h2mm_model' object, and two lists of numpy arrays as input. The lists of numpy arrays are the data to be optimized. Each array references a burst, the first list is the index of the photon stream, the second is the arrival time. Arrays should be of an integer type, and will be converted to np.uint32 and np.uint64 types for the stream and time respectively.
3. viterbi_path_PhotonByphoton takes the same inputs as EM_H2MM_C, but the 'h2mm_model' should be optimized through 'EM_H2MM_C' first, returns a tuple the:
	- path: the most likely state path
	- scale: the posterior probability of each photon
	- ll: the loglikelihood of the path for each burst
	- icl: the Integrated Complete Likelihood (ICL) of the state path given the model and data, provides an extremum based criterion for selecting the ideal number of states
4. viterbi_sort the viterbi algorithm but with additional parameters included:
	- icl: the Integrated Complete Likelihood (ICL) of teh state path given teh model and data, provides an extremum based criterion for selecting teh ideal number of states
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

## Sample Code
~~~
	# assume we have our data in 2 lists (real data should have hundreds to thousands of bursts)
	streams = [burst_stream1, burst_stream2, burst_streamN] # each element is a numpy array of indexes identifying the stream of each photon
	times = [burst_times1, burst_times2, burst_timesN] # each element is a numpy array of arrival times, must be in in order
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
~~~

And viterbi analysis

~~~
	# doing the fitting
	fitting = viterbi_sort(optimized_model,streams,times)
	
	print(fitting[0]) # print the ICL
	# the state path is in index 1
	print(fitting[1])
~~~

## Acknowledgements
Significant advice and help in understanding C code was provided by William Harris, who was also responsible for porting the code to Windows
## License and Copyright
This work falls under the MIT open source lisence
