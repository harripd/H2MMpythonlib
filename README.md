# H2MMpythonlib

Photon by Photon hidden Markov Modeling (H2MM)
Initially developed by Pirchi et. al. JPC 2016, 120, 13065

The original version of this code was written in Matlab by Pirchi et. al.
This repo is written in Python by Paul David Harris, with the objective of implementing the algorithm in Python to improve speed and make directly integrateable with the fretbursts Python library (Ingargiola et. al. (2016). http://dx.doi.org/10.1371/journal.pone.0160716)

The primary functions of this library are the EM_H2MM, and the equivalent EM_H2MM_par, which differ only in that EM_H2MM_par is accelerated by parallel processing.
These functions use Maximum Likelihood Estimators to optimize a hidden Markov Model of data produced primarily by diffusing smFRET measurements- which are composed of arrival times of single photons at two or more detectors. This method is built off of well established hidden Markov modeling (HMM) but extends it to efficiently handel variable times between observations.
Inputs to the functions are a h2mm_model object, which is composed of the prior or initial state vector, the transition probability matrix, and the observational likelihood matrix.
