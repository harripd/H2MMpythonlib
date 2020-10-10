# H2MMpythonlib

Photon by Photon hidden Markov Modeling (H2MM)
Initially developed by Pirchi et. al. JPC 2016, 120, 13065

The original version of this code was written in Matlab by Pirchi et. al.
This repo is written in Python by Paul David Harris, with the objective of implementing the algorithm in Python to improve speed and make directly integrateable with the fretbursts Python library (Ingargiola et. al. (2016). http://dx.doi.org/10.1371/journal.pone.0160716)

The primary functions of this library are the EM_H2MM, and the equivalent EM_H2MM_par, which differ only in that EM_H2MM_par is accelerated by parallel processing.
These functions use Maximum Likelihood Estimators to optimize a hidden Markov Model of data produced primarily by diffusing smFRET measurements- which are composed of arrival times of single photons at two or more detectors. This method is built off of well established hidden Markov modeling (HMM) but extends it to efficiently handel variable times between observations.
Inputs to the functions are:
    h2mm_model object: an object of the h2mm_model class specific to H2MMpythonlib, which contains three numpy arrays:
        prior: initial state distribution
        trans: the transition probability matrix
        obs: the emmision proability matrix, giving the probability of a photon arriving at a given detector in a given state
    ArrivalColors: a list of numpy int arrays correspondig to the colors (detectors)(indexed from 0) of the photons
    ArrivalTimes: a list of numpy int arrays, 1 array per burst, corresponding to the arrival times of the photons
The output of EM_H2MM and EM_H2MM_par is another h2mm_model object, which contains the optimized parameters.

This algorithm is both powerful and risky to use.
The main difficulty arrises due to the strong potential of overfitting.
Yet the potential to detect and quantify hidden dynamics is too good to pass up.
I ask any researcher to apply this method cautiously, but hopefully.
