About |H2MM|
============

What is |H2MM|?
---------------

|H2MM| is an extension of hiddenm Markov modeling, a broad set of methods for finding underlying behavior in noisy systems.
The basic assumptions of hidden Markov models are:

#. The system is described by a Markvov model
    a. The data describes a system that transitions between a set of states.
    b. When the system is in a given state, it has a distinct set of probabilities to transition to each of the other states
    c. The system is memoryless, that is the probability to transition to another state is only influenced by the current state, and not any of the previous states
#. The model is hidden, meaning
    a. The states cannot be directly observed, rather
    b. Each state has a certain probability to produce certain observable results, with each state having a different set of probabilities

The earliest applications of HMM were towards language processing, for which it was not very successful, but has found much greater use in signal processing.
HMM has also found extensive use in analyzing TIRF based smFRET trajectories.
All of these applications however, assume that there is a constant data rate, for a camera with a set frame rate so that there is an image every ms.

This is not the case for single molecule confocal based data using single photon detectors.
Here, data comes in sparsely, as individual photons, with varying interphoton times.
|H2MM| extends the HMM algorithm to accept these variable interphoton times, allowing application of the HMM machinery with confocal data without implementing an external time bin.

HMM methods use the Baum-Welch algorithm in an optimization process which finds the model of a set number of states that best describes the data.
Now since the number of states is fixed, other optimizations must be conducted with different numbers of states.
Then the different optimized models must be compared, and the best one choosen, and the ones with too many (overfit) and too few (underfit) states regected.

A final thing to understand about |H2MM|, is the use of indeces.
In |H2MM|, data comes in a set of detetector chanels, in the first iterations, these were exclusively the |DD| and |DA| channels, but with mp |H2MM|, this was extended to the |AA| channel, and was even suggested to be able to include the parralel and perpendicular channels in anisotropy based measurments.
A comparison to the original application of HMM methods is apt here: originally HMM was developed to anlayze word paterns, so each word was a unique index, and each state had a given probability to produce each word.
So in |H2MM| each photon can be compared to a word, each photon recieves a set of a limited nubmer of indeces.


A Brief History of |H2MM|
-------------------------

Application to confocal single molecule data started with `Gopich and Szabo 2009 <https://doi.org/10.1021/jp903671p>`_ who established the maximum likelihood estimator to calculated the likelihood of a model of transition rates and emission probabilities for a set of data.
`Pirchi and Tsukanov et. al. 2016 <https://doi.org/10.1021/acs.jpcb.6b10726>`_ then integrated the Baum-Welch algorithm, with some reformulation of Gopich and Szabo's original equations, which allowed for an optimization procedure ensuring that the likelihood of each iteration improves.
This made finding the ideal model a feasible undertaking.
However, discrimination between over and underfit models (models with too many or too few states), `Lerner et. al. 2018 <https://doi.org/10.1063/1.5004606>`_ introduced the first attempt at this, with the modified Bayes Information Criterion, and finally the Integrated Complete Likelihood was introduced in `Harris et. al. 2022 <https://doi.org/10.1038/s41467-022-28632-x>`_, which proved a more reliable statistical discriminator.
`Harris et. al. 2022 <https://doi.org/10.1038/s41467-022-28632-x>`_ also introduced the multiparameter approach, where the |AA| stream was integrated allowing discrimination of photophysical and FRET dynamics.

.. |H2MM| replace:: H\ :sup:`2`\ MM
.. |DD| replace:: D\ :sub:`ex`\ D\ :sub:`em`
.. |DA| replace:: D\ :sub:`ex`\ A\ :sub:`em`
.. |AA| replace:: A\ :sub:`ex`\ A\ :sub:`em`