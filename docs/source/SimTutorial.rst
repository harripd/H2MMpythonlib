Simulations Tutorial
====================

.. currentmodule:: H2MM_C

.. seealso::

    This can also be viewed as a Jupyter Notebook
    Download :download:`H2MM_Simulation_Tutorial.ipynb <notebooks/H2MM_Simulation_Tutorial.ipynb>`

    Download data file here: :download:`sample_data_3det.txt <notebooks/sample_data_3det.txt>`

The other side of H2MM_C is a set of functions for producing simulated data.

The simulation functions of H2MM_C are starte with :code:`sim_`.
Within |H2MM| there are several "levels" of data:

- **times**, treated as fixed points, part of normal data
- **state** of each data point, these can be simulated from a :class:`h2mm_model` and a set of **times**
- **index** of each data point- the observable of |H2MM|, these can be simulated with **times**, a :class:`h2mm_model` and (optionally) **states**

.. note::

   These simulates are based purely on hidden Markov modeling.
   They are **NOT** molecular, fluorescence or other sort of simulation.
   Use another package if you want to simulate something while explicitly handling such complexities in the simulation

First, let's get our imports out of the way.

.. code-block::

    import os
    import numpy as np
    from matplotlib import pyplot as plt

    import H2MM_C as hm

Basic Simulation
----------------

The most likely simulation function you will use is :func:`sim_phtraj_from_times`.
This function takes a :class:`h2mm_model` object and an array of arrival times (equivalent to 1 element of the :code:`times` list given to :func:`EM_H2MM_C`/:meth:`h2mm_model.optimize`, and nearly all the other non-simulation functions/methods, and returns a set of **states** and **indices** (the latter equivalent to one element of the :code:`indexes` list.

So, first lets generate a random distribution of times:

.. code-block::

    time = np.cumsum(np.random.exponential(100, size=50).astype(int))

Then a model (we'll make a rough approximation of teh model from the 3 detector setup from :doc:`FOptimizationTutorial` 

.. code-block::

    # define the arrays
    prior = np.array([0.63, 0.03, 0.19, 0.15])
    trans = np.array([[0.9997, 0.0001, 0.0001, 0.0001],
                      [2e-5, 1-3.2e-5, 1e-5, 2e-6],
                      [5e-6, 7e-6, 1-2.2e-5, 1e-5],
                      [3e-6, 3e-6, 4e-5, 1-4.6e-5]])
    obs = np.array([[0.62, 0.37, 0.01],
                    [0.14, 0.29, 0.57],
                    [0.44, 0.09, 0.47],
                    [0.84, 0.08, 0.08]])

    # make the model
    sim_model = hm.h2mm_model(prior, trans, obs)


And finally, simulate the data to get an array of **states** and **indices**.
These are simulations producing the resutls of *Viterbi* and the data point indices respectively.

.. code-block::

    simstate, simcolor = hm.sim_phtraj_from_times(sim_model, time)
    simstate, simcolor

| (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
|         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
|         0, 0, 0, 0, 0, 0], dtype=uint32),
|  array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0,
|         1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
|         1, 1, 0, 1, 1, 1], dtype=uint32))

Recreating Data: Multiple Arrays
--------------------------------

Now, if we want to simulate a full data set, you will want to repeat the previous process many times, which means using a loop.

.. code-block::

    # initiate lists
    simtimes = list()
    simstates = list()
    simcolors = list()

    # loop to create each set
    for _ in range(1000):
        # generate new time array
        simtime = np.cumsum(np.random.exponential(100, size=np.random.randint(50,150)).astype(int))
        # simulate data
        simstate, simcolor = hm.sim_phtraj_from_times(sim_model, simtime)
        # append arrays to lists
        simtimes.append(simtime)
        simstates.append(simstate)
        simcolors.append(simcolor)

Using Existing Times
--------------------

Another strategy, which can be a way to check the reasonableness of a model, is to use the actual times of an experiment, and the resulting :class:`h2mm_model`, and then compare the real data to the simulated data.

.. note::

   Markov processes are inherently random, you will need to come up with some metric along which to compare.
   You will certainly not get the same indices in the simulated data and real data.
   The ratios of the indices should be similar however- use you knowledge about the system to figure out a legitimate way to compare.

So let's load the times from the 3 detector data, and simulate the data from that:

.. code-block::

    ##############################################################
    # The code here is just for loading the data
    # load the data
    # color3 = list() # to save memory, we will not load this
    times3 = list()

    i = 0
    with open('sample_data_3det.txt','r') as f:
        for line in f:
            if i % 2 == 0:
                times3.append(np.array([int(x) for x in line.split()],dtype='Q'))
            # No need to load the color, so comment it out
    #         else:
    #             color3.append(np.array([int(x) for x in line.split()],dtype='L'))
            i += 1
    # End of data loading segment
    ##############################################################

    # initiate arrays
    simstates3 = list()
    simcolors3 = list()

    # loop over each time array and simulate each set
    for tm3 in times3:
        # conduct the simulation
        simstate3, simcolor3 = hm.sim_phtraj_from_times(sim_model, tm3)
        # append arrays to lists
        simstates3.append(simstate3)
        simcolors3.append(simcolor3)

Simulating from Components
--------------------------

It is also possible to simulate first **states**, and then with a separate function, simulate the times.
If you only want the state path, you can just do the first step, and then save the memory, computational time, and code complexity, or maybe you already have the state path, so just to the second half.

This is done with the :func:`sim_sparsestatepath` and :func:`sim_phtraj_from_state` functions.

.. note::

   :func:`sim_sparsestatepath` uses only the :attr:`h2mm_model.prior` and :attr:`h2mm_model.trans` arrays to calculate the states, because indices are not calcualted, the :attr:`h2mm_model.obs` array will have no influence on the results.

   Conversely, :func:`sim_phtraj_from_state` already has the states from input, and therefore the only important array for its simulation is the :attr:`h2mm_model.obs` array.

**Simulating a State Path**

.. code-block::

    simtime = np.cumsum(np.random.exponential(100, size=50).astype(int))

    statepath = hm.sim_sparsestatepath(sim_model, simtime)
    statepath

| array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
|        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
|        2, 2, 2, 2, 2, 2], dtype=uint32)

**Simulating Data from States**

.. code-block::

    color = hm.sim_phtraj_from_state(sim_model, statepath)
    color

| array([0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 2,
|        1, 2, 0, 0, 2, 1, 1, 2, 2, 2, 0, 0, 2, 2, 0, 2, 1, 0, 0, 2, 2, 0,
|        0, 2, 2, 2, 0, 0], dtype=uint32)


Setting the Random Seed for Reproducibility
-------------------------------------------

Since these simulations are based on a random number generator, results will be different each time.
However, if repeatability is desired, the seed of the random number generator can be set with the keyword argument :code:`seed`.
The same syntax is used across all three simulation functions.

.. note::

   The random seed is persitent, so it should only be set once.
   Each time the seed is set, the counter on the random number generator is reset.

.. code-block::

    simpath, simcolor = hm.sim_phtraj_from_times(sim_model, time, seed=100)
    simpath, simcolor

| (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
|         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
|         0, 0, 0, 0, 0, 0], dtype=uint32),
|  array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
|         0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
|         0, 0, 0, 0, 1, 1], dtype=uint32))



.. |H2MM| replace:: H\ :sup:`2`\ MM
