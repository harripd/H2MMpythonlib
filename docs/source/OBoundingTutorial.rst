Bounding |H2MM|
===============

.. currentmodule:: H2MM_C

.. seealso::

    This can also be viewed as a Jupyter Notebook
    Download :download:`H2MM_Bounds_Tutorial-OOP.ipynb <notebooks/H2MM_Bounds_Tutorial-OOP.ipynb>`

    The data file can be downloaded here: :download:`sample_data_3det.txt <notebooks/sample_data_3det.txt>`

Let's get our obligatory imports in order, this time we'll start with the 3 detector data.

.. code-block::

    import os
    import numpy as np
    from matplotlib import pyplot as plt

    import H2MM_C as hm

    # load the data
    def load_txtdata(filename):
        color = list()
        times = list()
        with open(filename,'r') as f:
            for i, line in enumerate(f):
                if i % 2 == 0:
                    times.append(np.array([int(x) for x in line.split()],dtype=np.int64))
                else:
                    color.append(np.array([int(x) for x in line.split()],dtype=np.uint8))
        return color, times
    
    color3, times3 = load_txtdata('sample_data_3det.txt')


Bounding Parameter Values
-------------------------

Sometimes we may want to restrain the range of possible values in an |H2MM| model (the :class:`h2mm_model`).
For instance, we may want to keep transition rates (:attr:`h2mm_model.trans`) within values reasonable for the duration of the experiment, or because you want restrain the values in the emission probability matrix (:attr:`h2mm_model.obs`) based on how we know the system should behave.

Bounds are defined using :class:`h2mm_limits` objects.
This object is passed to :meth:`h2mm_model.optimize` (or :func:`EM_H2MM_C` in a functional approach) through the :code:`bounds` keyword argument.
When this is specified, it is also necessary to specify another keyword argument, :code:`bounds_func`.

**So let's see an example**

.. code-block::

    alt_period = 4000 # a fake alternation period
    us_bounds = hm.h2mm_limits(max_trans = 1/(alt_period))

    prior = np.array([1/4, 1/4, 1/4, 1/4])
    trans = np.array([[1-3e-6, 1e-6, 1e-6, 1e-6],
                      [1e-6, 1-3e-6, 1e-6, 1e-6],
                      [1e-6, 1e-6, 1-3e-6, 1e-6],
                      [1e-6, 1e-6, 1e-6, 1-3e-6]])
    obs = np.array([[0.4,0.4,0.2],
                    [0.3,0.1,0.6],
                    [0.2,0.4,0.4],
                    [0.1,0.1,0.8]])

    us_opt_model4 = hm.h2mm_model(prior, trans, obs)

    us_opt_model4.optimize(color3, times3, bounds_func='revert', bounds=us_bounds)
    us_opt_model4


| The model converged after 631 iterations

| nstate: 4, ndet: 3, nphot: 436084, niter: 631, loglik: -408203.01780807425 converged state: 0x27
| prior:
| 0.19742522045704233, 0.5611254558625469, 0.24144932368041058, 7.251074733815892e-42
| trans:
| 0.9999562426518485, 2.620839826060098e-05, 1.818962272427181e-06, 1.5729987618497615e-05
| 7.049720131795914e-06, 0.9999698856343252, 6.991342045371946e-06, 1.607330349755956e-05
| 1.2716807355060168e-06, 1.7388217608512278e-05, 0.9999781791003083, 3.1610013477364933e-06
| 1.7301823234579738e-05, 0.00011452568669777342, 8.076641015599715e-06, 0.999860095849052
| obs:
| 0.849528664181505, 0.07564782657329712, 0.074823509245198
| 0.47168581743329263, 0.09134399902467148, 0.4369701835420359
| 0.14909987819343531, 0.31276918990273284, 0.5381309319038319
| 0.15084679777173912, 0.07681315977150306, 0.7723400424567578


So, what did we just to?
The :class:`h2mm_limits` object :code:`us_bounds` prevents any value (except on-diagonal values) of the **transition probability matrix** (:attr:`h2mm_model.trans`) from ever being larger (i.e. faster transition rate) than :code:`1/4000`

Bounds Process
--------------
When you use a bounds method, each iteration goes through the following steps:

#. Calculate the *loglikelihood* and a new model
#. Check if the optimization has converged
#. Analyze the new model, and correct if necessary.

   a. Check if any values are smaller or larger than the limits set in the :code:`bounds` argument.
   b. If values are out of bounds, apply a correction, based on the method defined by the argument passed to :code:`bounds_func`

#. Repeat optimization (return to step 1)

When creating a :class:`h2mm_limits` object, all arguments are passed as keyword arguments.
They come in the form of :code:`min/max_[name]` where :code:`[name]` is :code:`prior`, :code:`trans`, or :code:`obs` (the core arrays of :class:`h2mm_model` objects).
These specify the minimum/maximum values in the respective array.
If no value is specified for a given min/max array, then the values of that array can be as small or as large as possible for an unbounded model.

In all cases, these values can be specified as either a float or an array.
- If specified as a float, then the value is universal for all values in the given array (this is most useful for the :attr:`h2mm_model.trans` matrix). This means less granularity, but then the :class:`h2mm_limits` object can be used for any number of states/detectors
- If specified as an array, then the values in the array are matched with the corresponding array in :class:`h2mm_model`. This means greater granularity, but then you are locked into using the :class:`h2mm_limits` object only for optimizations with a specific number of states /detectors.

As mentioned in the above outline, you also need to specify the :code:`bounds_func` argument when using :code:`bounds`.
There are 3 options for this:

#. :code:`'minmax'`: The shallowest correction, sets the out of bounds value to its minimum or maximum
#. :code:`'revert'`: The preferred method, sets the out of bounds value to the value it was in the previous model
#. :code:`'revert_old'`: A more extreme form of :code:`'revert'` which goes to the model before the last in the optimization, and sets the out of bounds value to that "older" value.

Using :func:`factory_h2mm_model` with Bounds
--------------------------------------------

You will note that in the previous example, we specified the :class:`h2mm_model` explicity, instead of using the :func:`factory_h2mm_model` function.
This is because it is possible that the :func:`factory_h2mm_model` could return a :class:`h2mm_model` whose values are already out of bounds based for the :class:`h2mm_limits` object.
This could create odd behavior during optimization.

There is a way around this: you can call :func:`factory_h2mm_model` with the :class:`h2mm_limits` object passed through the :code:`bounds` keyword argument, and the function will automatically return a :class:`h2mm_model` object that is within the bound provided.

.. note::

    See the documentation of :func:`factory_h2mm_model` for a fill list of options for customizing the function's output.

.. code-block::

    us_bounds = hm.h2mm_limits(max_trans = 1/4000)
    # make factory_h2mm_model make a model within bounds
    us_model = hm.factory_h2mm_model(3,3, bounds=us_bounds)
    us_model.optimize(color3, times3, bounds=us_bounds, bounds_func='revert')

| The model converged after 198 iterations

Custom Bounds
-------------

There is a final option for specifying bounds: write  your own function and hand it to :code:`bounds_func`.

.. note::

   This feature was designed to allow the user to handle things/circumstances that the writers of H2MM_C had not anticipated.
   Therefore this example is very simple, and does not show a useful method.

So how does it work?

The function **must** take 4 input arguments, which will be handed to it ever iteration by :func:`EM_H2MM_C`, these are (in order):

#. :code:`new_model`: The model that will be optimized next, this is the one that should be bounded
#. :code:`current_model`: The model whose loglikelihood was just calculated, this is the model that :code:`'revert'` uses to reset an out of bounds value
#. :code:`old_model`: The model that was calculated before the current one, this is the model that :code:`'revert_old'` uses to reste an out of bounds value
#. :code:`bound`: The argument passed to the :code:`bound` keyword argument, if not specified, it will be :code:`None`, and can be anything (so long as you are passing a function to :code:`bounds_func`

Finally, the function must return a single :class:`h2mm_model` object, which is the function that will actually be calculated next- this is the equivalent of a "corrected" model based on the bounds.

.. warning::

    Make sure the model you return makes sense, otherwise the optimization will behave very strangely.
    Think of this as the "gloves off" appraoch, you might have a very powerful new method, or you might get something meaningless depending on how you code it.
    That's your responsibility.

**So let's see this in action**

.. code-block::

    def sample_bounds(new_model,current_model,old_model,bound):
        # it's usually best to just keep the function signature the same
        # grab the obs matrix
        obs = new_model.obs
        # set first row of obs matrix to bound
        obs[0,:] = bound
        # change the obs matrix of the new model
        new_model.obs = obs
        # return the adjusted model
        return new_model

.. code-block::

    bnd = np.array([0.09,0.01,0.9])
    prior = np.array([1/4, 1/4, 1/4, 1/4])
    trans = np.array([[1-3e-6, 1e-6, 1e-6, 1e-6],
                      [1e-6, 1-3e-6, 1e-6, 1e-6],
                      [1e-6, 1e-6, 1-3e-6, 1e-6],
                      [1e-6, 1e-6, 1e-6, 1-3e-6]])
    obs = np.array([bnd,
                    [0.3,0.1,0.6],
                    [0.2,0.4,0.4],
                    [0.1,0.1,0.8]])

    us_opt_model4 = hm.h2mm_model(prior, trans, obs)
    us_opt_model4.optimize(color3, times3, bounds_func=sample_bounds, bounds=bnd)

| The model converged after 712 iterations


.. |H2MM| replace:: H\ :sup:`2`\ MM
