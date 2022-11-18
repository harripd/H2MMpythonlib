H2MM_C Secondary Control Features
=================================

.. currentmodule:: H2MM_C

.. seealso::

    This can also be viewed as a Jupyter Notebook
    Download :download:`H2MM_Control_Optimization.ipynb <notebooks/H2MM_Control_Optimization.ipynb>`

    Download the data file here: :download:`sample_data_3det.txt <notebooks/sample_data_3det.txt>`

First, our obligatory imports and loading of 3 detector data:

.. code-block::

    import os
    import numpy as np
    from matplotlib import pyplot as plt

    import H2MM_C as hm

    # load the data
    color3 = list()
    times3 = list()

    i = 0
    with open('sample_data_3det.txt','r') as f:
        for line in f:
            if i % 2 == 0:
                times3.append(np.array([int(x) for x in line.split()],dtype='Q'))
            else:
                color3.append(np.array([int(x) for x in line.split()],dtype='L'))
            i += 1

Optimization Control
--------------------

Sometimes you want to control when optimizations stop, or how many cores the optimization uses.
Basically the course of the optimization is the same, you're just changing the thresholds for when to stop optimizing.

There are 4 of these "limits":

#. :code:`num_cores` The number of threads to use when optimizing/calculating a model or state-path.
#. :code:`max_iter` The maximum number of iterations to optimize a model until automatically quitting
#. :code:`converged_min` The threshold of improvement required to continue optimizing, i.e. if the new model improves the loglikelihood by less than this value, the optimization will stop.
#. :code:`max_time` The maximum duration to conduct optimization, after which optimization will automatically stop. **Uses inaccurate clock, by default is infinite, and recommended not to be changed**

Setting by Keword Arguments
***************************

These can be adjusted by passing these as keyword arguments to :func:`EM_H2MM_C` and :meth:`h2mm_model.optimize`.

:code:`num_cores` also works in :func:`H2MM_arr`, :meth:`h2mm_model.evaluate`, :func:`viterbi_path`, :func:`viterbi_sort`, in these there are no limits/thresholds that apply to these since they are not optimizations, however, they can be parallelized, and thus :code:`num_cores` is applicable.

Heres a quick example, where the number of optimizations is increased to 7200 iterations:

>>> model_5s3d = hm.EM_H2MM_C(hm.factory_h2mm_model(4,3), color3, times3, max_iter=7200)
Optimization reached maximum number of iterations

Setting Universal Defaults
**************************

The defaults of these are stored in the module variable :code:`H2MM_C.optimization_limits`.

.. note::

   This variable functions similarly to :code:`rcParams` in matplotlib.
   It's purpose is to make it easy to set the default value, instead of having to repeatebly input the same keyword arguments every time.

Values in :code:`H2MM_C.optimization_limits` can be accessed and set like both dictionary keys and attributes.
The default values are:
- :code:`H2MM_C.optimization_limits.num_cores = os.cpu_count() // 2` This value is set on :code:`import H2MM_C` this sets the number of *C* threads (which can run on different cores at the same time, making them like python processes in that regard, but they can share memory) the algorithms in H2MM_C will use. Since most of these algorithms are cpu intensive, they will generally not benefit from multi-threading. Since :code:`os.cpu_count()` actually returns the number of threads, and most CPUs are multi-threaded, :code:`os.cpu_count()` generally returns twice the number of CPUs than the machine actually has. Therefore the choice to set :code:`num_cores = os.cpu_count() //2`. If your machine is not multi-threaded or has some other oddity, consider setting this to a more reasonable value.
- :code:`H2MM_C.optimization_limits.max_iter = 3600` This is perhaps the most arbitrary parameter, set high enough that you are confident the model is good. 3600 was simply set because that is the number of seconds there are in an hour.
- :code:`H2MM_C.optimization_limits.converged_min = 1e-14` This value is very small, near the floating point error for most optimizations, in fact it is often smaller than the floating point error. For especially large data sets, (*roughly* >10,000 trajectories with >75 photons each) the floating point error is even larger, and so it would be recommended to set this to a larger value like :code:`1e-7` since when differences are less than that, changes in the value are less than the amount of error in the calculation itself.
- :code:`H2MM_C.optimization_limits.max_time = np.inf` The timer used in H2MM_C is the basic C-level clock, it tends to be inaccurate (and often runs fast), but it doesn't slow down the optimization much when checking the time each round. Therefore it is generally recommended to keep it at infinite, so that an optimization doesn't terminate at a random pont.

So lets see an example of setting these values with :code:`H2MM_C.optimization_limits`.
These settings will apply to all latter calls to H2MM_C functions/methods, unless a value is explicitly specified as a keyword argument in the function/method call.

.. code-block::

    hm.optimization_limits['num_cores'] = 2
    hm.optimization_limits['max_iter'] = 1000
    hm.optimization_limits['converged_min'] = 1e-7

    model_5s3d = hm.EM_H2MM_C(hm.factory_h2mm_model(4,3), color3, times3)

| Optimization reach maximum number of iterations

This is equivalent to:

.. code-block::

    hm.optimization_limits.num_cores = 2
    hm.optimization_limits.max_iter = 1000
    hm.optimization_limits.converged_min = 1e-7

    model_5s3d = hm.EM_H2MM_C(hm.factory_h2mm_model(4,3), color3, times3)

| Optimization reach maximum number of iterations

You can also view these values:

.. code-block::

    hm.optimization_limits.num_cores

| 2

Or as a whole:

.. code-block::

    hm.optimization limits

| Optimization limits:: num_cores: 2, max_iter: 1000, converged_min: 1e-7, max_time: inf

.. |H2MM| replace:: H\ :sup:`2`\ MM
