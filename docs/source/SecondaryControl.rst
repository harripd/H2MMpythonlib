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

Optimization Control
--------------------

Sometimes you want to control when optimizations stop, or how many cores the optimization uses.
Basically the course of the optimization is the same, you're just changing the thresholds for when to stop optimizing.

There are 4 of these "limits":

#. :code:`num_cores` The number of threads to use when optimizing/calculating a model or state-path.
#. :code:`max_iter` The maximum number of iterations to optimize a model until automatically quitting
#. :code:`converged_min` The threshold of improvement required to continue optimizing, i.e. if the new model improves the loglikelihood by less than this value, the optimization will stop.
#. :code:`max_time` The maximum duration to conduct optimization, after which optimization will automatically stop. **Uses inaccurate clock, by default is infinite, and recommended not to be changed**

.. note::

    The timer used in :code:`max_time` uses the inaccurate C clock, which tends to run fast, so
    youroptimizations will often end earlier than the time entered. Thus the use of :code:`max_time` is
    generally discouraged.


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

| Optimization limits:: num_cores: 2, max_iter: 1000, converged_min: 1e-07, max_time: inf,
| formatter: StdPrinter, outstream=<cyfunction get_stdout at 0x78b0ec59c580>

.. note::

    The parameters :attr:`formatter`` and :attr:`outstream` are disucssed in :ref:`print_formatter`


New v2.0
--------

Frozen models
*************

Version 2.0 introduced limited abilities to hash and use :class:`h2mm_model`
objects as keys in dictionaries.

By default, most models cannot be used in this way,
rather a model must first be put into a canonical form.
To generate such a model, call the method :meth:`h2mm_model.sort_states`
on any existing model. The returned model will be hashable.

.. code-block::

    sorted_model_5s3d = model_5s3d.sort_states()
    hash(sorted_model_5s3d)

| 4835756030495084663

Along with the hash function, :class:`h2mm_model` s can also be compared
to one another with the equality operator. When two models have
identical cannonical forms, they will evaluate to :code:`True` in
equality comparisons.

.. note::

    This comparison is very strict, ie it is **not** an
    almost equal comparison.
    This is so that using hashable models as dictionary
    keys will still function.
    The floating point numbers in two models must be
    identical.

.. code-block::

    if sorted_model_5s3d == model_5s3d:
        print("sorteted model are equivalent to unsroted models")
    else:
        print("sorteted model are not equivalent to unsroted models")
    model_dummy = hm.factory_h2mm_model(4,3)
    if model_dummy == model_5s3d:
        print("oops, unsorted models are comparable, and identical")
    else:
        print("unsoreted models are comparable, but not identical")

| sorteted model are equivalent to unsroted models
| unsoreted models are comparable, but not identical

.. note::

    Once a model is sorted so it can be hashed, it is fixed.
    It can no longer be assigned new :code:`prior` / :code:`trans` / :code:`obs`
    values, and optimizations must be performed with :code:`inplace=False`


``convcode`` bitmask
********************
The :attr:`h2mm_model.conv_code` property is a number that informs on the nature of the model.
While is is more common to interpret this with the :attr:`h2mm_model.conv_str`,

A full list is in the :ref:`constants` section of the documentation,
but we will provide a simple example of the usage here.

To test if a given model has been sorted and frozen, we can `&` (bitwise and) the constant `H2MM_C.convcode.frozen` with a model:

.. code-block::

    bool(model_5s3d.conv_code & hm.convcode.frozen), bool(sorted_model_5s3d.conv_code & hm.convcode.frozen)

| (False, True)


Model Serialization
*******************

Version 2.0.6 introduced the :meth:`h2mm_model.tobytes` and :meth:`h2mm_model.frombytes` 
method and classmethod respectively.
These generate and interpret a bytes objects that represent all values of 
an :class:`h2mm_model` object.
This is useful if you want to save your data for later:

.. code-block::

    model_bytes = model_5s3d.tobytes()
    print(f'tobytes creates a {type(model_bytes)}')
    model_loaded = hm.h2mm_model.frombytes(model_bytes)
    print(f'frombytes creates a {type(model_loaded)}')
    print("Original model:")
    print(repr(model_5s3d))
    print('--------------------------------')
    print("loaded model")
    print(repr(model_loaded))

| tobytes creates a <class 'bytes'>
| frombytes creates a <class 'H2MM_C.h2mm_model'>
| Original model:
| nstate: 4, ndet: 3, nphot: 436084, niter: 1000, loglik: -408237.71297778725 converged state: 0x47
| prior:
| 0.6245004645842184, 0.030662090712150807, 0.1924008209154963, 0.15243662378813444
| trans:
| 0.9432491631685289, 0.024409636842630065, 0.03234119998884105, 7.969094511822206e-25
| 2.055157130067666e-05, 0.9999671611112879, 1.0292524987281316e-05, 1.9947924240066577e-06
| 4.9177071302160525e-06, 6.376167245906585e-06, 0.999978781601267, 9.924524356814171e-06
| 3.0410544170313242e-06, 2.573965040036105e-06, 3.792681366573989e-05, 0.9999564581668772
| obs:
| 0.6174979502908565, 0.38250204970116924, 7.974116882904588e-12
| 0.14418825554260808, 0.2894580006333491, 0.5663537438240428
| 0.4416166011580735, 0.08698269744731542, 0.47140070139461104
| 0.840976670297041, 0.07645342127367338, 0.08256990842928548
| 
| --------------------------------
| loaded model
| nstate: 4, ndet: 3, nphot: 436084, niter: 1000, loglik: -408237.71297778725 converged state: 0x47
| prior:
| 0.6245004645842184, 0.030662090712150807, 0.1924008209154963, 0.15243662378813444
| trans:
| 0.9432491631685289, 0.024409636842630065, 0.03234119998884105, 7.969094511822206e-25
| 2.055157130067666e-05, 0.9999671611112879, 1.0292524987281316e-05, 1.9947924240066577e-06
| 4.9177071302160525e-06, 6.376167245906585e-06, 0.999978781601267, 9.924524356814171e-06
| 3.0410544170313242e-06, 2.573965040036105e-06, 3.792681366573989e-05, 0.9999564581668772
| obs:
| 0.6174979502908565, 0.38250204970116924, 7.974116882904588e-12
| 0.14418825554260808, 0.2894580006333491, 0.5663537438240428
| 0.4416166011580735, 0.08698269744731542, 0.47140070139461104
| 0.840976670297041, 0.07645342127367338, 0.08256990842928548


.. |H2MM| replace:: H\ :sup:`2`\ MM
