Bounding |H2MM|
===============

.. currentmodule:: H2MM_C

.. seealso::

    This can also be viewed as a Jupyter Notebook
    Downlaod :download:`H2MM_Bounds_Tutorial.ipynb <notebooks/H2MM_Bounds_Tutorial.ipynb>`

    The data file can be downloaded here: :download:`sample_data_3det.txt <notebooks/sample_data_3det.txt>`

Let's get our obligatory imports in order, this time we'll start with the 3 detector data.

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


Bounding Parameter Values
-------------------------

Sometimes we may want to restrain the range of possible values in an |H2MM| model (the :class:`h2mm_model`).
For instance, we may want to keep transition rates (:attr:`h2mm_model.trans`) within values reasonable for the duration of the experiment, or because you want restrain the values in the emission probability matrix (:attr:`h2mm_model.obs`) based on how we know the system should behave.

Bounds are defined using :class:`h2mm_limits` objects.
This object is passed to :func:`EM_H2MM_C` (or :meth:`h2mm_model.optimize` in an object-oriented approach) through the :code:`bounds` keyword argument.
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

    imodel_4s3d = hm.h2mm_model(prior, trans, obs)

    us_opt_model4 = hm.EM_H2MM_C(imodel_4s3d, color3, times3, bounds_func='revert', bounds=us_bounds)
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
- If specified as an array, then the values in the array are matched with the cooresponding array in :class:`h2mm_model`. This means greater granularity, but then you are locked into using the :class:`h2mm_limits` object only for optimizations with a specific number of states/detectors.

As mentioned in the above outline, you also need to specify the :code:`bounds_func` argument when using :code:`bounds`.
There are 3 options for this:

#. :code:`'minmax'`: The shallowest correction, sets the out of bounds value to its minimum or maximum
#. :code:`'revert'`: The preferred method, sets the out of bounds value to the value it was in the previous model
#. :code:`'revert_old'`: A more extreme form of :code:`'revert'` which goes to the model before the last in the optimization, and sets the out of bounds value to that "older" value.

Using :func:`factory_h2mm_model` with Bounds
--------------------------------------------

You will note that in the previous example, we specified the :class:`h2mm_model` explicity, instead of using the :func:`factory_h2mm_model` function.
This is because it is possible that the :func:`factory_h2mm_model` could return a :class:`h2mm_model` whose values are already out of bounds based for the :class:`h2mm_limits` object.
This could create odd behaviour during optimization.

There is a way around this: you can call :func:`factory_h2mm_model` with the :class:`h2mm_limits` object passed through the :code:`bounds` keyword argument, and the function will automatically return a :class:`h2mm_model` object that is within the bound provided.

.. note::

    See the documentation of :func:`factory_h2mm_model` for a fill list of options for customizing the function's output.

.. code-block::

    us_bounds = hm.h2mm_limits(max_trans = 1/4000)
    # make factory_h2mm_model make a model within bounds
    imodel = hm.factory_h2mm_model(3,3, bounds=us_bounds)
    us_model = hm.EM_H2MM_C(imodel, color3, times3, bounds=us_bounds, bounds_func='revert')

| The model converged after 198 iterations


Custom Bounds
-------------

Finally, it is possible to supply a custom bounding function to ``bounds_func``.

.. note::

    This feature was designed to allow the user to handle things/circumstances
    that the writers of H2MM_C had not anticipated.
    Therefore this example is very simple, and does not show a useful method.

This function must is called at the end of the optimization loop, after a given model's loglik has been 
calculated (and the standard |H2MM| next model for the next iteration produced).

This function takes the signature
``bounds_func(new:h2mm_model, current:h2mm_model, old:h2mm_model, *bounds, **bounds_func)->h2mm_model|int|tuple[h2mm_model,int]``

``new``, ``current`` and ``old`` are the :class:`h2mm_model` s of the current iteration.

#. ``new`` is the model suggested/produced by the current iteration
#. ``current`` is the model whose loglik was just computed
#. ``old`` is the model computed in the previous iteration

These are always supplied each iteration. ``bounds`` and ``bounds_kwargs`` come from the
identically named keyword arguments in :func:`EM_H2MM_C`. ``bounds`` by default is :code:`None`, which
is internally converted to a 0-size (empty) :code:`tuple`, likewise ``bounds_kwargs`` is by default :code:`None`
and is internally converted into an empty :code:`dict`.

The return value can either or both specify

#. The "bounded" `new` model
#. If the optimization has converged

If only the ``new`` model is specified, convergence will be determined like all other optimizaztions,
by the difference in loglik of ``current`` and ``old``.

**However,** if the bounds function returns a value specifying if the model has converged, then
:func:`EM_H2MM_C` will **not** separately check if the optimization has converged.

.. note::

    ``max_iter`` and ``max_time`` are enforced separetely from ``bounds_func``.


If specifying the converged state, this can be either a :code:`bool` or 0, 1, 2.

As a :code:`bool`, :code:`True` indicates that the optimization has converged, and thus
can stop, the ``old`` model will be returned as the "optimal" model. :code:`False`
will allow the optimization to proceed using the ``new`` model.

If specifying as :code:`0` is equivalent to :code:`False`, :code:`1` to :code:`True`, and :code:`2` will return
the ``current`` model as the optimal model.

If both the ``new`` model and converged state are specified, this must be done by returning a 2-tuple
of ``(new, converged_state)``.

.. warning::

    Make sure the model you return makes sense, otherwise the optimization will proceed unpredictably.
    Think of this as the “gloves off” approach, you might have a very powerful new method, or you might
    get something meaningless depending on how you code it. That’s your responsibility.

Below is a function that that re-implements the behavior of :code:`"minmax"` but now the limits
normally specified with a :class:`h2mm_limits` object supplied to ``bounds`` are replaced with kwargs:

.. code-block::

    def minmax_py(new, current, old, converged_min=1e-9, 
                      min_prior=None, max_prior=None,
                      min_trans=None, max_trans=None, 
                      min_obs=None, max_obs=None):
        # bounding of trans matrix
        if min_trans is not None or max_trans is not None:
            trans = new.trans
            idxs = np.arange(new.nstate)
            if isinstance(min_trans, float):
                trans[trans < min_trans*(~np.eye(new.nstate, dtype=np.bool_))] = min_trans
            elif isinstance(min_trans, np.ndarray):
                mask = trans < min_trans
                trans[mask] = min_trans[mask]
            if isinstance(max_trans, float):
                trans[trans > max_trans*(~np.eye(new.nstate, dtype=np.bool_))] = max_trans
            elif isinstance(max_trans, np.ndarray):
                mask = trans > max_trans
                trans[mask] = max_trans[mask]
            for i in range(trans.shape[0]):
                trans[i,i] = 1.0 - trans[i, idxs!=i].sum()
            new.trans = trans
        # bounding of obs matrix
        if min_obs is not None or max_obs is not None:
            obs = new.obs
            if min_obs is not None:
                minmask = obs < min_obs
                obs[minmask] = min_obs[minmask]
            else:
                minmask = np.zeros(obs.shape, dtype=np.bool_)
            if max_obs is not None:
                maxmask = obs > max_obs
                obs[maxmask] = max_obs[maxmask]
            else:
                maxmask = np.zeros(obs.shape, dtype=np.bool_)
            obsmask = minmask | maxmask
            for i in range(obs.shape[0]):
                obs[i,~obsmask[i,:]] += (1-obs[i,:].sum()) / (~obsmask).sum()
            new.obs = obs
        if min_prior is not None or max_prior is not None:
            prior = new.prior
            if min_prior is not None:
                minpmask = prior < min_prior
                prior[minpmask] = min_prior[minpmask]
            else:
                minpmask = np.zeros(new.nstate, base=np.bool_)
            if max_prior is not None:
                maxpmask = prior > max_prior
                prior[maxpmask] = max_prior[maxpmask]
            else:
                maxpmask = np.zeros(new.nstate, base=np.bool_)
            pmask = minpmask | maxpmask
            prior[~pmask] += (1-prior.sum()) / (~pmask).sum()
            new.prior = prior
        return new

Now let's see it in action

.. code-block::

    prior = np.array([1/4, 1/4, 1/4, 1/4])
    trans = np.array([[1-3e-6, 1e-6, 1e-6, 1e-6],
                      [1e-6, 1-3e-6, 1e-6, 1e-6],
                      [1e-6, 1e-6, 1-3e-6, 1e-6],
                      [1e-6, 1e-6, 1e-6, 1-3e-6]])
    obs = np.array([[0.09, 0.01,  0.9],
                    [0.3,   0.1,  0.6],
                    [0.2,   0.4,  0.4],
                    [0.1,   0.1,  0.8]])
    
    imodel4s3d = hm.h2mm_model(prior, trans, obs)
    us_opt_model4 = imodel4s3d.optimize(color3, times3, bounds_func=minmax_py, bounds_kwargs=dict(max_trans=1e-4))
    us_opt_model4

| The model converged after 350 iterations
| nstate: 4, ndet: 3, nphot: 436084, niter: 350, loglik: -408204.56206645 converged state: 0x27
| prior:
| 0.5555112046056888, 0.20187287854449681, 0.24261591684981437, 3.3481135599012533e-18
| trans:
| 0.999972194073311, 6.7564751089303695e-06, 6.949841706273052e-06, 1.4099609873856857e-05
| 2.578111151551638e-05, 0.9999561165677984, 1.8078717383206357e-06, 1.629444894776346e-05
| 1.7601657728916995e-05, 1.2573225584113972e-06, 0.9999780804917116, 3.0605280011286465e-06
| 0.0001, 1.9996845149615154e-05, 8.60786923267281e-06, 0.9998713952856177
| obs:
| 0.4703527236420553, 0.09123832328417568, 0.438408953073769
| 0.8487006012182848, 0.07578053476562936, 0.07551886401608592
| 0.14917490796760924, 0.3128863776835066, 0.5379387143488842
| 0.15243938881400212, 0.0771191789601975, 0.7704414322258003


Now let's re-implement how convergence of the optimization is handled

.. code-block::

    def limit_converged(new, current, old, conv_min):
        if current.loglik < old.loglik:
            return 1
        if (current.loglik - old.loglik) < conv_min:
            return 2
        return 0
    
    us_opt_model4 = imodel_4s3d.optimize(color3, times3, bounds_func=limit_converged, bounds=5e-8)
    us_opt_model4

| The model converged after 590 iterations
| nstate: 4, ndet: 3, nphot: 436084, niter: 590, loglik: -408203.0178092711 converged state: 0x27
| prior:
| 0.1974285837494489, 0.5611213924554068, 0.2414500237951443, 5.6637246878791966e-39
| trans:
| 0.999956242650991, 2.620764173899765e-05, 1.8189787919602346e-06, 1.5730728478121366e-05
| 7.049515187084443e-06, 0.9999698870139253, 6.991349840780767e-06, 1.6072121046823025e-05
| 1.271678650156712e-06, 1.7388310225536896e-05, 0.9999781790484651, 3.1609626592454596e-06
| 1.730359295110407e-05, 0.00011451967406835532, 8.076758043081446e-06, 0.9998600999749374
| obs:
| 0.8495279860407299, 0.07564793243468454, 0.07482408152458547
| 0.47168486928039943, 0.09134393961306524, 0.43697119110653526
| 0.14909992362596436, 0.31276915494455126, 0.5381309214294844
| 0.15084600961235772, 0.07681300825941331, 0.772340982128229


Finally, bellow is an example that re-implements the min-max procedure and checking for convergence

.. code-block::

    def minmax_conv_py(new, current, old, conv_min, **kwargs):
        return minmax_py(new, current, old, **kwargs), limit_converged(new, current, old, conv_min)
    
    us_opt_model4 = imodel_4s3d.optimize(color3, times3, bounds_func=minmax_conv_py, bounds=5e-8, bounds_kwargs=dict(max_trans=1e-4))
    us_opt_model4

| The model converged after 311 iterations
| nstate: 4, ndet: 3, nphot: 436084, niter: 311, loglik: -408204.56206731265 converged state: 0x27
| prior:
| 0.20187518613651614, 0.5555091603111122, 0.24261565355237166, 3.88786172217736e-17
| trans:
| 0.9999561169674445, 2.57788197862605e-05, 1.8080192889913584e-06, 1.629619348030374e-05
| 6.756063773343042e-06, 0.9999721942959813, 6.949977106380265e-06, 1.4099663139054262e-05
| 1.2573560248595137e-06, 1.7601808076185562e-05, 0.9999780806198352, 3.060216063830064e-06
| 1.999906525257268e-05, 0.0001, 8.606579409041898e-06, 0.9998713943553383
| obs:
| 0.8486997786227223, 0.07578060975327068, 0.07551961162400711
| 0.47035301854223116, 0.09123831157048941, 0.43840866988727945
| 0.14917480048855333, 0.3128859241466042, 0.5379392753648425
| 0.15244033018999176, 0.07711913168590599, 0.7704405381241023



.. |H2MM| replace:: H\ :sup:`2`\ MM
