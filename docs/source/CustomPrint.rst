Customizing Optimization Progress Display
=========================================

.. currentmodule:: H2MM_C

.. seealso::

    This can also be viewed as a Jupyter Notebook
    Download :download:`H2MM_DisplayProgress.ipynb  <notebooks/H2MM_DisplayProgress.ipynb>`

    The data file can be downloaded here: :download:`sample_data_3det.txt <notebooks/sample_data_3det.txt>`

As always, lets get the imports and loading data out of the way:

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


Basic :code:`print_func` Options
--------------------------------

.. note::

    These are best demonstrated in Jupyer notebooks

If you've been following along in a Jupyter notebook, you will probably have noticed that whenever :func:`EM_H2MM_C` or :meth:`h2mm_model.optimize` is run, you see a little display saying how many iterations the optimization has conducted so far.
This behavior can be modified in various ways.

The :code:`print_func` keyword argument let's us choose the format of the display.

The basic options are:

- :code:`'iter'` Prints only the iteration number- a compact way to track optimization progress (the default)
- :code:`'all'`: Prints the representation of the entire current model. This option is very verbose
- :code:`'diff'` Prints teh difference between the previous model and current model loglikelihoods, and the current loglikelihood
- :code:`'diff_time'` Same as :code:`'diff'`, but with additional information about the durration of the current iteration, adn the total time of the optimization. This is using the inaccurate C clock, so the times are not very reliable, and often fast
- :code:`'comp'` Print the old and current loglikelihoods
- :code:`'comp_time'` Similar to :code:`'diff_time'`, prints :code:`'comp'` with time information added
- :code:`None` Supress all printing (this is **NOT** a string, it is simply the python varaible :code:`None`

Passing one of these strings or :code:`None` to the :code:`print_func` keyword argument of either :func:`EM_H2MM_C` or :meth:`h2mm_model.optimize` will cause the respective display option while optimizing the model.

For instance, if we run the following in a jupyter notebooks, in the middle of the optimization, you will see this displayed:

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff')

| Iteration:   73, loglik:-4.093793e+05, improvement:3.552414e-02



Changing the Frequency of Display Updates
*****************************************

`print_freq` is used to specify how frequently the display updates, by passing an *integer* value into `print_freq`, then the display will only update after that many iterations.

Update the display every 10 iterations:

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_freq=10)

Will show:

| Iteration:   20, loglik:-4.138319e+05, improvement:2.561196e+02 

and then (errasing the previous):

| Iteration:   30, loglik:-4.114426e+05, improvement:2.344301e+02

and so on.


Custom Printing Functions
-------------------------

If you want to customize the display, you can define  your own printing function.

The function should have the following general signature:

:code:`print_func(niter:int, new_model:h2mm_model, current_model:h2mm_model, old_model:h2mm_model, iter_time:int, total_time:float)->str`

.. note::

    It is not necessary, but recommended to keep these variable names in the function declaration.

where:
- ``niter`` is the number of iterations
- ``new_model`` is a `h2mm_model` object that represents the next model to be optimized (**before** checking for out of bounds values) note that its `.loglik` will be irrelevant because it has not been calculated yet.
- ``current_model`` is a `h2mm_model` object that represents the model whose `.loglik` was just calculated
- ``old_model`` is a `h2mm_model` object that represents the model from the previous iteration.
- ``iter_time`` is a float which is the time in seconds based on the **inaccurate C clock** that it took to calculate the latest iteration
- ``total_time`` is a float which is the time in seconds based on the **inaccurate C clock** that the full optimization has taken

The output of `print_func` is converted to a string (`str()`) and displayed, unless `print_func` returns `None`, in which case nothing is printed (unless `print_func` internally calls its own print method)

So below is an example of a custom print function:

.. code-block::

    def silly_print(niter, new, current, old, titer, time):
        return (f"We haven't finished after {niter} iterations " 
               f"with {new.loglik - current.loglik} improvement "
               f"in loglik after {time} (inaccurate) seconds")

And an example of it's use in action:

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3, times3, print_func=silly_print, print_freq=10)

| We haven't finished after 10 iterations with 416078.5463625849 improvement in loglik after 1.258256 (inaccurate) secondsWe haven't finished after 20 iterations with 413831.9416083104 improvement in loglik after 2.373244 (inaccurate) secondsWe haven't finished after 30 iterations

Passing additional args to ``print_func`` with ``print_args``
*************************************************************

The true signature of print_args is

:code:`print_func(niter:int, new_model:h2mm_model, current_model:h2mm_model, old_model:h2mm_model, t_iter:int, t_total:float, *print_args, **print_kwargs)->str`

:func:`EM_H2MM_C` has two keyword arguments, ``print_args`` and ``print_kwargs``, only used when ``print_func`` is a callable, which are passed to ``print_func`` as
its args and kwargs. Note that in both cases, a value of :code:`None` is converted into an empty tuple/dict respectively. Further, if a non-tuple is passed
to ``print_args`` it is treated as a single argument.)

.. code-block::

    def silly_print_arg_kwarg(niter, new, current, old, titer, time, *args, modulus=None):
        return f"niter={niter} args={args}, modulus={modulus}"
    
    hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3, times3, 
                 print_func=silly_print_arg_kwarg, print_args="I'm very silly", print_kwargs=dict(modulus=1))

| niter=10, args=("I'm very silly", ), modulus=1


.. _print_formatter: 

Advanced options: Formatter
---------------------------

By default, optimization progress is printed to :code:`sys.stdout`, and each update over-writes the previous update.
This is done using the ``print_formatter``.

Keeping display of each iteration
*********************************

The default ``print_formatter`` has a single keyword argument: ``keep``.
We can set this using the ``print_fmt_kwargs`` keyword argument in :func:`EM_H2MM_C`,
if it is set to :code:`True`, then the text of each new iteration will be displayed on a new line.

>>> model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_fmt_kwargs=dict(keep=True))
Iteration:    0, loglik:-4.387899e+05, improvement:   inf
Iteration:    1, loglik:-4.203991e+05, improvement:1.839076e+04
Iteration:    2, loglik:-4.172495e+05, improvement:3.149621e+03
Iteration:    3, loglik:-4.168160e+05, improvement:4.334909e+02
Iteration:    4, loglik:-4.166697e+05, improvement:1.463433e+02
Iteration:    5, loglik:-4.165714e+05, improvement:9.831467e+01
Iteration:    6, loglik:-4.164868e+05, improvement:8.460410e+01
Iteration:    7, loglik:-4.164030e+05, improvement:8.370726e+01
Iteration:    8, loglik:-4.163116e+05, improvement:9.142352e+01
Iteration:    9, loglik:-4.162055e+05, improvement:1.061152e+02
Iteration:   10, loglik:-4.160785e+05, improvement:1.269588e+02
Iteration:   11, loglik:-4.159256e+05, improvement:1.529380e+02
Iteration:   12, loglik:-4.157447e+05, improvement:1.808611e+02
Iteration:   13, loglik:-4.155401e+05, improvement:2.046643e+02
Iteration:   14, loglik:-4.153189e+05, improvement:2.211944e+02
Iteration:   15, loglik:-4.150866e+05, improvement:2.322782e+02
Iteration:   16, loglik:-4.148451e+05, improvement:2.415008e+02
Iteration:   17, loglik:-4.145964e+05, improvement:2.486755e+02
Iteration:   18, loglik:-4.143435e+05, improvement:2.529542e+02
Iteration:   19, loglik:-4.140881e+05, improvement:2.554187e+02
Iteration:   20, loglik:-4.138319e+05, improvement:2.561196e+02
Iteration:   21, loglik:-4.135787e+05, improvement:2.532579e+02
Iteration:   22, loglik:-4.133311e+05, improvement:2.475722e+02
Iteration:   23, loglik:-4.130892e+05, improvement:2.418696e+02
Iteration:   24, loglik:-4.128514e+05, improvement:2.378508e+02
Iteration:   25, loglik:-4.126157e+05, improvement:2.356415e+02
Iteration:   26, loglik:-4.123811e+05, improvement:2.346992e+02
Iteration:   27, loglik:-4.121466e+05, improvement:2.344752e+02
Iteration:   28, loglik:-4.119119e+05, improvement:2.346358e+02
Iteration:   29, loglik:-4.116771e+05, improvement:2.348715e+02
Iteration:   30, loglik:-4.114426e+05, improvement:2.344301e+02
Iteration:   31, loglik:-4.112106e+05, improvement:2.320622e+02
Iteration:   32, loglik:-4.109840e+05, improvement:2.266024e+02
Iteration:   33, loglik:-4.107665e+05, improvement:2.175059e+02
Iteration:   34, loglik:-4.105615e+05, improvement:2.049792e+02
Iteration:   35, loglik:-4.103717e+05, improvement:1.897585e+02
Iteration:   36, loglik:-4.101993e+05, improvement:1.724524e+02
Iteration:   37, loglik:-4.100461e+05, improvement:1.531493e+02
Iteration:   38, loglik:-4.099140e+05, improvement:1.321656e+02
Iteration:   39, loglik:-4.098032e+05, improvement:1.107497e+02
Iteration:   40, loglik:-4.097127e+05, improvement:9.049932e+01
Iteration:   41, loglik:-4.096402e+05, improvement:7.254715e+01
Iteration:   42, loglik:-4.095828e+05, improvement:5.736204e+01
Iteration:   43, loglik:-4.095379e+05, improvement:4.492736e+01
Iteration:   44, loglik:-4.095029e+05, improvement:3.497448e+01
Iteration:   45, loglik:-4.094758e+05, improvement:2.713812e+01
Iteration:   46, loglik:-4.094547e+05, improvement:2.103855e+01
Iteration:   47, loglik:-4.094384e+05, improvement:1.632438e+01
Iteration:   48, loglik:-4.094257e+05, improvement:1.269319e+01
Iteration:   49, loglik:-4.094158e+05, improvement:9.897353e+00
Iteration:   50, loglik:-4.094081e+05, improvement:7.740962e+00
Iteration:   51, loglik:-4.094020e+05, improvement:6.072558e+00
Iteration:   52, loglik:-4.093972e+05, improvement:4.776702e+00
Iteration:   53, loglik:-4.093935e+05, improvement:3.766078e+00
Iteration:   54, loglik:-4.093905e+05, improvement:2.974800e+00
Iteration:   55, loglik:-4.093881e+05, improvement:2.353085e+00
Iteration:   56, loglik:-4.093863e+05, improvement:1.863156e+00
Iteration:   57, loglik:-4.093848e+05, improvement:1.476176e+00
Iteration:   58, loglik:-4.093836e+05, improvement:1.169979e+00
Iteration:   59, loglik:-4.093827e+05, improvement:9.274102e-01
Iteration:   60, loglik:-4.093820e+05, improvement:7.351031e-01
Iteration:   61, loglik:-4.093814e+05, improvement:5.825859e-01
Iteration:   62, loglik:-4.093809e+05, improvement:4.616147e-01
Iteration:   63, loglik:-4.093805e+05, improvement:3.656763e-01
Iteration:   64, loglik:-4.093803e+05, improvement:2.896107e-01
Iteration:   65, loglik:-4.093800e+05, improvement:2.293229e-01
Iteration:   66, loglik:-4.093798e+05, improvement:1.815592e-01
Iteration:   67, loglik:-4.093797e+05, improvement:1.437336e-01
Iteration:   68, loglik:-4.093796e+05, improvement:1.137899e-01
Iteration:   69, loglik:-4.093795e+05, improvement:9.009406e-02
Iteration:   70, loglik:-4.093794e+05, improvement:7.134806e-02
Iteration:   71, loglik:-4.093794e+05, improvement:5.652127e-02
Iteration:   72, loglik:-4.093793e+05, improvement:4.479606e-02
Iteration:   73, loglik:-4.093793e+05, improvement:3.552413e-02
Iteration:   74, loglik:-4.093793e+05, improvement:2.819196e-02
Iteration:   75, loglik:-4.093792e+05, improvement:2.239293e-02
Iteration:   76, loglik:-4.093792e+05, improvement:1.780534e-02
Iteration:   77, loglik:-4.093792e+05, improvement:1.417483e-02
Iteration:   78, loglik:-4.093792e+05, improvement:1.130034e-02
Iteration:   79, loglik:-4.093792e+05, improvement:9.023060e-03
Iteration:   80, loglik:-4.093792e+05, improvement:7.217583e-03
Iteration:   81, loglik:-4.093792e+05, improvement:5.784888e-03
Iteration:   82, loglik:-4.093792e+05, improvement:4.646831e-03
Iteration:   83, loglik:-4.093792e+05, improvement:3.741733e-03
Iteration:   84, loglik:-4.093792e+05, improvement:3.020921e-03
Iteration:   85, loglik:-4.093792e+05, improvement:2.445989e-03
Iteration:   86, loglik:-4.093792e+05, improvement:1.986612e-03
Iteration:   87, loglik:-4.093792e+05, improvement:1.618841e-03
Iteration:   88, loglik:-4.093792e+05, improvement:1.323802e-03
Iteration:   89, loglik:-4.093792e+05, improvement:1.086529e-03
Iteration:   90, loglik:-4.093792e+05, improvement:8.952317e-04
Iteration:   91, loglik:-4.093792e+05, improvement:7.405693e-04
Iteration:   92, loglik:-4.093792e+05, improvement:6.151437e-04
Iteration:   93, loglik:-4.093792e+05, improvement:5.131036e-04
Iteration:   94, loglik:-4.093791e+05, improvement:4.298040e-04
Iteration:   95, loglik:-4.093791e+05, improvement:3.615519e-04
Iteration:   96, loglik:-4.093791e+05, improvement:3.054271e-04
Iteration:   97, loglik:-4.093791e+05, improvement:2.590730e-04
Iteration:   98, loglik:-4.093791e+05, improvement:2.206524e-04
Iteration:   99, loglik:-4.093791e+05, improvement:1.886743e-04
Iteration:  100, loglik:-4.093791e+05, improvement:1.619282e-04
Iteration:  101, loglik:-4.093791e+05, improvement:1.394845e-04
Iteration:  102, loglik:-4.093791e+05, improvement:1.205552e-04
Iteration:  103, loglik:-4.093791e+05, improvement:1.045333e-04
Iteration:  104, loglik:-4.093791e+05, improvement:9.091164e-05
Iteration:  105, loglik:-4.093791e+05, improvement:7.928640e-05
Iteration:  106, loglik:-4.093791e+05, improvement:6.932841e-05
Iteration:  107, loglik:-4.093791e+05, improvement:6.076420e-05
Iteration:  108, loglik:-4.093791e+05, improvement:5.336886e-05
Iteration:  109, loglik:-4.093791e+05, improvement:4.697818e-05
Iteration:  110, loglik:-4.093791e+05, improvement:4.141859e-05
Iteration:  111, loglik:-4.093791e+05, improvement:3.657391e-05
Iteration:  112, loglik:-4.093791e+05, improvement:3.235607e-05
Iteration:  113, loglik:-4.093791e+05, improvement:2.864911e-05
Iteration:  114, loglik:-4.093791e+05, improvement:2.540177e-05
Iteration:  115, loglik:-4.093791e+05, improvement:2.255203e-05
Iteration:  116, loglik:-4.093791e+05, improvement:2.003304e-05
Iteration:  117, loglik:-4.093791e+05, improvement:1.781207e-05
Iteration:  118, loglik:-4.093791e+05, improvement:1.585321e-05
Iteration:  119, loglik:-4.093791e+05, improvement:1.411943e-05
Iteration:  120, loglik:-4.093791e+05, improvement:1.257751e-05
Iteration:  121, loglik:-4.093791e+05, improvement:1.121568e-05
Iteration:  122, loglik:-4.093791e+05, improvement:1.000549e-05
Iteration:  123, loglik:-4.093791e+05, improvement:8.921721e-06
Iteration:  124, loglik:-4.093791e+05, improvement:7.970841e-06
Iteration:  125, loglik:-4.093791e+05, improvement:7.111405e-06
Iteration:  126, loglik:-4.093791e+05, improvement:6.351736e-06
Iteration:  127, loglik:-4.093791e+05, improvement:5.681184e-06
Iteration:  128, loglik:-4.093791e+05, improvement:5.071750e-06
Iteration:  129, loglik:-4.093791e+05, improvement:4.537113e-06
Iteration:  130, loglik:-4.093791e+05, improvement:4.052301e-06
Iteration:  131, loglik:-4.093791e+05, improvement:3.622728e-06
Iteration:  132, loglik:-4.093791e+05, improvement:3.240188e-06
Iteration:  133, loglik:-4.093791e+05, improvement:2.902700e-06
Iteration:  134, loglik:-4.093791e+05, improvement:2.591172e-06
Iteration:  135, loglik:-4.093791e+05, improvement:2.317654e-06
Iteration:  136, loglik:-4.093791e+05, improvement:2.073532e-06
Iteration:  137, loglik:-4.093791e+05, improvement:1.855718e-06
Iteration:  138, loglik:-4.093791e+05, improvement:1.667067e-06
Iteration:  139, loglik:-4.093791e+05, improvement:1.481734e-06
Iteration:  140, loglik:-4.093791e+05, improvement:1.328648e-06
Iteration:  141, loglik:-4.093791e+05, improvement:1.191860e-06
Iteration:  142, loglik:-4.093791e+05, improvement:1.063221e-06
Iteration:  143, loglik:-4.093791e+05, improvement:9.535579e-07
Iteration:  144, loglik:-4.093791e+05, improvement:8.573988e-07
Iteration:  145, loglik:-4.093791e+05, improvement:7.635681e-07
Iteration:  146, loglik:-4.093791e+05, improvement:6.809714e-07
Iteration:  147, loglik:-4.093791e+05, improvement:6.131595e-07
Iteration:  148, loglik:-4.093791e+05, improvement:5.442416e-07
Iteration:  149, loglik:-4.093791e+05, improvement:4.927278e-07
Iteration:  150, loglik:-4.093791e+05, improvement:4.374888e-07
Iteration:  151, loglik:-4.093791e+05, improvement:3.930181e-07
Iteration:  152, loglik:-4.093791e+05, improvement:3.509340e-07
Iteration:  153, loglik:-4.093791e+05, improvement:3.150781e-07
Iteration:  154, loglik:-4.093791e+05, improvement:2.818415e-07
Iteration:  155, loglik:-4.093791e+05, improvement:2.491870e-07
Iteration:  156, loglik:-4.093791e+05, improvement:2.267188e-07
Iteration:  157, loglik:-4.093791e+05, improvement:2.040761e-07
Iteration:  158, loglik:-4.093791e+05, improvement:1.780572e-07
Iteration:  159, loglik:-4.093791e+05, improvement:1.619337e-07
Iteration:  160, loglik:-4.093791e+05, improvement:1.452281e-07
Iteration:  161, loglik:-4.093791e+05, improvement:1.309672e-07
Iteration:  162, loglik:-4.093791e+05, improvement:1.164735e-07
Iteration:  163, loglik:-4.093791e+05, improvement:1.042499e-07
Iteration:  164, loglik:-4.093791e+05, improvement:9.033829e-08
Iteration:  165, loglik:-4.093791e+05, improvement:8.364441e-08
Iteration:  166, loglik:-4.093791e+05, improvement:7.578637e-08
Iteration:  167, loglik:-4.093791e+05, improvement:6.600749e-08
Iteration:  168, loglik:-4.093791e+05, improvement:5.913898e-08
Iteration:  169, loglik:-4.093791e+05, improvement:5.075708e-08
Iteration:  170, loglik:-4.093791e+05, improvement:5.011680e-08
Iteration:  171, loglik:-4.093791e+05, improvement:4.476169e-08
Iteration:  172, loglik:-4.093791e+05, improvement:4.214235e-08
Iteration:  173, loglik:-4.093791e+05, improvement:2.951128e-08
Iteration:  174, loglik:-4.093791e+05, improvement:3.026798e-08
Iteration:  175, loglik:-4.093791e+05, improvement:2.945308e-08
Iteration:  176, loglik:-4.093791e+05, improvement:2.753222e-08
Iteration:  177, loglik:-4.093791e+05, improvement:1.909211e-08
Iteration:  178, loglik:-4.093791e+05, improvement:2.031447e-08
Iteration:  179, loglik:-4.093791e+05, improvement:1.798617e-08
Iteration:  180, loglik:-4.093791e+05, improvement:1.414446e-08
Iteration:  181, loglik:-4.093791e+05, improvement:1.664739e-08
Iteration:  182, loglik:-4.093791e+05, improvement:1.414446e-08
Iteration:  183, loglik:-4.093791e+05, improvement:8.265488e-09
Iteration:  184, loglik:-4.093791e+05, improvement:1.135049e-08
Iteration:  185, loglik:-4.093791e+05, improvement:9.138603e-09
Iteration:  186, loglik:-4.093791e+05, improvement:6.053597e-09
Iteration:  187, loglik:-4.093791e+05, improvement:4.889444e-09
Iteration:  188, loglik:-4.093791e+05, improvement:9.778887e-09
Iteration:  189, loglik:-4.093791e+05, improvement:4.423782e-09
Iteration:  190, loglik:-4.093791e+05, improvement:7.974450e-09
Iteration:  191, loglik:-4.093791e+05, improvement:5.064066e-09
Iteration:  192, loglik:-4.093791e+05, improvement:3.201421e-09
Iteration:  193, loglik:-4.093791e+05, improvement:2.619345e-09
Iteration:  194, loglik:-4.093791e+05, improvement:1.629815e-09
Iteration:  195, loglik:-4.093791e+05, improvement:3.550667e-09
Iteration:  196, loglik:-4.093791e+05, improvement:4.598405e-09
Iteration:  197, loglik:-4.093791e+05, improvement:1.979060e-09
Iteration:  198, loglik:-4.093791e+05, improvement:3.783498e-09
The model converged after 198 iterations


Changing output stream with ``print_stream``
--------------------------------------------

The ``print_formatter`` can be instructed to direct the output to another stream,
such as ``sys.stderr`` by specifying the `print_stream` keyword argument

.. code-block::

    import sys
    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_stream=sys.stderr)


Whats actually happeing
***********************

Now that simple options have been described, let's go through what actually is happeing.

Before an optimization begins, :func:`EM_H2MM_C` creates a *formatter* object.
These are typically a subclass of :class:`Printer`, the default being :class:`StdPrinter`.
It does so by calling ``print_formatter(print_stream, *print_fmt_args, **print_fmt_kwargs)``

.. note::

    If `print_formatter` is `None`, then `print_formatter` will be the default
    formater specified by `optimization_limits.formatter`.
    
    If ``print_stream`` is :code:`None`, it will be the default specified by
    *calling* ``hm.optmiziation_limits.outstream()``, this means that
    ``hm.optmiziation_limits.outstream`` should be a callable that takes no
    arguments. This allows for a "factory" function to be creates so that
    the stream can be dynamically assigned.


Then optimization begins.

Upon the completion of an iteration of optimization, :func:`EM_H2MM_C` first checks
if the iteration should be updated based on ``print_freq``.

If it is time to update the display, then :func:`EM_H2MM_C` calls ``print_func`` with the
cooresponding arguments, and hands the output as the single argument to
``formatter.update``. The total call look like:
``formatter.update(print_func(niter, new, current, old, iter_time, total_time, *print_args, **print_kwargs))``

At the end of the optimization ``formatter.close()`` is called, to ensure any finalization should be conducted.

The default :class:`StdPrinter` has ``__init__`` signature of ``(buffer, keep=False)``. From this you can see how
specifying ``print_stream=sys.stderr`` and ``print_fmt_kwargs=dict(keep=False)`` changes where the output is
printed and how it is printed respectively.

The :meth:`Printer.update` method takes the one argument, formats it according to ``keep`` and then
sends the formatted output to ``print_stream.write(text)``, and the
calls ``print_stream.flush()`` to ensure the output is actually displayed.


IPyPrinter
**********

Out of the box, ``H2MM_C`` comes with one other :class:`Printer` class: :class:`IPyPrinter`.
This class requires IPython(https://ipython.org/) to be installed, if it is not,
then this class will not exist.

:class:`IPyPrinter` is initiallized with the following signature: ``(handle, keep=False)``
Where ``handle`` is a `IPython.display.DisplayHandle <https://ipython.readthedocs.io/en/8.26.0/api/generated/IPython.display.html>`_
So the default ``sys.stdout`` will not work for the ``print_stream`` argument.
Instead, you will have to generate a ``DisplayHandle`` to use :class:`IPyPrinter`

Below see a simple call using :class:`IPyPrinter`.

.. code-block::

    from IPython.display import DisplayHandle
    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='all')

| ...
| Iteration time:0.1345, Total:0.682079
| nstate: 3, ndet: 3, nphot: 436084, niter: 6, loglik: -416486.75115250226 converged state: 0x3
| prior:
| 0.11801555761578152, 0.17674035701948754, 0.7052440853647308
| trans:
| 0.9999204691004234, 3.30832791970862e-05, 4.644762037947176e-05
| 4.215187645885762e-06, 0.9999758910815926, 1.989373076155992e-05
| 2.0649794030150514e-06, 7.668846607352397e-06, 0.9999902661739896
| obs:
| 0.1554361303878464, 0.4480538780317254, 0.3965099915804283
| 0.1569004844542735, 0.24341579368194147, 0.599683721863785
| 0.5223655621688783, 0.08363832319456292, 0.3939961146365587
| ...


What is the advantage of :class:`IPyPrinter`?

Because it uses a ``DisplayHandle``, it can fully clear the output each iteration, while :class:`StdPrinter` 

relies on ``\r`` characters to overwrite the previous output, which can only overwrite the last line.

So if you plan to use ``print_func='all'``, then :class:`IPyPrinter` is recomended.

On the other hand, using ``DisplayHandle`` means that the :class:`IPyPrinter` is slightly slower.


Custom Printers
---------------

``H2MM_C`` does not strictly enforce any typing on `print_formatter` objects, 
operating on the principle of `Duck Typing <https://en.wikipedia.org/wiki/Duck_typing>`_

All that is required is that calling
``print_formatter(print_stream, *print_args, **print_kwargs)`` will create an object
that has ``.update`` and ``.close`` methods that take 1 and 0 arguments respectively,
with the ``.update`` method accepting whatever the output of ``print_func`` is.

However, ``H2MM_C`` provides :class:`Printer` as an abstract base class, that is recomended as the
parent class of any ``print_formatter``.

Indeed, both :class:`StdPrinter` and :class:`IPyPrinter` are subclasses of :class:`Printer`

Below is the code for :class:`StdPrinter`

.. code-block::

    class StdPrinter(hm.Printer):
        __slots__ = ('buffer', 'width', 'keep',)
        def __init__(self, buffer, keep=False):
            self.buffer = buffer
            self.width = 0
            self.keep = bool(keep)
        
        def update(self, text):
            text = str(text)
            if self.keep:
                text = text + '\n'
            else:
                ln = len(text)
                text = '\r' + text + ' '*(self.width - ln)
                if ln > self.width:
                    self.width = ln
            self.buffer.write(text)
            self.buffer.flush()
        
        def close(self):
            self.buffer.write("\n")


>>> model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_formatter=StdPrinter, print_stream=sys.stdout)
The model converged after 198 iterations


