Customizing Optimization Progress Display
=========================================

.. currentmodule:: H2MM_C

.. seealso::

    This can also be viewed as a Jupyter Notebook
    Downlaod :download:`H2MM_DisplayProgress.ipynb  <notebooks/H2MM_DisplayProgress.ipynb>`

    The data file can be downloaded here: :download:`sample_data_3det.txt <notebooks/sample_data_3det.txt>`

As always, lets get the imports and loading data out of the way:

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

Recording Each Iteration with :code:`print_args`
************************************************

Usually it is most useful to just see the current state of the optimization.
But if you want to keep a record of each output, there's an option for that.

This is the :code:`print_args` keyword argument.
There are several options/methods for specifing this argument.

If you simply pass :code:`print_args = True` then the result of every iteration will be printed:

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_args=True)

| Iteration:    0, loglik:-4.387899e+05, improvement:   inf
| Iteration:    1, loglik:-4.203991e+05, improvement:1.839076e+04
| Iteration:    2, loglik:-4.172495e+05, improvement:3.149621e+03
| Iteration:    3, loglik:-4.168160e+05, improvement:4.334909e+02
| Iteration:    4, loglik:-4.166697e+05, improvement:1.463433e+02
| Iteration:    5, loglik:-4.165714e+05, improvement:9.831467e+01
| Iteration:    6, loglik:-4.164868e+05, improvement:8.460410e+01
| Iteration:    7, loglik:-4.164030e+05, improvement:8.370726e+01
| Iteration:    8, loglik:-4.163116e+05, improvement:9.142352e+01
| Iteration:    9, loglik:-4.162055e+05, improvement:1.061152e+02
| Iteration:   10, loglik:-4.160785e+05, improvement:1.269588e+02
| Iteration:   11, loglik:-4.159256e+05, improvement:1.529380e+02
| Iteration:   12, loglik:-4.157447e+05, improvement:1.808611e+02
| Iteration:   13, loglik:-4.155401e+05, improvement:2.046643e+02
| Iteration:   14, loglik:-4.153189e+05, improvement:2.211944e+02
| Iteration:   15, loglik:-4.150866e+05, improvement:2.322782e+02
| Iteration:   16, loglik:-4.148451e+05, improvement:2.415008e+02
| Iteration:   17, loglik:-4.145964e+05, improvement:2.486755e+02
| Iteration:   18, loglik:-4.143435e+05, improvement:2.529542e+02
| Iteration:   19, loglik:-4.140881e+05, improvement:2.554187e+02
| Iteration:   20, loglik:-4.138319e+05, improvement:2.561196e+02
| Iteration:   21, loglik:-4.135787e+05, improvement:2.532579e+02
| Iteration:   22, loglik:-4.133311e+05, improvement:2.475722e+02
| Iteration:   23, loglik:-4.130892e+05, improvement:2.418696e+02
| Iteration:   24, loglik:-4.128514e+05, improvement:2.378508e+02
| Iteration:   25, loglik:-4.126157e+05, improvement:2.356415e+02
| Iteration:   26, loglik:-4.123811e+05, improvement:2.346992e+02
| Iteration:   27, loglik:-4.121466e+05, improvement:2.344752e+02
| Iteration:   28, loglik:-4.119119e+05, improvement:2.346358e+02
| Iteration:   29, loglik:-4.116771e+05, improvement:2.348715e+02
| Iteration:   30, loglik:-4.114426e+05, improvement:2.344301e+02
| Iteration:   31, loglik:-4.112106e+05, improvement:2.320622e+02
| Iteration:   32, loglik:-4.109840e+05, improvement:2.266024e+02
| Iteration:   33, loglik:-4.107665e+05, improvement:2.175059e+02
| Iteration:   34, loglik:-4.105615e+05, improvement:2.049792e+02
| Iteration:   35, loglik:-4.103717e+05, improvement:1.897585e+02
| Iteration:   36, loglik:-4.101993e+05, improvement:1.724524e+02
| Iteration:   37, loglik:-4.100461e+05, improvement:1.531493e+02
| Iteration:   38, loglik:-4.099140e+05, improvement:1.321656e+02
| Iteration:   39, loglik:-4.098032e+05, improvement:1.107497e+02
| Iteration:   40, loglik:-4.097127e+05, improvement:9.049932e+01
| Iteration:   41, loglik:-4.096402e+05, improvement:7.254715e+01
| Iteration:   42, loglik:-4.095828e+05, improvement:5.736204e+01
| Iteration:   43, loglik:-4.095379e+05, improvement:4.492736e+01
| Iteration:   44, loglik:-4.095029e+05, improvement:3.497448e+01
| Iteration:   45, loglik:-4.094758e+05, improvement:2.713812e+01
| Iteration:   46, loglik:-4.094547e+05, improvement:2.103855e+01
| Iteration:   47, loglik:-4.094384e+05, improvement:1.632438e+01
| Iteration:   48, loglik:-4.094257e+05, improvement:1.269319e+01
| Iteration:   49, loglik:-4.094158e+05, improvement:9.897353e+00
| Iteration:   50, loglik:-4.094081e+05, improvement:7.740962e+00
| Iteration:   51, loglik:-4.094020e+05, improvement:6.072558e+00
| Iteration:   52, loglik:-4.093972e+05, improvement:4.776702e+00
| Iteration:   53, loglik:-4.093935e+05, improvement:3.766078e+00
| Iteration:   54, loglik:-4.093905e+05, improvement:2.974800e+00
| Iteration:   55, loglik:-4.093881e+05, improvement:2.353085e+00
| Iteration:   56, loglik:-4.093863e+05, improvement:1.863156e+00
| Iteration:   57, loglik:-4.093848e+05, improvement:1.476176e+00
| Iteration:   58, loglik:-4.093836e+05, improvement:1.169979e+00
| Iteration:   59, loglik:-4.093827e+05, improvement:9.274102e-01
| Iteration:   60, loglik:-4.093820e+05, improvement:7.351031e-01
| Iteration:   61, loglik:-4.093814e+05, improvement:5.825859e-01
| Iteration:   62, loglik:-4.093809e+05, improvement:4.616147e-01
| Iteration:   63, loglik:-4.093805e+05, improvement:3.656763e-01
| Iteration:   64, loglik:-4.093803e+05, improvement:2.896107e-01
| Iteration:   65, loglik:-4.093800e+05, improvement:2.293229e-01
| Iteration:   66, loglik:-4.093798e+05, improvement:1.815593e-01
| Iteration:   67, loglik:-4.093797e+05, improvement:1.437336e-01
| Iteration:   68, loglik:-4.093796e+05, improvement:1.137899e-01
| Iteration:   69, loglik:-4.093795e+05, improvement:9.009406e-02
| Iteration:   70, loglik:-4.093794e+05, improvement:7.134807e-02
| Iteration:   71, loglik:-4.093794e+05, improvement:5.652128e-02
| Iteration:   72, loglik:-4.093793e+05, improvement:4.479603e-02
| Iteration:   73, loglik:-4.093793e+05, improvement:3.552414e-02
| Iteration:   74, loglik:-4.093793e+05, improvement:2.819196e-02
| Iteration:   75, loglik:-4.093792e+05, improvement:2.239292e-02
| Iteration:   76, loglik:-4.093792e+05, improvement:1.780534e-02
| Iteration:   77, loglik:-4.093792e+05, improvement:1.417483e-02
| Iteration:   78, loglik:-4.093792e+05, improvement:1.130034e-02
| Iteration:   79, loglik:-4.093792e+05, improvement:9.023055e-03
| Iteration:   80, loglik:-4.093792e+05, improvement:7.217588e-03
| Iteration:   81, loglik:-4.093792e+05, improvement:5.784895e-03
| Iteration:   82, loglik:-4.093792e+05, improvement:4.646821e-03
| Iteration:   83, loglik:-4.093792e+05, improvement:3.741739e-03
| Iteration:   84, loglik:-4.093792e+05, improvement:3.020924e-03
| Iteration:   85, loglik:-4.093792e+05, improvement:2.445975e-03
| Iteration:   86, loglik:-4.093792e+05, improvement:1.986614e-03
| Iteration:   87, loglik:-4.093792e+05, improvement:1.618845e-03
| Iteration:   88, loglik:-4.093792e+05, improvement:1.323799e-03
| Iteration:   89, loglik:-4.093792e+05, improvement:1.086528e-03
| Iteration:   90, loglik:-4.093792e+05, improvement:8.952366e-04
| Iteration:   91, loglik:-4.093792e+05, improvement:7.405641e-04
| Iteration:   92, loglik:-4.093792e+05, improvement:6.151425e-04
| Iteration:   93, loglik:-4.093792e+05, improvement:5.131122e-04
| Iteration:   94, loglik:-4.093791e+05, improvement:4.298041e-04
| Iteration:   95, loglik:-4.093791e+05, improvement:3.615554e-04
| Iteration:   96, loglik:-4.093791e+05, improvement:3.054184e-04
| Iteration:   97, loglik:-4.093791e+05, improvement:2.590739e-04
| Iteration:   98, loglik:-4.093791e+05, improvement:2.206618e-04
| Iteration:   99, loglik:-4.093791e+05, improvement:1.886659e-04
| Iteration:  100, loglik:-4.093791e+05, improvement:1.619285e-04
| Iteration:  101, loglik:-4.093791e+05, improvement:1.394759e-04
| Iteration:  102, loglik:-4.093791e+05, improvement:1.205574e-04
| Iteration:  103, loglik:-4.093791e+05, improvement:1.045387e-04
| Iteration:  104, loglik:-4.093791e+05, improvement:9.091827e-05
| Iteration:  105, loglik:-4.093791e+05, improvement:7.927959e-05
| Iteration:  106, loglik:-4.093791e+05, improvement:6.932020e-05
| Iteration:  107, loglik:-4.093791e+05, improvement:6.076449e-05
| Iteration:  108, loglik:-4.093791e+05, improvement:5.337631e-05
| Iteration:  109, loglik:-4.093791e+05, improvement:4.697498e-05
| Iteration:  110, loglik:-4.093791e+05, improvement:4.142558e-05
| Iteration:  111, loglik:-4.093791e+05, improvement:3.657222e-05
| Iteration:  112, loglik:-4.093791e+05, improvement:3.234588e-05
| Iteration:  113, loglik:-4.093791e+05, improvement:2.865586e-05
| Iteration:  114, loglik:-4.093791e+05, improvement:2.540683e-05
| Iteration:  115, loglik:-4.093791e+05, improvement:2.255174e-05
| Iteration:  116, loglik:-4.093791e+05, improvement:2.003415e-05
| Iteration:  117, loglik:-4.093791e+05, improvement:1.780386e-05
| Iteration:  118, loglik:-4.093791e+05, improvement:1.586386e-05
| Iteration:  119, loglik:-4.093791e+05, improvement:1.410692e-05
| Iteration:  120, loglik:-4.093791e+05, improvement:1.258217e-05
| Iteration:  121, loglik:-4.093791e+05, improvement:1.121568e-05
| Iteration:  122, loglik:-4.093791e+05, improvement:1.000246e-05
| Iteration:  123, loglik:-4.093791e+05, improvement:8.932839e-06
| Iteration:  124, loglik:-4.093791e+05, improvement:7.965660e-06
| Iteration:  125, loglik:-4.093791e+05, improvement:7.115363e-06
| Iteration:  126, loglik:-4.093791e+05, improvement:6.358838e-06
| Iteration:  127, loglik:-4.093791e+05, improvement:5.671347e-06
| Iteration:  128, loglik:-4.093791e+05, improvement:5.068083e-06
| Iteration:  129, loglik:-4.093791e+05, improvement:4.545087e-06
| Iteration:  130, loglik:-4.093791e+05, improvement:4.037225e-06
| Iteration:  131, loglik:-4.093791e+05, improvement:3.644091e-06
| Iteration:  132, loglik:-4.093791e+05, improvement:3.237044e-06
| Iteration:  133, loglik:-4.093791e+05, improvement:2.882327e-06
| Iteration:  134, loglik:-4.093791e+05, improvement:2.597284e-06
| Iteration:  135, loglik:-4.093791e+05, improvement:2.326095e-06
| Iteration:  136, loglik:-4.093791e+05, improvement:2.074637e-06
| Iteration:  137, loglik:-4.093791e+05, improvement:1.864275e-06
| Iteration:  138, loglik:-4.093791e+05, improvement:1.654669e-06
| Iteration:  139, loglik:-4.093791e+05, improvement:1.485634e-06
| Iteration:  140, loglik:-4.093791e+05, improvement:1.323759e-06
| Iteration:  141, loglik:-4.093791e+05, improvement:1.185108e-06
| Iteration:  142, loglik:-4.093791e+05, improvement:1.082022e-06
| Iteration:  143, loglik:-4.093791e+05, improvement:9.350479e-07
| Iteration:  144, loglik:-4.093791e+05, improvement:8.596107e-07
| Iteration:  145, loglik:-4.093791e+05, improvement:7.714843e-07
| Iteration:  146, loglik:-4.093791e+05, improvement:6.818445e-07
| Iteration:  147, loglik:-4.093791e+05, improvement:6.159535e-07
| Iteration:  148, loglik:-4.093791e+05, improvement:5.329493e-07
| Iteration:  149, loglik:-4.093791e+05, improvement:4.933681e-07
| Iteration:  150, loglik:-4.093791e+05, improvement:4.384201e-07
| Iteration:  151, loglik:-4.093791e+05, improvement:3.998866e-07
| Iteration:  152, loglik:-4.093791e+05, improvement:3.416790e-07
| Iteration:  153, loglik:-4.093791e+05, improvement:3.172318e-07
| Iteration:  154, loglik:-4.093791e+05, improvement:2.878951e-07
| Iteration:  155, loglik:-4.093791e+05, improvement:2.462184e-07
| Iteration:  156, loglik:-4.093791e+05, improvement:2.315501e-07
| Iteration:  157, loglik:-4.093791e+05, improvement:2.052402e-07
| Iteration:  158, loglik:-4.093791e+05, improvement:1.821318e-07
| Iteration:  159, loglik:-4.093791e+05, improvement:1.493609e-07
| Iteration:  160, loglik:-4.093791e+05, improvement:1.412700e-07
| Iteration:  161, loglik:-4.093791e+05, improvement:1.399894e-07
| Iteration:  162, loglik:-4.093791e+05, improvement:1.111766e-07
| Iteration:  163, loglik:-4.093791e+05, improvement:1.127482e-07
| Iteration:  164, loglik:-4.093791e+05, improvement:9.097857e-08
| Iteration:  165, loglik:-4.093791e+05, improvement:7.415656e-08
| Iteration:  166, loglik:-4.093791e+05, improvement:7.945346e-08
| Iteration:  167, loglik:-4.093791e+05, improvement:5.890615e-08
| Iteration:  168, loglik:-4.093791e+05, improvement:7.031485e-08
| Iteration:  169, loglik:-4.093791e+05, improvement:4.714821e-08
| Iteration:  170, loglik:-4.093791e+05, improvement:5.762558e-08
| Iteration:  171, loglik:-4.093791e+05, improvement:3.405148e-08
| Iteration:  172, loglik:-4.093791e+05, improvement:3.521563e-08
| Iteration:  173, loglik:-4.093791e+05, improvement:4.225876e-08
| Iteration:  174, loglik:-4.093791e+05, improvement:2.479646e-08
| Iteration:  175, loglik:-4.093791e+05, improvement:2.601882e-08
| Iteration:  176, loglik:-4.093791e+05, improvement:2.625166e-08
| Iteration:  177, loglik:-4.093791e+05, improvement:2.695015e-08
| Iteration:  178, loglik:-4.093791e+05, improvement:1.856824e-08
| Iteration:  179, loglik:-4.093791e+05, improvement:1.600711e-08
| Iteration:  180, loglik:-4.093791e+05, improvement:1.257285e-08
| Iteration:  181, loglik:-4.093791e+05, improvement:1.612352e-08
| Iteration:  182, loglik:-4.093791e+05, improvement:1.705484e-08
| Iteration:  183, loglik:-4.093791e+05, improvement:9.720679e-09
| Iteration:  184, loglik:-4.093791e+05, improvement:6.577466e-09
| Iteration:  185, loglik:-4.093791e+05, improvement:1.059379e-08
| Iteration:  186, loglik:-4.093791e+05, improvement:1.006993e-08
| Iteration:  187, loglik:-4.093791e+05, improvement:1.292210e-08
| Iteration:  188, loglik:-4.093791e+05, improvement:5.529728e-09
| Iteration:  189, loglik:-4.093791e+05, improvement:6.519258e-09
| Iteration:  190, loglik:-4.093791e+05, improvement:4.773028e-09
| Iteration:  191, loglik:-4.093791e+05, improvement:-1.746230e-10
| The model converged after 191 iterations

As you can see, this is very verbose.

Changing the Frequecy of Display Updates
****************************************

If you would rather have periodic updates, then you can hand :code:`print_args` an integer instead.
Then the display will update only every x number of iterations.

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_args=10)

Will show:

| Iteration:   20, loglik:-4.138319e+05, improvement:2.561196e+02 

and then (errasing the previous):

| Iteration:   30, loglik:-4.114426e+05, improvement:2.344301e+02

and so on.

You can even combine the integer argument and :code:`True`/:code:`False` argument into a tuple like :code:`print_args = (10, True)` to keep the previous display, but only show certain numbers of iterations.

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3,  times3, print_func='diff', print_args=(10, True))

| Iteration:    0, loglik:-4.387899e+05, improvement:   inf
| Iteration:   10, loglik:-4.160785e+05, improvement:1.269588e+02
| Iteration:   20, loglik:-4.138319e+05, improvement:2.561196e+02
| Iteration:   30, loglik:-4.114426e+05, improvement:2.344301e+02
| Iteration:   40, loglik:-4.097127e+05, improvement:9.049932e+01
| Iteration:   50, loglik:-4.094081e+05, improvement:7.740962e+00
| Iteration:   60, loglik:-4.093820e+05, improvement:7.351031e-01
| Iteration:   70, loglik:-4.093794e+05, improvement:7.134807e-02
| Iteration:   80, loglik:-4.093792e+05, improvement:7.217589e-03
| Iteration:   90, loglik:-4.093792e+05, improvement:8.952374e-04
| Iteration:  100, loglik:-4.093791e+05, improvement:1.619274e-04
| Iteration:  110, loglik:-4.093791e+05, improvement:4.142273e-05
| Iteration:  120, loglik:-4.093791e+05, improvement:1.258438e-05
| Iteration:  130, loglik:-4.093791e+05, improvement:4.036992e-06
| Iteration:  140, loglik:-4.093791e+05, improvement:1.325563e-06
| Iteration:  150, loglik:-4.093791e+05, improvement:4.360336e-07
| Iteration:  160, loglik:-4.093791e+05, improvement:1.417357e-07
| Iteration:  170, loglik:-4.093791e+05, improvement:5.646143e-08
| Iteration:  180, loglik:-4.093791e+05, improvement:1.420267e-08
| Iteration:  190, loglik:-4.093791e+05, improvement:4.365575e-09
| The model converged after 191 iterations

Custom Printing Functions
-------------------------

If you want to customize the display evne further, you can define your own display function.

It must have at least 5 arguments, as :func:`EM_H2MM_C` or :meth:`h2mm_model.optimize` will call it like so:

``print_func(niter, new_model, current_model, olde_model, t_iter, t_total)```

Where 

- :code:`niter` is the number of iterations
- :code:`new_model` is the :class:`h2mm_model` object that whose loglikelihood will be evaluate next (the product of the latest round in the optiization)
- :code:`current_model` is the :class:`h2mm_model` whose loglikelihood was just calculated, the "current" round of the optimization
- :code:`old_model` is the :class:`h2mm_model` that was calculated the iteration just before.
- :code:`t_iter` is the time (using the inaccurate C clock) it took to calculate the current iteration
- :code:`t_total` 

Your function should return a string of what you want to be displayed.

You can still specify :code:`print_args` as before, controlling how often the display is updated and if the previous display is kept or overwritten.

Bellow is an example function:

.. code-block::

    def silly_print(niter, new, current, old, titer, time):
        return f"""We haven't finished after {niter} iterations
        with {new.loglik - current.loglik} improvement in loglik
        after {time} (inaccurate) seconds"""

And an example of it's use in action:

.. code-block::

    model_3d3c = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3, times3, print_func=silly_print, print_args=(10, True))

| We haven't finished after 0 iterations
|     with 438789.96130430285 improvement in loglik 
|     after 0.121331 (inaccurate) secondsWe haven't finished after 10 iterations
|     with 416078.5463625849 improvement in loglik 
|     after 1.258256 (inaccurate) secondsWe haven't finished after 20 iterations
|     with 413831.9416083104 improvement in loglik 
|     after 2.373244 (inaccurate) secondsWe haven't finished after 30 iterations
|     with 411442.6378319904 improvement in loglik 
|     after 3.452714 (inaccurate) secondsWe haven't finished after 40 iterations
|     with 409712.71341160056 improvement in loglik 
|     after 4.559021 (inaccurate) secondsWe haven't finished after 50 iterations
|     with 409408.0698347561 improvement in loglik 
|     after 5.632227 (inaccurate) secondsWe haven't finished after 60 iterations
|     with 409381.95478790614 improvement in loglik 
|     after 6.736231 (inaccurate) secondsWe haven't finished after 70 iterations
|     with 409379.4254526214 improvement in loglik 
|     after 7.767289 (inaccurate) secondsWe haven't finished after 80 iterations
|     with 409379.1785051314 improvement in loglik 
|     after 8.860826 (inaccurate) secondsWe haven't finished after 90 iterations
|     with 409379.1519537501 improvement in loglik 
|     after 9.932235 (inaccurate) secondsWe haven't finished after 100 iterations
|     with 409379.14815781976 improvement in loglik 
|     after 11.066373 (inaccurate) secondsWe haven't finished after 110 iterations
|     with 409379.1473511942 improvement in loglik 
|     after 12.210938 (inaccurate) secondsWe haven't finished after 120 iterations
|     with 409379.14712526737 improvement in loglik 
|     after 13.3879 (inaccurate) secondsWe haven't finished after 130 iterations
|     with 409379.1470543588 improvement in loglik 
|     after 14.505477 (inaccurate) secondsWe haven't finished after 140 iterations
|     with 409379.14703126845 improvement in loglik 
|     after 15.61771 (inaccurate) secondsWe haven't finished after 150 iterations
|     with 409379.14702367247 improvement in loglik 
|     after 16.747291 (inaccurate) secondsWe haven't finished after 160 iterations
|     with 409379.1470211701 improvement in loglik 
|     after 17.880479 (inaccurate) secondsWe haven't finished after 170 iterations
|     with 409379.14702032757 improvement in loglik 
|     after 18.939075 (inaccurate) secondsWe haven't finished after 180 iterations
|     with 409379.14702006435 improvement in loglik 
|     after 20.013219 (inaccurate) secondsWe haven't finished after 190 iterations
|     with 409379.1470199641 improvement in loglik 
|     after 21.129933 (inaccurate) secondsThe model converged after 192 iterations

Extra Arguments for :code:`print_func`
**************************************

Somtimes you might want even more flexibility in the display, maybe for instance you want to change display for several different optimizations, but would still like to use the same printing function.

For that, **H2MM_C** still has you covered.

For this, you can add additional argumetns to your printer function (probably best as :code:`*args`), and supply those additional arguments by adding more elements to the `print_args` tuple- just make sure to specify the frequency and whether or not you want to keep the previous display:

.. code-block::

    def silly_print(niter, new, current, old, titer, time, *args):
        return f"""
        We haven't finished after {niter} iterations
        with {new.loglik - current.loglik} improvement in loglik
        after {time} (inaccurate) seconds, {args}"""


.. code-block::

    model_3s3d = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3, times3, 
                              print_func=silly_print, print_args=(75, True, "I'm very silly"))


|     We haven't finished after 0 iterations
|     with 438789.88126171724 improvement in loglik 
|     after 0.118648 (inaccurate) seconds, ("I'm very silly",)
|     We haven't finished after 75 iterations
|     with 409379.2380262778 improvement in loglik 
|     after 8.351019 (inaccurate) seconds, ("I'm very silly",)
|     We haven't finished after 150 iterations
|     with 409379.14702367294 improvement in loglik 
|     after 16.491106 (inaccurate) seconds, ("I'm very silly",)The model converged after 191 iterations

Non-string Returning :code:`print_func` Functions
-------------------------------------------------

While :code:`print_func` always accepts the same arguments, it does not necessarily have to return a string.
It can also return nothing, in which case the user can call :code:`print` from within the function, or other form of display.

In this case H2MM_C will not clear previous print statements, you are stuck recording everything.

Let's see an example:

.. code-block::

    def silly_print(niter, new, current, old, titer, time, *args):
        return f"""
        We haven't finished after {niter} iterations
        with {new.loglik - current.loglik} improvement in loglik
        after {time} (inaccurate) seconds, {args}"""

.. code-block::

    model_3s3d = hm.EM_H2MM_C(hm.factory_h2mm_model(3,3), color3, times3, 
                              print_func=silly_fix_print, print_args=(75, True, "eggs"))

| The model converged after 191 iterations
| 
| We haven't finished after 1 iterations
|     with 0.0002 improvement in loglik 
|     after 0.2 (inaccurate) seconds, ('eggs',)
| We haven't finished after 0 iterations
|     with 438789.88126171665 improvement in loglik 
|     after 0.112852 (inaccurate) seconds, ('eggs',)
| We haven't finished after 75 iterations
|     with 409379.2380262789 improvement in loglik 
|     after 8.097362 (inaccurate) seconds, ('eggs',)
| We haven't finished after 150 iterations
|     with 409379.1470236713 improvement in loglik 
|     after 16.135376 (inaccurate) seconds, ('eggs',)

