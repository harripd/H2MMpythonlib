Change Notes
============

.. currentmodule:: H2MM_C


Changed in Version 2.1.1
------------------------

- Added ``loglikpath`` keyword argument to :func:`path_loglik`, if :code:`True`,
  the loglikelihood of each photon is return in an array of arrays.

- Fixed
    - Setting of :attr:`h2mm_model.prior`, :attr:`h2mm_model.trans` :attr:`h2mm_model.obs`, and :attr:`h2mm_model.niter` after model is sorted now raises an error.
    - :meth:`h2mm_model.optimize` after models sorted with option ``inplace=True`` now raises error and prevents modification of model.


Changed in version 2.0.6
------------------------

- Added :meth:`h2mm_model.tobytes` and :meth:`h2mm_model.frombytes` methods enabling serializing/loading :class:`h2mm_model`

Changed in version 2.0.5
------------------------

- Added beta ``IPyPlot`` class, for plotting with matplotlib
    - This class is not considered "official", and is not guaranteed to be backwards compatible with minor revision changes

.. note::

    When ``H2MM_C`` was first introduced, different version numbers were given in Zenodo and pip
    repositories, creating confusion, as one stated at 1.x.x, the other at 0.x.x.
    Version 2.0.5 marks the unification of these versioning systems.
    All version in Documentation refrence the pip versioning.

Changed in version 2.0.0
------------------------

- Interface changes
    - ``max_iter`` default now 2046 ( :math:`2^{11} - 2`, so that output includeing initial and last
      non-calculated modelof size :math:`2^{11} = 2048`)
    - `inplace` of :func:`EM_H2MM_C`, :meth:`h2mm_model.optimize`, and
      :meth:`h2mm_model.evaluate` now defaults to :code:`False`
    - Introduction of :class:`Printer` and subclasses :class:`StdPrinter` and :class:`IPyPrinter` to handle formatting of output through ``print_formatter`` in :func:`EM_H2MM_C`
        - New ``print_func``
        - ``print_args`` no longer used to set update frequency and keep
            - Use ``print_freq`` and ``print_fmt_kwargs`` instead (see further notes)
    - ``print_func`` as a function now takes signature ``(niter, new, current, old, iter_time, total_time, *print_args, **print_kwargs)``
    - ``print_func`` no longer called before optimization to check output type etc.
    - ``bounds_func`` now takes signature ``(niter, new, current, old, *bounds, **bounds_kwargs)``
    - ``bounds_func`` no longer called before optimization

- Internal Changes
    - Improved memory handling, indexes arrays no longer copied
    - Switched types of most C variables to fixed width <stdint.h> defined types
        - indexes from unsigned long to uint8_t
        - deltas (times) from unsigned long to int32_t
        - h2mm_mod.nstate/ndet to int64_t
