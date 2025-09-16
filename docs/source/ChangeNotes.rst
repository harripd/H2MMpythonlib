Change Notes
============

.. currentmodule:: H2MM_C

Changed in version 2.0.0
------------------------

- Interface changes
    - ``max_iter`` default now 2046 ( :math:`2^{11} - 2`, so that output includeing initial and last
      non-calculated modelof size :math:`2^{11} = 2048`)
    - `inplace` of :func:`EM_H2MM_C`, :meth:`h2mm_model.optimize`, and
      :meth:`h2mm_model.evaluate` now defaults to :code:`False`
    - Introduction of :class:`Printer` and subclasses :class:`StdPrinter` and :class:`IPyPrinter`
      to handle formatting of output through ``print_formatter`` in :func:`EM_H2MM_C`
        - New ``print_func
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