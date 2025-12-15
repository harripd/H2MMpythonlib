Documentation
=============

.. module:: H2MM_C


.. _analysis:

Data Analysis Functions
-----------------------

.. autofunction:: EM_H2MM_C

.. autofunction:: factory_h2mm_model

.. autofunction:: H2MM_arr

.. autofunction:: viterbi_path

.. autofunction:: viterbi_sort

.. autofunction:: path_loglik


.. _simfuncs:

Simulation Functions
--------------------

.. autofunction:: sim_phtraj_from_times

.. autofunction:: sim_phtraj_from_state

.. autofunction:: sim_sparsestatepath

.. autofunction:: sim_statepath


.. _classes:

Classes
-------

.. autoclass:: h2mm_model
    :members:

.. autoclass:: h2mm_limits
    :members:

.. autoclass:: Printer
    :members:

.. autoclass:: StdPrinter
    :members:

.. autoclass:: IPyPrinter
    :members:


.. _constants:

Constants
---------
The following are all constants that serve as a bitmask for :attr:`h2mm_model.conv_code` if
``&`` with a model and the given constant is non-zero, then the given condition is met

int H2MM_C.convcode.ll_computed
    Model has had loglik computed against some data (either during optimization or singly

int H2MM_C.convcode.from_opt
    Model was created during optimization

int H2MM_C.convcode.output
    Model is best model during optimization, by some criterion marked by other flags

int H2MM_C.convcode.converged
    Model has best logliklihood within converged_min threshold

int H2MM_C.convcode.max_iter
    Optimization reached maximum number of iterations (only set for models furing terminal iteration)

int H2MM_C.convcode.max_time
    Optimization reach maximum time (only set for models furing terminal iteration)

int H2MM_C.convcode.error
    Error occured during optimization

int H2MM_C.convcode.post_opt
    Model is either the model with poorer loglikelihood between current and old,
    or is new and loglik is not calculated

int H2MM_C.convcode.frozen
    Model cannot be modified in place