.. H2MM_C documentation master file, created by
   sphinx-quickstart on Sat Aug 13 08:15:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

H2MM_C is a python package that implements the |H2MM| algorithm.
This was initially designed for analyzing confocal based single-molecule FRET data to detect transitions between different states, characterizing the rate of transitions, and the FRET efficiencies.
The data comes in the form of photon arrival times, and indices to indicate at which detector the photon arrived.
While originally designed for confocal smFRET, in principle |H2MM| can be used to assess any other sort of data where data is composed of pairs of times and indices.

|H2MM| was originally developed by `Pirchi and Tsukanov et. al. 2016 <https://doi.org/10.1021/acs.jpcb.6b10726>`_ and this package was developed by `Harris et. al. 2022 <https://doi.org/10.1038/s41467-022-28632-x>`_ who also demonstrated additional statistical discriminators that clarified model selection.
If you use H2MM_C, please cite both of these papers.
For a more detailed description of the purpose and nature of |H2MM| see :doc:`AboutH2MM <AboutH2MM>`.

H2MM_C
======

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   AboutH2MM
   Tutorial
   HowTo
   Documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |H2MM| replace:: H\ :sup:`2`\ MM
.. |DD| replace:: D\ :sub:`ex`\ D\ :sub:`em`
.. |DA| replace:: D\ :sub:`ex`\ A\ :sub:`em`
.. |AA| replace:: A\ :sub:`ex`\ A\ :sub:`em`
