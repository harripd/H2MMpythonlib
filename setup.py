#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 6 14:01 2021
A full C level implementation of H2MM and python wrappers for access in Jupyter Notebooks
@author: Paul David Harris
"""
from setuptools import setup
from distutils.core import Extension
import numpy as np

ext = Extension("H2MM_C", sources=["H2MM_C.c","rho_calc.c","fwd_back_photonbyphoton_par.c","model_limits_funcs.c","C_H2MM.c","viterbi.c"],define_macros=[("NPY_NO_DEPRECATED_APY","NPY_1_7_API_VERSION")])
long_description = """
H2MM_C
======
***H2MM_C*** is a software package enabling rapid optimization of H2MM models, 
and calculation of ***Viterbi*** path. Including multi-parameter models.
Please cite Pirchi. et. al. J.Phys.Chem. B 2016, 120, 13065, DOI:10.1020/acs.jpcb.6b10726
And zenodo ...
"""

setup(name = "H2MM_C",
      version = "0.7",
      author = "Paul David Harris",
      author_email = "harripd@gmail.com",
      maintainer = "Paul David Harris",
      maintainer_email = "harripd@gmail.com",
      url = "https://github.com/harripd/H2MMpythonlib",
      download_url = "https://github.com/harripd/H2MMpythonlib",
      description = "C level implementation of H2MM algorithm by Pirchi. 2016",
      install_requires = ['numpy'],
      long_description = long_description,
      headers = ["C_H2MM.h"],
      ext_modules = [ext],
      license = "MIT",
      keywords = "single-molecule FRET",
      include_dirs=[np.get_include()])
