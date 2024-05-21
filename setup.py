#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 6 14:01 2021
A full C level implementation of H2MM and python wrappers for access in Jupyter Notebooks
@author: Paul David Harris
"""
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

import numpy as np

ext = [Extension("H2MM_C", sources=["H2MM_C/H2MM_C.pyx","H2MM_C/h2mm_functions.c","H2MM_C/model_array.c","H2MM_C/burst_threads.c","H2MM_C/fwd_back.c","H2MM_C/rho_calc.c","H2MM_C/viterbi.c","H2MM_C/model_limits_funcs.c","H2MM_C/utils.c","H2MM_C/state_path.c","H2MM_C/pathloglik.c"],
                 depends=["H2MM_C/C_H2MM.h"],
                 include_dirs = [np.get_include(),"H2MM_C/"])]
for e in ext:
    e.cython_directives = {'embedsignature':True}


setup(name = "H2MM_C",
      ext_modules = cythonize(ext),
      )
