#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 6 14:01 2021
A full C level implementation of H2MM and python wrappers for access in Jupyter Notebooks
@author: Paul David Harris
"""
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext

import numpy as np

ext = [Extension("H2MM_C", sources=["H2MM_C/H2MM_C.pyx","H2MM_C/rho_calc.c","H2MM_C/fwd_back_photonbyphoton_par.c","H2MM_C/model_limits_funcs.c","H2MM_C/C_H2MM.c","H2MM_C/viterbi.c","H2MM_C/state_path.c"],
                 depends=["H2MM_C/C_H2MM.h"],
                 include_dirs = [np.get_include(),"H2MM_C/"])]
with open("README.md",'r') as f:
    long_description = f.read()

setup(name = "H2MM_C",
      version = "0.8.1",
      author = "Paul David Harris",
      author_email = "harripd@gmail.com",
      maintainer = "Paul David Harris",
      maintainer_email = "harripd@gmail.com",
      url = "https://github.com/harripd/H2MMpythonlib",
      download_url = "https://github.com/harripd/H2MMpythonlib",
      install_requires = ['numpy>=1.20.0'],
      description = "C level implementation of H2MM algorithm by Pirchi. 2016",
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      ext_modules = ext,
      license = "MIT",
      keywords = "single-molecule FRET",
      classifiers=['Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Cython',
                   'Programming Language :: C',
                   'Topic :: Scientific/Engineering',
                   ],
      cmdclass = {'build_ext': build_ext}
      )
