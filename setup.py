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

ext = [Extension("H2MM_C", sources=["H2MM_C/H2MM_C.pyx","H2MM_C/h2mm_functions.c","H2MM_C/model_array.c","H2MM_C/burst_threads.c","H2MM_C/fwd_back.c","H2MM_C/rho_calc.c","H2MM_C/viterbi.c","H2MM_C/model_limits_funcs.c","H2MM_C/utils.c","H2MM_C/state_path.c","H2MM_C/pathloglik.c"],
                 depends=["H2MM_C/C_H2MM.h"],
                 include_dirs = [np.get_include(),"H2MM_C/"])]
for e in ext:
    e.cython_directives = {'embedsignature':True}


with open("README.md",'r') as f:
    long_description = f.read()

setup(name = "H2MM_C",
      version = "1.0.2",
      author = "Paul David Harris",
      author_email = "harripd@gmail.com",
      maintainer = "Paul David Harris",
      maintainer_email = "harripd@gmail.com",
      url = "https://github.com/harripd/H2MMpythonlib",
      download_url = "https://github.com/harripd/H2MMpythonlib",
      install_requires = ['numpy>=1.20.1','IPython'],
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
