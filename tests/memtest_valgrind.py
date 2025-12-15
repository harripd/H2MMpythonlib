#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 07:19:01 2025

@author: paul
"""
from itertools import product

import numpy as np
import H2MM_C as hm


def load_txtbursts(file):
    color = list()
    times = list()
    i = 0
    # load the data
    with open(file, 'r') as f:
        for line in f: # this reads each line one at a time
            if i % 2 == 0: # even lines are times
                times.append(np.array([int(x) for x in line.split()],dtype=np.int64))
            else: # odd lines are color
                color.append(np.array([int(x) for x in line.split()],dtype=np.uint8))
            i += 1
    return color, times


def limit_new(new, current, old, *args, **kwargs):
    return new

def limit_choice_int(new, current, old, *args, **kwargs):
    conv = kwargs.get('conv', None)
    if conv is None and args:
        conv = args[0]
    if conv is None:
        conv = 1e-3
    if old.loglik < current.loglik:
        return 1
    elif (old.loglik - current.loglik) < conv:
        return 2
    return 0

def limit_choice_bool(new, current, old, *args, **kwargs):
    return bool(limit_choice_int(new, current, old, *args, **kwargs))


def limit_choice_new_int(new, current, old, *args, **kwargs):
    return new, limit_choice_int(new, current, old, *args, **kwargs)


def limit_choice_new_bool(new, current, old, *args, **kwargs):
    return new, bool(limit_choice_int(new, current, old, *args, **kwargs))


def bad_limit(new, current, old, *args, **kwargs):
    return hm.factory_h2mm_model(current.nstate+1, current.ndet)


def limit_raise(new, current, old):
    raise ValueError("this function automatically raises an error")


def print_bad(niter, new, current, old, titer, t_time):
    raise ValueError("this print function automatically raises an error")


def print_read(niter, new, current, old, titer, t_time, *args, **kwargs):
    return f'{niter}, {args}, {kwargs}'

valid_bound = (limit_new, limit_choice_bool, limit_choice_int, limit_choice_new_bool, limit_choice_new_int)

valid_bound_str = ('minmax', 'revert', 'revert_old')

valid_print = ('all', 'diff', 'diff_time', 'comp', 'comp_time', 'iter', print_read)


if __name__ == "__main__":
    c2, t2 = load_txtbursts('../docs/source/notebooks/sample_data_2det.txt')
    mi2 = hm.factory_h2mm_model(2,2)
    bi2 = hm.h2mm_limits(max_trans=1e-5)
    try:
        mo1 = hm.factory_h2mm_model(3,3).optimize(c2,t2, bounds_func=bad_limit)
    except Exception as e:
        err = e
    else:
        raise ValueError("bad limit function did not raise an error")
    try:
        mo1 = mi2.optimize(c2, t2, bounds_func=limit_raise)
    except Exception as e:
        err = e
    else:
        raise ValueError("limit raise function did not raise an error")
    try:
        mo1 = mi2.optimize(c2, t2, print_func=print_bad)
    except Exception as e:
        err = e
    else:
        raise ValueError("bad print func did not raise an error")
    for vp in valid_print:
        ot = hm.EM_H2MM_C(hm.factory_h2mm_model(3,2), c2, t2, max_iter=3, print_func=vp)
        break
    for vp in valid_bound_str:
        print(vp)
        ot = hm.EM_H2MM_C(hm.factory_h2mm_model(3,2), c2, t2, max_iter=100, bounds_func=vp, bounds=bi2)
    for vp in valid_bound:
        print(vp, callable(vp))
        ot = hm.EM_H2MM_C(hm.factory_h2mm_model(3, 2), c2, t2, max_iter=100, bounds_func=vp, bounds_kwargs=dict(conv=1e-8))
    path, scale, ll, icl = hm.viterbi_path(ot, c2, t2)
    for bic, total_loglik, loglikarray, loglikpath in product(*((False, True) for _ in range(4))):
        if sum((bic, total_loglik, loglikarray, loglikpath)) == 0:
            continue
        _ = hm.path_loglik(ot, c2, t2, path, BIC=bic, total_loglik=total_loglik, loglikarray=loglikarray, loglikpath=loglikpath)
    md = [hm.factory_h2mm_model(3,2),  hm.factory_h2mm_model(3,2)]
    out = hm.H2MM_arr(md, c2, t2)
