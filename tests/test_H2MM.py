#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:12:55 2021
Tests for verification of H2MM_C functions
@author: Paul David Harris
"""

import numpy as np
from itertools import combinations_with_replacement

import pytest
import H2MM_C as h2

def model_almost_equal(model1,model2):
    """
    Compare two models are essentially the same
    """
    prior = np.all(np.abs(model1.prior - model2.prior) < 0.1)
    trans_check = model1.trans / model2.trans
    trans = np.all(trans_check < 5) * np.all(trans_check > 0.5)
    obs = np.all(np.abs(model1.obs - model2.obs) < 0.1)
    return prior * trans * obs

def true_size(inp):
    """
    Small function that returns the size of an object as a tuple
    """
    if isinstance(inp, (list, tuple)):
        return (len(inp), )
    elif isinstance(inp, np.ndarray):
        return inp.shape
    else:
        return (-1, )

def true_iter(inp):
    """
    Iterate over list, tuple or numpy array elements
    """
    if isinstance(inp, (list, tuple)):
        return iter(inp)
    elif isinstance(inp, np.ndarray):
        return inp.ravel()
    else:
        return iter([inp, ])

def limits_func_system_converged(new, current, old, bound):
    new.prior = np.ones(new.prior.size)/new.prior.size
    return new

def limits_func_user_converged(new, current, old, bound):
    if current.niter > bound:
        return new, 1
    if current.loglik <= old.loglik - 1e-6:
        return new, 2
    new.prior = np.ones(new.prior.size)/new.prior.size
    return new, 0

def limits_func_bad_size(new, current, old, bound):
    return h2.factory_h2mm_model(current.nstate+1, current.ndet+1)

def limits_func_bad_return(new, current, old, bound):
    return {'prior':np.array([1,2.]), 'trans':np.array([[2., 4,],[9.,3]]), 'obs':np.array([2,3])}

def print_func_str(new, current, old, titer, ttotal, *args):
    return str(new) + "\n"

def print_func_str_wrongarg(new, current, old):
    return 32

def print_func_str_wrongret(new, current, old, titer, ttotal, *args):
    return [32, ]

def data_gen_list():
    times = [np.array([   20,   50,   90,   95,  190,  230,  350,  800]),
             np.array([  820,  850,  990,  995, 1055, 1130, 1290, 1525]),
             np.array([ 1750, 1820, 1950, 1985, 2055, 2110, 2220, 2325])]
    dets =  [np.array([    1,    1,    0,    1,    0,    1,    0,    1]),
             np.array([    0,    0,    0,    1,    1,    1,    1,    0]),
             np.array([    1,    0,    1,    1,    0,    1,    0,    1])]
    nphot = sum(t.size for t in times)
    return (dets, times, nphot)

def data_gen_tuple():
    dets, times, nphot = data_gen_list()
    dets, times = tuple(dets), tuple(times)
    return (dets, times, nphot)

def data_gen_np():
    dets, times, nphot = data_gen_list()
    det, time = np.empty(len(dets), dtype=object), np.empty(len(dets), dtype=object)
    for i in range(len(dets)):
        det[i], time[i] = dets[i], times[i]
    return (det, time, nphot)

@pytest.fixture(scope="module")
def simple_data_list():
    return data_gen_list()

@pytest.fixture(scope="module")
def simple_data_tuple():
    return data_gen_tuple()

@pytest.fixture(scope="module")
def simple_data_np():
    return data_gen_np()
@pytest.fixture(scope="module", params=[data_gen_np, data_gen_list, data_gen_tuple])
def simple_data_all(request):
    dets, times, nphot = request.param()
    return dets, times, nphot

def test_MakeModel():
    """
    Test proper model values assigned when making new model
    """
    prior = np.ones(3)
    trans = np.array([[0.999,0.0005,0.005],[0.0005,0.999,0.0005],[0.0005,0.0005,0.999]])
    obs = np.array([[0.1,0.2,0.7],[0.5,0.3,0.2],[0.4,0.4,0.2]])
    model = h2.h2mm_model(prior,trans,obs)
    assert model.nstate == 3
    assert model.ndet == 3
    assert pytest.approx(1.0) == model.prior.sum()
    assert np.allclose(model.trans.sum(axis=1), 1.0)
    assert np.allclose(model.obs.sum(axis=1), 1.0)
    assert model.k == 14

def test_Factory():
    """
    Ensure factory_h2mm_model returns correct values
    """
    model = h2.factory_h2mm_model(3,2)
    assert type(model) == h2.h2mm_model
    assert model.nstate == 3
    assert model.ndet == 2
    assert model.k == 11

def test_Model_access():
    """
    Test setting of model arrays works
    """
    model = h2.factory_h2mm_model(4,3)
    with pytest.warns(UserWarning):
        model.prior = np.linspace(0,5,4)
    with pytest.warns(UserWarning):
        model.trans = np.array([[1.6, 1.2, 0.5, 0.1],[0.5, 9.8, 1.1, 0.2],[0.5,0.9,7.2, 1.2],[0.5,0.5,0.6,9.5]])
    with pytest.warns(UserWarning):
        model.obs = np.ones((4,3))
    assert pytest.approx(1.0) == model.prior.sum()
    assert np.allclose(model.trans.sum(axis=1), 1.0)
    assert np.allclose(model.obs.sum(axis=1), 1.0)

def test_InputType(simple_data_all):
    """
    Test all types of data input work
    """
    # test optimization
    dets, times, nphot = simple_data_all
    out = h2.EM_H2MM_C(h2.factory_h2mm_model(2,2), dets, times)
    assert isinstance(out, h2.h2mm_model)
    assert out.nphot == nphot
    assert out.loglik < 0.0
    assert out.conv_code > 2
    assert out.ndet == 2
    assert out.nstate == 2
    
def test_gamma(simple_data_all):
    """
    Test optimization returning gamma
    """
    out = h2.EM_H2MM_C(h2.factory_h2mm_model(3,2), simple_data_all[0], simple_data_all[1], opt_array=False, gamma=False, max_iter=10)
    assert isinstance(out, h2.h2mm_model)
    out = h2.EM_H2MM_C(h2.factory_h2mm_model(3,2), simple_data_all[0], simple_data_all[1], opt_array=True, gamma=False, max_iter=10)
    assert isinstance(out, np.ndarray) and out.dtype == object
    out, gamma = h2.EM_H2MM_C(h2.factory_h2mm_model(3,2), simple_data_all[0], simple_data_all[1], opt_array=False, gamma=True, max_iter=10)
    assert isinstance(out, h2.h2mm_model) and type(gamma) == type(simple_data_all[0]) and len(gamma) == len(simple_data_all[0])
    out, gamma = h2.EM_H2MM_C(h2.factory_h2mm_model(3,2), simple_data_all[0], simple_data_all[1], opt_array=True, gamma=True, max_iter=10)
    assert isinstance(out, np.ndarray) and out.dtype==object and type(gamma) == type(simple_data_all[0]) and len(gamma) == len(simple_data_all[0])

def test_gamma_oop(simple_data_all):
    """
    Test optimization returning gamma, using object oriented approach
    """
    out = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=False, gamma=False, max_iter=10)
    assert isinstance(out, h2.h2mm_model)
    out = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=True, gamma=False, max_iter=10)
    assert isinstance(out, np.ndarray) and out.dtype == object
    out, gamma = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=False, gamma=True, max_iter=10)
    assert isinstance(out, h2.h2mm_model) and type(gamma) == type(simple_data_all[0]) and len(gamma) == len(simple_data_all[0])
    out, gamma = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=True, gamma=True, max_iter=10)
    assert isinstance(out, np.ndarray) and out.dtype==object and type(gamma) == type(simple_data_all[0]) and len(gamma) == len(simple_data_all[0])
    out = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=False, gamma=False, max_iter=10, inplace=False)
    assert isinstance(out, h2.h2mm_model)
    out = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=True, gamma=False, max_iter=10, inplace=False)
    assert isinstance(out, np.ndarray) and out.dtype == object
    out, gamma = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=False, gamma=True, max_iter=10, inplace=False)
    assert isinstance(out, h2.h2mm_model) and type(gamma) == type(simple_data_all[0]) and len(gamma) == len(simple_data_all[0])
    out, gamma = h2.factory_h2mm_model(3,2).optimize(simple_data_all[0], simple_data_all[1], opt_array=True, gamma=True, max_iter=10, inplace=False)
    assert isinstance(out, np.ndarray) and out.dtype==object and type(gamma) == type(simple_data_all[0]) and len(gamma) == len(simple_data_all[0])
    # smoke test for optimizing model where optimization ended by max_iter
    out[-1].optimize(simple_data_all[0], simple_data_all[1], max_iter=10)

@pytest.fixture(scope="module")
def opt_model():
    dets, times, nphot = data_gen_list()
    return h2.EM_H2MM_C(h2.factory_h2mm_model(3,2), dets, times, max_iter=20)


def model_opt():
    dets, times, nphot = data_gen_list()
    out1 = h2.EM_H2MM_C(h2.factory_h2mm_model(2,2), dets, times, max_iter=10, opt_array=True)
    out2 = h2.EM_H2MM_C(h2.factory_h2mm_model(3,2), dets, times, max_iter=10, opt_array=True)
    out3 = np.concatenate([out1, out2])
    for model in reversed(out1):
        out = model
        if out.conv_code != 7:
            break
    yield out
    yield out2
    yield out3

@pytest.fixture(scope='module', params=[out for out in model_opt()])
def model_array(request):
    return request.param



def test_H2MM_arr(model_array,simple_data_all):
    """
    Test H2MM_arr function
    """
    dets, times, nphot = simple_data_all
    arr_out = h2.H2MM_arr(model_array, dets, times)
    assert type(arr_out) == type(model_array)
    assert true_size(arr_out) == true_size(model_array)
    assert np.allclose([m.loglik for m in true_iter(arr_out)], [m.loglik for m in true_iter(model_array)])

def test_H2MM_arr_gamma(model_array,simple_data_all):
    """
    Test model array returning gamma function
    """
    dets, times, nphot = simple_data_all
    arr_out, gamma = h2.H2MM_arr(model_array, dets, times, gamma=True)
    assert type(arr_out) == type(model_array)
    assert true_size(arr_out) == true_size(model_array)
    assert np.allclose([m.loglik for m in true_iter(arr_out)], [m.loglik for m in true_iter(model_array)])
 

def test_evaluate(opt_model, simple_data_tuple):
    mod = opt_model.copy()
    dets, times, nphot = simple_data_tuple
    mod_out = mod.evaluate(dets, times, inplace=False)
    assert isinstance(mod_out, h2.h2mm_model)
    assert np.allclose(mod_out.loglik, mod.loglik)
    mod_out, gamma = mod.evaluate(dets, times, gamma=True, inplace=False)
    assert isinstance(mod_out, h2.h2mm_model)
    assert np.allclose(mod_out.loglik, mod.loglik)
    for g in gamma:
        assert np.allclose(g.sum(axis=1), 1.0)
    mod_out = mod.evaluate(dets, times, inplace=True)
    assert isinstance(mod_out, h2.h2mm_model)
    assert mod_out.loglik == mod.loglik
    mod_out, gamma = mod.evaluate(dets, times, gamma=True, inplace=True)
    assert isinstance(mod_out, h2.h2mm_model)
    assert mod_out.loglik == mod.loglik
    for g in gamma:
        assert np.allclose(g.sum(axis=1), 1.0) 
    
def test_Viterbi(opt_model,simple_data_all):
    """
    Smoke test for viterbi and path_loglik functions
    """
    dets, times, nphot = simple_data_all
    path, scale, ll, icl = h2.viterbi_path(opt_model, dets, times)
    assert type(path) == type(scale) == type(times)
    assert true_size(path) == true_size(scale) == true_size(dets)
    for p, s, t in zip(true_iter(path), true_iter(scale),true_iter(times)):
        assert np.issubdtype(p.dtype, np.integer)
        assert np.all(p < opt_model.nstate)
        assert p.shape == s.shape == t.shape
        assert np.issubdtype(s.dtype, np.floating)
        assert np.all(s <= 1.0)
        assert np.all(s >= 0.0)
    for BIC, LL, log in combinations_with_replacement((True, False), 3):
        if sum((BIC, LL, log)) == 0:
            continue
        out = h2.path_loglik(opt_model, dets, times, path, BIC=BIC, total_loglik=LL, loglikarray=log)
        if sum((BIC, LL, log)) == 1:
            out = (out, )
        cur_pos = 0
        if BIC:
            bic = out[cur_pos]
            assert bic > 0.0
            cur_pos += 1
        if LL:
            ll = out[cur_pos]
            assert ll < 0.0
            cur_pos += 1
            if BIC and LL:
                assert np.allclose(bic, np.log(nphot)*opt_model.k-2*ll)
        if log:
            lg = out[cur_pos]
            assert np.all(lg < 0.0)
            assert true_size(lg) == true_size(dets)
            assert np.issubdtype(lg.dtype, np.floating)
            if log and LL:
                assert np.allclose(lg.sum(), ll)
        
        

def test_Limits(simple_data_list):
    """
    Ensure basic limits functions work properly
    """
    dets, times, nphot = simple_data_list
    limits = h2.h2mm_limits(min_trans=1e-7, max_trans=1e-1)
    for lim in ('minmax', 'revert', 'revert_old'):
        out = h2.EM_H2MM_C(h2.factory_h2mm_model(2,2), dets, times, bounds=limits, bounds_func=lim)
        assert np.all(out.trans[np.eye(2) < 1] < 1e-1)
        assert np.all(out.trans[np.eye(2) < 1] > 1e-7)
    

def test_Limits_usr(opt_model, simple_data_list):
    """
    Smoke test for user defined limits function
    """
    dets, times, nphot = simple_data_list
    out = h2.EM_H2MM_C(opt_model, dets, times, bounds_func=limits_func_system_converged)
    assert isinstance(out, h2.h2mm_model)
    out = h2.EM_H2MM_C(opt_model, dets, times, bounds_func=limits_func_user_converged, bounds=10)
    assert isinstance(out, h2.h2mm_model)
    assert out.niter == 10

def test_User_print_str(opt_model, simple_data_list):
    """
    Smoke test for print func returning a string
    """
    dets, times, nphot = simple_data_list
    h2.EM_H2MM_C(opt_model, dets, times, print_func=print_func_str)
    with pytest.raises(Exception):
        h2.EM_H2MM_C(opt_model, dets, times, print_func=print_func_str_wrongarg)
    with pytest.raises(Exception):
        h2.EM_H2MM_C(opt_model, dets, times, print_func=print_func_str_wrongret)


def test_Sim():
    """
    Simulate a trajectory with simulation function, then check if optimization finds
    a model with similar results
    """
    times = [np.cumsum(np.ceil(np.random.exponential(1000,size=(75)))).astype('L') for i in range(10000)]
    prior_init = np.array([1/3,1/3,1/3])
    trans_init = np.array([[1-1.5e-6,1e-6,5e-7],[1e-6,1-1.5e-6,5e-7],[5e-7,5e-7,1-1e-6]])
    obs_init = np.array([[0.2,0.8],[0.5,0.5],[0.8,0.2]])
    model_init = h2.h2mm_model(prior_init,trans_init,obs_init)
    sim_traj = [h2.sim_phtraj_from_times(model_init,tm) for tm in times]
    paths, colors = [], []
    for path, color in sim_traj:
        paths.append(path)
        colors.append(color)
    limits = h2.h2mm_limits(min_prior=0.001,max_prior=0.999,min_trans=1e-10,max_trans=1-1e-10,min_obs=0.001,max_obs=0.999)
    model_test = h2.factory_h2mm_model(3,2)
    model_test.optimize(colors,times,bounds=limits,bounds_func='revert',max_iter=3000)
    assert model_almost_equal(model_test,model_init)

def test_BadData(opt_model):
    """
    Test errors raised when bad data is passed to array
    """
    dets, time = data_gen_list()[0].copy(), data_gen_list()[1].copy()
    time[0][1] = 0
    with pytest.raises(ValueError):
        h2.EM_H2MM_C(opt_model, dets, time)
    dets = data_gen_list()[0].copy()
    dets[0][0] = 2
    with pytest.raises(Exception):
        h2.EM_H2MM_C(opt_model, dets, data_gen_list()[1])
    dets[0] = np.concatenate([dets[0], [1]])
    with pytest.raises(ValueError):
        h2.EM_H2MM_C(opt_model, dets, time)
    dets, time = data_gen_list()[0].copy(), data_gen_list()[1].copy()
    with pytest.raises(Exception):
        h2.EM_H2MM_C(opt_model, time, dets)

def test_BadLimits(opt_model, simple_data_list):
    """
    Check errors raised when bad limits func is passed
    """
    dets, times, nphot = simple_data_list
    with pytest.raises(ValueError):
        h2.EM_H2MM_C(opt_model, dets, times, bounds_func=limits_func_bad_size)
    with pytest.raises(TypeError):
        h2.EM_H2MM_C(opt_model, dets, times, bounds_func=limits_func_bad_return)

def test_OOp(simple_data_all):
    """
    Smoke tests to ensure basic oop operations on models work as expected
    """
    dets, times, nphot = simple_data_all
    in_mod = h2.factory_h2mm_model(3,2)
    out_mod = in_mod.optimize(dets, times, inplace=False)
    assert in_mod is not out_mod
    assert np.all(in_mod.prior != out_mod.prior)
    assert np.all(in_mod.trans != out_mod.trans)
    assert np.all(in_mod.obs != out_mod.obs)
    out_mod = in_mod.optimize(dets, times)
    assert np.all(in_mod.prior == out_mod.prior)
    assert np.all(in_mod.trans == out_mod.trans)
    assert np.all(in_mod.obs == out_mod.obs)
    in_mod = h2.factory_h2mm_model(3,2)
    ev_mod = in_mod.evaluate(dets, times, inplace=False)
    assert ev_mod is not in_mod
    assert ev_mod.loglik < 0 and not np.isinf(ev_mod.loglik)
    assert ev_mod.nphot == nphot
    ev_mod = in_mod.evaluate(dets, times)
    assert in_mod.loglik < 0 and not np.isinf(in_mod.loglik)
    assert in_mod.nphot == nphot
    
    

if __name__ == '__main__':
    pytest.main(['-x, -v, tests/test_H2MM.py'])
