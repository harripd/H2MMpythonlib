#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:12:55 2021
Tests for verification of H2MM_C functions
@author: paul
"""

import numpy as np
from itertools import repeat, chain

import pytest
import H2MM_C as h2

def model_almost_equal(model1,model2):
    prior = np.all(np.abs(model1.prior - model2.prior) < 0.1)
    trans_check = model1.trans / model2.trans
    trans = np.all(trans_check < 5) * np.all(trans_check > 0.5)
    obs = np.all(np.abs(model1.obs - model2.obs) < 0.1)
    return prior * trans * obs

@pytest.fixture
def simple_data():
    times = [np.array([   20,   50,   90,   95,  190,  230,  350,  800]),
             np.array([  820,  850,  990,  995, 1055, 1130, 1290, 1525]),
             np.array([ 1750, 1820, 1950, 1985, 2055, 2110, 2220, 2325])]
    dets =  [np.array([    1,    1,    0,    1,    0,    1,    0,    1]),
             np.array([    0,    0,    0,    1,    1,    1,    1,    0]),
             np.array([    1,    0,    1,    1,    0,    1,    0,    1])]
    return dets, times

def test_MakeModel():
    prior = np.ones(3)
    trans = np.array([[0.999,0.0005,0.005],[0.0005,0.999,0.0005],[0.0005,0.0005,0.999]])
    obs = np.array([[0.1,0.2,0.7],[0.5,0.3,0.2],[0.4,0.4,0.2]])
    model = h2.h2mm_model(prior,trans,obs)
    assert model.nstate == 3
    assert model.ndet == 3
    assert model.prior.sum() - 1 < 1e-10
    assert np.all(model.trans.sum(axis=1) - 1 < 1e-10)
    assert np.all(model.obs.sum(axis=1) -1 < 1e-10)
    assert model.k == 14

def test_Factory():
    model = h2.factory_h2mm_model(3,2)
    assert type(model) == h2.h2mm_model
    assert model.nstate == 3
    assert model.ndet == 2
    assert model.k == 11

def test_Model_access():
    model = h2.factory_h2mm_model(4,3)
    prior = model.prior
    prior[0] = 5.5
    trans = model.trans
    trans[0,0] = 5.5
    obs = model.obs
    obs[0,0] = 5.5
    assert model.prior.sum() - 1 < 1e-10
    assert np.all(model.trans.sum(axis=1) - 1 < 1e-10)
    assert np.all(model.obs.sum(axis=1) -1 < 1e-10)

def test_Sim():
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

def test_baddata(simple_data):
    dets, time = simple_data[0].copy(), simple_data[1].copy()
    time[0][1] = 0
    with pytest.raises(ValueError):
        h2.EM_H2MM_C(h2.factory_h2mm_model(2,2), dets, time)
    dets = simple_data[0].copy()
    dets[0][0] = 2
    with pytest.raises(Exception):
        h2.EM_H2MM_C(h2.factory_h2mm_model(2,2), dets, simple_data[1])
    dets[0] = np.concatenate([dets[0], [1]])
    with pytest.raises(ValueError):
        h2.EM_H2MM_C(h2.factory_h2mm_model(2,2), dets, time)

if __name__ == '__main__':
    pytest.main(['-x, -v, tests/test_H2MM.py'])
