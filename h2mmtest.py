#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 17:37:47 2020

@author: paul
"""
import numpy as np
from H2MM import *
from fretbursts import *

def H2MM_eval(h_model,data):    
    DAemDex_mask = data.get_ph_mask(ph_sel=Ph_sel(Dex='DAem'))
    AemDex_mask = data.get_ph_mask(ph_sel=Ph_sel(Dex='Aem'))
    AemDex_mask_red = AemDex_mask[DAemDex_mask]
    d_bursts = data.mburst[0]
    ph_d = data.get_ph_times(ph_sel=Ph_sel(Dex='DAem'))
    d_bursts_red = d_bursts.recompute_index_reduce(ph_d)
    data_color = []
    data_time = []
    for start, stop in zip(d_bursts.istart,d_bursts.istop+1):
        temp_ph_mask = AemDex_mask_red[start:stop]
        temp_color = np.zeros((temp_ph_mask.shpae),dtype=int)
        temp_color[temp_ph_mask] = 1
        data_time.append(ph_d[start:stop])
        data_color.append(temp_color)
    h_mod = EM_H2MM_par(h_model,data_color,data_time)
    return h_mod

if __name__ == "__main__" :
    filename = './data/minus8TA_minus6NTD_RPo_T25C_G150uW_R100uW_1.hdf5'
    if os.path.isfile(filename):
        print('Perfect, I found the file!')
    else:
        print('ERROR: file does not exist')
    d = loader.photon_hdf5(filename)
    d.add(det_donor_accept=(0, 1), 
          alex_period=4000, 
          D_ON=(2100, 3900), 
          A_ON=(150, 1900),
          offset=700)
    loader.usalex_apply_period(d)
    d.calc_bg(fun=bg.exp_fit,time_s=50.1, tail_min_us='auto', F_bg=1.7)
    d.burst_search(m=10, F=6, ph_sel=Ph_sel(Dex='DAem'))
    d.fuse_bursts(ms=0)
    d_all = Sel(d, select_bursts.naa, th1=50)
    d_all = Sel(d_all, select_bursts.size, th1=50)
    prior2 = np.array([0.1, 0.9])
    trans2 = np.array([[0.998, 0.002],[0.0001, 0.9999]])
    obs2 = np.array([[0.3, 0.7],[0.8, 0.2]])
    h_mod2i = h2mm_model(prior2,trans2,obs2)
    h_model = H2MM_eval(h_mod,d_all)