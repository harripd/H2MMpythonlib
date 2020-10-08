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
    h_mod = EM_H2MM(h_model,data_ph_short,data_time_short)
    

trans2 = np.array([[0.9999,0.0001],[0.001,0.999]])
prior2 = np.array([0.3,0.7])
obs2 = np.array([[0.1,0.9],[0.4,0.6]])
h_mod2 = h2mm_model(prior2,trans2,obs2)
trans3 = np.array([[0.98, 0.01, 0.01],[0.01, 0.98, 0.01],[0.01, 0.01, 0.98]])
prior3 = np.array([0.2, 0.6, 0.2])
obs3 = np.array([[0.1, 0.9],[0.5, 0.5],[0.7, 0.3]])
h_mod3 = h2mm_model(prior3, trans3, obs3)
data_time_short = np.zeros(12,dtype=int)

for i in range(1,12):
    if i != 4:
        data_time_short[i] = i + data_time_short[i-1]
    else:
        data_time_short[i] = i + data_time_short[i-1] + 3

data_time_short = np.array([0, 4, 6, 13, 19, 21, 50, 54, 55, 59, 70, 72])
data_ph_short = np.zeros(12,dtype=int)
for i in range(1,len(data_time_short)):
    data_ph_short[i] = data_time_short[i] % 2

deltas_short = np.diff(data_time_short)
R_short = ph_factors(deltas_short)
deltas_short = np.diff(data_time_short)
R_short = ph_factors(deltas_short)
transmat_t_short = CalculatePowerofTransMatrices(h_mod2,R_short.R)
transmat_t3_short = CalculatePowerofTransMatrices(h_mod3,R_short.R)

h_mod = EM_H2MM(h_mod2,[data_ph_short],[data_time_short])