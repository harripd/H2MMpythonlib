#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photon by photon hidden Markov modeling scripts, when using this sofware, cite
Pirchi et. al. Journal of Physical Chemistry B, 2016, 120, 13065-13075

Software was originally written in matlab, by Pirchi, and has been addapted to
python by Paul David Harris

@author: Paul David Harris
"""

import time
import concurrent.futures
import numpy as np
from numpy.linalg import matrix_power


"""
The h2mm_model class contains all the parameters needed to define a H2MM 
 model, the dimentionality is N=number of states (usually between 1 and 4),
 and M =  number of detectors (usually 2, the donor and acceptor)
 The model parameters and fields are as follows:
     prior: the initial state probability, a stochastic 1-D N element array
     trans: the transition probability matrix, an NxN array, which is row stochastic
     obs: an NxM matrix, N = number of states, M = number of detectors, which 
        is row stochastic
     loglik: the logliklihood value of the model, this is calculated with the 
        fwdback_photonByphoton_fast function, hence when initializing the model the
        loglik is automatically set to -inf
 """
class h2mm_model:
    def __init__(self,prior,trans,obs,loglik=-np.inf):
        self.nstate = prior.size
        self.nchannels = obs.shape[1]
        if self.nstate == prior.shape[0] == trans.shape[1] == trans.shape[0] == obs.shape[0]:
            self.prior = prior
            self.obs = obs
            self.trans = trans
            self.loglik = loglik
        else:
            print('Error: prior, trans and obs must have same number of states')
            self.prior = prior
            self.obs = obs
            self.trans = trans
            self.loglik = loglik
    def normalize(self): # this method is used to make each matrix/array properly stochastic
        self.prior = self.prior / np.sum(self.prior)
        for i in range(0,self.nstate):
            self.trans[i,:] = self.trans[i,:] / np.sum(self.trans[i,:])
            self.obs[i,:] = self.obs[i,:] / np.sum(self.obs[i,:])
            

"""
The fwdback_values class is used like a precurrsor to update the h2mm model.
 functionally it serves just as a containter for the fwdback_photonByphoton_fast
 function to return its values.
alpha (forward variable) and beta (backward variable) are technically unnecessary
 as the information in these is then encapsulated into gamma, and it is gamma
 and xi_summed which are actually needed to update the h2mm_model
 however, as alpha and beta are potentially usefull, these are incorporated into
 the class for future flexibility
"""
class fwdback_values:
    def __init__(self,alpha,beta,gamma,loglik,xi_summed,ex):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loglik = loglik
        self.xi_summed = xi_summed
        self.ex = ex


"""
The ph_factors class is used for breaking down the different arrival deltas into
 their relevant components (partitioning).
It contains 2 key fields:
    deltas: a 1-D array array of all unique interphoton times, sorted from least to greatest
        this array is equivalent to TotalArrivalDelta in matlab
    R: a list of arrays, which identify a partition of deltas
        this list is equivalent to R in matlab
This class serves the same function as Factorization in matlab.
The iniialization function, when given the list of interphoton times automatically
calls the appropriate methods to create both deltas and R
"""
class ph_factors:
    def __init__(self,deltas):
        deltas = np.unique(deltas)
        deltas = deltas[deltas!=0]
        if deltas[0] != 1:
            deltas = np.append(1,deltas)
        self.deltas = deltas
        self.R = self.factorize(deltas)
        self.expand()
    def factorize(self,deltas): # this method is used to create the R array from deltas
        deltas = np.unique(deltas)
        deltas = deltas[deltas!=0]
        if deltas[0] != 1:
            deltas = np.append(1,deltas)
        R = [np.array([[0,1]])]
        for i in range(1,deltas.size):
            d = deltas[i] // deltas[i-1]
            m = deltas[i] % deltas[i-1]
            R.append(np.array([[i-1,d]]))
            while m != 0:
                k = np.argwhere(deltas == m)
                if k.size != 0:
                    R[i] = np.append(R[i],np.array([[k[0,0], 1]]),axis=0)
                    m = 0
                else:
                    k = np.argwhere(deltas < m)[-1,0]
                    d = m // deltas[k]
                    m = m % deltas[k]
                    R[i] = np.append(R[i],np.array([[k,d]],dtype=int),axis=0)
        return R
    def expand(self): # this method is used to add extra deltas, which nominally increases computational efficiency
        B = np.array([],dtype=int)
        for i in range(0,len(self.R)):
            if self.R[i].shape[0] > 1:
                k = np.argwhere(self.R[i][:,1]>1)
                for j in range(0,k.shape[0]):
                    temp_ind = self.R[i][k[j,0],0]
                    temp_d = self.R[i][k[j,0],1]
                    temp_t = self.deltas[temp_ind]
                    B = np.append(B,temp_t*temp_d)
        B = np.unique(B)
        deltas = np.append(self.deltas,B)
        deltas = np.unique(deltas)
        if deltas.size != self.deltas.size:
            self.deltas = deltas
            self.R = self.factorize(deltas)
            S = True
        else:
            S = False
        return S


"""
CalculatePowerofTransMatrices is equivalent to similarly names matlab function
Its purpose is to create transmat_t, a XxNxN array, a stack of trans matrices
reflecting the power of the trans matrix in the Xth position in the deltas array
"""    
def CalculatePowerofTransMatrices(h_model,R):
    nfact = len(R)
    M = np.zeros((nfact,h_model.nstate,h_model.nstate))
    M[0,:,:] = h_model.trans
    for ii in range(1,nfact):
        M_temp = np.eye(h_model.nstate)
        for iii in range(0,R[ii].shape[0]):
            M[ii,:,:] = M_temp @ matrix_power(M[R[ii][iii,0],:,:],R[ii][iii,1])
            M_temp = M[ii,:,:]
    return M

def Rho_product_fast(A_dt1,A_dt2,Rho1,Rho2):
    N = Rho1.shape[0]
    Rho12 = np.zeros((N,N,N,N))
    for m in range(0,N):
        for n in range(0,N):
            Rho12[m,n,:,:] = Rho1[m,n,:,:]@A_dt2 + A_dt1@Rho2[m,n,:,:]
    return Rho12

def Rho_power_fast(A_dt0,Rho0,power):
    N = A_dt0.shape[0]
    A0 = np.eye(N)
    Rho = Rho0
    for i in range(0,power-1):
        A0 = A0@A_dt0
        Rho = Rho_product_fast(A0,A_dt0,Rho,Rho0)
    return Rho


"""
Calc_Rho is equivalent to the matlab function of the same name
Calc_Rho calculates the sum of P(x(t)=i,x(t)=j, x(to(n+1))=m | x(to(n))=k) (Rho).
over all t in the range to(n)<=t<to(n+1).
"""
def Calc_Rho(transmat_t,R):
    len_t = transmat_t.shape[0]
    Ns = transmat_t.shape[1]
    Rho = np.zeros((len_t,Ns,Ns,Ns,Ns))
    # the follwoing nested for loop is the initiation (eq. 24) step of the h2mm Rho calculation
    for i in range(0,Ns):
        for j in range(0,Ns):
            Rho[0,i,j,i,j] = transmat_t[0,i,j]
    # this is the recursion step
    for t in range(1,len_t):
        R_temp = R[t]
        tempRsize = R_temp.shape[0]
        if tempRsize > 1:
            for Rind in range(0,tempRsize-1):
                if Rind == 0:
                    # this side of the if statement is used to initiate the initiation step
                    transmat_1 = transmat_t[R_temp[Rind,0],:,:]
                    if R_temp[Rind,1] == 1:
                        Rho_1 = Rho[R_temp[Rind,0],:,:,:,:]
                    else:
                        Rho_1 = Rho_product_fast(transmat_1,Rho_1,R_temp[Rind,1])
                        transmat_1 = matrix_power(transmat_1,R_temp[Rind,1])
                    transmat_2 = transmat_t[R_temp[Rind+1,0],:,:]
                    if R_temp[Rind+1,1] == 1:
                        Rho_2 = Rho[R_temp[Rind+1,0],:,:,:,:]
                    else:
                        Rho_2 = Rho_power_fast(transmat_2,Rho_2,R_temp[Rind,1])
                        transmat_2 = matrix_power(transmat_2,Rho_2,R_temp[Rind,1])
                    Rho_temp = Rho_product_fast(transmat_1,transmat_2,Rho_1,Rho_2)
                else:
                    Rho_1 = Rho_temp
                    transmat_1 = transmat_2 @ transmat_2
                    transmat_2 = transmat_t[R_temp[Rind+1,0],:,:]
                    if R_temp[Rind+1,1] == 1:
                        Rho_2 = Rho[R_temp[Rind+1,0],:,:,:,:]
                    else:
                        Rho_2 = Rho_power_fast(transmat_2,Rho[R_temp[Rind+1,0],:,:,:,:],R_temp[Rind+1,1])
                        transmat_2 = matrix_power(transmat_2,R_temp[Rind+1,1])
                    Rho_temp = Rho_product_fast(transmat_1,transmat_2,Rho_1,Rho_2)
            Rho[t,:,:,:,:] = Rho_temp
        else:
            Rho_1 = Rho[R_temp[0,0],:,:,:,:]
            transmat_1 = transmat_t[R_temp[0,0],:,:]
            Rho[t,:,:,:,:] = Rho_power_fast(transmat_1,Rho[R_temp[0,0],:,:,:,:],R_temp[0,1])
    return Rho


"""
fwdback_photonByphoton_fast is equivalent to the matlab function of the same name
It calculaets the posterior probabilites using the forward backward algorithm
"""
def fwdback_photonByphoton_fast(h_model,ph_color,arrivalinds,Rho,transmat_t,ex):
    loglik = 0
    T = arrivalinds.shape[0]
    S = h_model.nstate
    alpha = np.zeros((T,S))
    beta = np.zeros((T,S))
    gamma = np.zeros((T,S))
    xi_summed = np.zeros((S,S))
    obslik = np.zeros((T,S))
    scale = np.zeros(T)
    # Setting up obslik (equivalent to multinomial_prob)
    for t in range(0,T):
        obslik[t,:] = h_model.obs[:,ph_color[t]]
    # alpha initiation
    alpha[0,:] = h_model.prior*obslik[0,:]
    scale[0] = alpha[0,:].sum()
    alpha[0,:] = alpha[0,:]/scale[0]
    # alpha recursion
    for t in range(1,T):
        trans = transmat_t[arrivalinds[t],:,:]
        a = trans.T @ alpha[t-1,:]
        a2 = a * obslik[t,:]
        scale[t] = a2.sum()
        alpha[t,:] = a2 / scale[t]
    if np.any(scale == 0):
        loglik = -np.inf
    else:
        loglik = np.log(scale).sum()
    # beta initiation
    beta[T-1,:] = np.ones(S)
    gamma[T-1,:] = alpha[T-1,:] * beta[T-1,:] / (alpha[T-1,:]*beta[T-1,:]).sum()
    # beta recusion
    for t in range(T-2,-1,-1):
        b = beta[t+1,:]*obslik[t+1,:]
        trans = transmat_t[arrivalinds[t+1],:,:]
        trans0 = trans
        trans0[trans0==0] = 1
        beta[t,:] = (trans @ b) / (trans @ b).sum()
        gamma[t,:] = alpha[t,:]*beta[t,:]/(alpha[t,:]*beta[t,:]).sum()
        xi_temp = trans * np.outer(alpha[t,:],b)
        xi_temp = xi_temp/xi_temp.sum()
        fxi = ((xi_temp / trans0)* Rho[arrivalinds[t+1],:,:,:,:]).sum(axis=3).sum(axis=2)
        xi_summed += fxi
    return fwdback_values(alpha,beta,gamma,loglik,xi_summed,ex)


"""
compute_ess_dhmm is the single processor version of the matlab function of the
same name.
It accpets the data, and h2mm model, and computes teh loglikelihood of the 
model and produces an updated model based on the gamma and xi_summed values
"""
def compute_ess_dhmm(h_model,ph_color,ArrivalInds,R):
    numex = len(ArrivalInds)
    Nd = h_model.obs.shape[1]
    exp_num_trans = np.zeros((h_model.nstate,h_model.nstate))
    exp_num_visits = np.zeros(h_model.nstate)
    exp_num_emits = np.zeros((numex,h_model.nstate,Nd))
    loglik = 0
    transmat_t = CalculatePowerofTransMatrices(h_model,R.R)
    Rho = Calc_Rho(transmat_t,R.R)
    for ex in range(0,numex):
        T = ph_color[ex].size
        param_temp = fwdback_photonByphoton_fast(h_model,ph_color[ex],ArrivalInds[ex],Rho,transmat_t,ex)
        loglik += param_temp.loglik
        exp_num_trans += param_temp.xi_summed
        exp_num_visits += param_temp.gamma[0,:]
        if T < Nd:
            for t in range(0,T):
                p = ph_color[ex][t]
                exp_num_emits[ex,p,:] += param_temp.gamma[t,:]
                print('Triggered T<Ns')
        else:
            for n in range(0,Nd):
                ndx = np.argwhere(ph_color[ex]==n)[:,0]
                if ndx.size != 0:
                    exp_num_emits[ex,:,n] = param_temp.gamma[ndx,:].sum(axis=0)
    exp_num_emit = exp_num_emits.sum(axis=0)
    return h2mm_model(exp_num_visits,exp_num_trans,exp_num_emit,loglik=loglik)

"""
compute_ess_dhmm_par is the parallelized version of the matlab function
    compute_ess_dhmm
It accpets the data, and h2mm model, and computes teh loglikelihood of the 
model and produces an updated model based on the gamma and xi_summed values
"""
def compute_ess_dhmm_par(h_model,ph_color,ArrivalInds,R):
    numex = len(ArrivalInds)
    Nd = h_model.obs.shape[1]
    exp_num_trans = np.zeros((h_model.nstate,h_model.nstate))
    exp_num_visits = np.zeros(h_model.nstate)
    exp_num_emits = np.zeros((numex,h_model.nstate,Nd))
    loglik = 0
    transmat_t = CalculatePowerofTransMatrices(h_model,R.R) # calculate powers of transistion matrix for all unique interphoton times
    Rho = Calc_Rho(transmat_t,R.R) # calculate Rho for all interphoton times
    with concurrent.futures.ProcessPoolExecutor() as executor: # open context manager for parallel processing of all bursts by fwdback_photonByphoton
        param_temp = {executor.submit(fwdback_photonByphoton_fast, h_model,ph_color[ex],ArrivalInds[ex],Rho,transmat_t,ex) for ex in range(0,numex)} # calculating in parallel the gamma and xi_summed factors for each burst
        for params in concurrent.futures.as_completed(param_temp): # for loop loops over results from all bursts to generate new loglik, prior, trans and obs values note: prior trans and obs are not normalized at this point
            param = params.result()
            loglik += param.loglik
            exp_num_trans += param.xi_summed
            exp_num_visits += param.gamma[0,:]
            if ph_color[param.ex].size < Nd:
                for t in range(0,ph_color[param.ex].size):
                    p = ph_color[param.ex][t]
                    exp_num_emits[param.ex,p,:] += param.gamma[t,:]
            else:
                for n in range(0,Nd):
                    ndx = np.argwhere(ph_color[param.ex]==n)[:,0]
                    if ndx.size != 0:
                        exp_num_emits[param.ex,:,n] = param.gamma[ndx,:].sum(axis=0)
    exp_num_emit = exp_num_emits.sum(axis=0)
    return h2mm_model(exp_num_visits,exp_num_trans,exp_num_emit,loglik=loglik)


"""
EM_H2MM is the single process version equivalent to the matlab function of the
 same name.
inputs are as follows:
    h_mod: a h2mm_model class, containing prior, trans and obs initiation values
    burst_colors: a list of 1-D numpy integer arrays, each array contains the
        detector (color) of the photons in the burst
    burst_times: a list of 1-D numpy integer arrays, each array contains the 
        arrival times of the photons, list length and size of each array must
        match burst_colors
        ### futuer versions might consider a special class or other way of
        ### linking these two sets of data more tighly together
    ### optional arguments ###
        max_iter: the number of iterations at which point the calculation will 
            terminate, default is 3600, which is much higher than should be necessary
        max_time: time limit for calcuation, similar to max_iter, but set to inf
            by default, allowing max_iter to set the time limit
        converged_min: minimum difference between previous loglik and current
            loglik to consider the calculation converged, sets a limit to account
            for floating point inaccuracies, and the fact that models do not
            vary significantly when variations of loglik are small
"""
def EM_H2MM(h_mod,burst_colors,burst_times,max_iter=3600,max_time=np.inf,converged_min=1e-14):
    # Initiate varable and do some initial calculations
    assert len(burst_times) == len(burst_colors)
    TotalArrivalDelta = np.empty((0),dtype=int)
    burst_deltas = []
    for i in range(0,len(burst_times)):
        deltas_temp = np.diff(burst_times[i])
        deltas_temp[deltas_temp==0] = 1
        TotalArrivalDelta = np.append(TotalArrivalDelta,deltas_temp)
        burst_deltas.append(np.append(1,deltas_temp))
        assert burst_colors[i].size == burst_deltas[i].size
    assert len(burst_deltas) == len(burst_colors)
    R = ph_factors(TotalArrivalDelta)
    burst_inds = []
    for i in range(0,len(burst_deltas)):
        burst_inds_temp = np.ones(burst_deltas[i].shape,dtype=int)
        for t in range(0,burst_deltas[i].size):
            burst_inds_temp[t] = np.argwhere(burst_deltas[i][t] == R.deltas)[0,0]
        burst_inds.append(burst_inds_temp)
        assert burst_inds[i].size == burst_colors[i].size
    assert len(burst_inds) == len(burst_colors)
    
    # Initialize variable for main optimization loop
    h_mod_current = h_mod
    LL = np.empty((0),dtype=float)
    niter = 0
    cont = True
    tm_start = time.perf_counter()
    tm_iter = tm_start
    print('Begin compute_ess_dhmm')
    while cont: # while loop loops until one of the end conditions is met...
        h_mod_temp = compute_ess_dhmm(h_mod_current,burst_colors,burst_inds,R) # calculates new model
        if h_mod_temp.loglik - h_mod_current.loglik > converged_min: # check main condition of convergence
            h_mod_temp.normalize()
            h_mod_current = h_mod_temp
            LL = np.append(LL,h_mod_current.loglik)
            tm_prev = tm_iter
            tm_iter = time.perf_counter()
            tm_elapse = tm_iter - tm_start
            tm_cycle = tm_iter - tm_prev
            print(f'Iteration {niter}, loglik {h_mod_current.loglik}, iteration time {tm_cycle}, total time {tm_elapse}')
            niter += 1
            if max_iter < niter: # checking alternative end conditions, 
                h_mod_current = h_mod_temp
                print('Maximum iterations reached')
                cont = False
            if max_time*3600 < tm_elapse:
                print('Maximum time reached')
                cont = False
        else:
            print('Converged')
            cont = False
    return h_mod_current


"""
EM_H2MM is the parallel version equivalent to the matlab function EM_H2MM.
inputs are as follows:
    h_mod: a h2mm_model class, containing prior, trans and obs initiation values
    burst_colors: a list of 1-D numpy integer arrays, each array contains the
        detector (color) of the photons in the burst
    burst_times: a list of 1-D numpy integer arrays, each array contains the 
        arrival times of the photons, list length and size of each array must
        match burst_colors
        ### futuer versions might consider a special class or other way of
        ### linking these two sets of data more tighly together
    ### optional arguments ###
        max_iter: the number of iterations at which point the calculation will 
            terminate, default is 3600, which is much higher than should be necessary
        max_time: time limit for calcuation, similar to max_iter, but set to inf
            by default, allowing max_iter to set the time limit
        converged_min: minimum difference between previous loglik and current
            loglik to consider the calculation converged, sets a limit to account
            for floating point inaccuracies, and the fact that models do not
            vary significantly when variations of loglik are small
"""
def EM_H2MM_par(h_mod,burst_colors,burst_times,max_iter=3600,max_time=np.inf,converged_min=1e-14):
    # Initiate varable and do some initial calculations
    assert len(burst_times) == len(burst_colors)
    TotalArrivalDelta = np.empty((0),dtype=int)
    burst_deltas = []
    for i in range(0,len(burst_times)):
        deltas_temp = np.diff(burst_times[i])
        deltas_temp[deltas_temp==0] = 1
        TotalArrivalDelta = np.append(TotalArrivalDelta,deltas_temp)
        burst_deltas.append(np.append(1,deltas_temp))
        assert burst_colors[i].size == burst_deltas[i].size
    assert len(burst_deltas) == len(burst_colors)
    R = ph_factors(TotalArrivalDelta)
    burst_inds = []
    for i in range(0,len(burst_deltas)):
        burst_inds_temp = np.ones(burst_deltas[i].shape,dtype=int)
        for t in range(0,burst_deltas[i].size):
            burst_inds_temp[t] = np.argwhere(burst_deltas[i][t] == R.deltas)[0,0]
        burst_inds.append(burst_inds_temp)
        assert burst_inds[i].size == burst_colors[i].size
    assert len(burst_inds) == len(burst_colors)
    
    # Initialize variable for main optimization loop
    h_mod_current = h_mod
    LL = np.empty((0),dtype=float)
    niter = 0
    cont = True
    tm_start = time.perf_counter()
    tm_iter = tm_start
    print('Begin compute_ess_dhmm')
    while cont: # while loop loops until one of the end conditions is met...
        h_mod_temp = compute_ess_dhmm_par(h_mod_current,burst_colors,burst_inds,R) # calculates new model
        if h_mod_temp.loglik - h_mod_current.loglik > converged_min: # check main condition of convergence
            h_mod_temp.normalize()
            h_mod_current = h_mod_temp
            LL = np.append(LL,h_mod_current.loglik)
            tm_prev = tm_iter
            tm_iter = time.perf_counter()
            tm_elapse = tm_iter - tm_start
            tm_cycle = tm_iter - tm_prev
            print(f'Iteration {niter}, loglik {h_mod_current.loglik}, iteration time {tm_cycle}, total time {tm_elapse}')
            niter += 1
            if max_iter < niter: # checking alternative end conditions, 
                h_mod_current = h_mod_temp
                print('Maximum iterations reached')
                cont = False
            if max_time*3600 < tm_elapse:
                print('Maximum time reached')
                cont = False
        else:
            print('Converged')
            cont = False
    return h_mod_current

def viterbi_path_PhotonByPhoton(h_model,ph_color,ph_arrivaltime):
    assert ph_color.size == ph_arrivaltime.size
    T = ph_color.shape[0]
    S = h_model.nstate
    obslik = np.zeros((T,S),dtype=float)
    for t in range(0,T):
        obslik[t,:] = h_model.obs[:,ph_color[t]]
    ph_arrivaldiff = np.diff(ph_arrivaltime)-1
    ph_arrivaldiff_max = np.max(ph_arrivaldiff)+1
    transmat_t = np.zeros((ph_arrivaldiff_max,S,S))
    transmat_t[0,:,:] = h_model.trans
    for t in range(1,ph_arrivaldiff_max):
        transmat_t[t,:,:] = transmat_t[t-1,:,:] @ h_model.trans
    delta = np.zeros((T,S))
    psi = np.zeros((T,S))
    scale = np.ones(T)
    path = np.zeros(T,dtype=int)
    t = 0
    delta[t,:] = h_model.prior * obslik[t,:]
    scale[t] = 1/delta[t,:].sum()
    delta[t,:] = delta[t,:]/delta[t,:].sum()
    psi[t,:] = 0
    for t in range(1,T):
        trans = transmat_t[ph_arrivaldiff[t-1],:,:]
        for j in range(0,S):
            delta_temp = trans[:,j] * delta[t-1,:]
            delta[t,j] = np.max(delta_temp)
            psi[t,j] = np.argwhere(delta[t,j]==delta_temp)[0]
            delta[t,j] = delta[t,j] * obslik[t,j]
        scale[t] = 1/delta[t,:].sum()
        delta[t,:] = delta[t,:]/delta[t,:].sum()
    t = T-1
    path[t] = np.argwhere(np.max(delta[t,:])==delta[t,:])[0]
    for t in range(T-2,-1,-1):
        path[t] = psi[t+1,path[t+1]]
    return path