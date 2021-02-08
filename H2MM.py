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
import h5py



class h2mm_model:
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
    def __init__(self,prior,trans,obs,loglik=-np.inf):
        self.nstate = prior.size
        self.nstream = obs.shape[1]
        if self.nstate == prior.shape[0] == trans.shape[1] == trans.shape[0] == obs.shape[0]:
            self.prior = prior
            self.obs = obs
            self.trans = trans
            self.loglik = loglik
            self.converged = False
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
    def set_converged(self,set=True):
        self.converged = set
            


class fwdback_values:
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
    def __init__(self,alpha,beta,gamma,loglik,xi_summed,ex):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loglik = loglik
        self.xi_summed = xi_summed
        self.ex = ex



class ph_factors:
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



def CalculatePowerofTransMatrices(h_model,R):
    """
    CalculatePowerofTransMatrices is equivalent to similarly names matlab function
    Its purpose is to create transmat_t, a XxNxN array, a stack of trans matrices
    reflecting the power of the trans matrix in the Xth position in the deltas array
    """    
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



def Calc_Rho(transmat_t,R):
    """
    Calc_Rho is equivalent to the matlab function of the same name
    Calc_Rho calculates the sum of P(x(t)=i,x(t)=j, x(to(n+1))=m | x(to(n))=k) (Rho).
    over all t in the range to(n)<=t<to(n+1).
    """
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
                        Rho_1 = Rho_power_fast(transmat_1,Rho[R_temp[Rind,0],:,:,:,:],R_temp[Rind,1])
                        transmat_1 = matrix_power(transmat_1,R_temp[Rind,1])
                    transmat_2 = transmat_t[R_temp[Rind+1,0],:,:]
                    if R_temp[Rind+1,1] == 1:
                        Rho_2 = Rho[R_temp[Rind+1,0],:,:,:,:]
                    else:
                        Rho_2 = Rho_power_fast(transmat_2,Rho[R_temp[Rind+1,0],:,:,:,:],R_temp[Rind+1,1])
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


def fwdback_photonByphoton_fast(h_model,ph_color,arrivalinds,Rho,transmat_t,ex):
    """
    fwdback_photonByphoton_fast is equivalent to the matlab function of the same name
    It calculaets the posterior probabilites using the forward backward algorithm
    """
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


def compute_ess_dhmm(h_model,ph_color,ArrivalInds,R):
    """
    compute_ess_dhmm is the single processor version of the matlab function of the
    same name.
    It accpets the data, and h2mm model, and computes teh loglikelihood of the 
    model and produces an updated model based on the gamma and xi_summed values
    """
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


def compute_ess_dhmm_par(h_model,ph_color,ArrivalInds,R):
    """
    compute_ess_dhmm_par is the parallelized version of the matlab function
        compute_ess_dhmm
    It accpets the data, and h2mm model, and computes teh loglikelihood of the 
    model and produces an updated model based on the gamma and xi_summed values
    """
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



def EM_H2MM(h_mod,burst_colors,burst_times,max_iter=3600,max_time=np.inf,converged_min=1e-14):
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
            h_mod_current.set_converged()
            cont = False
    return h_mod_current



def EM_H2MM_par(h_mod,burst_colors,burst_times,max_iter=3600,max_time=np.inf,converged_min=1e-14,save_file=None,save_freq=100):
    """
    EM_H2MM_par is the parallel version equivalent to the matlab function EM_H2MM.
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
                if save_file != None:
                    with h5py.File(save_file,'w') as f:
                        dset = f.create_dataset('prior',data=h_mod_current.prior)
                        dset = f.create_dataset('trans',data=h_mod_current.trans)
                        dset = f.create_dataset('obs',data=h_mod_current.obs)
                        dset = f.create_dataset('loglik',data=h_mod_temp.loglik)
            if max_time*3600 < tm_elapse:
                print('Maximum time reached')
                cont = False
                if save_file != None:
                    with h5py.File(save_file,'w') as f:
                        dset = f.create_dataset('prior',data=h_mod_current.prior)
                        dset = f.create_dataset('trans',data=h_mod_current.trans)
                        dset = f.create_dataset('obs',data=h_mod_current.obs)
                        dset = f.create_dataset('loglik',data=h_mod_temp.loglik)
            if save_file != None and niter % save_freq == 0:
                with h5py.File(save_file,'w') as f:
                    dset = f.create_dataset('prior',data=h_mod_current.prior)
                    dset = f.create_dataset('trans',data=h_mod_current.trans)
                    dset = f.create_dataset('obs',data=h_mod_current.obs)
                    dset = f.create_dataset('loglik',data=h_mod_temp.loglik)
        else:
            print('Converged')
            h_mod_current.set_converged()
            cont = False
            if save_file != None:
                with h5py.File(save_file,'w') as f:
                    dset = f.create_dataset('prior',data=h_mod_current.prior)
                    dset = f.create_dataset('trans',data=h_mod_current.trans)
                    dset = f.create_dataset('obs',data=h_mod_current.obs)
                    dset = f.create_dataset('loglik',data=h_mod_temp.loglik)
    return h_mod_current



def viterbi_path_PhotonByPhoton(h_model,ph_color,ph_arrivaltime):
    """
    viteri_path_PhotonByPhoton takes the following inputs:
    h_model: an h2mm_model object, this should be optimized to have the highest
    loglik for the input data
    ph_color: 1-D numpy int array, representing the detector of photons in a single
    burst, represented as indexes
    ph_arrivaltime: a 1-D numpy int array, of the same size as ph_color, showing the
    arrival times of the photons in ph_color
    
    This function uses the viterbi algorithm to determine the most likely path given
    the photon stream and the hidden markov model.
    This function returns one int and one float numpy array, the int array returns
    the most likely state of each photon, while the float array returns the max value
    of the delta array at that time.
    """
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
    prob = np.zeros(T,dtype=float)
    for t in range(T-2,-1,-1):
        path[t] = psi[t+1,path[t+1]]
        prob[t] = delta[t,path[t]]
    return path, prob

def viterbi_sort(h_model, ph_color, ph_time):
    """
    Compute the viterbi path of a single burst, returns additional information
    about dwell photons and times

    Parameters
    ----------
    h_model : OBJECT h2mm_model
        This should be the optimized H2MM model parameters for the input data
        in the proceeding arguments
    ph_color : NUMPY.NDARRY(int, 1D)
        A list of numpy integer arrays with elements representing the index of
        the detector at which the photon arrived, one numpy array per burst
    ph_time : NUMPY.NDARRY(int, 1D)
        A lis of numpy integer arrays with elements representing the arrival
        time of each photon, one numpy array per burst
    Returns
    -------
    path : NUMPY.NDARRAY(int, 1D)
        Viterbi path, same dimension as ph_color.
    prob : NUMPY.NDARRAY(flaot, 1D)
        Probability of photon being from given state, from delta in viterbi,
        same dimension as ph_color
    dwell_ph_count : LIST of NUMPY.NDARRAY(int, 2D)
        numpy arrays containing counts of photons in each state per dwell
        each element in the list corresponds to different state
    dwell_length : LIST of LIST of NUMPY.NDARRAY(int, 1D)
        numpy arrays counting the length of each dwell, discarding dwells at
        beginning and end of burst. Lists are nessted as [origin state][end state]

    """
    nstate = h_model.nstate
    nstream = h_model.nstream
    path, prob = viterbi_path_PhotonByPhoton(h_model,ph_color,ph_time)
    demar = np.diff(path)
    demar_ind = np.argwhere(demar!=0)[:,0]
    demar_beg = np.append(0,demar_ind+1)
    demar_end = np.append(demar_ind+1,path.size+1)
    dwell_ph_count = [np.empty((0,nstream)) for i in range(nstate)]
    dwell_length = [[np.empty(0,dtype=int) for i in range(nstate)] for k in range(nstate)]
    for beg, end in zip(demar_beg, demar_end):
        state_ind = path[beg] 
        ph_cnt = np.array([[(ph_color[beg:end] == stream).sum() for stream in range(nstream)]])
        dwell_ph_count[state_ind] = np.concatenate((dwell_ph_count[state_ind],ph_cnt.reshape((1,nstream))))
        # if statement filters out first and last 
        if beg != 0 and end != path.size+1:
            dwell_length[path[beg]][path[end]] = np.append(dwell_length[path[beg]][path[end]], ph_time[end] - ph_time[beg])
    return path, prob, dwell_ph_count, dwell_length
    



def viterbi_all(h_mod, ArrivalColor, ArrivalTime):
    """
    viteri_all takes the following inputs:
    h_model: an h2mm_model object, this should be optimized to have the highest
    loglik for the input data
    ph_color: a list of 1-D numpy int arrays, representing the detector of photons in a single
    burst, represented as indexes
    ph_arrivaltime: a list of 1-D numpy int arrays, of the same size as ph_color, showing the
    arrival times of the photons in ph_color
    
    This function uses the viterbi algorithm to determine the most likely path given
    the photon stream and the hidden markov model.
    This function returns one int and one float numpy array, the int array returns
    the most likely state of each photon, while the float array returns the max value
    of the delta array at that time.
    viterbi_all allows the user to run the viterbi algorithm of viterbi_path_PhotonByPhoton
    """
    nburst = len(ArrivalColor)
    path = [np.zeros(ArrivalColor[b].shape) for b in range(0,nburst)]
    prob = [np.zeros(ArrivalColor[b].shape) for b in range(0,nburst)]
    assert len(ArrivalColor) == len(ArrivalTime)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for burst, param_temp in enumerate(executor.map(viterbi_path_PhotonByPhoton,[h_mod for i in range(0,nburst)],ArrivalColor,ArrivalTime)):
            path[burst] = param_temp[0]
            prob[burst] = param_temp[1]
    return path, prob



def viterbi_error(h_mod, ArrivalColor, ArrivalTime):
    """
    Caclculate viterbi path, and uses it to compute the standard error of
    obs and trans parameters.

    Parameters
    ----------
    h_mod : OBJECT h2mm_model
        This should be the optimized H2MM model parameters for the input data
        in the proceeding arguments
    ArrivalColor : LIST of NUMPY.NDARRY(int)
        A list of numpy integer arrays with elements representing the index of
        the detector at which the photon arrived, one numpy array per burst
    ArrivalTime : LIST of NUMPY.NDARRY(int)
        A lis of numpy integer arrays with elements representing the arrival
        time of each photon, one numpy array per burst

    Returns
    -------
    TUPLE of NUMPY float arrays
        (path, prob, P_w, P_w_err, t_av, t_err, E_w, E_w_err)
        path: LSIT of NUMPY.NDARRAY(int, 1D)
            the most likey path, as determined using the viterbi algorithm
            through each burst, each element of the list is same dimensions as
            input ArrivalColor or ArrivalTime
        prob: LIST of NUMPY.NDARRAY(float, 1D)
            the value of the delta variable, indicating the probability that
            the given photon originated from the state assigned by the viterbi
            algorithm. each element of the list is same dimensions as input 
            ArrivalColor or ArrivalTime
        P_w: NUMPY.NDARRAY(float, 2D)
            weighted average of state probability matrix derived from viterbi
            fit, generally less accurate compared to the state probability 
            matrix derived by H2MM algorithm. Dimensions: nstate x nstream
        P_w_err: NUPMY.NDARRAY(flaot, 2D)
            standard error of P_w
        t_av: NUMPY.NDARRAY(flaot, 2D)
            mean transition rates derived from viterbi, generally less
            accurate that transition rate derived from H2MM algorithm.
            Dimensions: nstate x nstate
        t_err: NUMPY.NDARRAY(float, 2D)
            standard error of values in t_av. Dimensions: nstate x nstate
        E_w: NUMPY.NDARRAY(flaot, 1D) 
            weighted average of E values derived from viterbi algorithm,
            generally less accurate than values derived from H2MM algorithm,
            this assumes that index 0 is the donor stream, and index 1 is
            the acceptor stream, will give erroneous results otherwise.
            Length: nstate 
        E_w_err:  NUMPY.NDARRAY(float, 1D)
            standard error of E_w. Length: nstate
        
        Note: there may be 2 additional variables in the output, wait for 
        further documentations and publications for explanation

    """
    nburst = len(ArrivalColor)
    nstate = h_mod.nstate
    nstream = h_mod.nstream
    path = [[] for i in range(nburst)]
    prob = [[] for i in range(nburst)]
    dwell_lengths = [[] for i in range(nburst)]
    dwell_ph_counts = [[] for i in range(nburst)]
    # retrieve burst by burst information on dwell_lengths and photon counts
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for burst, dwell_temp in enumerate(executor.map(viterbi_sort,[h_mod for i in range(0,nburst)],ArrivalColor,ArrivalTime)):
            path[burst] = dwell_temp[0]
            prob[burst] = dwell_temp[1]
            dwell_ph_counts[burst] = dwell_temp[2]
            dwell_lengths[burst] = dwell_temp[3]
    dwell_ph_counts_cat = [np.empty((0,nstream),dtype=int) for i in range(nstate)]
    dwell_lengths_cat = [[np.empty(0,dtype=int) for i in range(nstate)] for k in range(nstate)]
    # switch from list of arrays to singular arrays
    for state in range(nstate):
        dwell_lengths_cat[state][state] = np.array([1],dtype=int)
    for burst in range(nburst):
        for state in range(nstate):
            dwell_ph_counts_cat[state] = np.concatenate((dwell_ph_counts_cat[state], dwell_ph_counts[burst][state]))
            for state2 in range(nstate):
                if state != state2:
                    dwell_lengths_cat[state][state2] = np.append(dwell_lengths_cat[state][state2], dwell_lengths[burst][state][state2])
    # clear up some memory
    del(dwell_lengths)
    del(dwell_ph_counts)
    # find total number of dwells, and total number of photons observed for each state
    dwell_num = [dwell_ph_counts_cat[state].shape[0] for state in range(nstate)]
    dwell_ph_counts_total = np.concatenate([dwell_ph_counts_cat[state].sum(axis=0).reshape(1,nstream) for state in range(nstate)])
    # calculate errors for channels in each state
    P_w = dwell_ph_counts_total / dwell_ph_counts_total.sum(axis=1).reshape((nstate,1))
    P_dwell = [dwell_ph_counts_cat[state] / dwell_ph_counts_cat[state].sum(axis=1).reshape((dwell_num[state],1)) for state in range(nstate)]
    P_w_sqdiff = [dwell_ph_counts_cat[state].sum(axis=1).reshape((dwell_num[state],1))*(P_dwell[state] - P_w[state,:])**2 for state in range(nstate)]
    P_w_err = np.concatenate([P_w_sqdiff[state].sum(axis=0).reshape(1,nstream)/dwell_ph_counts_total[state,:].sum()/np.sqrt(dwell_num[state]) for state in range(nstate)])
    # calculate E_w of each state from the photons identified with the viterbi algorithm
    E_filt = [dwell_ph_counts_cat[state][:,0:2].sum(axis=1) != 0 for state in range(nstate)]
    E_num = [E_filt[state].sum() for state in range(nstate)]
    E_w = dwell_ph_counts_total[:,1] / dwell_ph_counts_total[:,0:2].sum(axis=1)
    E_dwell = [dwell_ph_counts_cat[state][E_filt[state],1]/dwell_ph_counts_cat[state][E_filt[state],0:2].sum(axis=1) for state in range(nstate)]
    E_w_sqdiff = [dwell_ph_counts_cat[state][E_filt[state],0:2].sum(axis=1)*(E_dwell[state] - E_w[state])**2 for state in range(nstate)]
    E_w_err = np.array([E_w_sqdiff[state].sum()/dwell_ph_counts_total[state,0:2].sum()/np.sqrt(E_num[state]) for state in range(nstate)])
    # For H2MM using more that 2 photon streams
    if nstream >= 3:
        S_w = dwell_ph_counts_total[:,0:2].sum(axis=1) / dwell_ph_counts_total[:,0:3].sum(axis=1)
        S_dwell = [dwell_ph_counts_cat[state][:,0:2].sum(axis=1)/dwell_ph_counts_cat[state][:,0:3].sum(axis=1) for state in range(nstate)]
        S_w_sqdiff = [dwell_ph_counts_cat[state][:,0:3].sum(axis=1)*(S_dwell[state] - S_w[state])**2 for state in range(nstate)]
        S_w_err = np.array([S_w_sqdiff[state].sum()/dwell_ph_counts_total[state,0:3].sum()/np.sqrt(dwell_num[state]) for state in range(nstate)])
    # calculating error for dwell time
    dwell_av = np.array([[np.mean(dwell_lengths_cat[state1][state2]) for state2 in range(nstate)] for state1 in range(nstate)])
    for state in range(nstate):
        dwell_av[state,state] = np.array([1.0])
    dwell_err = [[((dwell_lengths_cat[state1][state2] - dwell_av[state1,state2])**2).sum()/(dwell_lengths_cat[state1][state2].size**(3/2)) for state2 in range(nstate)] for state1 in range(nstate)]
    t_av = 1/dwell_av
    t_err = dwell_err/(dwell_av**2)
    if nstream >= 3:
        return path, prob, P_w, P_w_err, t_av, t_err, E_w, E_w_err, S_w, S_w_err
    else:
        return path, prob, P_w, P_w_err, t_av, t_err, E_w, E_w_err