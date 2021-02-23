#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:12:41 2021

@author: sjguo
"""

import pennylane as qml
import autograd.numpy as np
from time import time
import matplotlib.pyplot as plt

dev = qml.device('default.qubit', wires=2) # Define two-qubit circuit

######## USING SANDBOX SIMULATOR ########

# import remote_cirq

# API_KEY = "AIzaSyAFny1gXO9IE35nXKmLVTxLMPZysawkrJ0"
# sim = remote_cirq.RemoteSimulator(API_KEY)

# dev = qml.device("cirq.simulator",
#                   wires=2,
#                   simulator=sim,
#                   analytic=False)

##########################################

def noisy_CNot(noise):
    '''
    Noised CNOT gate (aka source gate), Assume noise is unknown.
    noise.shape = (4,3) range = [-noise level/2, noise level/2]

    '''
    qml.Rot(*noise[0],wires=0)
    qml.Rot(*noise[1],wires=1)
    qml.CNOT(wires=[0,1])
    qml.Rot(*noise[2],wires=0)
    qml.Rot(*noise[3],wires=1)

def random_state_ansatz(state_params, noise=None, improve=None): #TODO: For real quantum device, may need to use: arXiv:1608.00263
    '''
    Ansatz that generate random state with pre-randomized state_params. 
    state_params.shape = (8) range = [-np.pi, np.pi]
    noise = noise in noisy_CNot function or False. If False, apply true CNOT.
    improve = params in improving_CNot function or False.
    '''
    qml.Rot(*state_params[0], wires=0)
    qml.Rot(*state_params[1], wires=1)
    
    if noise is None:
        qml.CNOT(wires=[0,1])
    elif improve is None:
        noisy_CNot(noise)
    else:
        improving_CNot(improve, noise)

    qml.Rot(*state_params[2], wires=0)
    qml.Rot(*state_params[3], wires=1)


def improving_CNot(params, noise):
    '''
    Ansatz that try to improve noisy_CNot with varibles params. 
    params.shape = (6,3) range = [-np.pi, np.pi]
    noise = noise in noisy_CNot function

    '''
    qml.Rot(*params[0],wires=0)
    qml.Rot(*params[1],wires=1)
    noisy_CNot(noise)
    qml.Rot(*params[2],wires=0)
    qml.Rot(*params[3],wires=1)
    noisy_CNot(noise)
    qml.Rot(*params[4],wires=0)
    qml.Rot(*params[5],wires=1)

def bias_test(noise=None):
    '''
    Test if the distribution is biased for random state ansatz.

    '''
    
    sample_size = 10000
    state_params = np.random.uniform(low=-np.pi, high=np.pi, size=(sample_size, 4, 3))
    if noise is None:
        @qml.qnode(dev)
        def random_state(state_params, noise):
            random_state_ansatz(state_params)
            return qml.state()
    else:
        @qml.qnode(dev)
        def random_state(state_params, noise):
            random_state_ansatz(state_params, noise=noise)
            return qml.state()
    
    state_amp = []
    state_phase = []
    for p in state_params:
        state = np.array(random_state(p, noise))
        state_amp.append(np.abs(state)**2)
        state_phase.append(np.angle(state))
    
    state_amp = np.array(state_amp)
    state_phase = np.array(state_phase)
    # print(state_amp.mean(axis=0))
    # print(state_phase.mean(axis=0))
    
    fig, axs = plt.subplots(2, 4,figsize=(20,6))
    for i in range(4):
        axs[0,i].hist(state_amp[:,i], bins=20, range=(0,1))
        axs[1,i].hist(state_phase[:,i], bins=20, range=(-np.pi,np.pi))
        axs[0,i].set_ylim([0,sample_size/3])
        axs[1,i].set_ylim([0,sample_size/15])

def GramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q

def find_unitary(target_state):
    """
    Find a unitary that tranform the |00> state to target_state

    """
    A = np.identity(4, dtype=complex)
    a = np.argmax(target_state)
    A[:, a] = target_state
    A[:, [a, 0]] = A[:, [0, a]]
    return GramSchmidt(A)

def get_agi(params, noise, state_prep=True, alpha=0.0): #TODO: For real quantum device, may use: arXiv:1104.4695 
    '''
    Calculate the Average Gate Infidelity, with measurement on prepared state.
    params, noise = params, noise in improving_CNot function
    If params = None, Calculate AGI of noisy_CNot
    
    state_prep: If true, do both random state preparation with real CNOT gates.
                If False, do experimental state preparation with noisy_CNot.
                If parmas-like: do experimental state preparation with improving_CNot.
    alpha: Parameter related to number of shots measuring the prepared state. 
           Range in [0, 1), higher alpha requires more shots. alpha = 1 means infinite shots.
           Defined as: simulated_state = sqrt(1-alpha) * expected state + sqrt(alpha) * prepared_state
           If alpha = 0.0: Execute original get_agi described in VQGO paper.
    '''
    # CNOT True:
    if alpha == 0:
         @qml.qnode(dev)
         def CNOT_true(state_params):
             random_state_ansatz(state_params)
             qml.CNOT(wires=[0,1])
             return qml.state()
    else:
        @qml.qnode(dev)
        def state_prepared(state_params): 
            # Ramdom state preparation
            if type(state_prep) == bool:
                if state_prep == True: 
                    random_state_ansatz(state_params)
                else:
                    random_state_ansatz(state_params, noise=noise)
            else:
                random_state_ansatz(state_params, noise=noise, improve=state_prep)            
            # Measurements
            return qml.state()
        
        @qml.qnode(dev)
        def state_expected(state_params): 
            random_state_ansatz(state_params)   
            # Measurements
            return qml.state()
        
        @qml.qnode(dev)
        def CNOT_true_updated(U):
            qml.QubitUnitary(U,wires=[0,1])
            qml.CNOT(wires=[0,1])
            return qml.state()
    
    # CNot Test:
    @qml.qnode(dev)
    def CNot_test(state_params): 
        # Ramdom state preparation
        if type(state_prep) == bool:
            if state_prep == True: 
                random_state_ansatz(state_params)
            else:
                random_state_ansatz(state_params, noise=noise)
        else:
            random_state_ansatz(state_params, noise=noise, improve=state_prep)
        
        # Apply testing CNot
        if params is None: 
            noisy_CNot(noise)
        else:
            improving_CNot(params, noise)
            
        return qml.state()
    
    
    sample_size = 100
    state_params = np.random.uniform(low=-np.pi, high=np.pi, size=(sample_size, 4, 3))
    fidelities = []
    if alpha != 0:
        a1 = np.sqrt(1-alpha)
        a2 = np.sqrt(alpha)

    for p in state_params:
        if alpha == 0:
            state_true = CNOT_true(p)
        else:
            u = find_unitary(a1*state_expected(p) + a2*state_prepared(p))
            state_true = CNOT_true_updated(u)
            
        state_test = CNot_test(p)
        fidelities.append(np.abs(np.dot(state_true.conj(),state_test))**2)
        
    return 1 - np.mean(np.array(fidelities))

def vqgo(prep_params, noise, previous_params=None, get_history=False, alpha=0.0, start_time=None):
    '''
    VQGO algorithm from arXiv:1810.12745. With state preparation params for QSI.

    '''
    def cost_fn(params):
        return np.log(get_agi(params, noise, state_prep=prep_params, alpha=alpha))
    
    if previous_params is None:
        params = np.random.uniform(low=-np.pi, high=np.pi, size=(6,3))
    else:
        params = previous_params
    
    max_iterations = 150
    opt = qml.AdamOptimizer(stepsize=0.3)
    agi_list = []
    params_list = []
    if get_history:
        agi_true_list = []
        
    for n in range(max_iterations+1):
        params = opt.step(cost_fn, params)
        params_list.append(params)
        agi_list.append(cost_fn(params))
        if get_history:
            agi_true_list.append(get_agi(params,noise))
        if n % 10 == 0:
            if start_time is None:
                print("VQGO: Iteration = {:}, Log(AGI) = {:.8f} ".format(n, agi_list[-1]))
            else:
                print("VQGO: Iteration = {:}, AGI = {:.8f}, Exp_AGI = {:.8f}, Time = {:.0f} ".format(0, agi[0], agi_exp[0], time()-t))
    if get_history:
        return params_list[np.argmin(agi_list)], np.array(agi_true_list)
    else:
        return params_list[np.argmin(agi_list)]
    
def plot_result(VQGO_AGI, QSI_AGI):
    return 0
    
#%% Test random state preparation 

if __name__ == "__main__":
    noise = np.random.uniform(low=-np.pi/10, high=np.pi/10, size=(4,3))
    bias_test(noise)
    bias_test(False)
    print(get_agi(noise))


#%% VQGO (Assume perfect state preparation)

if __name__ == "__main__":
    noise = np.random.normal(loc=0.0, scale=0.01, size=(4,3))
    agi = vqgo(True, noise)


#%% VQGO with Noisy preparation (One iteration of QSI)

if __name__ == "__main__":
    noise = np.random.normal(loc=0.0, scale=0.1, size=(4,3))
    agi_noise = get_agi(None, noise)
    print("QSI: Iteration = {:}, AGI = {:.8f} ".format(0, agi_noise))
    params, agi_true_list = vqgo(False, noise, get_history=True, alpha=0.9)
    agi_improved = get_agi(params, noise)
    print("QSI: Iteration = {:}, AGI = {:.8f} ".format(1, agi_improved))
    plt.plot(agi_true_list)
    plt.yscale('log')
    plt.axhline(y=agi_noise, color='r', linestyle='-')
    plt.axhline(y=agi_improved, color='g', linestyle='-')
    
#%% QSI (Not tested)

if __name__ == "__main__":
    t = time()
    noise = np.random.normal(loc=0.0, scale=0.15, size=(4,3))
    iteration = 3
    
    agi = np.zeros((iteration+1)) # True AGI
    agi_exp = np.zeros((iteration+1)) # Experimental AGI 
    
    #AGIs of Noisy_CNot
    agi[0] = get_agi(None, noise) 
    agi_exp[0] = get_agi(None, noise, state_prep=False)
    print("QSI: Iteration = {:}, AGI = {:.8f}, Exp_AGI = {:.8f}, Time = {:.0f} ".format(0, agi[0], agi_exp[0], time()-t))
    
    # First Iteration: Use noisy_CNot to prepare
   
    prep_params = vqgo(False, noise)
    agi[1] = get_agi(prep_params, noise)
    agi_exp[1] = get_agi(prep_params, noise, state_prep=prep_params) 
    print("QSI: Iteration = {:}, True_AGI = {:.8f}, Exp_AGI = {:.8f}, Time = {:.0f} ".format(1, agi[1], agi_exp[1], time()-t))
    
    # Second Iteration and on: Use improved_CNot to prepare
    for i in range(iteration-1):
        prep_params = vqgo(prep_params, noise, previous_params=prep_params)
        agi[i+2] = get_agi(prep_params, noise)
        agi_exp[i+2] = get_agi(prep_params, noise, state_prep=prep_params) 
        print("QSI: Iteration = {:}, True_AGI = {:.8f}, Exp_AGI = {:.8f}, Time = {:.0f} ".format(i+2, agi[i+2], agi_exp[i+2], time()-t))
    
    plt.plot(agi)
    
        