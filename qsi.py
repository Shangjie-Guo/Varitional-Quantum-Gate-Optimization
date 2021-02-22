#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:12:41 2021

@author: sjguo
"""

import pennylane as qml
import autograd.numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dev = qml.device('default.qubit', wires=2)

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

def random_state_ansatz(state_params, noise=False, improve=False): #TODO: phase of qubit 0 for no noise seems to have a distribution, why?
    '''
    Ansatz that generate random state with pre-randomized state_params. 
    state_params.shape = (8) range = [-np.pi, np.pi]
    noise = noise in noisy_CNot function or False. If False, apply true CNOT.
    improve = params in improving_CNot function or False.
    '''
    qml.RY(state_params[0], wires=0)
    qml.RZ(state_params[1], wires=0)
    qml.RY(state_params[2], wires=1)
    qml.RZ(state_params[3], wires=1)
    if type(noise) == bool:
        qml.CNOT(wires=[0,1])
    elif type(improve) == bool:
        noisy_CNot(noise)
    else:
        improving_CNot(improve, noise)
    qml.RY(state_params[4], wires=0)
    qml.RY(state_params[5], wires=1)
    qml.RZ(state_params[6], wires=0)
    qml.RZ(state_params[7], wires=1)
    if type(noise) == bool:
        qml.CNOT(wires=[0,1])
    elif type(improve) == bool:
        noisy_CNot(noise)
    else:
        improving_CNot(improve, noise)

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

def bias_test(noise=False):
    '''
    Test if the distribution is biased for random state ansatz.

    '''
    
    sample_size = 10000
    state_params = np.random.uniform(low=-np.pi, high=np.pi, size=(sample_size, 8))
    if type(noise) == bool:
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
    
def get_agi(params, noise, state_prep=True):
    '''
    Calculate the Average Gate Infidelity.
    params, noise = params, noise in improving_CNot function
    state_prep: If true, do both random state preparation with real CNOT gates.
                If False, do experimental state preparation with noisy_CNot.
                If parmas-like: do experimental state preparation with improving_CNot.

    '''
    if type(state_prep) == bool:
        if state_prep == True:
            @qml.qnode(dev)
            def CNot_test(state_params):
                random_state_ansatz(state_params)
                improving_CNot(params, noise)
                return qml.state()
        else:
            @qml.qnode(dev)
            def CNot_test(state_params):
                random_state_ansatz(state_params, noise=noise)
                improving_CNot(params, noise)
                return qml.state()
    else:
        @qml.qnode(dev)
        def CNot_test(state_params):
            random_state_ansatz(state_params, noise=noise, improve=state_prep)
            improving_CNot(params, noise)
            return qml.state()
    
    @qml.qnode(dev)
    def CNOT_true(state_params):
        random_state_ansatz(state_params)
        qml.CNOT(wires=[0,1])
        return qml.state()
    
    sample_size = 10
    state_params = np.random.uniform(low=-np.pi, high=np.pi, size=(sample_size, 8))
    #fidelities = np.zeros(sample_size)
    fidelities = []
    for i, p in enumerate(state_params):
        state_test = CNot_test(p)
        state_true = CNOT_true(p)
        fidelities.append(np.abs(np.dot(state_true.conj(),state_test))**2)
    return 1-np.mean(np.array(fidelities))

def vqgo(prep_params, noise):
    
    def cost_fn(params):
        return get_agi(params, noise, state_prep=prep_params)
    
    params = np.random.uniform(low=-np.pi, high=np.pi, size=(6,3))
    max_iterations = 1000
    conv_tol = 1e-6
    opt = qml.AdamOptimizer(stepsize=0.3)
    for n in range(max_iterations+1):
        params = opt.step(cost_fn, params)
        if n % 10 == 0:
            agi = cost_fn(params)
            print("VQGO: Iteration = {:}, AGI = {:.8f} ".format(n, agi))
            if agi < conv_tol:
                break
    return params
    
#%% test randomness

if __name__ == "__main__":
    noise = np.random.uniform(low=-np.pi/10, high=np.pi/10, size=(4,3))
    bias_test(noise)
    bias_test(False)
    print(get_agi(noise))


#%% VQGO (Assume perfect state preparation)

if __name__ == "__main__":
    noise = np.random.uniform(low=-np.pi/10, high=np.pi/10, size=(4,3))
    agi = vqgo(True, noise)


#%% QSI

if __name__ == "__main__":
    noise = np.random.uniform(low=-np.pi/1000, high=np.pi/1000, size=(4,3))
    params = np.zeros((6,3))
    it = 10
    agi = np.zeros((it+1))
    agi[0] = get_agi(params, noise)
    print("QSI: Iteration = {:}, AGI = {:.8f} ".format(0, agi[0]))
    for i in range(10):
        params = vqgo(params, noise)
        agi[i+1] = get_agi(params, noise)
        print("QSI: Iteration = {:}, AGI = {:.8f} ".format(i+1, agi[i+1]))
    
        