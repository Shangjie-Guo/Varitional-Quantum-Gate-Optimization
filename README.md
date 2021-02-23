# Quantum-Self-Improvement
Quantum Self-Improvement Algorithm source code for QHack Open Hackathon 2021


---
## Implementation of VQGO

Variational Quantum Gate Optimization(VQGO) [[1]](#1) is a gate optimization method aiming to construct a target multi-qubit gate. It approximates the target gate by optimizing a parametrized quantum circuit which consists of tunable single-qubit gates with high fidelities and fixed multi-qubit gates with limited controlabilities.  

Our implementation of VQGO is included in ```openqsi.py``` and the same as in [[1]](#1), we choose CNOT gate as our target gate <em>U<sub>target</sub></em> since it is the only needed multi-qubit gate in the universal gate set.  

---
### Parametrized Quantum Circuit

![Figure 1. Parametrized Quantum Circuit in VQGO](https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration%20figures/parametrized_quantum_circuit.png)

Figure 1 shows the design of the parametrized quantum circuit we are optimizing. <em>U<sub>source</sub></em> are source gates for the construction and <em>u<sub>ij</sub>(\theta)</em> are single-qubit gates the optimization is taking place.

Function ```noisy_CNot``` realizes <em>U<sub>source</sub></em>. Since in reality, we do not have perfect CNOT, then we can only construct CNOT based on imperfect CNOT with noise. So we choose noisy CNOT to be the source gate.

Function ```improving_CNot``` constructs the parametrized quantum circuit shown in Figure 1. We choose rotation operators to be those singe-qubit gates.

---
### Iterative Optimization Protocol

![Figure 2. Iterative Optimization Protocol in VQGO](https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration%20figures/iterative%20optimization%20protocol.png)

Figure 2 shows the iterative optimization protocol, which presents the core idea of  VQGO. <em>\rho<sub>in</sub></em> is supposed to be an uniformly sampled input state and the cost function <em>h(\theta)</em> measures average gate infidelity(AGI) of <em>U<sub>target</sub></em> and <em>U(\theta_l)</em>.

Function ```get_agi``` constructs such cost function. We choose the cost function to be <em>1-<O<sub>ideal</sub>|O<sub>exp</sub>></em>. The smaller the value is, the larger overlap between output states between the ideal and the experiment, which gives a better approximation of CNOT.

Function ```vqgo``` implements the iterative optimization protocol as shown in Figure 2. We choose AdamOptimizer to perform the optimization, which can also be replaced by others.

##### Side Note: Noisy State Preparation

Here, <em>\rho<sub>in</sub></em> is supposed to be an uniformly sampled input state so that its output state <em>O<sub>exp</sub></em> can be used to judge the performance of <em>U(\theta)</em>. The problem is that such state is also prepared using CNOT gate which introduces entanglement. In [[1]](#1), it seems that their prepared state are prepared by perfect CNOT. But in real experiment, as we do not have perfect CNOT, it is not realizable.

So in our implementation, we prepare <em>\rho<sub>in</sub></em> also using noisy CNOT, together with rotation gates to introduce randomness. ```random_state_ansatz``` shows such construction. ```bias_test``` is used to test the randomness of our input state prepared by noisy CNOT, and we have tested that they are very close to uniformly sampled state.

This also inspires us towards the idea of quantum self-improvement(QSI). In short, after implementing one round of VQGO, we obtain a better approximation of CNOT. Then what if we use this better 'CNOT' to prepare <em>\rho<sub>in</sub></em> and implement VQGO again? Details are explained in Section QSI.

---
### Experiment Result



---
## Quantum self-improvement (QSI)


---
## VQGO on REAL quantum computer


---
## QSI on REAL quantum computer

---
### References
<a id="1">[1]</a>
Heya, Kentaro, et al.
"Variational quantum gate optimization."
arXiv preprint arXiv:1810.12745 (2018).
