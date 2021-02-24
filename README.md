# Quantum-Self-Improvement
Quantum Self-Improvement Algorithm source code for QHack Open Hackathon 2021


---
## Implementation of VQGO

Variational Quantum Gate Optimization(VQGO) [[1]](#1) is a gate optimization method aiming to construct a target multi-qubit gate. It approximates the target gate by optimizing a parametrized quantum circuit which consists of tunable single-qubit gates with high fidelities and fixed multi-qubit gates with limited controlabilities.  

Our implementation of VQGO is included in ```openqsi.py``` and the same as in [[1]](#1), we choose CNOT gate as our target gate <em>U<sub>target</sub></em> since it is the only needed multi-qubit gate in the universal gate set.  

---
### Variational Quantum Circuit

<p align="center">
  <img alt="Variational Quantum Circuit in VQGO" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/parametrized_quantum_circuit.png" width="300">
  <br>
    <em>Figure 1. Variational Quantum Circuit in VQGO</em>
</p>

Figure 1 shows the design of the variational quantum circuit we are optimizing. <em>U<sub>source</sub></em> are source gates for the construction and <em>u<sub>ij</sub>(\theta)</em> are single-qubit gates the optimization is taking place.

Function ```noisy_CNot``` realizes <em>U<sub>source</sub></em>. Since in reality, we do not have perfect CNOT, then we can only construct CNOT based on imperfect CNOT with noise. So we choose noisy CNOT to be the source gate.

Function ```improving_CNot``` constructs the parametrized quantum circuit shown in Figure 1. We choose rotation operators to be those singe-qubit gates.

---
### Iterative Optimization Protocol

<p align="center">
  <img alt="Iterative Optimization Protocol in VQGO" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/iterative_optimization_protocol.png" width="300">
  <br>
    <em>Figure 2. Iterative Optimization Protocol in VQGO</em>
</p>


Figure 2 shows the iterative optimization protocol, which presents the core idea of  VQGO. <em>\rho<sub>in</sub></em> is supposed to be an uniformly sampled input state and the cost function <em>h(\theta)</em> measures average gate infidelity(AGI) of <em>U<sub>target</sub></em> and <em>U(\theta_l)</em>.

Function ```get_agi``` constructs such cost function. We choose the cost function to be <em>1-<O<sub>ideal</sub>|O<sub>exp</sub>></em>. The smaller the value is, the larger overlap between output states between the ideal and the experiment, which gives a better approximation of CNOT.

Function ```vqgo``` implements the iterative optimization protocol as shown in Figure 2. We choose AdamOptimizer to perform the optimization, which can also be replaced by others.

##### Side Note: Noisy State Preparation

Here, <em>\rho<sub>in</sub></em> is supposed to be an uniformly sampled input state so that its output state <em>O<sub>exp</sub></em> can be used to judge the performance of <em>U(\theta)</em>. The problem is that such state is also prepared using CNOT gate which introduces entanglement. In [[1]](#1), it seems that their prepared state are prepared by perfect CNOT. But in real experiment, as we do not have perfect CNOT, it is not realizable.

So in our implementation, we prepare <em>\rho<sub>in</sub></em> also using noisy CNOT, together with rotation gates to introduce randomness. ```random_state_ansatz``` shows such construction. ```bias_test``` is used to test the randomness of our input state prepared by noisy CNOT, and we have tested that they are very close to uniformly sampled state.

This also inspires us towards the idea of quantum self-improvement(QSI). In short, after implementing one round of VQGO, we obtain a better approximation of CNOT. Then what if we use this better 'CNOT' to prepare <em>\rho<sub>in</sub></em> and implement VQGO again? Details are explained in Section QSI.

---
### Experimental Result

An experimental result of VQGO is shown below.

<p align="center">
  <img alt="VQGO result" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/VQGO_result.jpg" width="300">
  <br>
    <em>Figure 3. AGI of Approximated CNOT</em>
</p>

The red star in Figure 3 represents the infidelity of a imperfect CNOT with random noise, while the starting point of the orange line represents the infidelity of the first approximated CNOT constructed from the imperfect noisy CNOT. The green star corresponds to the lowest point of the orange line, which represents the best approximated CNOT during the optimization.

It can be seen that the optimal approximated CNOT constructed from VQGO has infidelity ~10<sup>-4</sup>, decreased from ~10<sup>-3</sup> of a random noisy CNOT.


---
## Quantum self-improvement (QSI)

The design of quantum self-improvement is shown in Figure 4.

<p align="center">
  <img alt="QSI Design" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/QSI_result.jpg" width="300">
  <br>
    <em>Figure 4. Iterative VQGO Realizing Quantum Self-Inproment</em>
</p>

Given a noisy CNOT and a uniformly sampled quantum state <em>\rho<sub>in</sub></em> prepared with this noisy CNOT, VQGO outputs a better approximation of perfect CNOT. Then such VQGO is performed recursively with <em>\rho<sub>in</sub></em> in the experiment being prepared by better imperfect CNOT obtained from previous VQGO.

Still ```vqgo``` is used to implement this procedure with input <em>get_history</em> set to true to get previous VQGO's result.

### Preliminary Experimental Result

The preliminary experimental result of QSI is shown below, with stars and lines represent same thing as before.

<p align="center">
  <img alt="VQGO result" src="" width="300">
  <br>
    <em>Figure 3. AGI of Approximated CNOT</em>
</p>

We chooses to set the iteration of a single round VQGO to be 10. Detailedly, the QSI algorithm checks infidelity of current 'CNOT' after every 10 iterations, and if it performs better than previous one, QSI will replace it in the next run of VQGO. The number we choose should make sure both that the training procedure is enough to approximate a better CNOT and that the update of CNOT is frequent enough for a better prepared input state.

This result seems extremely good. The infidelity of our imperfect CNOT is decreased from ~10<sup>-1</sup> to  ~10<sup>-9</sup>, which indicates that we have obtained a very well approximated CNOT.

More experiments are needed to be performed to establish this.


---
## VQGO on REAL quantum computer


---
## QSI on REAL quantum computer

---
## References
<a id="1">[1]</a>
Heya, Kentaro, et al.
"Variational quantum gate optimization."
arXiv preprint arXiv:1810.12745 (2018).
