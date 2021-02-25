# Improved Variational Quantum Gate Optimization
Improved Variational Quantum Gate Optimization Algorithm source code for QHack Open Hackathon 2021

In this repo, we implement the Variational Quantum Gate Optimization (VQGO) algorithm described in [[1]](#1). In addition, we improved VQGO and make it implmentable to a noisy quantum computer (NQC).

The task of VQGO is to optimize a variational quantum gate with source gate <em>U<sub>S</sub></em> and paramters <em>**θ**</em>, with cost function <em>||U(U<sub>S</sub>, **θ**) - U<sub>T</sub>||</em>, where <em>U<sub>T</sub></em> is target gate that is previously unachievable with experiment, and <em>U<sub>S</sub></em> is the best experimental proxy of <em>U<sub>T</sub></em>. Practically, <em>U</em> is usually a multi qubit gate.

The original VQGO describes as below:
1. Start with random intitial parameter <em>**θ**</em>;
2. Prepare a ramdom sample <em>|i></em>;
3. Simulate <em>U<sub>T</sub>|i></em>;
4. Measure fidelity  <em>F(|i>, **θ**) = |<i|U<sup>†</sup>(U<sub>S</sub>, **θ**) U<sub>T</sub>|i>|<sup>2</sup></em>;
5. Do step 2-4 with <em>N</em> random samples  <em>{|i>}</em>, calculate average gate infidelity <em>AGI(**θ**) = 1 - [Σ<sub>{|i>}</sub> F(|i>, **θ**)]/N </em>;
6. Optimize <em>**θ**</em> with cost function <em>AGI(**θ**)</em>.

We found with pennylane, we can easily implement VQGO.

<p align="center">
  <img alt="VQGO" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/VQGO.png" width="400">
</p>

---
### Problem with VQGO
The problem of VQGO is in step 2, it is impoosible to random state sample without fiducial multi qubit gate. To show that, we introduce ansatz <em>A(U, ϕ)</em>, s.t. if <em>ϕ</em> is a random sample in real space, then <em>|i> = A(U<sub>T</sub>, ϕ)|0></em> is also a random sample to Haar measure, with fiducial state <em>|0></em>. We show that if change step 2 to:

2'. Prepare a ramdom sample <em>A(U<sub>S</sub>, ϕ)|0></em>;

then the VQGO hits noise floor after few iterations.
<p align="center">
  <img alt="VQGO_noise" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/VQGO_noise.png" width="400">
</p>

---
### Our improvement to VQGO
Here we provide our solution to the problem above, The improved VQGO describes as below:
1. Start with random intitial parameter <em>**θ**</em>;
2. Prepare a sample <em>A(U<sub>S</sub>, ϕ)|0></em> with random <em>ϕ</em>, make a state estimation with few shots <em>|est> ≃ A(U<sub>S</sub>, ϕ)|0> + 𝛿|dψ></em>;
3. Simulate the reference state with estimation: <em>|sim> = (1-α)<sup>1/2</sup>A(U<sub>T</sub>, ϕ)|0> + α<sup>1/2</sup>|est></em>;
4. Measure fidelity <em>F(ϕ, **θ**) = |<0|A<sup>†</sup>(U<sub>S</sub>, ϕ)U<sup>†</sup>(U<sub>S</sub>, **θ**) U<sub>T</sub>|sim>|<sup>2</sup></em>;
5. Do step 2-4 with <em>N</em> random samples  <em>{ϕ}</em>, calculate <em>AGI(**θ**) = 1 - [Σ<sub>{ϕ}</sub> F(ϕ, **θ**)]/N </em>;
6. Optimize <em>**θ**</em> with cost function <em>AGI(**θ**)</em>;
7. If optimized<em>U(U<sub>S</sub>, **θ**')</em> is better<sup>*</sup> than previous <em>U<sub>S</sub></em>, then update <em>A(U<sub>S</sub>, ϕ)</em> to <em>A(U(U<sub>S</sub>, **θ**'), ϕ)</em>.

where <em>𝛿|dψ></em> describes the uncertainty of the estimation, and paramter <em>α</em> describes the confidence to the estimated state. We show that our improved algorithm successfully optimizes AGI even with noisy state preparation.

<p align="center">
  <img alt="VQGO_improve" src="https://github.com/Shangjie-Guo/Quantum-Self-Improvement/blob/main/illustration_figures/VQGO_improve.png" width="400">
</p>


<sup>*</sup> Better here is experimentally testable, defined as: 

<em>Test_AGI[U(U<sub>S</sub>, **θ**')] < Test_AGI[U<sub>S</sub>]</em>,

where <em>Test_AGI[U] = 1-[Σ<sub>{ϕ}</sub>|<0|A<sup>†</sup>(U, ϕ)U<sup>†</sup> U<sub>T</sub>A(U<sub>T</sub>, ϕ)|0>|<sup>2</sup>]/N}</em>

## References
<a id="1">[1]</a>
Heya, Kentaro, et al.
"Variational quantum gate optimization."
arXiv preprint arXiv:1810.12745 (2018).
