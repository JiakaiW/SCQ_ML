<!---
README.md, the decription of the whole project
-->

I decide to to proceed to the GNN model building now. To accomplish automated qubit discovery, we have two phases: ## 1. How to automatically evaluate the performance of a complex superconducting qubit? (There's no easy way of determining whether a qubit is good or bad, it depends on what levels to use as computational states, what is the gate protocol, gate time, measurement time, what is the gate error rate, is the error biased or support erasure conversion? Biased error is easy to correct, erasure conversion makes the effective error rates one magitude lower. We want there to be an automated way of proposing ways to utilize a superconducting circuit as a qubit, so the GCPN agent in phase 2's outputs can be evaluated.) ## 2. Train a GCPN using the "circuit user trained in phase 1 as the environment", and let the GCPN propose good superconducting qubits and test them in fab with the gate protocols proposed by the first agent in phase 1. 

For the first phase, I want to divide it into two parts: ### a. We should create a dataset and a benchmark system, and let other people create RL policies to output good gate protocols that achieve good scores on a wide range of superconducting circuits.
### b. We use the best policy to perform high-throughput in silico qubit screening in phase 2.

Does it make sense? How do people provide benchmarks? How do people host the datasets?

The purpose is to use ML to understand and predict properties of complex superconducting circuits, with the hope that the resulting ML model can help discover more performant superconducting qubits or more interesting superconducting qubits. Some interesting properties are: 
1. Selection rule as in natrual qubits. Fluxonium has some pseudo-selection rule that originates from its wavefunction overlaps. It would be interesting to see if there are superconducting qubits that exhibit natural-atom like selection rules that enable optical pumping and other interesting protocols.
2. Multiple long-lived subspaces. the "omg" (optical, metastable, ground) structure of natural atoms like $^{171}Yb$ enables interesting gate protocols in trapped ion qubits. It would be interesting to see if we can create something like that with superconducting qubits.


# Two facets of the holy grail:
## 1. How to automatically evaluate the performance of a complex superconducting qubit?
### a. We should create a dataset and a benchmark, and let other people create RL policies that achieve good scores on random qubits.
### b. We use the best policy to perform high-throughput in silico qubit screening.
## 2. Use GCPN to generate good superconducting qubits and test them in fab

# Literature review
### Graph Neural Networks-based Parameter Design towards Large-Scale Superconducting Quantum Circuits for Crosstalk Mitigation (Nov 2024) 
It's about freuqency allocation of a processor, not about qubits.

### SQuADDS: A validated design database and simulation workflow for superconducting qubit design
It's about physical realization of qubits, like length of inductance, so not relevanbt


# Pipeline of analyzing a single superconducting circuit
1. Circuit decription as a graph
   ```Python
    zero_pi = scq.Circuit("""# zero-pi
    branches:
    - ["JJ", 1,2, EJ = 10, 20]
    - ["JJ", 3,4, EJ, 20]
    - ["L", 2,3, 0.008]
    - ["L", 4,1, 0.008]
    - ["C", 1,3, 0.02]
    - ["C", 2,4, 0.02]
    """,from_file=False)
   ```
2. Symmbolic Lagrangian, Symmbolic Hamiltonian with transformed variables, Symmbolic multi-dimensional Potential
   ```Python
   zero_pi.sym_lagrangian(return_expr=True,vars_type="new")
   # (25.0*\dot{θ_2}**2 + 0.00625*\dot{θ_3}**2 + 6.25625*\dot{θ_1}**2) + (-0.008*θ3**2 - 0.032*θ2**2 + EJ*cos(θ1 + θ3) + EJ*cos(θ1 + 1.0*(2πΦ_{1}) - 1.0*θ3))

   zero_pi.sym_hamiltonian(return_expr=True)
   #(40.0*Q3**2 + 0.03996*n1**2 + 0.03996*n_g1**2 + 0.01*Q2**2 + 0.07992*n1*n_g1) + (0.008*θ3**2 + 0.032*θ2**2 - EJ*cos(θ1 + θ3) - EJ*cos((2πΦ_{1}) + θ1 - 1.0*θ3))

   zero_pi.sym_potential(return_expr=True,print_latex=True)
    # 0.008 θ_{3}^{2} + 0.032 θ_{2}^{2} - EJ \cos{\left(θ_{1} + θ_{3} \right)} - EJ \cos{\left((2πΦ_{1}) + θ_{1} - 1.0 θ_{3} \right)}
   ```
3. Hierarchical diagonalization to ensure convergence in diagonalization with smaller truncation
   ```Python
   system_hierarchy = [[1,3], [2]] #explicit by grouping circuit variable indices in a nested list
   zero_pi.cutoff_n_1 = 15
   zero_pi.cutoff_ext_2 = 50
   zero_pi.cutoff_ext_3 = 100
   zero_pi.configure(system_hierarchy=system_hierarchy, subsystem_trunc_dims=[150, 30])
   ```
4. Evaluate the qubit performance: level structure, matrix elements, frequencies.
   ```Python
   zero_pi.eigenvals()
   zero_pi.plot_potential(θ1=np.linspace(-np.pi, np.pi),
                       θ3=np.linspace(-6*np.pi, 6*np.pi, 200),
                       θ2 = 0.) 
   zero_pi.plot_wavefunction(which=0, var_indices=(1,3));# scqubits can plot potential or wavefunctions on 2d grid, but maybe the ML model can consider the full dimension.
   for method_name in zero_pi.__dir__():
    if method_name.endswith("operator"):
        print(method_name)
   '''  # => return a bunch of sparse matrices
    sinθ1_operator
    cosθ1_operator
    n1_operator
    θ2_operator
    θ3_operator
    cosθ2_operator
    cosθ3_operator
    sinθ2_operator
    sinθ3_operator
    Q2_operator
    Q3_operator
    Qs2_operator
    Qs3_operator
    I_operator
   '''
   ```

# Developmentsin GNN after spring 2021
1. Graph Transformers like Graphormer with spatial encoding
2. Higher-order GNNs like k-GNN and spectral methods
3. Physics-informed models like HOPF nets for graphs with physics
4. Diffusion-based graph generation (e.g., Graph Diffuser, Diffusion-based modeling)
5. Enhanced GNN training with methods like GraphMAE and masking
6. 

# Specific NN architecture choice
### Graph Transformer
### Physics-informed NN
### Message-Passing GNN (e.g. MPNN, GatedGCN)
(GNN stack)

        └─→  Global readout → shared 256-d latent

                ├── Linear → ω predictions (MSE)

                ├── Linear → matrix-element tensor (MSE)

                └── Linear → selection-rule flag (BCE)

### Variational Autoencoder / Diffusion (generator)

# Project phases
1. Implement a way to describe the circuits, and use it to bridge ML model to the qubit properties like matrix element, eigenenergies.
2. Choose a model architecture
   1. Review of relavent projects:
   2. Pick one
3. Train ML model on small dataset (3-node circuits)
4. Evaluate ML model
5. Test the ML model
