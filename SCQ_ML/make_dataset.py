# ──────────────────────────────────────────────────────────────────────────────
# Generate a supervised dataset for GNN-based property prediction of
# two-node superconducting circuits           (c) 2025  - MIT   Licence
# ──────────────────────────────────────────────────────────────────────────────
import itertools, random, json, math, pathlib, multiprocessing as mp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import scqubits as scq               # ≥1.4.0

# ----------------------------- hyper-parameters ------------------------------ #
NUM_SAMPLES_PER_TOPOLOGY = 400       # → total 7×400 = 2800 graphs
N_JOBS                = 1  # parallel scqubits runs
RNG_SEED              = 42
torch.manual_seed(RNG_SEED);  random.seed(RNG_SEED);  np.random.seed(RNG_SEED)

# Component ranges  (GHz units for EJ, EC, EL → SC-Qubits default)
RANGES = dict(
    EJ=(  5.0,  30.0),   # Josephson energy
    EC=(0.05,   3.0),    # Capacitive energy  (E_C = e²/2C; pick broad span)
    EL=(0.01,   1.0),    # Inductive energy   (E_L = Φ₀²/2L)
)

ONE_HOT = dict(J=[1,0,0], C=[0,1,0], L=[0,0,1])

# ---------------------------------------------------------------------------- #
def sample_component(comp_type: str) -> float:
    low, high = RANGES[f'E{comp_type}']
    return random.uniform(low, high)

def build_scqubits_circuit(topology: str, params: dict) -> scq.Circuit:
    """
    topology = string like 'JL' etc.
    params   = dict {'J0':val, 'L0':val, 'L1':val, ...}
    """
    branch_lines = []
    # All elements connect node 1 ↔ 2 (SC-Qubits counts from 1)
    for k, comp in enumerate(topology):
        par_val = params[f'{comp}{k}']
        if   comp == 'J':
            branch_lines.append(f"- [JJ, 1,2,EJ,{par_val}GHz]")
        elif comp == 'C':
            branch_lines.append(f"- [C,  1,2,EC,{par_val}GHz]")
        elif comp == 'L':
            branch_lines.append(f"- [L,  1,2,EL,{par_val}GHz]")
    yaml_descr = "branches:\n" + "\n".join(branch_lines)
    print(yaml_descr)
    print("end yaml")
    return scq.Circuit(yaml_descr, from_file=False)

def graph_from_topology(topology: str, params: dict,
                        targets: tuple[float,float]) -> Data:
    """
    Returns PyG Data object:
      node feature dim = 5  (3-way one-hot + 2 values) for element nodes,
                           5 zeros for each of the two circuit nodes.
                           For C, L components, the second value is 0.
    """
    # Node-ordering: [cir_node0, cir_node1, elem0, elem1, ...]
    num_circuit_nodes = 2
    elem_features = []
    for i, c in enumerate(topology):
        one_hot_vec = ONE_HOT[c]
        par_val = params[f'{c}{i}']
        if c == 'J':
            ej, ecj = par_val
            elem_features.append(one_hot_vec + [ej, ecj])
        else:
            elem_features.append(one_hot_vec + [par_val, 0.0])

    circuit_features = [[0,0,0,0,0] for _ in range(num_circuit_nodes)]
    x = torch.tensor(circuit_features + elem_features, dtype=torch.float)

    # Edges: element_i ↔ circuit_0 and element_i ↔ circuit_1
    edge_index = []
    for i in range(len(topology)):
        elem_idx = num_circuit_nodes + i
        for cir in (0, 1):
            edge_index += [[elem_idx, cir], [cir, elem_idx]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor(targets, dtype=torch.float)  # (ω01, ω12)

    return Data(x=x, edge_index=edge_index, y=y)

# ---------------------------------------------------------------------------- #
def make_one_sample(topology: str):
    """Worker function for pool.map"""
    # (1) sample parameters
    params = {}
    for idx, element in enumerate(topology):
        params[f'{element}{idx}'] = sample_component(element)
    # (2) spectrum
    circ = build_scqubits_circuit(topology, params)
    evals = circ.eigenvals()
    evals -= evals[0]
    w01, w12 = float(evals[1]), float(evals[2]-evals[1])
    # (3) graph object
    data = graph_from_topology(topology, params, (w01, w12))
    return data

def generate_dataset():
    topo_list = ['J', 'C', 'L', 'JC', 'JL', 'CL', 'JCL']
    chunks = []
    for topo in topo_list:
        for _ in range(NUM_SAMPLES_PER_TOPOLOGY):
            print(f"Generating {topo}...")
            chunks += make_one_sample([topo])
            print(f"Done {topo}...")
    return chunks

# ---------------------------------------------------------------------------- #
class CircuitDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):   # one file is enough
        return ['circuit_graphs.pt']
    
    def download(self):               # nothing to download
        pass

    def process(self):
        data_list = generate_dataset()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    out_dir = pathlib.Path("./gnn_circuit_data")
    ds = CircuitDataset(root=out_dir)
    print(ds)
    print("Example graph:", ds[0])
