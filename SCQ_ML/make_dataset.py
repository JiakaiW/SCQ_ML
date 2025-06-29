# make_dataset.py  ────────────────────────────────────────────────────────────
import itertools, random, pathlib, multiprocessing as mp, argparse
import shutil
from uuid import uuid4
import math
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import scqubits as scq
from tqdm import tqdm

from utils import write_chunk

# --------------------------- PyTorch Multiprocessing -------------------------
torch.multiprocessing.set_sharing_strategy('file_system')

# --------------------------- global settings ---------------------------------
RNG_SEED = 42
# This will now be set by the command-line argument
# NUM_SAMPLES_PER_TOPOLOGY = 10000 
COMP_RANGES = dict( EJ=(  1.0, 10.0),
                    EC=(0.01,  10.0),
                    EL=(0.01,  10.0) )
ONE_HOT = dict(J=[1,0,0], C=[0,1,0], L=[0,0,1])

# ─────────────────────────── your helper ────────────────────────────────────
def get_yaml_for_one_circuit_edge(i: int, j: int,
                                  EJ: float = 0, EC: float = 0, EL: float = 0):
    """
    Returns *list* of YAML lines (one per branch).
    """
    branch_lines = []
    if EJ != 0:
        if EC != 0:                          # JJ with explicit EJC
            branch_lines.append(
                f'- ["JJ", {i},{j},EJ={EJ}GHz,EJC={EC}GHz]'
            )
        else:                                # JJ with default EJC
            EJC = 100                        # internal capacitance (GHz)
            branch_lines.append(
                f'- ["JJ", {i},{j},EJ={EJ}GHz,EJC={EJC}GHz]'
            )
    elif EC != 0:                            # standalone capacitor
        branch_lines.append(f'- ["C", {i},{j},EC={EC}GHz]')

    if EL != 0:                              # (possibly in parallel) inductor
        branch_lines.append(f'- ["L", {i},{j},EL={EL}GHz]')
    return branch_lines
# ─────────────────────────────────────────────────────────────────────────────

# ----------------------------- helpers ---------------------------------------
def rand_val(label):          # uniform sampling inside the specified range
    lo, hi = COMP_RANGES[label]
    return random.uniform(lo, hi)

def build_sc_circuit(topology: str, params: dict) -> scq.Circuit:
    """
    topology string e.g. 'JL', params names like 'J0', 'L1', ...
    Uses get_yaml_for_one_circuit_edge(1,2, ...)
    """
    yaml_lines = []
    for idx, kind in enumerate(topology):
        if kind == 'J':
            yaml_lines += get_yaml_for_one_circuit_edge(
                1, 2, EJ=params[f'J{idx}'] )
        elif kind == 'C':
            yaml_lines += get_yaml_for_one_circuit_edge(
                1, 2, EC=params[f'C{idx}'] )
        elif kind == 'L':
            yaml_lines += get_yaml_for_one_circuit_edge(
                1, 2, EL=params[f'L{idx}'] )
    circ_yaml = "branches:\n" + "\n".join(yaml_lines)
    return scq.Circuit(circ_yaml, from_file=False)

def pyg_graph(topology: str, params: dict, w01: float, w12: float) -> Data:
    n_circ = 2
    elem_feats = [ONE_HOT[k] + [params[f'{k}{i}']]
                  for i, k in enumerate(topology)]
    x = torch.tensor([[0,0,0,0]]*n_circ + elem_feats, dtype=torch.float)

    edges = []
    for i in range(len(topology)):
        e_idx = n_circ + i
        for c in (0, 1):
            edges += [[e_idx, c], [c, e_idx]]
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    y = torch.tensor([[math.log(w01), math.log(w12)]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

def make_sample(topology: str) -> Data:
    p = {f'{k}{i}': rand_val(f'E{k}') for i, k in enumerate(topology)}
    circ = build_sc_circuit(topology, p)
    evals = circ.eigenvals();  evals -= evals[0]
    w01, w12 = float(evals[1]), float(evals[2]-evals[1])
    return pyg_graph(topology, p, w01, w12)
    
# --------------------------- dataset class -----------------------------------
class CircuitDataset(InMemoryDataset):
    topologies = ['J', 'C', 'L', 'JC', 'JL', 'CL', 'JCL']

    def __init__(self, root, num_samples):
        self.num_samples = num_samples
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): 
        # Filename is now dynamic based on num_samples
        return [f'circuit_graphs_n{self.num_samples}_v8.pt']
    
    def download(self): pass

    def process(self):
        # Use a unique temporary directory for each processing run to avoid conflicts
        tmp_dir = self.root / f"tmp_{uuid4()}"

        if tmp_dir.exists(): # Should not happen with uuid, but good practice
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True)

        N_WORKERS = 6
        BATCH_SIZE = 1000
        wanted_topologies = [t for t in self.topologies if "J" in t]
        
        chunk_count = 0
        with mp.Pool(N_WORKERS) as pool:
            for topo in wanted_topologies:
                print(f"Generating {self.num_samples} samples for topology: {topo}")
                buffer = []
                
                tasks = [topo] * self.num_samples
                
                for sample in tqdm(pool.imap_unordered(make_sample, tasks), total=len(tasks)):
                    buffer.append(sample)
                    if len(buffer) == BATCH_SIZE:
                        chunk_path = tmp_dir / f"chunk_{chunk_count}.pt"
                        write_chunk(buffer, chunk_path)
                        buffer.clear()
                        chunk_count += 1
                
                if buffer: # Write any remaining samples
                    chunk_path = tmp_dir / f"chunk_{chunk_count}.pt"
                    write_chunk(buffer, chunk_path)
                    chunk_count += 1

        print("Collating sample chunks...")
        data_list = []
        for chunk_file in tqdm(list(tmp_dir.glob("*.pt"))):
            # Each chunk file is now just a list of Data objects
            data_chunk = torch.load(chunk_file, weights_only=False)
            data_list.extend(data_chunk)
        
        shutil.rmtree(tmp_dir)

        print(f"Total samples generated: {len(data_list)}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# --------------------------- run once ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate circuit graph dataset.')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate per topology.')
    args = parser.parse_args()

    random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)
    
    ds_root = pathlib.Path("gnn_circuit_data")
    # We pass the num_samples to the constructor now
    ds = CircuitDataset(root=ds_root, num_samples=args.num_samples)
    print(ds, "\nExample:", ds[0])
