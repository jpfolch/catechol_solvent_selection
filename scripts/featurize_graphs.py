import pandas as pd
from gauche.dataloader import MolPropLoader
import numpy as np
from catechol.models.Graph_GP.graph import graph
from rdkit.Chem import MolFromSmiles

# load the data
path = "data/single_solvent/catechol_single_solvent_yields.csv"
df = pd.read_csv(path)
atom_encoder = {
        "C": 0,
        "N": 1,
        "O": 2,
        "F": 3
    }
def featurize_to_graph(smiles):
    mol = MolFromSmiles(smiles)
    N = mol.GetNumAtoms()
    node_labels = {i: atom_encoder[mol.GetAtomWithIdx(i).GetSymbol()] for i in range(N)}
    F = np.zeros((N, len(atom_encoder)))
    for node, label in node_labels.items():
        F[node, label] = 1
    A = np.eye(N)
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        A[start_idx, end_idx] = 1
        A[end_idx, start_idx] = 1
    return graph(A=A, F=F)