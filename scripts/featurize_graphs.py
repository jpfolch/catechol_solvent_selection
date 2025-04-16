import pandas as pd
from gauche.dataloader import MolPropLoader

# load the data
path = 'data/full_data/catechol_full_data_yields.csv'
df = pd.read_csv(path)

def graph_represenation(solvent: str = 'A'):
    if solvent == 'A':
        smiles_column = 'SOLVENT A SMILES'
    elif solvent == 'B':
        smiles_column = 'SOLVENT B SMILES'
    else:
        raise ValueError('solvent must be A or B')
    # extract the solvent names
    loader = MolPropLoader()
    loader.read_csv(path, smiles_column=smiles_column, label_column='Product 2')
    loader.featurize('molecular_graphs')

    return loader.features

graphs_a = graph_represenation('A')
graphs_b = graph_represenation('B')
