import pandas as pd
from catechol.data.featurizations import get_featurization
# load the data
path = 'data/full_data/catechol_full_data_yields.csv'
df = pd.read_csv(path)

feat = get_featurization(df["SOLVENT A NAME"], "acs_pca_descriptors")
print(feat)