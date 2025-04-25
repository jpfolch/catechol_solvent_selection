import pandas as pd
from catechol.data.featurizations import (
    featurize_input_df,
)

# load the data
path = "data/full_data/catechol_full_data_yields.csv"
df = pd.read_csv(path).iloc[:1000:100]

# replace solvent names with featurization
featurized_df = featurize_input_df(df, "acs_pca_descriptors")
print(featurized_df)

# you can also remove any features that are constant across all solvents
featurized_df = featurize_input_df(df, "fragprints", remove_constant=True)
print(featurized_df)
