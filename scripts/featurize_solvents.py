import pandas as pd
from catechol.data.featurizations import (
    featurize_input_df,
    get_featurization_from_series,
)

# load the data
path = "data/full_data/catechol_full_data_yields.csv"
df = pd.read_csv(path).iloc[:1000:100]

feat = get_featurization_from_series(df["SOLVENT A NAME"], "acs_pca_descriptors")
print(feat)

featurized_df = featurize_input_df(df, "acs_pca_descriptors")
print(featurized_df)
