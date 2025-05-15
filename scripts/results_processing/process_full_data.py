from pathlib import Path
from catechol.data.featurizations import FEATURIZATIONS

import pandas as pd

OUTPUT_DIR = Path("results")
FULL_DATA_RESULTS_DIR = Path("results/full_data")
SINGLE_SOLVENT_RESULTS_DIR = Path("results/single_solvent")
TRANSFER_LEARNING_RESULTS_DIR = Path("results/transfer_learning")
ALL_MODELS = ["GPModel", "MLPModel", "LLMModel"]

REGRESSION_TABLE = [
    ("BaselineGPModel", "", ""),
    ("GPModel", "acs", "indep"),
    ("GPModel", "drfps", "indep"),
    ("GPModel", "fragprints", "indep"),
    ("GPModel", "spange", "indep"),
    ("MLPModel", "acs", ""),
    ("MLPModel", "drfps", ""),
    ("MLPModel", "fragprints", ""),
    ("MLPModel", "spange", ""),
    ("LLMModel", "smiles", "rxnfp"),
    ("LLMModel", "smiles", "chemberta"),
    ("NODEModel", "spange", ""),
    ("EODEModel", "spange", ""),
    ("LODEModel", "spange", ""),
]
EXTENSIONS_TABLE = [
    ("BaselineGPModel", "", ""),
    ("GPModel", "spange", "indep"),
    ("GPModel", "spange", "indep-learnmean"),
    ("GPModel", "spange", "indep-separated"),
    ("GPModel", "spange", "multi"),
    ("GPModel", "spange", "indep-warp"),
]


def parse_model_filename(model_str: str) -> dict[str, str]:
    """Parse the saved name of the results from a model."""
    splits = model_str.split("-")
    if len(splits) == 1:
        splits = [*splits, "", ""]
    elif len(splits) == 2 and splits[-1] == "transfer":
        # special case for when there are no featurizations, but the detail matters
        splits = [*splits, ""]

    model_name, *details, featurization = splits
    featurization = featurization.split("_")[0]
    return {
        "name": model_name,
        "details": "-".join(details),
        "featurization": featurization,
    }


def load_results(dir: Path):
    all_results_lst = []
    for result_path in dir.iterdir():
        model_idx = parse_model_filename(result_path.stem)
        result_df_full = pd.read_csv(result_path, index_col=0)
        has_acetonitrile = result_df_full.index.str.contains("Acetonitrile")
        result_df_full = result_df_full.loc[~has_acetonitrile]
        data = result_df_full[["mse", "nlpd"]].mean().to_numpy().reshape(1, 2)
        result_df = pd.DataFrame(
            data=data,
            index=pd.MultiIndex.from_tuples(
                [(model_idx["name"], model_idx["details"], model_idx["featurization"])],
                names=("Model", "Details", "Featurization"),
            ),
            columns=("MSE ($\downarrow$)", "NLPD ($\downarrow$)"),
        )

        all_results_lst.append(result_df)

    return pd.concat(all_results_lst)


def filter_and_sort_results(all_results: pd.DataFrame, normalize_nlpd: bool = True):
    def sorter(idx: pd.Index):
        sorted_order = {
            "Model": ["BaselineModel", "BaselineGPModel", "GPModel", "MLPModel", "LLMModel", "LODEModel", "EODEModel", "NODEModel"],
            "Featurization": [f.split("_")[0] for f in FEATURIZATIONS],
            "Details": [],
        }
        return idx.map({model: i for i, model in enumerate(sorted_order[idx.name])})
    
    # remove all of the warps
    warp_idcs = all_results.index.get_level_values("Details").str.contains("warp")
    spange_idcs = all_results.index.get_level_values("Featurization") == "spange"
    all_results = all_results[(~warp_idcs) | spange_idcs]

    # remove the old baseline models
    all_results = all_results.drop(("BaselineModel", "", ""))

    # normalize the NLPD
    if normalize_nlpd:
        # difficult indexing here to make sure that we only subtract from the NLPD
        baseline = all_results.loc["BaselineGPModel", "", ""]
        baseline_nlpd = baseline[baseline.index.get_level_values("Metric").str.contains("NLPD")]
        idx = pd.IndexSlice
        all_results.loc[idx[:, :, :], idx[:, "NLPD ($\downarrow$)"]] -= baseline_nlpd
        # all_results = all_results.drop(("BaselineGPModel", "", ""))

        # rename NLPD columns to reflect that they have been normalized
        levels = all_results.columns.get_level_values("Metric")[:2]
        new_levels = levels.str.replace("NLPD", "SNLPD")
        all_results.columns = all_results.columns.set_levels(new_levels, level=1)

    return all_results.reorder_levels(
        ["Model", "Featurization", "Details"]
    ).sort_index(key=sorter, level=[0, 2])


def get_latex_table(all_results: pd.DataFrame):
    all_results = all_results.fillna("-")
    # remove the labels for the header
    all_results.columns.names = [None, None]
    styler = all_results.style.format(precision=3)
    return styler.to_latex(
        hrules=True,
        multicol_align="c",
        multirow_align="t"
    )


if __name__ == "__main__":
    # Table: regression
    full_data_results = load_results(FULL_DATA_RESULTS_DIR)
    single_solvent_results = load_results(SINGLE_SOLVENT_RESULTS_DIR)

    all_results = pd.concat(
        {
            "Full data": full_data_results,
            "Single solvent": single_solvent_results,
        },
        axis="columns",
        names=["Dataset", "Metric"]
    )
    all_results = filter_and_sort_results(all_results, normalize_nlpd=False)
    with open(OUTPUT_DIR / "regression.tex", "w") as f:
        f.write(get_latex_table(all_results.loc[REGRESSION_TABLE]))

    with open(OUTPUT_DIR / "extensions.tex", "w") as f:
        f.write(get_latex_table(all_results.loc[EXTENSIONS_TABLE]))

    # Table: transfer learning
    tl_results = load_results(TRANSFER_LEARNING_RESULTS_DIR)
    transfer_idcs = tl_results.index.get_level_values("Details").str.contains("transfer")
    clean_transfer_fn = lambda s: s.replace("-transfer", "").replace("transfer", "")
    tl_results = pd.concat(
        {
            "Catechol": tl_results.loc[~transfer_idcs],
            "Catechol + Claisen": tl_results.loc[transfer_idcs].rename(index=clean_transfer_fn),
        },
        axis="columns",
        names=["Dataset", "Metric"]
    )
    tl_results = filter_and_sort_results(tl_results, normalize_nlpd=False)
    with open(OUTPUT_DIR / "transfer_learning.tex", "w") as f:
        f.write(get_latex_table(tl_results))
