from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("results")
FULL_DATA_RESULTS_DIR = Path("results/full_data")
SINGLE_SOLVENT_RESULTS_DIR = Path("results/single_solvent")
ALL_MODELS = ["GPModel", "MLPModel", "LLMModel"]


def parse_model_filename(model_str: str) -> dict[str, str]:
    """Parse the saved name of the results from a model."""
    model_name, *details, featurization = model_str.split("-")
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
        result_df_full = pd.read_csv(result_path)

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


def filter_and_sort_results(all_results: pd.DataFrame):
    def sorter(idx):
        return idx.map({model: i for i, model in enumerate(ALL_MODELS)})
    
    # remove all of the warps
    warp_idcs = all_results.index.get_level_values("Details").str.contains("warp")
    spange_idcs = all_results.index.get_level_values("Featurization") == "spange"
    all_results = all_results[(~warp_idcs) | spange_idcs]

    return all_results.reorder_levels(
        ["Model", "Featurization", "Details"]
    ).sort_index()


def get_latex_table(all_results: pd.DataFrame):
    all_results = all_results.fillna("-")
    styler = all_results.style.format(precision=3)
    return styler.to_latex(hrules=True)


if __name__ == "__main__":
    full_data_results = load_results(FULL_DATA_RESULTS_DIR)
    single_solvent_results = load_results(SINGLE_SOLVENT_RESULTS_DIR)

    all_results = pd.concat(
        {
            "Full data": full_data_results,
            "Single solvent": single_solvent_results,
        },
        axis="columns"
    )
    all_results = filter_and_sort_results(all_results)
    # print(all_results)
    with open(OUTPUT_DIR / "regression.tex", "w") as f:
        f.write(get_latex_table(all_results))

    # indep = all_results.index.get_level_values("Details").str.contains("indep")
    # print(indep)
