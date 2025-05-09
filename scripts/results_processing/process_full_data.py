import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results/full_data")
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

def load_results():
    all_results_lst = []
    for result_path in RESULTS_DIR.iterdir():
        model_idx = parse_model_filename(result_path.stem)
        result_df_full = pd.read_csv(result_path)

        data = result_df_full[["mse", "nlpd"]].mean().to_numpy().reshape(1, 2)
        result_df = pd.DataFrame(
            data=data,
            index=pd.MultiIndex.from_tuples(
                [(model_idx["name"], model_idx["details"], model_idx["featurization"])],
                names=("Model", "Details", "Featurization"),
            ), 
            columns=("MSE ($\downarrow$)", "NLPD ($\downarrow$)")
        )

        all_results_lst.append(result_df)

    return pd.concat(all_results_lst)

def sort_results(all_results: pd.DataFrame):
    def sorter(idx):
        return idx.map({model: i for i, model in enumerate(ALL_MODELS)})
    return all_results.reorder_levels(["Model", "Featurization", "Details"]).sort_index()

def get_latex_table(all_results: pd.DataFrame):
    styler = all_results.style.format(precision=3)
    return styler.to_latex(
        hrules=True
    )

if __name__ == "__main__":
    all_results = load_results()
    all_results = sort_results(all_results)
    print(get_latex_table(all_results))

    # indep = all_results.index.get_level_values("Details").str.contains("indep")
    # print(indep)

