import argparse
import textwrap

import pandas as pd
from pathlib import Path
import tqdm
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_GRAPH_GP, TARGET_LABELS
from catechol.data.featurizations import FeaturizationType
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import get_model
from catechol.script_utils import StoreDict
from catechol.models import GraphGPModel

def main(model_name: str, kernel, kwargs, split = 0):
    
    assert kernel in ["BoGrape_SP", "BoGrape_ESP", "RW", "SW", "WL"], f"Kernel {kernel} not implemented"

    model = GraphGPModel(target_labels=TARGET_LABELS, kernel_name=kernel)  
    
    X, Y = load_single_solvent_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_GRAPH_GP + ['SOLVENT NAME']]

    results = pd.DataFrame(columns=["Test solvent", "mse", "nlpd"])
    out_dir = Path(f"results/single_solvent/{model_name}_{kernel}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.get_model_name()

    # this will generate all of the possible leave-one-out splits of the dataset
    split_generator = generate_leave_one_out_splits(X, Y)


    all_splits = list(split_generator)
    (train_X, train_Y), (test_X, test_Y) = all_splits[split]

    model.train(train_X[INPUT_LABELS_GRAPH_GP], train_Y)

    test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
    predictions = model.predict(test_X[INPUT_LABELS_GRAPH_GP])

    # calculate some metrics
    mse = metrics.mse(predictions, test_Y)
    nlpd = metrics.nlpd(predictions, test_Y)
    test_solvent = test_X.iloc[0]["SOLVENT NAME"]

    result = pd.DataFrame(
        {"Test solvent": test_solvent, "mse": mse, "nlpd": nlpd}, index=[split]
    )
    results = pd.concat((results, result))

    # store the results as you go
    # make the directory if it doesn't exist
    results.to_csv(out_dir / f"split_{split}.csv", index=False)

    return results

main('GraphGPModel', 'BoGrape_SP', {}, 0)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate a model on the single solvent dataset.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_single_solvents.py -m "GPModel" -f "drfps" -c multitask=True
            """
        ),
    )
    argparser.add_argument("-m", "--model", type=str)
    argparser.add_argument("-k", "--kernel", type=str)
    argparser.add_argument("-s", "--split", type=int, default=0)
    argparser.add_argument(
        "-c",
        "--config",
        action=StoreDict,
        nargs="+",
        help="Store kwargs-style dictionary to support arbitrary config to models.",
    )

    args = argparser.parse_args()
    # if no config is passed, create an empty dictionary
    config = args.config or {}
    results = main(args.model, args.kernel, config, args.split)
