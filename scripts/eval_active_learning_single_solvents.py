import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.featurizations import FeaturizationType
from catechol.data.loader import (
    generate_active_learning_train_test_split,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import get_model
from catechol.script_utils import StoreDict


def main(
    model_name: str, featurization: FeaturizationType, kwargs, init_set_size: int = 3
):
    model = get_model(model_name=model_name, featurization=featurization, **kwargs)
    X, Y = load_single_solvent_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_SINGLE_SOLVENT + model.extra_input_columns]

    results = pd.DataFrame(
        columns=["Number of solvents", "mse", "nlpd", "Solvent chosen"]
    )
    out_dir = Path("results/active_learning/")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.get_model_name()

    # get the solvent list
    solvent_list = X["SOLVENT NAME"].unique()
    # random initial sample
    initial_solvents = np.random.choice(solvent_list, size=init_set_size, replace=False)
    solvents_to_train = [solvent for solvent in initial_solvents]

    iteration = 0
    num_of_solvents = 0
    for solvent in initial_solvents:
        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of solvents": num_of_solvents,
                "mse": None,
                "nlpd": None,
                "Solvent chosen": solvent,
            },
            index=[num_of_solvents],
        )
        results = pd.concat((results, result))
        num_of_solvents += 1

    while len(solvents_to_train) < len(solvent_list):
        iteration += 1
        (
            (train_X, train_Y),
            (test_X, test_Y),
        ) = generate_active_learning_train_test_split(
            X, Y, solvents_to_train, solvent_list
        )
        model.train(train_X, train_Y)

        test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
        predictions = model.predict(test_X)

        # calculate some metrics
        mse = metrics.mse(predictions, test_Y)
        nlpd = metrics.nlpd(predictions, test_Y)

        # select the next solvent
        next_solvent = model.select_next_solvent(solvents_to_train, solvent_list, X)
        solvents_to_train.append(next_solvent)

        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of solvents": len(solvents_to_train),
                "mse": mse,
                "nlpd": nlpd,
                "Solvent chosen": next_solvent,
            },
            index=[num_of_solvents],
        )
        results = pd.concat((results, result))

        # store the results as you go
        results.to_csv(out_dir / f"{model_name}.csv", index=False)

        num_of_solvents += 1

    return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate active learning on the single solvents data.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_active_learning_single_solvents.py -m "GPModel" -f "drfps" -c multitask=True
            """
        ),
    )
    argparser.add_argument("-m", "--model", type=str)
    argparser.add_argument("-f", "--featurization", type=str)
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
    results = main(args.model, args.featurization, config)
