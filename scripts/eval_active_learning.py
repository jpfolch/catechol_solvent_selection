import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_ACTIVE_LEARNING
from catechol.data.featurizations import FeaturizationType
from catechol.data.loader import (
    generate_active_learning_train_test_split,
    load_solvent_ramp_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import get_model
from catechol.script_utils import StoreDict


def main(
    model_name: str,
    featurization: FeaturizationType,
    kwargs,
    init_set_size: int,
    seed: int,
    strategy: str,
):
    model = get_model(
        model_name=model_name,
        featurization=featurization,
        al_strategy=strategy,
        **kwargs,
    )
    X, Y = load_solvent_ramp_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_ACTIVE_LEARNING + model.extra_input_columns]

    results = pd.DataFrame(columns=["Number of ramps", "mse", "nlpd", "Ramp chosen"])
    model_name = model.get_model_name()
    out_dir = Path(f"results/active_learning/{model_name}/{strategy}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # get the ramp list
    ramp_list = X["RAMP NUM"].unique()
    # random initial sample
    rng = np.random.default_rng(seed)
    initial_ramps = rng.choice(ramp_list, size=init_set_size, replace=False)
    ramps_to_train = [ramp for ramp in initial_ramps]

    iteration = 0
    num_of_ramps = 0
    for ramp in ramps_to_train:
        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of ramps": num_of_ramps,
                "mse": None,
                "nlpd": None,
                "Ramp chosen": ramp,
            },
            index=[num_of_ramps],
        )
        results = pd.concat((results, result))
        num_of_ramps += 1

    while len(ramps_to_train) < int(len(ramp_list)):
        iteration += 1
        (
            (train_X, train_Y),
            (test_X, test_Y),
        ) = generate_active_learning_train_test_split(X, Y, ramps_to_train)
        model.train(train_X, train_Y)

        test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
        predictions = model.predict(test_X)

        # calculate some metrics
        mse = metrics.mse(predictions, test_Y)
        nlpd = metrics.nlpd(predictions, test_Y)

        # select the next ramp
        next_ramp = model.select_next_ramp(ramps_to_train, ramp_list, X)
        ramps_to_train.append(next_ramp)

        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of ramps": len(ramps_to_train),
                "mse": mse,
                "nlpd": nlpd,
                "Ramp chosen": next_ramp,
            },
            index=[num_of_ramps],
        )
        results = pd.concat((results, result))

        # store the results as you go
        results.to_csv(out_dir / f"{seed}.csv", index=False)

        num_of_ramps += 1

    return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate active learning on the full data_set.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_active_learning.py -m "GPModel" -f "drfps_catechol" -c multitask=True
            """
        ),
    )
    argparser.add_argument("-m", "--model", type=str)
    argparser.add_argument("-f", "--featurization", type=str)
    argparser.add_argument("-s", "--seed", type=int, default=239)
    argparser.add_argument("-i", "--initset", type=int, default=5)
    argparser.add_argument("-st", "--strategy", type=str, default="mutual_information")
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
    results = main(
        args.model, args.featurization, config, args.initset, args.seed, args.strategy
    )
