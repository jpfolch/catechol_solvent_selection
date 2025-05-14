import argparse
import textwrap
from pathlib import Path

import pandas as pd
import tqdm
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_FULL_DATA
from catechol.data.featurizations import FeaturizationType
from catechol.data.loader import (
    generate_leave_one_ramp_out_splits,
    load_claisen_data,
    load_solvent_ramp_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import get_model
from catechol.script_utils import StoreDict


def main(model_name: str, featurization: FeaturizationType, transfer: bool, kwargs):
    model = get_model(
        model_name=model_name,
        featurization=featurization,
        # transfer_learning=transfer,
        **kwargs,
    )
    X, Y = load_solvent_ramp_data()
    X_c, Y_c = load_claisen_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_FULL_DATA + model.extra_input_columns]
    X_c = X_c[INPUT_LABELS_FULL_DATA + model.extra_input_columns]

    # transform to simpler problem
    Y = Y[["Product 2", "Product 3"]].sum(axis="columns").to_frame(name="Product")
    Y_c = Y_c[["Product"]]

    results = pd.DataFrame(columns=["Test solvent", "mse", "nlpd"])
    out_dir = Path("results/transfer_learning/")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.get_model_name()
    if transfer and model_name == "BaselineModel":
        model_name = f"{model_name}-transfer"


    # this will generate all of the possible leave-one-out splits of the dataset
    split_generator = generate_leave_one_ramp_out_splits(X, Y)
    for i, split in tqdm.tqdm(enumerate(split_generator)):
        (train_X, train_Y), (test_X, test_Y) = split
        model.train(
            pd.concat((train_X, X_c)) if transfer else train_X,
            pd.concat((train_Y, Y_c)) if transfer else train_Y,
        )

        test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
        predictions = model.predict(test_X)

        # calculate some metrics
        mse = metrics.mse(predictions, test_Y)
        nlpd = metrics.nlpd(predictions, test_Y)
        test_solvent = "-".join(test_X.iloc[0][["SOLVENT A NAME", "SOLVENT B NAME"]])

        result = pd.DataFrame(
            {"Test solvent": test_solvent, "mse": mse, "nlpd": nlpd}, index=[i]
        )
        results = pd.concat((results, result))

        # store the results as you go
        results.to_csv(out_dir / f"{model_name}.csv", index=False)

    return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate a model on the solvent ramps with transfer learning.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_transfer_learning.py -m "GPModel" -f "drfps" -t -c multitask=True
            """
        ),
    )
    argparser.add_argument("-m", "--model", type=str)
    argparser.add_argument("-f", "--featurization", type=str)
    argparser.add_argument(
        "-t", "--transfer", action=argparse.BooleanOptionalAction, default=False
    )
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
    results = main(args.model, args.featurization, args.transfer, config)
