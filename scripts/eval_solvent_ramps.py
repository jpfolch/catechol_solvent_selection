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
    load_solvent_ramp_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import get_model


class StoreDict(argparse.Action):
    """Custom action to support passing kwargs.

    https://stackoverflow.com/a/11762020"""

    def __call__(self, parser, namespace, values, option_string=None):
        # Create or retrieve an existing dictionary from the namespace.
        kwargs_dict = {}
        # Allow values to be passed as a list (supporting multiple key-value pairs)
        for value in values:
            try:
                key, _, val = value.partition("=")
                if val.lower() in ["true", "false"]:
                    val = val.lower() == "true"
            except ValueError:
                message = f"Value '{value}' is not in key=value format"
                raise argparse.ArgumentError(self, message)
            kwargs_dict[key] = val
        setattr(namespace, self.dest, kwargs_dict)


def main(model_name: str, featurization: FeaturizationType, kwargs):
    model = get_model(model_name=model_name, featurization=featurization, **kwargs)
    X, Y = load_solvent_ramp_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_FULL_DATA]

    results = pd.DataFrame(columns=["Test solvent", "mse", "nlpd"])
    out_dir = Path("results/full_data/")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.get_model_name()

    # this will generate all of the possible leave-one-out splits of the dataset
    split_generator = generate_leave_one_ramp_out_splits(X, Y)
    for i, split in tqdm.tqdm(enumerate(split_generator), total=13):
        (train_X, train_Y), (test_X, test_Y) = split
        model.train(train_X, train_Y)

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
        description="Evaluate a model on the solvent ramp dataset.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_solvent_ramps.py -m "GPModel" -f "drfps" -c multitask=True
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

