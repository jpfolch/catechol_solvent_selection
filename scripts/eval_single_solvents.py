import argparse
import textwrap

import pandas as pd
from pathlib import Path
import tqdm
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.featurizations import FeaturizationType
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
    replace_repeated_measurements_with_average,
)
from catechol.models import get_model
from catechol.script_utils import StoreDict
from catechol.models.learn_mean import LearnMean
from gpytorch.means import ZeroMean

def main(model_name: str, featurization: FeaturizationType, kwargs, learn_prior_mean: bool = False):
    model = get_model(model_name=model_name, featurization=featurization, **kwargs)
    X, Y = load_single_solvent_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_SINGLE_SOLVENT + model.extra_input_columns]

    results = pd.DataFrame(columns=["Test solvent", "mse", "nlpd"])
    out_dir = Path("results/single_solvent/")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.get_model_name()

    # this will generate all of the possible leave-one-out splits of the dataset
    split_generator = generate_leave_one_out_splits(X, Y)
    for i, split in tqdm.tqdm(enumerate(split_generator)):
        (train_X, train_Y), (test_X, test_Y) = split
        if learn_prior_mean:
            prior_mean = LearnMean(train_X, train_Y, **kwargs)
        else:
            prior_mean = ZeroMean()
        model.train(train_X, train_Y, prior_mean=prior_mean)

        test_X, test_Y = replace_repeated_measurements_with_average(test_X, test_Y)
        predictions = model.predict(test_X)

        # calculate some metrics
        mse = metrics.mse(predictions, test_Y)
        nlpd = metrics.nlpd(predictions, test_Y)
        test_solvent = test_X.iloc[0]["SOLVENT NAME"]

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
        description="Evaluate a model on the single solvent dataset.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_single_solvents.py -m "GPModel" -f "drfps" -c multitask=True
            """
        ),
    )
    argparser.add_argument("-m", "--model", type=str)
    argparser.add_argument("-f", "--featurization", type=str)
    argparser.add_argument("-l", "--learn_mean", type=bool)
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
    results = main(args.model, args.featurization, config, args.learn_mean)
