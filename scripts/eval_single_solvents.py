import pandas as pd
from catechol import metrics
from catechol.data.data_labels import INPUT_LABELS_SINGLE_SOLVENT
from catechol.data.loader import (
    generate_leave_one_out_splits,
    load_single_solvent_data,
)
from catechol.models import get_model
from catechol.plots.plot_solvent_prediction import plot_solvent_prediction
import argparse
from catechol.data.featurizations import FeaturizationType
import textwrap

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
    X, Y = load_single_solvent_data()
    # remove unnecessary columns
    X = X[INPUT_LABELS_SINGLE_SOLVENT]

    results = pd.DataFrame(columns=["Test catalyst", "mse", "nlpd"])

    # this will generate all of the possible leave-one-out splits of the dataset
    split_generator = generate_leave_one_out_splits(X, Y)
    for i, split in enumerate(split_generator):
        (train_X, train_Y), (test_X, test_Y) = split
        model.train(train_X, train_Y)

        predictions = model.predict(test_X)

        # calculate some metrics
        mse = metrics.mse(predictions, test_Y)
        nlpd = metrics.nlpd(predictions, test_Y)
        test_solvent = test_X.iloc[0]["SOLVENT NAME"]

        result = pd.DataFrame({"Test solvent": test_solvent, "mse": mse, "nlpd": nlpd}, index=[i])
        results = pd.concat((results, result))

    return results

if __name__ == "__main__":
    argparser  = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate a model on the single solvent dataset.",
        epilog=textwrap.dedent(
            """To pass in arbitrary options, use the -c flag.
            Example usage:
                python scripts/eval_single_solvents.py -m "GPModel" -f "drfps" -c multitask=True
            """
        )
    )
    argparser.add_argument("-m", "--model", type=str)
    argparser.add_argument("-f", "--featurization", type=str)
    argparser.add_argument("-c", "--config", action=StoreDict, nargs="+",
        help="Store kwargs-style dictionary to support arbitrary config to models.")

    args = argparser.parse_args()
    results = main(args.model, args.featurization, args.config)