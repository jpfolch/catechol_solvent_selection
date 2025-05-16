import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from catechol.data.featurizations import FeaturizationType
from catechol.data.bo_benchmark import BOBenchmark
from catechol.models import get_model
from catechol.script_utils import StoreDict


def main(
    model_name: str, featurization: FeaturizationType, kwargs, init_set_size: int, seed: int, strategy: str
):
    model = get_model(model_name=model_name, featurization=featurization, bo_strategy = strategy, **kwargs)
    bench = BOBenchmark(model.featurization, **kwargs)

    results = pd.DataFrame(
        columns=["Iteration", "Number of data points", "Best value", "Point chosen"]
    )
    model_name = model.get_model_name()
    out_dir = Path(f"results/bo/{model_name}/{strategy}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # get the search space
    X = bench.get_search_space()
    # random initial sample
    rng = np.random.default_rng(seed)
    initial_points = rng.choice(X.index, size=init_set_size, replace=False)
    train_idx = list(initial_points)

    iteration = 0
    num_of_points = 1

    train_X = []
    train_Y = []

    for idx in initial_points:
        train_X.append(X.iloc[idx])
        train_Y.append(bench.objective_function(idx))

        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of data points": num_of_points,
                "Best value": np.max(train_Y),
                "Point chosen": X.iloc[idx],
            },
            index=[iteration],
        )

        num_of_points += 1
        iteration += 1



    for iteration in range(1, 100):
        if model.bo_strategy not in  ["random"]:
            # select a random point
            model.train(pd.DataFrame(train_X), pd.DataFrame(train_Y))
        # select the next ramp
        next_x_idx = model.select_next_bo(train_idx, X)

        next_x = X.iloc[next_x_idx]
        next_y = bench.objective_function(next_x_idx)

        train_X.append(next_x)
        train_Y.append(next_y)
        train_idx.append(next_x_idx)

        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of data points": num_of_points,
                "Best value": np.max(train_Y),
                "Point chosen": next_x,
            },
            index=[iteration + init_set_size],
        )
        results = pd.concat((results, result))

        # store the results as you go
        results.to_csv(out_dir / f"{seed}.csv", index=False)

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
    argparser.add_argument("-st", "--strategy", type=str, default="ei")
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
    results = main(args.model, args.featurization, config, args.initset, args.seed, args.strategy)