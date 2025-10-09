import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from catechol.data.bo_benchmark import MOBOBenchmark
from catechol.data.featurizations import FeaturizationType
from catechol.models import get_model
from catechol.data.loader import load_green_scores
from catechol.script_utils import StoreDict, calculate_euclidean_generational_distance, calculate_maximum_pareto_frontier_error, calculate_inverted_generational_distance, calculate_pareto_set


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
        bo_strategy=strategy,
        **kwargs,
    )
    bench = MOBOBenchmark(model.featurization, **kwargs)

    results = pd.DataFrame(
        columns=["Iteration", "Number of data points", "Number of pareto points queries", "GD", "IGD", "MPFE", "Point chosen"]
    )
    model_name = model.get_model_name()
    out_dir = Path(f"results/mobo/{model_name}/{strategy}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # get the search space
    X = bench.get_search_space()
    # get the green scores
    green_scores = load_green_scores(X)
    green_scores = green_scores["Green Score"]
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

        # get the number of pareto points
        _, pareto_vals, pareto_set = bench.get_optimum()

        estimated_pareto_set = calculate_pareto_set(train_Y)
        gd = calculate_euclidean_generational_distance(
            estimated_pareto_set, pareto_vals
        )
        igd = calculate_inverted_generational_distance(
            estimated_pareto_set, pareto_vals
        )
        mpfe = calculate_maximum_pareto_frontier_error(
            estimated_pareto_set, pareto_vals
        )

        num_pareto_points = len(set([p for p in initial_points if p in pareto_set]))

        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of data points": num_of_points,
                "Number of pareto points queries": num_pareto_points,
                "GD": gd,
                "IGD": igd,
                "MPFE": mpfe,
                "Point chosen": idx,
            },
            index=[iteration],
        )

        num_of_points += 1
        iteration += 1

    for iteration in range(1, 100):
        if model.bo_strategy not in ["random"]:
            # select a random point
            model.train(pd.DataFrame(train_X), pd.DataFrame(train_Y))
        # select the next ramp
        next_x_idx = model.select_next_mobo(train_idx, X, green_scores)

        next_x = X.iloc[next_x_idx]
        next_y = bench.objective_function(next_x_idx)

        train_X.append(next_x)
        train_Y.append(next_y)
        train_idx.append(next_x_idx)

        # get the number of pareto points
        # get the number of pareto points
        _, pareto_vals, pareto_set = bench.get_optimum()
        
        num_pareto_points = len(set([p for p in train_idx if p in pareto_set]))

        num_of_points = len(train_X)

        estimated_pareto_set = calculate_pareto_set(train_Y)
        gd = calculate_euclidean_generational_distance(
            estimated_pareto_set, pareto_vals
        )
        igd = calculate_inverted_generational_distance(
            estimated_pareto_set, pareto_vals
        )
        mpfe = calculate_maximum_pareto_frontier_error(
            estimated_pareto_set, pareto_vals
        )

        result = pd.DataFrame(
            {
                "Iteration": iteration,
                "Number of data points": num_of_points,
                "Number of pareto points queries": num_pareto_points,
                "GD": gd,
                "IGD": igd,
                "MPFE": mpfe,
                "Point chosen": next_x_idx,
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
    argparser.add_argument("-m", "--model", type=str, default="GPModel")
    argparser.add_argument("-f", "--featurization", type=str, default="spange_descriptors")
    argparser.add_argument("-s", "--seed", type=int, default=239)
    argparser.add_argument("-i", "--initset", type=int, default=5)
    argparser.add_argument("-st", "--strategy", type=str, default="random")
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
