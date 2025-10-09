import argparse
from typing import TypeVar
import numpy as np

T = TypeVar("T")
T_ = TypeVar("T")


def map_str_to_bool(s: T_) -> T_ | bool:
    if not isinstance(s, str):
        return s
    if s.lower() in ["true", "false"]:
        return s.lower() == "true"
    return s


def map_str_to_type(s: T_, type: type[T]) -> T_ | T:
    if not isinstance(s, str):
        return s
    try:
        out = type(s)
    except ValueError:
        out = s
    return out


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
                val = map_str_to_bool(val)
                val = map_str_to_type(val, int)
                val = map_str_to_type(val, float)
            except ValueError:
                message = f"Value '{value}' is not in key=value format"
                raise argparse.ArgumentError(self, message)
            kwargs_dict[key] = val
        setattr(namespace, self.dest, kwargs_dict)


def calculate_euclidean_generational_distance(
        estimated_pareto_set,
        true_pareto_set):
    """Calculate the Euclidean generational distance between estimated and true Pareto sets."""

    estimated_pareto_set = np.array(estimated_pareto_set)
    true_pareto_set = np.array(true_pareto_set)

    if estimated_pareto_set.shape[1] != true_pareto_set.shape[1]:
        raise ValueError("Pareto sets must have the same number of objectives.")
    
    # Calculate the Euclidean distance from each point in the estimated Pareto set to the nearest point in the true Pareto set
    distances = np.linalg.norm(
        estimated_pareto_set[:, np.newaxis] - true_pareto_set[np.newaxis, :],
        axis=2
    )
    min_distances = np.min(distances, axis=1)

    # calculate the generational distance
    generational_distance = np.mean(min_distances)

    return generational_distance

def calculate_inverted_generational_distance(
        estimated_pareto_set,
        true_pareto_set):
    """Calculate the inverted generational distance between estimated and true Pareto sets."""
    estimated_pareto_set = np.array(estimated_pareto_set)
    true_pareto_set = np.array(true_pareto_set)

    if estimated_pareto_set.shape[1] != true_pareto_set.shape[1]:
        raise ValueError("Pareto sets must have the same number of objectives.")
    
    # Calculate the Euclidean distance from each point in the true Pareto set to the nearest point in the estimated Pareto set
    distances = np.linalg.norm(
        true_pareto_set[:, np.newaxis] - estimated_pareto_set[np.newaxis, :],
        axis=2
    )
    min_distances = np.min(distances, axis=1)

    # calculate the inverted generational distance
    inverted_generational_distance = np.mean(min_distances)

    return inverted_generational_distance

def calculate_maximum_pareto_frontier_error(
        estimated_pareto_set,
        true_pareto_set):
    
    """Calculate the maximum Pareto frontier error between estimated and true Pareto sets."""
    estimated_pareto_set = np.array(estimated_pareto_set)
    true_pareto_set = np.array(true_pareto_set)

    if estimated_pareto_set.shape[1] != true_pareto_set.shape[1]:
        raise ValueError("Pareto sets must have the same number of objectives.")
    
    # Calculate the maximum distance from each point in the estimated Pareto set to the true Pareto set
    distances = np.linalg.norm(
        estimated_pareto_set[:, np.newaxis] - true_pareto_set[np.newaxis, :],
        axis=2
    )
    max_distance = np.max(np.min(distances, axis=1))

    return max_distance

def calculate_pareto_set(objective_values):

    def is_pareto(point, points):
        """Check if a point is Pareto optimal (maximization)."""
        for p in points:
            if all(p_i >= q_i for p_i, q_i in zip(p, point)) and any(p_i > q_i for p_i, q_i in zip(p, point)):
                return False
        
        return True
    
    pareto_set = []
    for point in objective_values:
        if is_pareto(point, objective_values):
            pareto_set.append(point)
    
    return pareto_set