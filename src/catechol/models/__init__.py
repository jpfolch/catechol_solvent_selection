from .base_model import Model
from .gp import GPModel
from .lode import LantentODE, ExplicitODE

ALL_MODELS = {
    "GPModel": GPModel,
    "LatentODE": LantentODE,
    "ExplicitODE": ExplicitODE,
}


def get_model(model_name: str, **kwargs) -> Model:
    """Get a model instance by name."""
    if model_name not in ALL_MODELS:
        raise ValueError(f"Model {model_name} is not recognized.")
    return ALL_MODELS[model_name](**kwargs)
