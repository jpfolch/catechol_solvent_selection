from .base_model import Model
from .gp import GPModel
from .llm import LLMModel
from .mlp import MLPModel

ALL_MODELS = {
    "GPModel": GPModel,
    "LLMModel": LLMModel,
    "MLPModel": MLPModel,
}


def get_model(model_name: str, **kwargs) -> Model:
    """Get a model instance by name."""
    if model_name not in ALL_MODELS:
        raise ValueError(
            f"Model `{model_name}` is not recognized. Remember to add your model to the `ALL_MODELS` lookup dictionary in `catechol/models/__init__.py`."
        )
    return ALL_MODELS[model_name](**kwargs)
