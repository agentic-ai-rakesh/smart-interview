import json
import os
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, TypeAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = PROJECT_ROOT / "data" / "models.json"
MODEL_ROUTING_FILE = PROJECT_ROOT / "data" / "model_routing.json"

Provider = Literal["gemini", "openai", "openrouter"]
StructuredOutputMethod = Literal["json_mode", "function_calling"]
QuestionType = Literal["General", "Code problem", "Scenario problem"]
AUTO_MODEL_ID = "auto"
KNOWLEDGE_BASE_MODEL_ID = "knowledge_base"


class ModelConfig(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider: Provider
    model_name: str = Field(min_length=1)
    model_name_env: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    api_key_env: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    structured_output_method: StructuredOutputMethod
    default: bool = False
    # What the model is good at; the agent reads this when it chooses a model.
    description: str = ""


class ModelNotFoundError(ValueError):
    pass


def load_models() -> list[ModelConfig]:
    with MODELS_FILE.open(encoding="utf-8") as models_file:
        raw_models = json.load(models_file)

    models = TypeAdapter(list[ModelConfig]).validate_python(raw_models)
    model_ids = [model.id for model in models]
    if not models or len(model_ids) != len(set(model_ids)):
        raise ValueError(f"{MODELS_FILE} must contain models with unique IDs.")
    if sum(model.default for model in models) != 1:
        raise ValueError(f"{MODELS_FILE} must contain exactly one default model.")

    return models


def get_model_config(model_id: Optional[str]) -> ModelConfig:
    models = load_models()
    if model_id is None:
        return next(model for model in models if model.default)

    for model in models:
        if model.id == model_id:
            return model
    raise ModelNotFoundError(f"Model '{model_id}' is not configured.")


def get_agent_model_config() -> ModelConfig:
    """The model that runs the agent's own tasks (choosing a model, refining).

    AGENT_MODEL_ID in .env picks one from data/models.json; otherwise the default model.
    """
    return get_model_config(os.getenv("AGENT_MODEL_ID") or None)


def resolve_model_config(
    model_id: Optional[str], question_type: QuestionType
) -> ModelConfig:
    if model_id is not None and model_id != AUTO_MODEL_ID:
        return get_model_config(model_id)

    with MODEL_ROUTING_FILE.open(encoding="utf-8") as routing_file:
        routes = TypeAdapter(dict[QuestionType, str]).validate_python(
            json.load(routing_file)
        )
    if set(routes) != {"General", "Code problem", "Scenario problem"}:
        raise ValueError(
            f"{MODEL_ROUTING_FILE} must define a model for every question type."
        )

    try:
        routed_model_id = routes[question_type]
    except KeyError as error:
        raise ValueError(
            f"{MODEL_ROUTING_FILE} has no model for question type "
            f"'{question_type}'."
        ) from error

    try:
        return get_model_config(routed_model_id)
    except ModelNotFoundError as error:
        raise ValueError(
            f"{MODEL_ROUTING_FILE} maps '{question_type}' to an unknown model "
            f"'{routed_model_id}'."
        ) from error


def get_model_options() -> list[dict[str, Union[str, bool]]]:
    return [
        {
            "id": AUTO_MODEL_ID,
            "name": "Auto (AI agent chooses the model)",
            "default": False,
        },
        {
            "id": KNOWLEDGE_BASE_MODEL_ID,
            "name": "Own knowledge base (saved history)",
            "default": True,
        },
        *[
            {"id": model.id, "name": model.name, "default": False}
            for model in load_models()
        ],
    ]


def _create_gemini(config: ModelConfig, api_key: str) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=(
            os.getenv(config.model_name_env) if config.model_name_env else None
        )
        or config.model_name,
        temperature=config.temperature,
        google_api_key=api_key,
    )


def _create_openai(config: ModelConfig, api_key: str) -> BaseChatModel:
    kwargs = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    return ChatOpenAI(
        model=(
            os.getenv(config.model_name_env) if config.model_name_env else None
        )
        or config.model_name,
        temperature=config.temperature,
        api_key=api_key,
        **kwargs,
    )


PROVIDER_FACTORIES: dict[Provider, Callable[[ModelConfig, str], BaseChatModel]] = {
    "gemini": _create_gemini,
    "openai": _create_openai,
    "openrouter": _create_openai,
}


def create_chat_model(
    config: ModelConfig,
    api_key: str,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    if temperature is not None:
        config = config.model_copy(update={"temperature": temperature})
    return PROVIDER_FACTORIES[config.provider](config, api_key)


def create_structured_model(
    config: ModelConfig,
    api_key: str,
    schema: type[BaseModel],
    temperature: Optional[float] = None,
) -> Runnable:
    model = create_chat_model(config, api_key, temperature)
    return model.with_structured_output(
        schema,
        method=config.structured_output_method,
        include_raw=True,
    )
