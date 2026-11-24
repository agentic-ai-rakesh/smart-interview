"""Smart Interview agent: a LangGraph workflow for every LLM task in the app.

    START -> route_task -+- "generate" -------> select_model -> generate -> validate -+-> END
                         |                                       ^               |
                         |                                       +--- retry -----+  (invalid output)
                         +- "ask" ------------> select_ask_model -+-> answer -> END
                         +- "quick_topic" ----> select_ask_model -+-> topic_card -> END
                         +- "refine_question" / "refine_answer" -> refine -> END

- select_model: with "Auto" the LLM chooses the model from the question parameters
  (falls back to data/model_routing.json if its choice is unusable); a model picked by
  the user is always respected.
- validate: checks the number of questions and their subtopics; on a problem the agent
  retries once and tells the model what was wrong.
- ask: answers the candidate's own question (model chosen the same way as above; the
  default model is the fallback).
- quick_topic: writes a Quick Topic box (overview, methodologies, types, other technical
  points, small example) for a topic; the model is chosen like for "ask".
- refine: rewrites a question in easier words, or explains an answer in a new, easier way.

Prompts live in prompt_interview.py. Errors are raised as AgentError(status_code, detail).
"""

import json
import logging
import os
import time
from typing import Any, Literal, Optional, TypedDict

from google.api_core.exceptions import GoogleAPIError
from langchain_core.exceptions import OutputParserException
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError
from pydantic import ValidationError

from .model_selection import (
    AUTO_MODEL_ID,
    KNOWLEDGE_BASE_MODEL_ID,
    ModelConfig,
    create_chat_model,
    create_structured_model,
    get_agent_model_config,
    get_model_config,
    load_models,
    resolve_model_config,
)
from .progress_info import ProcessInfo, _process_info
from .prompt_interview import (
    ASK_ANSWER_PROMPT,
    ASK_MODEL_SELECTION_PROMPT,
    INTERVIEW_PROMPT,
    MODEL_SELECTION_PROMPT,
    QUICK_TOPIC_PROMPT,
    PREVIOUS_REFINEMENT,
    REFINE_ANSWER_PROMPT,
    REFINE_QUESTION_PROMPT,
    RETRY_FEEDBACK,
)
from .schemas import (
    AgentError,
    GeneratedInterview,
    InterviewQuestion,
    InterviewRequest,
    ModelChoice,
    RefineTarget,
    TopicCard,
    safe_error_message,
)

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 2  # first try + one retry with feedback
MODEL_ERRORS = (
    ChatGoogleGenerativeAIError,
    GoogleAPIError,
    OpenAIError,
    OutputParserException,
    ValidationError,
    ValueError,
)

Task = Literal["generate", "ask", "quick_topic", "refine_question", "refine_answer"]


class AgentState(TypedDict, total=False):
    task: Task
    # generate
    request: InterviewRequest
    request_validation_time: float
    model_config: ModelConfig
    model_selection: str  # how the model was chosen, shown in the process info
    selection_time: float  # seconds the agent spent choosing the model (an LLM call)
    attempts: int
    feedback: str  # why the previous attempt was rejected
    model_result: Any
    timings: dict[str, float]
    questions: list[InterviewQuestion]
    process_info: ProcessInfo
    # ask
    ask_question: str
    ask_model: Optional[str]  # model picked by the user, or None / "auto" for the agent
    answer: str
    # quick topic (the topic text is passed in ask_question, for model selection)
    topic_card: TopicCard
    # refine
    refine_input: dict[str, str]
    refined: str


def _api_key(config: ModelConfig) -> str:
    api_key = os.getenv(config.api_key_env)
    if not api_key or not api_key.strip():
        raise AgentError(
            503, f"{config.api_key_env} is not configured. Set it in smart_interview/.env."
        )
    return api_key


# --- nodes ---------------------------------------------------------------------------


def route_task(state: AgentState) -> str:
    return {
        "generate": "select_model",
        "ask": "select_ask_model",
        "quick_topic": "select_ask_model",
    }.get(state["task"], "refine")


def after_ask_model(state: AgentState) -> str:
    return "topic_card" if state["task"] == "quick_topic" else "answer"


def _configured_models() -> dict[str, ModelConfig]:
    """Models whose API key is set, by id."""
    return {
        model.id: model for model in load_models() if (os.getenv(model.api_key_env) or "").strip()
    }


def _agent_pick(prompt_template, values: dict[str, Any], candidates: dict[str, ModelConfig]):
    """Let the agent LLM pick one of the candidates. Returns (config, reason) or None."""
    if not candidates:
        return None
    try:
        choice = _ask_agent_for_model(prompt_template, values, list(candidates.values()))
    except Exception as error:  # any failure here just means "use the fallback"
        logger.warning("Agent model selection failed: %s", safe_error_message(error))
        return None
    if choice.model_id not in candidates:
        logger.warning("Agent chose an unavailable model: %s", choice.model_id)
        return None
    return candidates[choice.model_id], choice.reason


def select_model(state: AgentState) -> dict:
    request = state["request"]
    if request.model is not None and request.model != AUTO_MODEL_ID:
        config = get_model_config(request.model)  # ModelNotFoundError -> 422 in the API
        return {"model_config": config, "model_selection": f"Chosen by you: {config.name}"}

    # Auto (or no model given): let the LLM decide, among models whose API key is set.
    started_at = time.perf_counter()
    picked = _agent_pick(
        MODEL_SELECTION_PROMPT,
        {
            "topic": request.topic,
            "sub_topics": json.dumps(request.sub_topics, ensure_ascii=False),
            "experience_level": request.experience_level,
            "question_type": request.question_type,
            "question_count": request.question_count,
        },
        _configured_models(),
    )
    selection_time = time.perf_counter() - started_at
    if picked:
        config, reason = picked
        return {
            "model_config": config,
            "model_selection": f"Chosen by AI agent: {config.name}. {reason}",
            "selection_time": selection_time,
        }

    config = resolve_model_config(AUTO_MODEL_ID, request.question_type)
    return {
        "model_config": config,
        "model_selection": f"Rule-based fallback: {config.name}",
        "selection_time": selection_time,
    }


def _ask_agent_for_model(
    prompt_template, values: dict[str, Any], models: list[ModelConfig]
) -> ModelChoice:
    agent_config = get_agent_model_config()
    model = create_structured_model(agent_config, _api_key(agent_config), ModelChoice, 0)
    prompt = prompt_template.invoke(
        {
            **values,
            "available_models": json.dumps(
                [
                    {"id": m.id, "name": m.name, "description": m.description}
                    for m in models
                ],
                ensure_ascii=False,
            ),
        }
    )
    result = model.invoke(prompt)
    choice = result.get("parsed") if isinstance(result, dict) else result
    if isinstance(choice, dict):
        choice = ModelChoice.model_validate(choice)
    if not isinstance(choice, ModelChoice):
        raise ValueError("no model choice returned")
    return choice


def generate(state: AgentState) -> dict:
    request = state["request"]
    config = state["model_config"]
    api_key = _api_key(config)
    attempts = state.get("attempts", 0) + 1
    try:
        setup_started_at = time.perf_counter()
        model = create_structured_model(
            config, api_key, GeneratedInterview, temperature=request.temperature
        )
        model_setup_time = time.perf_counter() - setup_started_at

        prompt_started_at = time.perf_counter()
        prompt = INTERVIEW_PROMPT.invoke(
            {
                "topic": request.topic,
                "sub_topics": json.dumps(request.sub_topics, ensure_ascii=False),
                "experience_level": request.experience_level,
                "question_type": request.question_type,
                "question_count": request.question_count,
                "feedback": state.get("feedback", ""),
            }
        )
        prompt_preparation_time = time.perf_counter() - prompt_started_at

        llm_started_at = time.perf_counter()
        model_result = model.invoke(prompt)
        llm_processing_time = time.perf_counter() - llm_started_at
    except MODEL_ERRORS as error:
        logger.exception("%s interview question generation failed", config.id)
        raise AgentError(
            502,
            f"{config.name} failed to generate a valid response: {safe_error_message(error)}",
        ) from error

    return {
        "attempts": attempts,
        "model_result": model_result,
        "timings": {
            "model_setup_time": model_setup_time,
            "prompt_preparation_time": prompt_preparation_time,
            "llm_processing_time": llm_processing_time,
        },
    }


def validate(state: AgentState) -> dict:
    """Check the model output; ask for a retry (feedback) or build the final result."""
    started_at = time.perf_counter()
    request = state["request"]
    config = state["model_config"]
    model_result = state["model_result"]

    if not isinstance(model_result, dict) or "raw" not in model_result:
        raise AgentError(502, "The interview question service returned invalid response metadata.")

    parsing_error = model_result.get("parsing_error")
    generated = model_result.get("parsed")
    problem: Optional[str] = None
    if parsing_error is not None:
        logger.error("%s response could not be parsed: %s", config.id, parsing_error)
        problem = (
            "the response was not valid structured output "
            f"({safe_error_message(parsing_error)})"
        )
    else:
        if isinstance(generated, dict):
            try:
                generated = GeneratedInterview.model_validate(generated)
            except ValidationError:
                logger.exception("%s returned invalid structured output", config.id)
                problem = "the response did not match the required question format"
        if problem is None and not isinstance(generated, GeneratedInterview):
            problem = "the response did not contain any questions"
        elif problem is None and len(generated.questions) != request.question_count:
            problem = (
                f"it had {len(generated.questions)} questions instead of exactly "
                f"{request.question_count}"
            )
        elif problem is None and any(
            question.sub_topic not in request.sub_topics for question in generated.questions
        ):
            problem = "it used a subtopic that was not in the selected subtopics"

    if problem is not None:
        if state["attempts"] < MAX_GENERATION_ATTEMPTS:
            logger.warning("Agent retrying generation: %s", problem)
            return {"feedback": RETRY_FEEDBACK.format(reason=problem)}
        raise AgentError(502, f"{config.name} returned an invalid response: {problem}.")

    questions = [
        InterviewQuestion(
            id=index,
            category=question.category,
            sub_topic=question.sub_topic,
            text=question.text,
            answer=question.answer,
        )
        for index, question in enumerate(generated.questions, start=1)
    ]
    timings = state["timings"]
    process_info = _process_info(
        config,
        model_result["raw"],
        state["request_validation_time"],
        timings["model_setup_time"],
        timings["prompt_preparation_time"],
        # includes the agent's model-choice call, which is also LLM time
        timings["llm_processing_time"] + state.get("selection_time", 0.0),
        time.perf_counter() - started_at,
    ).model_copy(update={"model_selection": state.get("model_selection")})
    return {"questions": questions, "process_info": process_info, "feedback": ""}


def after_validate(state: AgentState) -> str:
    return END if state.get("questions") else "generate"


def select_ask_model(state: AgentState) -> dict:
    requested = state.get("ask_model")
    if requested not in (None, AUTO_MODEL_ID, KNOWLEDGE_BASE_MODEL_ID):
        config = get_model_config(requested)  # ModelNotFoundError -> 422 in the API
        return {"model_config": config, "model_selection": f"Chosen by you: {config.name}"}

    started_at = time.perf_counter()
    picked = _agent_pick(
        ASK_MODEL_SELECTION_PROMPT, {"question": state["ask_question"]}, _configured_models()
    )
    selection_time = time.perf_counter() - started_at
    if picked:
        config, reason = picked
        return {
            "model_config": config,
            "model_selection": f"Chosen by AI agent: {config.name}. {reason}",
            "selection_time": selection_time,
        }
    config = get_model_config(None)
    return {
        "model_config": config,
        "model_selection": f"Default model (fallback): {config.name}",
        "selection_time": selection_time,
    }


def answer(state: AgentState) -> dict:
    config = state["model_config"]
    api_key = _api_key(config)
    try:
        setup_started_at = time.perf_counter()
        model = create_chat_model(config, api_key)
        model_setup_time = time.perf_counter() - setup_started_at

        prompt_started_at = time.perf_counter()
        prompt = ASK_ANSWER_PROMPT.invoke({"question": state["ask_question"]})
        prompt_preparation_time = time.perf_counter() - prompt_started_at

        llm_started_at = time.perf_counter()
        result = model.invoke(prompt)
        llm_processing_time = time.perf_counter() - llm_started_at
    except MODEL_ERRORS as error:
        logger.exception("%s could not answer the question", config.id)
        raise AgentError(
            502, f"{config.name} could not answer the question: {safe_error_message(error)}"
        ) from error

    response_started_at = time.perf_counter()
    text = _message_text(result)
    if not text:
        raise AgentError(502, f"{config.name} returned an empty answer.")
    process_info = _process_info(
        config,
        result,
        state.get("request_validation_time", 0.0),
        model_setup_time,
        prompt_preparation_time,
        # includes the agent's model-choice call, which is also LLM time
        llm_processing_time + state.get("selection_time", 0.0),
        time.perf_counter() - response_started_at,
    ).model_copy(update={"model_selection": state.get("model_selection")})
    return {"answer": text, "process_info": process_info}


def topic_card(state: AgentState) -> dict:
    """Write the Quick Topic box as structured bullet sections."""
    config = state["model_config"]
    api_key = _api_key(config)
    try:
        setup_started_at = time.perf_counter()
        model = create_structured_model(config, api_key, TopicCard)
        model_setup_time = time.perf_counter() - setup_started_at

        prompt_started_at = time.perf_counter()
        prompt = QUICK_TOPIC_PROMPT.invoke({"topic": state["ask_question"]})
        prompt_preparation_time = time.perf_counter() - prompt_started_at

        llm_started_at = time.perf_counter()
        result = model.invoke(prompt)
        llm_processing_time = time.perf_counter() - llm_started_at
    except MODEL_ERRORS as error:
        logger.exception("%s could not write the topic box", config.id)
        raise AgentError(
            502, f"{config.name} could not explain the topic: {safe_error_message(error)}"
        ) from error

    response_started_at = time.perf_counter()
    card = result.get("parsed") if isinstance(result, dict) else None
    if isinstance(card, dict):
        try:
            card = TopicCard.model_validate(card)
        except ValidationError:
            card = None
    if not isinstance(card, TopicCard) or not (card.overview or card.methodologies):
        raise AgentError(502, f"{config.name} returned an invalid topic box.")
    process_info = _process_info(
        config,
        result["raw"],
        state.get("request_validation_time", 0.0),
        model_setup_time,
        prompt_preparation_time,
        # includes the agent's model-choice call, which is also LLM time
        llm_processing_time + state.get("selection_time", 0.0),
        time.perf_counter() - response_started_at,
    ).model_copy(update={"model_selection": state.get("model_selection")})
    return {"topic_card": card, "process_info": process_info}


def _message_text(result: Any) -> str:
    """Plain text of a chat model reply (some providers return a list of content parts)."""
    content = getattr(result, "content", result)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content.strip() if isinstance(content, str) else ""


def refine(state: AgentState) -> dict:
    data = state["refine_input"]
    config = get_agent_model_config()
    prompt_template = (
        REFINE_QUESTION_PROMPT if state["task"] == "refine_question" else REFINE_ANSWER_PROMPT
    )
    previous = (
        PREVIOUS_REFINEMENT.format(previous=data["previous"]) if data.get("previous") else ""
    )
    try:
        model = create_chat_model(config, _api_key(config))
        result = model.invoke(prompt_template.invoke({**data, "previous": previous}))
    except MODEL_ERRORS as error:
        logger.exception("Refine with %s failed", config.id)
        raise AgentError(
            502, f"{config.name} could not refine the text: {safe_error_message(error)}"
        ) from error
    refined = _message_text(result)
    if not refined:
        raise AgentError(502, f"{config.name} returned an empty refinement.")
    return {"refined": refined}


# --- graph and public entry points ---------------------------------------------------


def build_interview_graph():
    graph = StateGraph(AgentState)
    graph.add_node("select_model", select_model)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("select_ask_model", select_ask_model)
    graph.add_node("answer", answer)
    graph.add_node("topic_card", topic_card)
    graph.add_node("refine", refine)

    graph.add_conditional_edges(START, route_task, ["select_model", "select_ask_model", "refine"])
    graph.add_edge("select_model", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", after_validate, ["generate", END])
    graph.add_conditional_edges("select_ask_model", after_ask_model, ["answer", "topic_card"])
    graph.add_edge("answer", END)
    graph.add_edge("topic_card", END)
    graph.add_edge("refine", END)
    return graph.compile()


interview_agent = build_interview_graph()


def run_generation(
    request: InterviewRequest, request_validation_time: float
) -> tuple[list[InterviewQuestion], ProcessInfo]:
    """Choose a model, generate the questions and validate them."""
    final_state = interview_agent.invoke(
        {
            "task": "generate",
            "request": request,
            "request_validation_time": request_validation_time,
        }
    )
    return final_state["questions"], final_state["process_info"]


def run_ask(
    question: str, model: Optional[str], request_validation_time: float = 0.0
) -> tuple[str, ProcessInfo]:
    """Answer the candidate's own question. `model` None / "auto" lets the agent choose."""
    final_state = interview_agent.invoke(
        {
            "task": "ask",
            "ask_question": question,
            "ask_model": model,
            "request_validation_time": request_validation_time,
        }
    )
    return final_state["answer"], final_state["process_info"]


def run_quick_topic(
    topic: str, model: Optional[str], request_validation_time: float = 0.0
) -> tuple[TopicCard, ProcessInfo]:
    """Write a Quick Topic box. `model` None / "auto" lets the agent choose."""
    final_state = interview_agent.invoke(
        {
            "task": "quick_topic",
            "ask_question": topic,
            "ask_model": model,
            "request_validation_time": request_validation_time,
        }
    )
    return final_state["topic_card"], final_state["process_info"]


def run_refine(target: RefineTarget, question: dict[str, Any]) -> str:
    """Rewrite a saved question (target="question") or its answer (target="answer")."""
    previous_key = "refined_text" if target == "question" else "refined_answer"
    final_state = interview_agent.invoke(
        {
            "task": "refine_question" if target == "question" else "refine_answer",
            "refine_input": {
                "topic": question.get("topic", ""),
                "experience_level": question.get("experience_level", ""),
                "question_type": question.get("question_type", ""),
                "question": question["text"],
                "answer": question["answer"],
                "previous": question.get(previous_key) or "",
            },
        }
    )
    return final_state["refined"]
