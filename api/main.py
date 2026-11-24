import json
import logging
import os
import random
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Callable, Optional, TypeVar, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from .agent_interview import run_ask, run_generation, run_quick_topic, run_refine
from .model_selection import (
    KNOWLEDGE_BASE_MODEL_ID,
    ModelNotFoundError,
    get_model_options,
)
from .progress_info import (
    ProcessInfo,
    collect_runtime_metrics,
    runtime_metrics,
)
from .schemas import (
    AgentError,
    ExperienceLevel,
    InterviewQuestion,
    InterviewRequest,
    QuestionStatus,
    QuickTopic,
    QuestionType,
    RefineTarget,
)
from .schemas import safe_error_message as _safe_error_message

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPICS_FILE = PROJECT_ROOT / "data" / "topics.json"
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)
R = TypeVar("R")

_mongo_client: Optional[MongoClient] = None
_mongo_client_uri: Optional[str] = None
_mongo_client_lock = Lock()


def handle_processing_errors(
    operation: str, status_code: int = 500
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    def decorator(function: Callable[..., R]) -> Callable[..., R]:
        @wraps(function)
        def wrapped(*args: object, **kwargs: object) -> R:
            started_at = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            except HTTPException as error:
                elapsed = time.perf_counter() - started_at
                runtime_metrics.record_process(operation, False, elapsed)
                logger.warning(
                    "%s failed with HTTP %s after %.3f seconds",
                    operation,
                    error.status_code,
                    elapsed,
                )
                raise
            except Exception as error:
                elapsed = time.perf_counter() - started_at
                runtime_metrics.record_process(operation, False, elapsed)
                logger.exception("%s failed after %.3f seconds", operation, elapsed)
                raise HTTPException(
                    status_code=status_code,
                    detail=f"{operation} failed: {_safe_error_message(error)}",
                ) from error
            elapsed = time.perf_counter() - started_at
            runtime_metrics.record_process(operation, True, elapsed)
            logger.info("%s completed in %.3f seconds", operation, elapsed)
            return result

        return wrapped

    return decorator


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    yield
    close_mongo_client()


app = FastAPI(title="Smart Interview API", version="1.0.0", lifespan=lifespan)
app.middleware("http")(collect_runtime_metrics)


@handle_processing_errors("Loading interview topics")
def load_topics() -> dict[str, list[str]]:
    with TOPICS_FILE.open(encoding="utf-8") as topics_file:
        topics = json.load(topics_file)

    if (
        not isinstance(topics, dict)
        or not topics
        or any(not isinstance(topic, str) or not topic.strip() for topic in topics)
        or any(
            not isinstance(sub_topics, list)
            or not sub_topics
            or any(
                not isinstance(sub_topic, str) or not sub_topic.strip()
                for sub_topic in sub_topics
            )
            or len({sub_topic.strip() for sub_topic in sub_topics}) != len(sub_topics)
            for sub_topics in topics.values()
        )
    ):
        raise ValueError(
            f"{TOPICS_FILE} must map unique topic names to non-empty lists of unique subtopic strings."
        )

    return {
        topic.strip(): [sub_topic.strip() for sub_topic in sub_topics]
        for topic, sub_topics in topics.items()
    }


class SavedQuestionsRequest(BaseModel):
    topic: str = Field(min_length=1)
    sub_topics: list[str] = Field(min_length=1)
    experience_level: ExperienceLevel
    question_type: QuestionType
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=5, ge=1, le=50)


class SavedQuestionsPage(BaseModel):
    total: int  # all matching saved questions
    checked_total: int  # how many of those are "Checked"
    page: int
    page_size: int
    total_pages: int
    questions: list[InterviewQuestion]  # this page only; `id` is the position in the full list


USER_QUESTION_KIND = "user_question"  # marks saved documents holding a user's own question


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    model: Optional[str] = None  # None / "auto" (or "knowledge_base"): the agent chooses


class AskResponse(BaseModel):
    question: InterviewQuestion
    process_info: ProcessInfo


QUICK_TOPIC_KIND = "quick_topic"  # marks saved Quick Topic boxes


class QuickTopicRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    model: Optional[str] = None  # None / "auto" (or "knowledge_base"): the agent chooses


class QuickTopicResponse(BaseModel):
    topic: QuickTopic
    process_info: ProcessInfo


class QuickTopicPage(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    topics: list[QuickTopic]  # newest first


class RefineRequest(BaseModel):
    target: RefineTarget  # "question" or "answer"


class RefineResponse(BaseModel):
    question_id: str
    target: RefineTarget
    refined: str


class QuestionStatusUpdate(BaseModel):
    status: QuestionStatus


class QuestionStatusResponse(BaseModel):
    question_id: str
    status: QuestionStatus


class InterviewResponse(BaseModel):
    topic: str
    sub_topics: list[str]
    experience_level: ExperienceLevel
    question_type: QuestionType
    question_count: int
    questions: list[InterviewQuestion]
    process_info: ProcessInfo


@handle_processing_errors("Connecting to MongoDB", status_code=503)
def _get_mongo_collection() -> Collection:
    global _mongo_client, _mongo_client_uri

    mongo_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE")
    collection_name = os.getenv("MONGODB_COLLECTION")
    missing_settings = [
        name
        for name, value in (
            ("MONGODB_URI", mongo_uri),
            ("MONGODB_DATABASE", database_name),
            ("MONGODB_COLLECTION", collection_name),
        )
        if not value or not value.strip()
    ]
    if missing_settings:
        raise HTTPException(
            status_code=503,
            detail=(
                "MongoDB is not configured. Set "
                f"{', '.join(missing_settings)} in smart_interview/.env."
            ),
        )

    assert mongo_uri is not None
    assert database_name is not None
    assert collection_name is not None

    with _mongo_client_lock:
        if _mongo_client is None or _mongo_client_uri != mongo_uri:
            if _mongo_client is not None:
                _mongo_client.close()
            _mongo_client = MongoClient(mongo_uri)
            _mongo_client_uri = mongo_uri

        return _mongo_client[database_name][collection_name]


@handle_processing_errors("Saving interview response to MongoDB", status_code=503)
def _save_interview_response(response: InterviewResponse) -> None:
    _get_mongo_collection().insert_one(response.model_dump(mode="json"))


def _load_saved_questions(
    topic: str,
    experience_level: str,
    question_type: str,
    sub_topics: list[str],
) -> list[InterviewQuestion]:
    """All saved questions matching the selection: newest first, duplicate texts removed."""
    query = {
        "topic": topic,
        "experience_level": experience_level,
        "question_type": question_type,
        "sub_topics": {"$in": sub_topics},
    }
    try:
        history = _get_mongo_collection().find(
            query,
            {"questions": 1},
        ).sort("_id", -1)
        seen_texts: set[str] = set()
        questions: list[InterviewQuestion] = []
        for saved_response in history:
            for saved_question in saved_response["questions"]:
                if saved_question["sub_topic"] not in sub_topics:
                    continue
                normalized_text = saved_question["text"].strip().casefold()
                if normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                questions.append(
                    InterviewQuestion.model_validate(
                        {
                            **saved_question,
                            "id": len(questions) + 1,
                        }
                    )
                )
    except PyMongoError as error:
        logger.exception("Failed to retrieve interview questions from MongoDB")
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not retrieve saved interview questions from MongoDB: "
                f"{_safe_error_message(error)}"
            ),
        ) from error
    except (KeyError, TypeError, ValidationError) as error:
        logger.exception("Saved interview response has an invalid document shape")
        raise HTTPException(
            status_code=500,
            detail=(
                "Saved interview history contains an invalid response document: "
                f"{_safe_error_message(error)}"
            ),
        ) from error
    return questions


@handle_processing_errors("Retrieving saved interview questions", status_code=503)
def get_questions_from_history(
    request: InterviewRequest,
    request_validation_time: float,
) -> tuple[list[InterviewQuestion], ProcessInfo]:
    started_at = time.perf_counter()
    questions = _load_saved_questions(
        request.topic,
        request.experience_level,
        request.question_type,
        request.sub_topics,
    )

    if len(questions) < request.question_count:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Only {len(questions)} unique saved questions match these "
                f"selections; {request.question_count} are required. Generate "
                "more questions with a model or reduce the question count."
            ),
        )

    questions = [
        question.model_copy(update={"id": index})
        for index, question in enumerate(
            random.sample(questions, request.question_count),
            start=1,
        )
    ]

    history_lookup_time = time.perf_counter() - started_at
    return questions, ProcessInfo(
        model="Own knowledge base (saved history)",
        token_size=None,
        input_tokens=None,
        output_tokens=None,
        processed_time=0,
        request_validation_time=round(request_validation_time, 3),
        model_setup_time=0,
        prompt_preparation_time=0,
        llm_processing_time=0,
        response_processing_time=round(history_lookup_time, 3),
        api_processing_time=0,
        total_processing_time=0,
        tokens_per_second=None,
        estimated_cost_usd=None,
    )


@handle_processing_errors("Closing MongoDB connection")
def close_mongo_client() -> None:
    global _mongo_client, _mongo_client_uri

    with _mongo_client_lock:
        if _mongo_client is not None:
            _mongo_client.close()
            _mongo_client = None
            _mongo_client_uri = None


@app.get("/health")
@handle_processing_errors("Checking API health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/topics", response_model=dict[str, list[str]])
@handle_processing_errors("Loading interview topics")
def get_topics() -> dict[str, list[str]]:
    return load_topics()


@app.get("/api/models", response_model=list[dict[str, Union[str, bool]]])
@handle_processing_errors("Loading configured models")
def get_models() -> list[dict[str, Union[str, bool]]]:
    return get_model_options()


@app.get("/api/metrics", response_model=dict[str, Union[int, float, str]])
@handle_processing_errors("Collecting runtime metrics")
def get_runtime_metrics() -> dict[str, Union[int, float, str]]:
    return runtime_metrics.snapshot()


@handle_processing_errors("Generating interview questions")
def generate_questions(
    request: InterviewRequest,
    request_validation_time: float,
) -> tuple[list[InterviewQuestion], ProcessInfo]:
    """Hand the request to the interview agent (model choice, generation, validation)."""
    try:
        return run_generation(request, request_validation_time)
    except ModelNotFoundError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AgentError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@app.post("/api/interview/questions", response_model=InterviewResponse)
@handle_processing_errors("Processing interview request")
def create_interview_questions(
    request: InterviewRequest,
) -> InterviewResponse:
    request_started_at = time.perf_counter()
    topics = load_topics()
    if request.topic not in topics:
        raise HTTPException(status_code=422, detail="Select a topic from the available topics.")
    if (
        len(set(request.sub_topics)) != len(request.sub_topics)
        or any(sub_topic not in topics[request.topic] for sub_topic in request.sub_topics)
    ):
        raise HTTPException(
            status_code=422,
            detail="Select unique subtopics that belong to the selected topic.",
        )

    if request.model == KNOWLEDGE_BASE_MODEL_ID:
        request_validation_time = time.perf_counter() - request_started_at
        questions, process_info = get_questions_from_history(
            request,
            request_validation_time,
        )
    else:
        request_validation_time = time.perf_counter() - request_started_at
        questions, process_info = generate_questions(
            request,
            request_validation_time,
        )
    response = InterviewResponse(
        topic=request.topic,
        sub_topics=request.sub_topics,
        experience_level=request.experience_level,
        question_type=request.question_type,
        question_count=request.question_count,
        questions=questions,
        process_info=process_info,
    )
    total_processing_time = time.perf_counter() - request_started_at
    llm_processing_time = process_info.llm_processing_time
    response.process_info = process_info.model_copy(
        update={
            "total_processing_time": round(total_processing_time, 3),
            "api_processing_time": round(
                max(total_processing_time - llm_processing_time, 0), 3
            ),
        }
    )
    if request.model != KNOWLEDGE_BASE_MODEL_ID:
        _save_interview_response(response)
    return response


@app.patch(
    "/api/questions/{question_id}/status",
    response_model=QuestionStatusResponse,
)
@handle_processing_errors("Updating question status", status_code=503)
def update_question_status(
    question_id: str,
    update: QuestionStatusUpdate,
) -> QuestionStatusResponse:
    try:
        result = _get_mongo_collection().update_one(
            {"questions.question_id": question_id},
            {"$set": {"questions.$.status": update.status}},
        )
    except PyMongoError as error:
        logger.exception("Failed to update question status in MongoDB")
        raise HTTPException(
            status_code=503,
            detail=f"Could not update the question status: {_safe_error_message(error)}",
        ) from error
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Saved question not found.")
    return QuestionStatusResponse(question_id=question_id, status=update.status)


@app.post("/api/questions/saved", response_model=SavedQuestionsPage)
@handle_processing_errors("Listing saved questions", status_code=503)
def list_saved_questions(request: SavedQuestionsRequest) -> SavedQuestionsPage:
    questions = _load_saved_questions(
        request.topic,
        request.experience_level,
        request.question_type,
        request.sub_topics,
    )
    return _paginate(questions, request.page, request.page_size)


def _paginate(
    questions: list[InterviewQuestion], page: int, page_size: int
) -> SavedQuestionsPage:
    total = len(questions)
    total_pages = max(1, -(-total // page_size))  # ceiling division
    page = min(page, total_pages)  # a page past the end shows the last page
    start = (page - 1) * page_size
    return SavedQuestionsPage(
        total=total,
        checked_total=sum(question.status == "Checked" for question in questions),
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        questions=questions[start : start + page_size],
    )


@app.post("/api/ask", response_model=AskResponse)
@handle_processing_errors("Answering user question", status_code=503)
def ask_question(request: AskRequest) -> AskResponse:
    """The agent answers the user's own question (choosing the model unless one is given)."""
    started_at = time.perf_counter()
    question_text = request.question.strip()
    if len(question_text) < 3:
        raise HTTPException(status_code=422, detail="Type a question first.")
    try:
        answer, process_info = run_ask(
            question_text, request.model, time.perf_counter() - started_at
        )
    except ModelNotFoundError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AgentError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    total_processing_time = time.perf_counter() - started_at
    process_info = process_info.model_copy(
        update={
            "total_processing_time": round(total_processing_time, 3),
            "api_processing_time": round(
                max(total_processing_time - process_info.llm_processing_time, 0), 3
            ),
        }
    )
    question = InterviewQuestion(
        id=1,
        category="Your question",
        sub_topic="Asked by you",
        text=question_text,
        answer=answer,
    )
    try:
        _get_mongo_collection().insert_one(
            {
                "kind": USER_QUESTION_KIND,
                # topic / level / type give the agent context when this is refined later
                "topic": "Your own question",
                "experience_level": "",
                "question_type": "General",
                "questions": [question.model_dump(mode="json")],
                "process_info": process_info.model_dump(mode="json"),
            }
        )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not save your question: {_safe_error_message(error)}",
        ) from error
    return AskResponse(question=question, process_info=process_info)


@app.get("/api/user-questions", response_model=SavedQuestionsPage)
@handle_processing_errors("Listing user questions", status_code=503)
def list_user_questions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=50),
) -> SavedQuestionsPage:
    """All questions the user asked, newest first, one page at a time."""
    try:
        documents = (
            _get_mongo_collection()
            .find({"kind": USER_QUESTION_KIND}, {"questions": 1})
            .sort("_id", -1)
        )
        questions = []
        for document in documents:
            for saved_question in document.get("questions", []):
                questions.append(
                    InterviewQuestion.model_validate(
                        {**saved_question, "id": len(questions) + 1}
                    )
                )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load your questions: {_safe_error_message(error)}",
        ) from error
    return _paginate(questions, page, page_size)


@app.post("/api/questions/{question_id}/refine", response_model=RefineResponse)
@handle_processing_errors("Refining question", status_code=503)
def refine_question(question_id: str, request: RefineRequest) -> RefineResponse:
    """The agent rewrites a saved question (or its answer) in easier words and saves it."""
    field = "refined_text" if request.target == "question" else "refined_answer"
    collection = _get_mongo_collection()
    try:
        document = collection.find_one(
            {"questions.question_id": question_id},
            {"topic": 1, "experience_level": 1, "question_type": 1, "questions.$": 1},
        )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load the question: {_safe_error_message(error)}",
        ) from error
    if not document or not document.get("questions"):
        raise HTTPException(status_code=404, detail="Saved question not found.")

    question = {
        **document["questions"][0],
        "topic": document.get("topic", ""),
        "experience_level": document.get("experience_level", ""),
        "question_type": document.get("question_type", ""),
    }
    try:
        refined = run_refine(request.target, question)
    except AgentError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    try:
        collection.update_one(
            {"questions.question_id": question_id},
            {"$set": {f"questions.$.{field}": refined}},
        )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not save the refined text: {_safe_error_message(error)}",
        ) from error
    return RefineResponse(question_id=question_id, target=request.target, refined=refined)


@app.post("/api/quick-topics", response_model=QuickTopicResponse)
@handle_processing_errors("Creating quick topic", status_code=503)
def create_quick_topic(request: QuickTopicRequest) -> QuickTopicResponse:
    """The agent explains a topic as a box of bullet sections; the box is saved."""
    started_at = time.perf_counter()
    topic_text = request.topic.strip()
    if len(topic_text) < 2:
        raise HTTPException(status_code=422, detail="Type a topic first.")
    try:
        card, process_info = run_quick_topic(
            topic_text, request.model, time.perf_counter() - started_at
        )
    except ModelNotFoundError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except AgentError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error

    total_processing_time = time.perf_counter() - started_at
    process_info = process_info.model_copy(
        update={
            "total_processing_time": round(total_processing_time, 3),
            "api_processing_time": round(
                max(total_processing_time - process_info.llm_processing_time, 0), 3
            ),
        }
    )
    topic = QuickTopic(query=topic_text, **card.model_dump())
    try:
        _get_mongo_collection().insert_one(
            {
                "kind": QUICK_TOPIC_KIND,
                "topic": topic.model_dump(mode="json"),
                "process_info": process_info.model_dump(mode="json"),
            }
        )
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not save the topic: {_safe_error_message(error)}",
        ) from error
    return QuickTopicResponse(topic=topic, process_info=process_info)


@app.get("/api/quick-topics", response_model=QuickTopicPage)
@handle_processing_errors("Listing quick topics", status_code=503)
def list_quick_topics(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=5, ge=1, le=50),
) -> QuickTopicPage:
    """Saved Quick Topic boxes, newest first, one page at a time."""
    try:
        documents = (
            _get_mongo_collection()
            .find({"kind": QUICK_TOPIC_KIND}, {"topic": 1})
            .sort("_id", -1)
        )
        topics = [QuickTopic.model_validate(document["topic"]) for document in documents]
    except PyMongoError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load your topics: {_safe_error_message(error)}",
        ) from error
    total = len(topics)
    total_pages = max(1, -(-total // page_size))  # ceiling division
    page = min(page, total_pages)  # a page past the end shows the last page
    start = (page - 1) * page_size
    return QuickTopicPage(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        topics=topics[start : start + page_size],
    )

