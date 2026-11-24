import json
import logging
import re
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pymongo.errors import PyMongoError

from smart_interview.api import main as interview_api
from smart_interview.api import model_selection
from smart_interview.api.main import app, load_topics
from smart_interview.api.prompt_interview import (
    AGENT_SCOPE,
    ASK_ANSWER_PROMPT,
    INTERVIEW_PROMPT,
    MODEL_SELECTION_PROMPT,
    REFINE_ANSWER_PROMPT,
    REFINE_QUESTION_PROMPT,
)
from smart_interview.api.progress_info import (
    RuntimeMetrics,
    _process_info,
    runtime_metrics,
)

client = TestClient(app)


def _rule_based_pick(prompt: str) -> str:
    question_type = re.search(r"Question type: (.+)", prompt).group(1).strip()
    return "gemini" if question_type == "General" else "claude-sonnet"


# Controls and records the fake LLM's agent calls (model choice and refine).
# "choice": model id the fake agent picks, a function(prompt) -> id, or an Exception.
AGENT: dict[str, object] = {}
PYTHON_SUBTOPIC = "Python fundamentals and data types"
PYTHON_OOP_SUBTOPIC = "Object-oriented programming"


def make_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "topic": "Python",
        "sub_topics": [PYTHON_SUBTOPIC],
        "experience_level": "Intermediate",
        "question_type": "General",
    }
    request.update(overrides)
    return request


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, object], str]]:
    calls: list[tuple[dict[str, object], str]] = []
    AGENT.clear()
    AGENT.update(
        {"choice": _rule_based_pick, "router": [], "refine": [], "topic": [], "topic_card": None}
    )

    class FakeMongoCollection:
        def insert_one(self, document: dict[str, object]) -> None:
            assert isinstance(document, dict)

    class FakeMongoDatabase:
        def __getitem__(self, collection_name: str) -> FakeMongoCollection:
            assert collection_name == "interview_responses"
            return FakeMongoCollection()

    class FakeMongoClient:
        def __init__(self, uri: str) -> None:
            assert uri == "mongodb://test-host:27017"

        def __getitem__(self, database_name: str) -> FakeMongoDatabase:
            assert database_name == "test_interviews"
            return FakeMongoDatabase()

        def close(self) -> None:
            pass

    class FakeChatOpenAI:
        def __init__(
            self,
            *,
            model: str,
            temperature: float,
            api_key: str,
            base_url: Optional[str] = None,
            max_tokens: Optional[int] = None,
        ) -> None:
            self.config = {
                "model": model,
                "temperature": temperature,
                "api_key": api_key,
            }
            if base_url:
                self.config["base_url"] = base_url
            if max_tokens is not None:
                self.config["max_tokens"] = max_tokens

        def invoke(self, prompt_value) -> AIMessage:  # plain call, used by "Refine"
            AGENT["refine"].append((dict(self.config), prompt_value.to_string()))
            return AIMessage(content="  Simpler version from the agent.  ")

        def with_structured_output(
            self,
            schema: type,
            *,
            method: str = "function_calling",
            include_raw: bool = False,
        ) -> RunnableLambda:
            return self._structured_output(schema, method, include_raw)

        def _structured_output(
            self, schema: type, method: str, include_raw: bool
        ) -> RunnableLambda:
            self.config["structured_output_method"] = method

            def generate(prompt_value):
                prompt = prompt_value.to_string()
                if schema.__name__ == "ModelChoice":  # the agent choosing a model
                    AGENT["router"].append((dict(self.config), prompt))
                    pick = AGENT["choice"]
                    if isinstance(pick, Exception):
                        raise pick
                    pick = pick(prompt) if callable(pick) else pick
                    parsed_choice = schema(model_id=pick, reason="Fake agent reason.")
                    if include_raw:
                        return {
                            "raw": AIMessage(content=""),
                            "parsed": parsed_choice,
                            "parsing_error": None,
                        }
                    return parsed_choice
                if schema.__name__ == "TopicCard":  # the agent writing a Quick Topic box
                    AGENT["topic"].append((dict(self.config), prompt))
                    card = AGENT["topic_card"] or schema(
                        title="Retrieval-Augmented Generation (RAG)",
                        overview=["Gives an LLM facts it looks up first."],
                        methodologies=["Embed documents, retrieve the closest, add to prompt."],
                        types=["Naive RAG", "Advanced RAG"],
                        other_technical=["Vector database"],
                        example=["Chatbot answers from company PDFs."],
                    )
                    return {"raw": AIMessage(content=""), "parsed": card, "parsing_error": None}
                calls.append((self.config, prompt))
                topic = re.search(r"Topic: (.+)", prompt).group(1)
                selected_subtopics = json.loads(
                    re.search(r"Selected subtopics \(JSON array\): (.+)", prompt).group(1)
                )
                experience = re.search(
                    r"Experience level: (\w+)", prompt
                ).group(1)
                question_type = re.search(
                    r"Question type: (.+)", prompt
                ).group(1)
                count = int(
                    re.search(r"Number of questions: (.+)", prompt).group(1)
                )
                category_by_type = {
                    "General": "experience",
                    "Code problem": "implementation",
                    "Scenario problem": "system-design",
                }
                generated = schema(
                    questions=[
                        {
                            "category": category_by_type[question_type],
                            "sub_topic": selected_subtopics[index % len(selected_subtopics)],
                            "text": (
                                f"Question about {selected_subtopics[index % len(selected_subtopics)]} "
                                f"in {topic} at {experience} level ({question_type})?"
                            ),
                            "answer": (
                                f"Suggested {question_type} answer for "
                                f"{selected_subtopics[index % len(selected_subtopics)]}."
                            ),
                        }
                        for index in range(count)
                    ]
                )
                if method == "json_mode":
                    parsed = generated.model_dump()
                else:
                    parsed = generated
                if include_raw:
                    return {
                        "raw": AIMessage(
                            content="",
                            usage_metadata={
                                "input_tokens": 24,
                                "output_tokens": 16,
                                "total_tokens": 40,
                            },
                            response_metadata={
                                "token_usage": {
                                    "prompt_tokens": 24,
                                    "completion_tokens": 16,
                                    "total_tokens": 40,
                                    "cost": 0.0001,
                                }
                            },
                        ),
                        "parsed": parsed,
                        "parsing_error": None,
                    }
                return parsed

            return RunnableLambda(generate)

    class FakeChatGoogleGenerativeAI(FakeChatOpenAI):
        def __init__(
            self, *, model: str, temperature: float, google_api_key: str
        ) -> None:
            super().__init__(
                model=model,
                temperature=temperature,
                api_key=google_api_key,
            )
            self.config["google_api_key"] = self.config.pop("api_key")

    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-api-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-api-key")
    monkeypatch.setenv("MONGODB_URI", "mongodb://test-host:27017")
    monkeypatch.setenv("MONGODB_DATABASE", "test_interviews")
    monkeypatch.setenv("MONGODB_COLLECTION", "interview_responses")
    monkeypatch.setattr(model_selection, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        model_selection, "ChatGoogleGenerativeAI", FakeChatGoogleGenerativeAI
    )
    monkeypatch.setattr(interview_api, "MongoClient", FakeMongoClient)
    monkeypatch.setattr(interview_api, "_mongo_client", None)
    monkeypatch.setattr(interview_api, "_mongo_client_uri", None)
    return calls


def test_health_check() -> None:
    response = client.get("/health")
    metrics_response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert metrics_response.status_code == 200
    assert metrics_response.json()["process_checking_api_health_run_count"] >= 1
    assert metrics_response.json()["process_checking_api_health_success_count"] >= 1


def test_runtime_metrics_track_requests_and_latency_percentiles() -> None:
    metrics = RuntimeMetrics()
    metrics.request_started()
    metrics.request_started()
    metrics.request_finished(200, 1.0)
    metrics.request_finished(500, 3.0)
    metrics.request_started()
    metrics.request_finished(200, 5.0)

    snapshot = metrics.snapshot()

    assert snapshot["request_count_since_start"] == 3
    assert snapshot["server_error_count_since_start"] == 1
    assert snapshot["server_error_rate_percent_since_start"] == pytest.approx(33.33)
    assert snapshot["in_flight_requests"] == 0
    assert snapshot["max_concurrent_requests_observed"] == 2
    assert snapshot["completed_latency_sample_size"] == 3
    assert snapshot["average_latency_seconds_last_sample"] == 3.0
    assert snapshot["p50_latency_seconds_last_sample"] == 3.0
    assert snapshot["p95_latency_seconds_last_sample"] == 5.0
    assert snapshot["requests_last_60_seconds"] == 3
    assert snapshot["observed_requests_per_second_last_60_seconds"] == 0.05


def test_runtime_metrics_track_process_runs_and_failures() -> None:
    metrics = RuntimeMetrics()
    metrics.record_process("Loading interview topics", True, 0.2)
    metrics.record_process("Loading interview topics", False, 0.4)

    snapshot = metrics.snapshot()

    assert snapshot["process_loading_interview_topics_run_count"] == 2
    assert snapshot["process_loading_interview_topics_success_count"] == 1
    assert snapshot["process_loading_interview_topics_failure_count"] == 1
    assert snapshot["process_loading_interview_topics_average_elapsed_seconds"] == 0.3
    assert snapshot["process_loading_interview_topics_last_elapsed_seconds"] == 0.4
    assert snapshot["process_loading_interview_topics_max_elapsed_seconds"] == 0.4


def test_processing_decorator_logs_and_monitors_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    metrics = RuntimeMetrics()
    monkeypatch.setattr(interview_api, "runtime_metrics", metrics)

    @interview_api.handle_processing_errors("Test process")
    def succeeds() -> str:
        return "complete"

    @interview_api.handle_processing_errors("Failing test process")
    def fails() -> None:
        raise ValueError("bad processing input")

    with caplog.at_level(logging.INFO):
        assert succeeds() == "complete"
        with pytest.raises(interview_api.HTTPException) as error:
            fails()

    snapshot = metrics.snapshot()
    assert error.value.status_code == 500
    assert error.value.detail == "Failing test process failed: bad processing input"
    assert "Test process completed in" in caplog.text
    assert "Failing test process failed after" in caplog.text
    assert snapshot["process_test_process_success_count"] == 1
    assert snapshot["process_failing_test_process_failure_count"] == 1


def test_metrics_endpoint_does_not_count_itself() -> None:
    request_count_before = runtime_metrics.request_count

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["request_count_since_start"] == request_count_before
    assert runtime_metrics.request_count == request_count_before


def _prompt_text(prompt) -> str:
    return " ".join(
        message.prompt.template
        for message in prompt.messages
        if hasattr(message.prompt, "template")
    )


def test_interview_prompt_is_defined_in_prompt_module() -> None:
    assert INTERVIEW_PROMPT.input_variables == [
        "experience_level",
        "feedback",
        "question_count",
        "question_type",
        "sub_topics",
        "topic",
    ]
    prompt_text = _prompt_text(INTERVIEW_PROMPT)
    assert "2-3 concise lines" in prompt_text
    assert "fenced code block with the language identified" in prompt_text
    assert "one-line bullet points only, with no extra details" in prompt_text
    assert "at most 3 one-line bullets" in prompt_text  # Code problem answers


def test_agent_scope_is_shared_by_every_agent_prompt() -> None:
    for name in ("Scope:", "Default helper tasks:", "TCS", "Infosys", "HCL"):
        assert name in AGENT_SCOPE
    assert "Match the selected experience level exactly" in AGENT_SCOPE
    # One-line bullets for everything except coding; details only when asked for.
    assert "Answer format for every question except coding questions" in AGENT_SCOPE
    assert "each bullet is a single line" in AGENT_SCOPE
    assert "no introduction, no conclusion, no extra details" in AGENT_SCOPE
    assert "Coding questions" in AGENT_SCOPE and "at most 3 one-line bullets" in AGENT_SCOPE
    assert "Details only on request" in AGENT_SCOPE
    assert "unless the user explicitly asks for one" in AGENT_SCOPE  # examples / diagrams
    assert "explicitly asks for detail" in _prompt_text(ASK_ANSWER_PROMPT)
    for prompt in (
        INTERVIEW_PROMPT,
        MODEL_SELECTION_PROMPT,
        REFINE_QUESTION_PROMPT,
        REFINE_ANSWER_PROMPT,
    ):
        assert AGENT_SCOPE in _prompt_text(prompt)


def test_topics_are_loaded_from_json_file() -> None:
    topics = load_topics()

    assert list(topics) == [
        "Python",
        "Advanced Python",
        "LLM",
        "Generative AI",
        "Artificial Intelligence",
        "RAG",
        "NLP",
        "Transformer",
        "AI Evals",
    ]
    assert all(sub_topics for sub_topics in topics.values())
    assert all(len(sub_topics) == len(set(sub_topics)) for sub_topics in topics.values())


def test_get_topics() -> None:
    response = client.get("/api/topics")

    assert response.status_code == 200
    assert response.json() == load_topics()


def test_topic_loading_failure_returns_processing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topics_file = tmp_path / "invalid-topics.json"
    topics_file.write_text("{", encoding="utf-8")
    monkeypatch.setattr(interview_api, "TOPICS_FILE", topics_file)

    response = client.get("/api/topics")

    assert response.status_code == 500
    assert "Loading interview topics failed" in response.json()["detail"]
    assert "Expecting property name" in response.json()["detail"]


def test_get_models_lists_configured_models() -> None:
    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "auto",
            "name": "Auto (AI agent chooses the model)",
            "default": False,
        },
        {
            "id": "knowledge_base",
            "name": "Own knowledge base (saved history)",
            "default": True,
        },
        {
            "id": "gemini",
            "name": "Gemini 2.5 Flash-Lite (OpenRouter)",
            "default": False,
        },
        {
            "id": "openai",
            "name": "OpenAI GPT-4o mini (OpenRouter)",
            "default": False,
        },
        {
            "id": "claude-sonnet",
            "name": "Claude Sonnet 4.6 (OpenRouter)",
            "default": False,
        },
    ]


def test_additional_model_is_loaded_from_models_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_openai: list[tuple[dict[str, object], str]],
) -> None:
    models_file = tmp_path / "models.json"
    models = json.loads(model_selection.MODELS_FILE.read_text(encoding="utf-8"))
    models.append(
        {
            "id": "gemini-pro",
            "name": "Gemini Pro",
            "provider": "gemini",
            "model_name": "gemini-2.5-pro",
            "api_key_env": "GEMINI_API_KEY",
            "temperature": 0.3,
            "structured_output_method": "json_mode",
            "default": False,
        }
    )
    models_file.write_text(json.dumps(models), encoding="utf-8")
    monkeypatch.setattr(model_selection, "MODELS_FILE", models_file)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    response = client.post(
        "/api/interview/questions",
        json=make_request(model="gemini-pro"),
    )

    assert response.status_code == 200
    assert any(item["id"] == "gemini-pro" for item in client.get("/api/models").json())
    assert mock_openai[-1][0]["model"] == "gemini-2.5-pro"


def test_topics_and_questions_use_updated_json_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topics_file = tmp_path / "topics.json"
    topics_file.write_text(
        json.dumps({"Custom Topic": ["Custom Subtopic"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(interview_api, "TOPICS_FILE", topics_file)

    topics_response = client.get("/api/topics")
    questions_response = client.post(
        "/api/interview/questions",
        json={
            "topic": "Custom Topic",
            "sub_topics": ["Custom Subtopic"],
            "experience_level": "Beginner",
            "question_type": "General",
        },
    )

    assert topics_response.status_code == 200
    assert topics_response.json() == {"Custom Topic": ["Custom Subtopic"]}
    assert questions_response.status_code == 200
    assert questions_response.json()["topic"] == "Custom Topic"
    assert questions_response.json()["sub_topics"] == ["Custom Subtopic"]


def test_create_interview_questions_and_answers_rotate_through_subtopics() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(
            sub_topics=[PYTHON_SUBTOPIC, PYTHON_OOP_SUBTOPIC],
            question_count=3,
        ),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["topic"] == "Python"
    assert result["sub_topics"] == [PYTHON_SUBTOPIC, PYTHON_OOP_SUBTOPIC]
    assert result["experience_level"] == "Intermediate"
    assert result["question_type"] == "General"
    assert result["question_count"] == 3
    assert result["process_info"]["model"] == "Gemini 2.5 Flash-Lite (OpenRouter)"
    assert result["process_info"]["token_size"] == 40
    assert result["process_info"]["input_tokens"] == 24
    assert result["process_info"]["output_tokens"] == 16
    assert result["process_info"]["processed_time"] >= 0
    assert result["process_info"]["request_validation_time"] >= 0
    assert result["process_info"]["model_setup_time"] >= 0
    assert result["process_info"]["prompt_preparation_time"] >= 0
    assert result["process_info"]["llm_processing_time"] >= 0
    assert result["process_info"]["response_processing_time"] >= 0
    assert result["process_info"]["api_processing_time"] >= 0
    assert result["process_info"]["total_processing_time"] >= (
        result["process_info"]["llm_processing_time"]
    )
    assert result["process_info"]["api_processing_time"] + result["process_info"][
        "llm_processing_time"
    ] == pytest.approx(result["process_info"]["total_processing_time"], abs=0.003)
    assert result["process_info"]["tokens_per_second"] is not None
    assert result["process_info"]["estimated_cost_usd"] == 0.0001
    assert len(result["questions"]) == 3
    assert [question["sub_topic"] for question in result["questions"]] == [
        PYTHON_SUBTOPIC,
        PYTHON_OOP_SUBTOPIC,
        PYTHON_SUBTOPIC,
    ]
    assert all(sub_topic in question["text"] for sub_topic, question in zip(
        [PYTHON_SUBTOPIC, PYTHON_OOP_SUBTOPIC, PYTHON_SUBTOPIC],
        result["questions"],
    ))
    assert "Intermediate level" in result["questions"][0]["text"]
    assert all(question["answer"] for question in result["questions"])


def test_knowledge_base_returns_matching_history_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
    mock_openai: list[tuple[dict[str, object], str]],
) -> None:
    saved_question = {
        "id": 7,
        "question_id": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
        "category": "experience",
        "sub_topic": PYTHON_SUBTOPIC,
        "text": "How do Python generators work?",
        "answer": "They produce values lazily using yield.",
        "status": "Checked",  # status saved earlier must come back unchanged
    }

    class FakeCursor:
        def sort(self, key: str, direction: int) -> "FakeCursor":
            assert key == "_id"
            assert direction == -1
            return self

        def __iter__(self):
            return iter([{"questions": [saved_question]}])

    class FakeCollection:
        def find(
            self,
            query: dict[str, object],
            projection: dict[str, int],
        ) -> FakeCursor:
            assert query == {
                "topic": "Python",
                "experience_level": "Intermediate",
                "question_type": "General",
                "sub_topics": {"$in": [PYTHON_SUBTOPIC]},
            }
            assert projection == {"questions": 1}
            return FakeCursor()

    monkeypatch.setattr(
        interview_api,
        "_get_mongo_collection",
        lambda: FakeCollection(),
    )
    monkeypatch.setattr(
        interview_api,
        "_save_interview_response",
        lambda response: pytest.fail("History retrieval should not save a duplicate"),
    )

    response = client.post(
        "/api/interview/questions",
        json=make_request(model="knowledge_base", question_count=1),
    )

    assert response.status_code == 200
    result = response.json()
    assert mock_openai == []
    assert result["questions"] == [
        {
            **saved_question,
            "id": 1,
            "refined_text": None,  # nothing refined yet
            "refined_answer": None,
        }
    ]
    assert result["process_info"]["model"] == "Own knowledge base (saved history)"
    assert result["process_info"]["llm_processing_time"] == 0
    assert result["process_info"]["token_size"] is None


def test_knowledge_base_randomly_selects_matching_questions_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_questions = [
        {
            "id": index,
            "category": "experience",
            "sub_topic": PYTHON_SUBTOPIC,
            "text": f"Saved question {index}",
            "answer": f"Saved answer {index}",
        }
        for index in range(1, 4)
    ]

    class FakeCursor:
        def sort(self, key: str, direction: int) -> "FakeCursor":
            return self

        def __iter__(self):
            return iter([{"questions": saved_questions}])

    class FakeCollection:
        def find(
            self,
            query: dict[str, object],
            projection: dict[str, int],
        ) -> FakeCursor:
            return FakeCursor()

    sample_call = 0
    sampled_candidate_counts: list[int] = []

    def choose_questions(
        population: list[interview_api.InterviewQuestion], count: int
    ) -> list[interview_api.InterviewQuestion]:
        nonlocal sample_call
        sample_call += 1
        sampled_candidate_counts.append(len(population))
        assert count == 2
        return (
            [population[2], population[0]]
            if sample_call == 1
            else [population[1], population[2]]
        )

    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: FakeCollection())
    monkeypatch.setattr(interview_api.random, "sample", choose_questions)

    results = [
        client.post(
            "/api/interview/questions",
            json=make_request(model="knowledge_base", question_count=2),
        )
        for _ in range(2)
    ]

    assert all(response.status_code == 200 for response in results)
    first_texts = [question["text"] for question in results[0].json()["questions"]]
    second_texts = [question["text"] for question in results[1].json()["questions"]]
    assert first_texts == ["Saved question 3", "Saved question 1"]
    assert second_texts == ["Saved question 2", "Saved question 3"]
    assert sampled_candidate_counts == [3, 3]


def test_knowledge_base_requires_enough_matching_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyCursor:
        def sort(self, key: str, direction: int) -> "EmptyCursor":
            return self

        def __iter__(self):
            return iter([])

    class EmptyCollection:
        def find(
            self,
            query: dict[str, object],
            projection: dict[str, int],
        ) -> EmptyCursor:
            return EmptyCursor()

    monkeypatch.setattr(
        interview_api,
        "_get_mongo_collection",
        lambda: EmptyCollection(),
    )

    response = client.post(
        "/api/interview/questions",
        json=make_request(model="knowledge_base", question_count=1),
    )

    assert response.status_code == 404
    assert "Only 0 unique saved questions" in response.json()["detail"]


def test_complete_interview_response_is_saved_to_mongodb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inserted_documents: list[dict[str, object]] = []

    class FakeCollection:
        def insert_one(self, document: dict[str, object]) -> None:
            inserted_documents.append(document)

    class FakeDatabase:
        def __getitem__(self, collection_name: str) -> FakeCollection:
            assert collection_name == "interview_responses"
            return FakeCollection()

    class FakeMongoClient:
        def __init__(self, uri: str) -> None:
            assert uri == "mongodb://test-host:27017"

        def __getitem__(self, database_name: str) -> FakeDatabase:
            assert database_name == "test_interviews"
            return FakeDatabase()

        def close(self) -> None:
            pass

    monkeypatch.setenv("MONGODB_URI", "mongodb://test-host:27017")
    monkeypatch.setenv("MONGODB_DATABASE", "test_interviews")
    monkeypatch.setenv("MONGODB_COLLECTION", "interview_responses")
    monkeypatch.setattr(interview_api, "MongoClient", FakeMongoClient)
    monkeypatch.setattr(interview_api, "_mongo_client", None)
    monkeypatch.setattr(interview_api, "_mongo_client_uri", None)

    response = client.post(
        "/api/interview/questions",
        json=make_request(question_count=1),
    )

    assert response.status_code == 200
    assert inserted_documents == [response.json()]
    assert "questions" in inserted_documents[0]
    assert "process_info" in inserted_documents[0]


def test_interview_response_is_not_returned_when_mongodb_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)
    monkeypatch.delenv("MONGODB_COLLECTION", raising=False)

    response = client.post(
        "/api/interview/questions",
        json=make_request(question_count=1),
    )

    assert response.status_code == 503
    assert "MONGODB_URI" in response.json()["detail"]


def test_mongodb_insert_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCollection:
        def insert_one(self, document: dict[str, object]) -> None:
            assert isinstance(document, dict)
            raise PyMongoError("MongoDB unavailable")

    monkeypatch.setattr(
        interview_api,
        "_get_mongo_collection",
        lambda: FailingCollection(),
    )

    response = client.post(
        "/api/interview/questions",
        json=make_request(question_count=1),
    )

    assert response.status_code == 503
    assert "Saving interview response to MongoDB failed" in response.json()["detail"]
    assert "MongoDB unavailable" in response.json()["detail"]


def test_process_info_marks_unreported_token_usage_unavailable() -> None:
    process_info = _process_info(
        model_selection.get_model_config("gemini"),
        AIMessage(content=""),
        0,
        0,
        0,
        0.25,
        0,
    )

    assert process_info.token_size is None
    assert process_info.input_tokens is None
    assert process_info.output_tokens is None
    assert process_info.processed_time == 0.25
    assert process_info.request_validation_time == 0
    assert process_info.model_setup_time == 0
    assert process_info.prompt_preparation_time == 0
    assert process_info.llm_processing_time == 0.25
    assert process_info.response_processing_time == 0
    assert process_info.tokens_per_second is None
    assert process_info.estimated_cost_usd is None


def test_reject_invalid_question_count() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(question_count=6),
    )

    assert response.status_code == 422


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_reject_temperature_outside_supported_range(temperature: float) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(temperature=temperature),
    )

    assert response.status_code == 422


def test_reject_missing_experience_level() -> None:
    request = make_request()
    request.pop("experience_level")

    response = client.post("/api/interview/questions", json=request)

    assert response.status_code == 422


def test_reject_unsupported_experience_level() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(experience_level="entry"),
    )

    assert response.status_code == 422


def test_reject_unsupported_topic() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(topic="Data Analyst", sub_topics=["Unlisted subtopic"]),
    )

    assert response.status_code == 422


def test_reject_missing_topic() -> None:
    request = make_request()
    request.pop("topic")

    response = client.post("/api/interview/questions", json=request)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("question_type", "expected_category"),
    [
        ("General", "experience"),
        ("Code problem", "implementation"),
        ("Scenario problem", "system-design"),
    ],
)
def test_generate_questions_by_type(
    question_type: str, expected_category: str
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(question_type=question_type, question_count=1),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["question_type"] == question_type
    assert result["questions"][0]["category"] == expected_category
    assert PYTHON_SUBTOPIC in result["questions"][0]["text"]


@pytest.mark.parametrize(
    ("question_type", "model_id", "expected_model_id", "expected_provider_model"),
    [
        ("General", None, "gemini", "google/gemini-2.5-flash-lite"),
        ("Code problem", None, "claude-sonnet", "anthropic/claude-sonnet-4.6"),
        ("Scenario problem", None, "claude-sonnet", "anthropic/claude-sonnet-4.6"),
        ("General", "auto", "gemini", "google/gemini-2.5-flash-lite"),
        ("Code problem", "auto", "claude-sonnet", "anthropic/claude-sonnet-4.6"),
        ("Scenario problem", "auto", "claude-sonnet", "anthropic/claude-sonnet-4.6"),
    ],
)
def test_auto_model_routes_by_question_type(
    question_type: str,
    model_id: Optional[str],
    expected_model_id: str,
    expected_provider_model: str,
    mock_openai: list[tuple[dict[str, object], str]],
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(model=model_id, question_type=question_type),
    )

    assert response.status_code == 200
    assert mock_openai[-1][0]["model"] == expected_provider_model
    assert response.json()["process_info"]["model"] == {
        "gemini": "Gemini 2.5 Flash-Lite (OpenRouter)",
        "claude-sonnet": "Claude Sonnet 4.6 (OpenRouter)",
    }[expected_model_id]


def test_llm_prompt_uses_every_interview_parameter(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(
            model="openai",
            topic="Advanced Python",
            sub_topics=["Decorators and closures", "Generators, iterators, and lazy evaluation"],
            experience_level="Advanced",
            question_type="Code problem",
            question_count=2,
        ),
    )

    assert response.status_code == 200
    config, prompt = mock_openai[-1]
    assert config == {
        "model": "openai/gpt-4o-mini",
        "temperature": 0.3,
        "api_key": "test-openrouter-api-key",
        "base_url": "https://openrouter.ai/api/v1",
        "structured_output_method": "function_calling",
    }
    for parameter in (
        "Advanced Python",
        "Decorators and closures",
        "Generators, iterators, and lazy evaluation",
        "Experience level: Advanced",
        "Code problem",
        "Number of questions: 2",
    ):
        assert parameter in prompt


def test_gemini_model_can_generate_questions(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(model="gemini"),
    )

    assert response.status_code == 200
    config, _ = mock_openai[-1]
    assert config == {
        "model": "google/gemini-2.5-flash-lite",
        "temperature": 0.3,
        "api_key": "test-openrouter-api-key",
        "base_url": "https://openrouter.ai/api/v1",
        "structured_output_method": "function_calling",
    }


def test_selected_temperature_is_used_by_model(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(model="gemini", temperature=0.7),
    )

    assert response.status_code == 200
    config, _ = mock_openai[-1]
    assert config["temperature"] == 0.7


def test_claude_model_can_generate_questions(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(model="claude-sonnet"),
    )

    assert response.status_code == 200
    config, _ = mock_openai[-1]
    assert config == {
        "model": "anthropic/claude-sonnet-4.6",
        "temperature": 0.3,
        "api_key": "test-openrouter-api-key",
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 2048,
        "structured_output_method": "function_calling",
    }


def test_default_model_is_gemini(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(),
    )

    assert response.status_code == 200
    config, _ = mock_openai[-1]
    assert config["model"] == "google/gemini-2.5-flash-lite"
    assert config["api_key"] == "test-openrouter-api-key"
    assert config["base_url"] == "https://openrouter.ai/api/v1"
    assert config["structured_output_method"] == "function_calling"


@pytest.mark.parametrize("model_id", ["gemini", "openai", "claude-sonnet"])
def test_missing_openrouter_key_returns_configuration_error(
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
    mock_openai: list[tuple[dict[str, object], str]],
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")

    response = client.post(
        "/api/interview/questions",
        json=make_request(model=model_id),
    )

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]
    assert mock_openai == []


def test_reject_unsupported_model() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(model="other"),
    )

    assert response.status_code == 422


def test_reject_missing_question_type() -> None:
    request = make_request()
    request.pop("question_type")

    response = client.post("/api/interview/questions", json=request)

    assert response.status_code == 422


def test_reject_unsupported_question_type() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(question_type="Behavioral"),
    )

    assert response.status_code == 422


def test_reject_subtopic_from_another_topic() -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(sub_topics=["Transformer training objectives"]),
    )

    assert response.status_code == 422


@pytest.mark.parametrize("sub_topics", [[], [PYTHON_SUBTOPIC, PYTHON_SUBTOPIC]])
def test_reject_empty_or_duplicate_subtopics(sub_topics: list[str]) -> None:
    response = client.post(
        "/api/interview/questions",
        json=make_request(sub_topics=sub_topics),
    )

    assert response.status_code == 422


def test_reject_missing_subtopics() -> None:
    request = make_request()
    request.pop("sub_topics")

    response = client.post("/api/interview/questions", json=request)

    assert response.status_code == 422


def test_generated_questions_get_ids_and_default_unchecked_status(
    mock_openai: list[tuple[dict[str, object], str]],
) -> None:
    response = client.post("/api/interview/questions", json=make_request(question_count=3))

    assert response.status_code == 200
    questions = response.json()["questions"]
    assert [question["status"] for question in questions] == ["Unchecked"] * 3
    ids = [question["question_id"] for question in questions]
    assert len(set(ids)) == 3
    assert all(re.fullmatch(r"[0-9a-f]{32}", question_id) for question_id in ids)


class FakeUpdateResult:
    def __init__(self, matched_count: int) -> None:
        self.matched_count = matched_count


@pytest.mark.parametrize("status", ["Checked", "Unchecked"])
def test_update_question_status(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    calls = []

    class FakeCollection:
        def update_one(self, query, update):
            calls.append((query, update))
            return FakeUpdateResult(matched_count=1)

    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: FakeCollection())

    response = client.patch("/api/questions/abc123/status", json={"status": status})

    assert response.status_code == 200
    assert response.json() == {"question_id": "abc123", "status": status}
    assert calls == [
        (
            {"questions.question_id": "abc123"},
            {"$set": {"questions.$.status": status}},
        )
    ]


def test_update_question_status_unknown_question(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCollection:
        def update_one(self, query, update):
            return FakeUpdateResult(matched_count=0)

    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: FakeCollection())

    response = client.patch("/api/questions/missing/status", json={"status": "Checked"})

    assert response.status_code == 404


def test_update_question_status_rejects_unknown_status() -> None:
    response = client.patch("/api/questions/abc123/status", json={"status": "Done"})

    assert response.status_code == 422



def _saved_listing_collection(questions: list[dict[str, object]]):
    class FakeCursor:
        def sort(self, key: str, direction: int) -> "FakeCursor":
            return self

        def __iter__(self):
            return iter([{"questions": questions}])

    class FakeCollection:
        def find(self, query, projection) -> FakeCursor:
            return FakeCursor()

    return FakeCollection()


def _saved(count: int, checked: int = 0) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "question_id": f"q{index}",
            "category": "experience",
            "sub_topic": PYTHON_SUBTOPIC,
            "text": f"Saved question {index}",
            "answer": f"Saved answer {index}",
            "status": "Checked" if index <= checked else "Unchecked",
        }
        for index in range(1, count + 1)
    ]


def _list_saved(page: int, page_size: int = 5):
    return client.post(
        "/api/questions/saved",
        json={
            "topic": "Python",
            "sub_topics": [PYTHON_SUBTOPIC],
            "experience_level": "Intermediate",
            "question_type": "General",
            "page": page,
            "page_size": page_size,
        },
    )


def test_saved_questions_are_paginated(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = _saved_listing_collection(_saved(12, checked=3))
    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: collection)

    first = _list_saved(page=1).json()
    assert (first["total"], first["total_pages"], first["page"]) == (12, 3, 1)
    assert first["checked_total"] == 3
    assert [q["question_id"] for q in first["questions"]] == ["q1", "q2", "q3", "q4", "q5"]
    assert [q["id"] for q in first["questions"]] == [1, 2, 3, 4, 5]

    last = _list_saved(page=3).json()
    assert [q["question_id"] for q in last["questions"]] == ["q11", "q12"]
    assert [q["id"] for q in last["questions"]] == [11, 12]

    past_end = _list_saved(page=9).json()
    assert past_end["page"] == 3
    assert [q["question_id"] for q in past_end["questions"]] == ["q11", "q12"]


def test_saved_questions_page_with_no_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = _saved_listing_collection([])
    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: collection)

    result = _list_saved(page=1).json()

    assert result == {
        "total": 0,
        "checked_total": 0,
        "page": 1,
        "page_size": 5,
        "total_pages": 1,
        "questions": [],
    }


def test_saved_questions_rejects_invalid_page() -> None:
    assert _list_saved(page=0).status_code == 422
    assert _list_saved(page=1, page_size=0).status_code == 422


# --- agent: model selection ----------------------------------------------------------


def test_agent_llm_chooses_the_model_for_auto(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    AGENT["choice"] = "openai"

    response = client.post(
        "/api/interview/questions",
        json=make_request(model="auto", question_type="Code problem", experience_level="Expert"),
    )

    assert response.status_code == 200
    assert mock_openai[-1][0]["model"] == "openai/gpt-4o-mini"  # the agent's pick was used
    selection = response.json()["process_info"]["model_selection"]
    assert selection.startswith("Chosen by AI agent: OpenAI GPT-4o mini (OpenRouter)")
    assert "Fake agent reason." in selection

    [(router_config, router_prompt)] = AGENT["router"]
    assert router_config["model"] == "google/gemini-2.5-flash-lite"  # the agent model
    assert router_config["temperature"] == 0
    for text in ("Question type: Code problem", "Experience level: Expert", '"claude-sonnet"'):
        assert text in router_prompt
    assert "Strongest reasoning and coding" in router_prompt  # model descriptions


@pytest.mark.parametrize("choice", ["not-a-model", RuntimeError("router down")])
def test_agent_falls_back_to_rules_when_its_choice_is_unusable(
    choice: object, mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    AGENT["choice"] = choice

    response = client.post(
        "/api/interview/questions",
        json=make_request(model="auto", question_type="Code problem"),
    )

    assert response.status_code == 200
    assert mock_openai[-1][0]["model"] == "anthropic/claude-sonnet-4.6"  # model_routing.json
    assert response.json()["process_info"]["model_selection"] == (
        "Rule-based fallback: Claude Sonnet 4.6 (OpenRouter)"
    )


def test_model_picked_by_user_skips_agent_selection(
    mock_openai: list[tuple[dict[str, object], str]]
) -> None:
    response = client.post("/api/interview/questions", json=make_request(model="openai"))

    assert response.status_code == 200
    assert AGENT["router"] == []
    assert response.json()["process_info"]["model_selection"] == (
        "Chosen by you: OpenAI GPT-4o mini (OpenRouter)"
    )


# --- agent: validation and retry -----------------------------------------------------


def _fake_generation_model(monkeypatch: pytest.MonkeyPatch, counts: list[int]):
    """Generation returns len(counts) answers in turn, each with that many questions."""
    from smart_interview.api import agent_interview
    from smart_interview.api.schemas import GeneratedInterview

    prompts: list[str] = []

    def fake_create(config, api_key, schema, temperature=None):
        def generate(prompt_value):
            prompts.append(prompt_value.to_string())
            count = counts[len(prompts) - 1]
            parsed = GeneratedInterview(
                questions=[
                    {
                        "category": "experience",
                        "sub_topic": PYTHON_SUBTOPIC,
                        "text": f"Question {index}?",
                        "answer": "- Answer bullet",
                    }
                    for index in range(count)
                ]
            )
            return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}

        return RunnableLambda(generate)

    monkeypatch.setattr(agent_interview, "create_structured_model", fake_create)
    return prompts


def test_agent_retries_once_with_feedback_after_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = _fake_generation_model(monkeypatch, counts=[1, 2])

    response = client.post(
        "/api/interview/questions", json=make_request(model="gemini", question_count=2)
    )

    assert response.status_code == 200
    assert len(response.json()["questions"]) == 2
    assert len(prompts) == 2
    assert "rejected" not in prompts[0]
    assert "it had 1 questions instead of exactly 2" in prompts[1]


def test_agent_gives_up_after_second_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _fake_generation_model(monkeypatch, counts=[1, 1])

    response = client.post(
        "/api/interview/questions", json=make_request(model="gemini", question_count=2)
    )

    assert response.status_code == 502
    assert len(prompts) == 2
    assert "instead of exactly 2" in response.json()["detail"]


# --- agent: refine -------------------------------------------------------------------


def _refine_collection(question: dict[str, object]):
    updates: list[tuple[dict, dict]] = []

    class FakeCollection:
        def find_one(self, query, projection):
            if query != {"questions.question_id": question["question_id"]}:
                return None
            return {
                "topic": "Python",
                "experience_level": "Beginner",
                "question_type": "General",
                "questions": [question],
            }

        def update_one(self, query, update):
            updates.append((query, update))

    return FakeCollection(), updates


@pytest.mark.parametrize(
    ("target", "field", "task_text"),
    [
        ("question", "refined_text", "Current task: refine a question."),
        ("answer", "refined_answer", "Current task: explain a suggested answer again."),
    ],
)
def test_refine_uses_agent_llm_and_saves_result(
    monkeypatch: pytest.MonkeyPatch, target: str, field: str, task_text: str
) -> None:
    question = {
        "question_id": "q1",
        "text": "Explain list comprehensions.",
        "answer": "- They build lists in one line.",
    }
    collection, updates = _refine_collection(question)
    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: collection)

    response = client.post("/api/questions/q1/refine", json={"target": target})

    assert response.status_code == 200
    assert response.json() == {
        "question_id": "q1",
        "target": target,
        "refined": "Simpler version from the agent.",
    }
    [(config, prompt)] = AGENT["refine"]
    assert config["model"] == "google/gemini-2.5-flash-lite"  # the agent model
    assert task_text in prompt
    assert "Explain list comprehensions." in prompt
    assert "Experience level: Beginner" in prompt
    assert "earlier simplified version" not in prompt
    assert updates == [
        (
            {"questions.question_id": "q1"},
            {"$set": {f"questions.$.{field}": "Simpler version from the agent."}},
        )
    ]


def test_refine_again_asks_for_a_different_version(monkeypatch: pytest.MonkeyPatch) -> None:
    question = {
        "question_id": "q1",
        "text": "Explain list comprehensions.",
        "answer": "- They build lists in one line.",
        "refined_answer": "- A short way to make a list.",
    }
    collection, _ = _refine_collection(question)
    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: collection)

    response = client.post("/api/questions/q1/refine", json={"target": "answer"})

    assert response.status_code == 200
    [(_, prompt)] = AGENT["refine"]
    assert "An earlier simplified version was:\n- A short way to make a list." in prompt
    assert "different and even easier" in prompt


def test_refine_unknown_question_and_bad_target(monkeypatch: pytest.MonkeyPatch) -> None:
    collection, _ = _refine_collection({"question_id": "q1", "text": "t", "answer": "a"})
    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: collection)

    assert client.post("/api/questions/missing/refine", json={"target": "question"}).status_code == 404
    assert client.post("/api/questions/q1/refine", json={"target": "everything"}).status_code == 422
    assert AGENT["refine"] == []


# --- agent: the user's own questions -------------------------------------------------


def _capture_inserts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    inserted: list[dict[str, object]] = []

    class FakeCollection:
        def insert_one(self, document):
            inserted.append(document)

    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: FakeCollection())
    return inserted


def test_ask_lets_the_agent_choose_the_model_and_saves_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inserted = _capture_inserts(monkeypatch)
    AGENT["choice"] = "openai"

    response = client.post(
        "/api/ask", json={"question": "  What is a REST API?  ", "model": "auto"}
    )

    assert response.status_code == 200
    result = response.json()
    question = result["question"]
    assert question["text"] == "What is a REST API?"
    assert question["answer"] == "Simpler version from the agent."
    assert question["status"] == "Unchecked"
    assert result["process_info"]["model"] == "OpenAI GPT-4o mini (OpenRouter)"
    assert result["process_info"]["model_selection"].startswith("Chosen by AI agent: OpenAI")

    [(_, router_prompt)] = AGENT["router"]
    assert "Candidate question:\nWhat is a REST API?" in router_prompt
    [(answer_config, answer_prompt)] = AGENT["refine"]  # plain (non-structured) model calls
    assert answer_config["model"] == "openai/gpt-4o-mini"
    assert "Current task: answer the candidate's own question." in answer_prompt

    [document] = inserted
    assert document["kind"] == "user_question"
    assert document["questions"][0]["question_id"] == question["question_id"]
    assert document["questions"][0]["answer"] == "Simpler version from the agent."


def test_ask_uses_the_model_the_user_picked(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_inserts(monkeypatch)

    response = client.post(
        "/api/ask", json={"question": "Explain SQL joins.", "model": "claude-sonnet"}
    )

    assert response.status_code == 200
    assert AGENT["router"] == []
    assert response.json()["process_info"]["model_selection"] == (
        "Chosen by you: Claude Sonnet 4.6 (OpenRouter)"
    )


@pytest.mark.parametrize("model", ["knowledge_base", None])
def test_ask_treats_knowledge_base_and_no_model_as_auto(
    monkeypatch: pytest.MonkeyPatch, model: Optional[str]
) -> None:
    _capture_inserts(monkeypatch)
    AGENT["choice"] = "gemini"

    response = client.post("/api/ask", json={"question": "What is Docker?", "model": model})

    assert response.status_code == 200
    assert len(AGENT["router"]) == 1
    assert response.json()["process_info"]["model"] == "Gemini 2.5 Flash-Lite (OpenRouter)"


def test_ask_falls_back_to_default_model_when_agent_choice_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_inserts(monkeypatch)
    AGENT["choice"] = RuntimeError("router down")

    response = client.post("/api/ask", json={"question": "What is Docker?"})

    assert response.status_code == 200
    assert response.json()["process_info"]["model_selection"] == (
        "Default model (fallback): Gemini 2.5 Flash-Lite (OpenRouter)"
    )


@pytest.mark.parametrize("question", ["", "  ", "hi"])
def test_ask_rejects_too_short_questions(question: str) -> None:
    response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    assert AGENT["refine"] == []


def test_user_questions_are_listed_newest_first_with_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        {"questions": [{**question, "category": "Your question"}]}
        for question in _saved(7, checked=2)
    ]
    queries = []

    class FakeCursor:
        def sort(self, key, direction):
            assert (key, direction) == ("_id", -1)
            return self

        def __iter__(self):
            return iter(documents)

    class FakeCollection:
        def find(self, query, projection):
            queries.append(query)
            return FakeCursor()

    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: FakeCollection())

    first = client.get("/api/user-questions").json()
    second = client.get("/api/user-questions", params={"page": 2}).json()

    assert queries[0] == {"kind": "user_question"}
    assert (first["total"], first["checked_total"], first["total_pages"]) == (7, 2, 2)
    assert [q["id"] for q in first["questions"]] == [1, 2, 3, 4, 5]
    assert [q["id"] for q in second["questions"]] == [6, 7]
    assert client.get("/api/user-questions", params={"page": 0}).status_code == 422


# --- agent: Quick Topic boxes --------------------------------------------------------


def test_quick_topic_prompt_asks_for_every_section_and_an_example() -> None:
    from smart_interview.api.prompt_interview import QUICK_TOPIC_PROMPT

    text = _prompt_text(QUICK_TOPIC_PROMPT)
    assert AGENT_SCOPE in text
    for part in ("methodologies", "types", "other_technical", "example", "one-line bullet"):
        assert part in text
    assert "always includes the small example" in text


def test_quick_topic_is_created_by_the_agent_and_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    inserted = _capture_inserts(monkeypatch)
    AGENT["choice"] = "openai"

    response = client.post("/api/quick-topics", json={"topic": " What is RAG? ", "model": "auto"})

    assert response.status_code == 200
    topic = response.json()["topic"]
    assert topic["query"] == "What is RAG?"
    assert topic["title"] == "Retrieval-Augmented Generation (RAG)"
    assert topic["types"] == ["Naive RAG", "Advanced RAG"]
    assert topic["example"] == ["Chatbot answers from company PDFs."]
    assert re.fullmatch(r"[0-9a-f]{32}", topic["topic_id"])
    assert response.json()["process_info"]["model_selection"].startswith(
        "Chosen by AI agent: OpenAI"
    )

    [(_, router_prompt)] = AGENT["router"]
    assert "Candidate question:\nWhat is RAG?" in router_prompt
    [(topic_config, topic_prompt)] = AGENT["topic"]
    assert topic_config["model"] == "openai/gpt-4o-mini"
    assert "Current task: write a Quick Topic box." in topic_prompt
    assert "Topic:\nWhat is RAG?" in topic_prompt

    [document] = inserted
    assert document["kind"] == "quick_topic"
    assert document["topic"]["topic_id"] == topic["topic_id"]


def test_quick_topic_uses_the_model_the_user_picked(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_inserts(monkeypatch)

    response = client.post("/api/quick-topics", json={"topic": "Kafka", "model": "gemini"})

    assert response.status_code == 200
    assert AGENT["router"] == []
    assert AGENT["topic"][0][0]["model"] == "google/gemini-2.5-flash-lite"


def test_empty_quick_topic_box_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from smart_interview.api.schemas import TopicCard

    inserted = _capture_inserts(monkeypatch)
    AGENT["topic_card"] = TopicCard(title="Kafka")  # no bullets at all

    response = client.post("/api/quick-topics", json={"topic": "Kafka", "model": "gemini"})

    assert response.status_code == 502
    assert inserted == []


@pytest.mark.parametrize("topic", ["", " ", "x"])
def test_quick_topic_rejects_too_short_topics(topic: str) -> None:
    assert client.post("/api/quick-topics", json={"topic": topic}).status_code == 422
    assert AGENT["topic"] == []


def test_quick_topics_are_listed_newest_first_with_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        {"topic": {"topic_id": f"t{i}", "query": f"Topic {i}", "title": f"Topic {i}"}}
        for i in range(1, 8)
    ]
    queries = []

    class FakeCursor:
        def sort(self, key, direction):
            assert (key, direction) == ("_id", -1)
            return self

        def __iter__(self):
            return iter(documents)

    class FakeCollection:
        def find(self, query, projection):
            queries.append(query)
            return FakeCursor()

    monkeypatch.setattr(interview_api, "_get_mongo_collection", lambda: FakeCollection())

    first = client.get("/api/quick-topics").json()
    last = client.get("/api/quick-topics", params={"page": 2}).json()

    assert queries[0] == {"kind": "quick_topic"}
    assert (first["total"], first["total_pages"]) == (7, 2)
    assert [t["topic_id"] for t in first["topics"]] == ["t1", "t2", "t3", "t4", "t5"]
    assert [t["topic_id"] for t in last["topics"]] == ["t6", "t7"]

