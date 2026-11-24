"""Data models and errors shared by the API (main.py) and the interview agent."""

import os
import re
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

ExperienceLevel = Literal["Beginner", "Intermediate", "Proficient", "Advanced", "Expert"]
QuestionType = Literal["General", "Code problem", "Scenario problem"]
# Set by the user once they have read and understood a question.
QuestionStatus = Literal["Unchecked", "Checked"]
DEFAULT_QUESTION_STATUS: QuestionStatus = "Unchecked"
# What the agent's "Refine" task rewrites.
RefineTarget = Literal["question", "answer"]


class GeneratedQuestion(BaseModel):
    category: str = Field(min_length=1)
    sub_topic: str = Field(min_length=1)
    text: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class GeneratedInterview(BaseModel):
    questions: list[GeneratedQuestion] = Field(min_length=1)


class ModelChoice(BaseModel):
    """The agent's model-selection decision."""

    model_id: str = Field(min_length=1, description="id of one of the available models")
    reason: str = Field(min_length=1, description="one short sentence explaining the choice")


class TopicCard(BaseModel):
    """A Quick Topic box written by the agent: every section is a list of one-line bullets."""

    title: str = Field(
        min_length=1, description="short topic name, with the full form of any acronym"
    )
    overview: list[str] = Field(default_factory=list, description="what it is and why it is used")
    methodologies: list[str] = Field(
        default_factory=list, description="how it works: key techniques, steps or methods"
    )
    types: list[str] = Field(default_factory=list, description="kinds or variants, if any")
    other_technical: list[str] = Field(
        default_factory=list,
        description="other technical points if relevant: components, tools, key terms, pros/cons",
    )
    example: list[str] = Field(
        default_factory=list, description="one small example in 1-3 bullets"
    )


class QuickTopic(TopicCard):
    # permanent id, stored in MongoDB
    topic_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str  # what the user typed


class InterviewRequest(BaseModel):
    topic: str = Field(min_length=1)
    sub_topics: list[str] = Field(min_length=1)
    experience_level: ExperienceLevel
    question_type: QuestionType
    question_count: int = Field(default=5, ge=1, le=5)
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)


def _new_question_id() -> str:
    return uuid.uuid4().hex


class InterviewQuestion(BaseModel):
    id: int  # position in this response (1, 2, ...)
    question_id: str = Field(default_factory=_new_question_id)  # permanent, stored in MongoDB
    category: str
    sub_topic: str
    text: str
    answer: str
    status: QuestionStatus = DEFAULT_QUESTION_STATUS
    # Easier versions written by the agent's "Refine" task; the originals are kept.
    refined_text: Optional[str] = None
    refined_answer: Optional[str] = None


class AgentError(Exception):
    """A failure inside the agent, with the HTTP status the API should return."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def safe_error_message(error: Exception) -> str:
    """Error text with database credentials and API keys removed."""
    message = str(error).strip() or type(error).__name__
    message = re.sub(
        r"(mongodb(?:\+srv)?://)[^/@\s]+@",
        r"\1[REDACTED]@",
        message,
        flags=re.IGNORECASE,
    )
    for name, value in os.environ.items():
        if value and ("API_KEY" in name or "PASSWORD" in name):
            message = message.replace(value, "[REDACTED]")
    return message
