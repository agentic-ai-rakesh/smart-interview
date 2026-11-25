import html
import os
from typing import Optional

import requests
import streamlit as st

from progress_info import (
    fetch_runtime_metrics,
    render_current_metrics,
    render_overall_metrics,
)

API_URL = os.getenv("SMART_INTERVIEW_API_URL", "http://127.0.0.1:8000").rstrip("/")
GENERATION_TIMEOUT_SECONDS = 120
TEMPERATURE_OPTIONS = (0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
EXPERIENCE_LEVELS = ("Beginner", "Intermediate", "Proficient", "Advanced", "Expert")
DEFAULT_EXPERIENCE_LEVEL = "Advanced"
QUESTION_TYPES = ("General", "Code problem", "Scenario problem")
DEFAULT_QUESTION_TYPE = "Scenario problem"

DEFAULT_QUESTION_STATUS = "Unchecked"
QUESTIONS_PER_PAGE = 5
# status -> (button label, tooltip). Colors live in the CSS below (.q-status-*).
QUESTION_STATUS_BUTTONS = {
    "Unchecked": ("● Unchecked", "Click when you have read and understood this question"),
    "Checked": ("✔ Checked", "Click to mark this question as unchecked again"),
}


def api_error_detail(error: requests.HTTPError) -> str:
    response = error.response
    if response is None:
        return str(error)

    try:
        payload = response.json()
    except ValueError:
        return response.text or str(error)

    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return response.text or str(error)


def question_header_html(number: int, category: str, status: str) -> str:
    """Question title. Its q-status-* class colors the status button placed right after it."""
    return (
        f'<div class="q-title q-status-{status.lower()}" style="margin-top:0.5rem;">'
        f"<strong>{number}. {html.escape(category.title())}</strong></div>"
    )


def fetch_saved_questions_page(
    listing_filter: dict[str, object], page: int
) -> tuple[Optional[dict[str, object]], Optional[str]]:
    """One page of the saved questions matching the selection, or an error message."""
    try:
        response = requests.post(
            f"{API_URL}/api/questions/saved",
            json={**listing_filter, "page": page, "page_size": QUESTIONS_PER_PAGE},
            timeout=15,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        return None, f"Could not load saved questions: {api_error_detail(error)}"
    except requests.RequestException as error:
        return None, f"Could not reach the interview API at {API_URL}. Details: {error}"
    return response.json(), None


def refine_saved_question(question_id: str, target: str) -> Optional[str]:
    """Ask the agent to rewrite a question or its answer more simply. Returns an error or None."""
    try:
        response = requests.post(
            f"{API_URL}/api/questions/{question_id}/refine",
            json={"target": target},
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        return f"Could not refine the {target}: {api_error_detail(error)}"
    except requests.RequestException as error:
        return f"Could not reach the interview API at {API_URL}. Details: {error}"
    return None


def refine_button(question_id: str, target: str, key_prefix: str = "") -> None:
    """A "Refine" button; on click the agent rewrites the text and the page reloads."""
    label, tooltip, waiting = {
        "question": (
            "✨ Refine question",
            "Rewrite this question in simpler words",
            "The agent is simplifying the question…",
        ),
        "answer": (
            "✨ Refine answer",
            "Explain this answer again in a different, easier way",
            "The agent is explaining the answer more simply…",
        ),
    }[target]
    if st.button(label, key=f"{key_prefix}refine_{target}_{question_id}", help=tooltip):
        with st.spinner(waiting):
            error = refine_saved_question(question_id, target)
        if error:
            st.session_state["refine_error"] = error
        else:
            st.session_state.pop("refine_error", None)
        st.experimental_rerun()


def render_question_card(
    question: dict[str, object], answer_label: str = "Suggested answer", key_prefix: str = ""
) -> None:
    """One question: title, status pill, (refined) text, Refine buttons and the answer.

    key_prefix keeps widget keys unique when the same question is shown twice on a page.
    """
    status = question.get("status", DEFAULT_QUESTION_STATUS)
    st.markdown(
        question_header_html(question["id"], question["category"], status),
        unsafe_allow_html=True,
    )
    # The one status control: shows the status and switches it when clicked.
    # It must come right after the title so the CSS can color it.
    label, tooltip = QUESTION_STATUS_BUTTONS[status]
    st.button(
        label,
        key=f"{key_prefix}status_{question['question_id']}",
        help=tooltip,
        on_click=set_question_status,
        args=(
            question["question_id"],
            "Unchecked" if status == "Checked" else "Checked",
        ),
    )
    st.caption(f"Subtopic: {question['sub_topic']}")
    if question.get("refined_text"):
        st.caption("✨ Simplified by the AI agent")
        st.markdown(question["refined_text"])
        with st.expander("Original question"):
            st.markdown(question["text"])
    else:
        st.markdown(question["text"])
    refine_button(question["question_id"], "question", key_prefix)
    with st.expander(answer_label):
        if question.get("refined_answer"):
            st.caption("✨ Explained in an easier way by the AI agent")
            st.markdown(question["refined_answer"])
            st.markdown("---")
            st.caption("Original answer")
        st.markdown(question["answer"])
        refine_button(question["question_id"], "answer", key_prefix)


def ask_agent(question: str, model: str) -> tuple[Optional[dict[str, object]], Optional[str]]:
    """Send the user's own question to the agent. Returns (result, error message)."""
    try:
        response = requests.post(
            f"{API_URL}/api/ask",
            json={"question": question, "model": model},
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        return None, f"Could not answer your question: {api_error_detail(error)}"
    except requests.RequestException as error:
        return None, f"Could not reach the interview API at {API_URL}. Details: {error}"
    return response.json(), None


def fetch_user_questions_page(
    page: int, page_size: int = QUESTIONS_PER_PAGE
) -> tuple[Optional[dict[str, object]], Optional[str]]:
    """One page of the user's own questions (newest first), or an error message."""
    try:
        response = requests.get(
            f"{API_URL}/api/user-questions",
            params={"page": page, "page_size": page_size},
            timeout=15,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        return None, f"Could not load your questions: {api_error_detail(error)}"
    except requests.RequestException as error:
        return None, f"Could not reach the interview API at {API_URL}. Details: {error}"
    return response.json(), None


TOPIC_SECTIONS = (
    ("overview", "Overview"),
    ("methodologies", "Methodologies"),
    ("types", "Types"),
    ("other_technical", "Other technical details"),
    ("example", "Example"),
)


def topic_box_markdown(topic: dict[str, object]) -> str:
    """A Quick Topic box: title, then each non-empty section as bullet points."""
    parts = [f"#### 📦 {topic['title']}", f"_You asked: {topic['query']}_"]
    for key, label in TOPIC_SECTIONS:
        items = [str(item) for item in topic.get(key) or [] if str(item).strip()]
        if any(item.strip().startswith("```") for item in items):
            # The model sent a code block line by line: show it as text, not as bullets.
            parts.append(f"**{label}**\n\n" + "\n".join(items))
            continue
        bullets = [item.strip().lstrip("-*• ").strip() for item in items]
        if bullets:
            parts.append(f"**{label}**\n\n" + "\n".join(f"- {bullet}" for bullet in bullets))
    return "\n\n".join(parts)


def render_topic_box(topic: dict[str, object]) -> None:
    # The hidden marker lets the CSS draw the next element as a bordered box.
    st.markdown('<span class="topic-box-marker"></span>', unsafe_allow_html=True)
    st.markdown(topic_box_markdown(topic))


def create_quick_topic(
    topic: str, model: str
) -> tuple[Optional[dict[str, object]], Optional[str]]:
    """Ask the agent for a Quick Topic box. Returns (result, error message)."""
    try:
        response = requests.post(
            f"{API_URL}/api/quick-topics",
            json={"topic": topic, "model": model},
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        return None, f"Could not create the topic box: {api_error_detail(error)}"
    except requests.RequestException as error:
        return None, f"Could not reach the interview API at {API_URL}. Details: {error}"
    return response.json(), None


def fetch_quick_topics_page(page: int) -> tuple[Optional[dict[str, object]], Optional[str]]:
    """One page of saved Quick Topic boxes (newest first), or an error message."""
    try:
        response = requests.get(
            f"{API_URL}/api/quick-topics",
            params={"page": page, "page_size": QUESTIONS_PER_PAGE},
            timeout=15,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        return None, f"Could not load your topics: {api_error_detail(error)}"
    except requests.RequestException as error:
        return None, f"Could not reach the interview API at {API_URL}. Details: {error}"
    return response.json(), None


def change_listing_page(step: int, state_key: str = "listing_page") -> None:
    """Previous / Next button callback."""
    st.session_state[state_key] = max(1, st.session_state.get(state_key, 1) + step)


def render_pager(
    listing: dict[str, object],
    position: str,
    state_key: str = "listing_page",
    items_key: str = "questions",
    noun: str = "question",
) -> None:
    """Page info, then one row with Previous on the left and Next on the right.

    Streamlit 1.12 can't nest columns here, so the hidden .pager-marker lets the CSS place
    the two buttons that follow it on one row. Both are always drawn (greyed out at the
    first / last page) so the row never changes shape.
    """
    first = (listing["page"] - 1) * listing["page_size"] + 1
    last = first + len(listing[items_key]) - 1
    shown = f"{noun} {first}" if first == last else f"{noun}s {first}–{last}"
    st.caption(
        f"Page {listing['page']} of {listing['total_pages']} · "
        f"{shown} of {listing['total']}"
    )
    st.markdown('<span class="pager-marker"></span>', unsafe_allow_html=True)
    st.button(
        "⬅ Previous",
        key=f"{state_key}_previous_{position}",
        help="Previous page",
        on_click=change_listing_page,
        args=(-1, state_key),
        disabled=listing["page"] <= 1,
    )
    st.button(
        "Next ➡",
        key=f"{state_key}_next_{position}",
        help="Next page",
        on_click=change_listing_page,
        args=(1, state_key),
        disabled=listing["page"] >= listing["total_pages"],
    )


def set_question_status(question_id: str, status: str) -> None:
    """Button callback: save the new status. The page is reloaded from the API afterwards."""
    try:
        response = requests.patch(
            f"{API_URL}/api/questions/{question_id}/status",
            json={"status": status},
            timeout=15,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        st.session_state["status_error"] = (
            f"Could not update the question status: {api_error_detail(error)}"
        )
        return
    except requests.RequestException as error:
        st.session_state["status_error"] = (
            f"Could not reach the interview API at {API_URL}. Details: {error}"
        )
        return
    st.session_state.pop("status_error", None)


st.set_page_config(page_title="Smart Interview", page_icon="🎤", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: var(--primary-color) !important;
    }
    /* Status button: a yellow / green pill, colored by the title right before it. */
    div.element-container:has(.q-status-unchecked) + div.element-container .stButton button,
    div.element-container:has(.q-status-checked) + div.element-container .stButton button {
        border-radius: 999px;
        padding: 0.1rem 0.9rem;
        min-height: 0;
        font-size: 15px !important;
        font-weight: 600;
    }
    div.element-container:has(.q-status-unchecked) + div.element-container .stButton button {
        color: #92400E !important;
        background: #FEF3C7;
        border: 1px solid #F59E0B !important;
    }
    div.element-container:has(.q-status-unchecked) + div.element-container .stButton button:hover {
        background: #FDE68A;
    }
    div.element-container:has(.q-status-checked) + div.element-container .stButton button {
        color: #166534 !important;
        background: #DCFCE7;
        border: 1px solid #22C55E !important;
    }
    div.element-container:has(.q-status-checked) + div.element-container .stButton button:hover {
        background: #BBF7D0;
    }
    /* Pager row: hide the marker, then Previous on the left and Next pulled up beside it on the right.
       Both buttons get a fixed height so the pull-up (height + the 1rem gap) is exact. */
    div.element-container:has(.pager-marker) {
        display: none;
    }
    div.element-container:has(.pager-marker) + div.element-container,
    div.element-container:has(.pager-marker) + div.element-container + div.element-container {
        width: auto !important;
    }
    div.element-container:has(.pager-marker) + div.element-container .stButton,
    div.element-container:has(.pager-marker) + div.element-container + div.element-container .stButton {
        width: auto !important;  /* Streamlit sets a full-width inline style */
    }
    div.element-container:has(.pager-marker) + div.element-container button,
    div.element-container:has(.pager-marker) + div.element-container + div.element-container button {
        height: 2.5rem;
        min-height: 2.5rem;
        padding: 0 1.25rem;
        font-size: 1rem;
        font-weight: 600;
        color: #1D4ED8;
        background: #EFF6FF;
        border: 1px solid #93C5FD;
        border-radius: 999px;
    }
    div.element-container:has(.pager-marker) + div.element-container button:hover:enabled,
    div.element-container:has(.pager-marker) + div.element-container + div.element-container button:hover:enabled {
        color: #1E40AF;
        background: #DBEAFE;
        border-color: #3B82F6;
    }
    div.element-container:has(.pager-marker) + div.element-container button:disabled,
    div.element-container:has(.pager-marker) + div.element-container + div.element-container button:disabled {
        color: #9CA3AF;
        background: #F3F4F6;
        border-color: #E5E7EB;
    }
    div.element-container:has(.pager-marker) + div.element-container {
        align-self: flex-start;
    }
    div.element-container:has(.pager-marker) + div.element-container + div.element-container {
        align-self: flex-end;
        margin-top: calc(-2.5rem - 1rem);
    }
    /* Quick Topic box: the element after the hidden marker gets a border and background. */
    div.element-container:has(.topic-box-marker) {
        display: none;
    }
    div.element-container:has(.topic-box-marker) + div.element-container {
        box-sizing: border-box;
        width: 100% !important;
        overflow-wrap: anywhere;
        border: 1px solid #C7D2FE;
        border-left: 4px solid #6366F1;
        background: #F8FAFF;
        border-radius: 12px;
        padding: 0.75rem 1.1rem 0.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    div.element-container:has(.topic-box-marker) + div.element-container > div {
        width: 100% !important;  /* Streamlit sets a fixed inline width on the markdown */
    }
    /* Less empty space at the top: Streamlit adds 6rem above the page and a white header
       bar. Keep a small gap, make the bar transparent, and let clicks pass through it to
       the content below (only its menu button stays clickable). */
    .block-container {
        padding-top: 1rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        pointer-events: none;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"],
    header[data-testid="stHeader"] [data-testid="stToolbar"] * {
        pointer-events: auto;
    }
    /* Middle panel only (all tabs): larger, darker, easier-to-read text.
       The left and right panels keep Streamlit's default sizes. */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stMarkdownContainer"] p,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stMarkdownContainer"] li {
        font-size: 18px;
        line-height: 1.65;
        color: #111827;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stMarkdownContainer"] li {
        margin-bottom: 0.25rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stMarkdownContainer"] code {
        font-size: 0.95em;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stCaptionContainer"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stCaptionContainer"] p {
        font-size: 16px !important;
        color: #374151 !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) h3 {
        font-size: 28px;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) h4 {
        font-size: 22px;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [role="tablist"] {
        flex-wrap: wrap;  /* larger tab labels wrap instead of scrolling out of view */
        row-gap: 0.25rem;
    }
    /* The sliding underline can't follow wrapped tabs, so mark the active tab with a
       rounded background instead. */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-baseweb="tab-highlight"] {
        display: none;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button[role="tab"] {
        border-radius: 8px;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button[role="tab"][aria-selected="true"] {
        background: #FEE2E2;
        color: #B91C1C;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) .q-title {
        font-size: 19px;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) button[role="tab"] {
        font-size: 16px;
        font-weight: 600;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) label,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) label p {
        font-size: 16px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) input,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) textarea {
        font-size: 17px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) .streamlit-expanderHeader,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) .streamlit-expanderHeader p {
        font-size: 17px;
        font-weight: 600;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) .stButton button,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) .stButton button p,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stFormSubmitButton"] button,
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) [data-testid="stFormSubmitButton"] button p {
        font-size: 16px;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) .stAlert p {
        font-size: 17px;
    }
    /* Don't tint a just-clicked button red (clashes with the green "Checked" state). */
    .stButton > button:focus:not(:active) {
        border-color: rgba(49, 51, 63, 0.2);
        color: inherit;
        box-shadow: none;
    }
    @media (min-width: 768px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            height: calc(100vh - 2rem);
            overflow-y: auto;
            align-self: flex-start;
            padding-right: 0.5rem;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
            height: calc(100vh - 2rem);
            overflow-y: auto;
            align-self: flex-start;
            padding-right: 0.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings_column, response_column, model_column = st.columns(
    [25, 55, 20], gap="small"
)

try:
    topics_response = requests.get(f"{API_URL}/api/topics", timeout=15)
    topics_response.raise_for_status()
    topics = topics_response.json()
except requests.HTTPError as error:
    topics = {}
    topics_error = f"Interview API returned an error: {api_error_detail(error)}"
except requests.RequestException as error:
    topics = {}
    topics_error = (
        "Could not load topics from the interview API. Make sure the FastAPI "
        f"server is running at {API_URL}. Details: {error}"
    )
else:
    topics_error = ""

try:
    models_response = requests.get(f"{API_URL}/api/models", timeout=15)
    models_response.raise_for_status()
    models = models_response.json()
except requests.RequestException as error:
    models = []
    if isinstance(error, requests.HTTPError):
        models_error = f"Interview API returned an error: {api_error_detail(error)}"
    else:
        models_error = (
            f"Could not load models from the interview API. Details: {error}"
        )
else:
    models_error = ""

runtime_metrics, runtime_metrics_error = fetch_runtime_metrics(API_URL)

with settings_column:
    st.title("Smart Interview")
    st.subheader("Interview setup")
    st.caption("Choose a topic, experience level, and question type.")
    if topics_error:
        st.error(topics_error)
        submitted = False
        topic = ""
        sub_topics = []
        experience_level = ""
        question_type = ""
        question_count = 5
    else:
        topic_options = ("Select a topic", *topics)
        default_topic_index = (
            topic_options.index("Generative AI")
            if "Generative AI" in topic_options
            else 0
        )
        topic = st.selectbox(
            "Topic",
            topic_options,
            index=default_topic_index,
        )
        if topic in topics:
            sub_topics = st.multiselect(
                "Sub Topic (select one or more)",
                topics[topic],
                default=(
                    ["Prompting and controllability"]
                    if topic == "Generative AI"
                    and "Prompting and controllability" in topics[topic]
                    else []
                ),
            )
        else:
            st.multiselect(
                "Sub Topic (select one or more)",
                [],
                disabled=True,
            )
            sub_topics = []

        with st.form("interview_form"):
            # Collapsed panel; its title shows the current choice. Values in a form only
            # update on submit, so the title reflects the last submitted selection.
            current_level = st.session_state.get(
                "experience_level_select", DEFAULT_EXPERIENCE_LEVEL
            )
            current_type = st.session_state.get(
                "question_type_select", DEFAULT_QUESTION_TYPE
            )
            needs_selection = current_level.startswith("Select") or current_type.startswith(
                "Select"
            )
            with st.expander(
                f"Level & type: {current_level} · {current_type}",
                expanded=needs_selection,
            ):
                experience_level = st.selectbox(
                    "Experience level",
                    ("Select an experience level", *EXPERIENCE_LEVELS),
                    index=EXPERIENCE_LEVELS.index(DEFAULT_EXPERIENCE_LEVEL) + 1,
                    key="experience_level_select",
                )
                question_type = st.selectbox(
                    "Question type",
                    ("Select a question type", *QUESTION_TYPES),
                    index=QUESTION_TYPES.index(DEFAULT_QUESTION_TYPE) + 1,
                    key="question_type_select",
                )
            question_count = st.slider(
                "Number of questions", min_value=1, max_value=5, value=2
            )
            submitted = st.form_submit_button("Generate questions")

    st.markdown("### Practice tips")
    st.caption(
        "Answer each question aloud, use specific examples, and explain the "
        "impact of your decisions."
    )

with model_column:
    st.subheader("Model")
    if models_error:
        st.error(models_error)
        model = None
    elif not models:
        st.error("No models are configured. Check data/models.json.")
        model = None
    else:
        model = st.selectbox(
            "Choose a model",
            [item["id"] for item in models],
            index=next(
                (
                    index
                    for index, item in enumerate(models)
                    if item["default"]
                ),
                0,
            ),
            format_func=lambda model_id: next(
                item["name"] for item in models if item["id"] == model_id
            ),
        )
        if model == "knowledge_base":
            st.caption(
                "Uses matching questions saved in your local history. No LLM "
                "call is made. (For Ask a question, the AI agent chooses the model.)"
            )
            temperature = None
        else:
            st.caption(
                "Auto: the AI agent chooses the best model for each request. "
                "Choose a model to override it."
            )
            temperature = st.selectbox(
                "Temperature",
                TEMPERATURE_OPTIONS,
                index=TEMPERATURE_OPTIONS.index(0.3),
                format_func=lambda value: f"{value:.1f}",
                help="Lower values are more consistent; higher values are more varied.",
            )
    current_metrics_placeholder = st.empty()
    render_current_metrics(
        current_metrics_placeholder,
        st.session_state.get("process_info"),
    )
    st.markdown("---")
    runtime_metrics_placeholder = st.empty()
    render_overall_metrics(
        runtime_metrics_placeholder,
        runtime_metrics,
        runtime_metrics_error,
    )

with response_column:
    interview_tab, ask_tab, my_questions_tab, quick_topic_tab = st.tabs(
        ["📋 Interview questions", "💬 Ask a question", "📚 My questions", "⚡ Quick Topic"]
    )

    with interview_tab:
        st.subheader("Your interview questions")

        def generate_interview(payload: dict[str, object]) -> None:
            try:
                response = requests.post(
                    f"{API_URL}/api/interview/questions",
                    json=payload,
                    timeout=GENERATION_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                if (
                    payload.get("model") == "knowledge_base"
                    and isinstance(error, requests.HTTPError)
                    and error.response is not None
                    and error.response.status_code == 404
                ):
                    st.session_state["pending_llm_fallback"] = {
                        **payload,
                        "model": "auto",
                        "temperature": None,
                    }
                    st.session_state["pending_llm_fallback_detail"] = api_error_detail(
                        error
                    )
                elif isinstance(error, requests.HTTPError):
                    st.error(
                        "Interview API returned an error: "
                        f"{api_error_detail(error)}"
                    )
                else:
                    st.error(
                        "Could not reach the interview API. Make sure the FastAPI "
                        f"server is running at {API_URL}. Details: {error}"
                    )
            else:
                result = response.json()
                st.session_state["interview"] = result
                st.session_state["process_info"] = result["process_info"]
                # The list shows every saved question for this selection, from page 1.
                st.session_state["listing_filter"] = {
                    key: payload[key]
                    for key in ("topic", "sub_topics", "experience_level", "question_type")
                }
                st.session_state["listing_page"] = 1
                render_current_metrics(
                    current_metrics_placeholder,
                    result["process_info"],
                )

            updated_metrics, metrics_error = fetch_runtime_metrics(API_URL)
            render_overall_metrics(
                runtime_metrics_placeholder,
                updated_metrics,
                metrics_error,
            )

        if submitted:
            st.session_state.pop("pending_llm_fallback", None)
            st.session_state.pop("pending_llm_fallback_detail", None)
            if topic == "Select a topic":
                st.error("Select a topic.")
            elif not sub_topics:
                st.error("Select at least one subtopic.")
            elif experience_level == "Select an experience level":
                st.error("Select your experience level.")
            elif question_type == "Select a question type":
                st.error("Select a question type.")
            elif model is None:
                st.error("Select a configured model.")
            else:
                generate_interview(
                    {
                        "topic": topic,
                        "sub_topics": sub_topics,
                        "experience_level": experience_level,
                        "question_type": question_type,
                        "question_count": question_count,
                        "model": model,
                        "temperature": temperature,
                    }
                )

        pending_fallback = st.session_state.get("pending_llm_fallback")
        if pending_fallback:
            st.warning(
                "There aren't enough saved questions for this combination of topic, "
                "subtopics, experience level, question type, and question count. "
                f"{st.session_state.get('pending_llm_fallback_detail', '')}"
            )
            st.info(
                "**Change your selections, or explicitly confirm an LLM call "
                "using the automatically selected model.**"
            )
            # Stacked, not side by side: this already sits inside a column, and Streamlit
            # 1.12 does not allow nested columns.
            if st.button("Generate with Auto (LLM)", key="confirm_llm_fallback"):
                st.session_state.pop("pending_llm_fallback", None)
                st.session_state.pop("pending_llm_fallback_detail", None)
                generate_interview(pending_fallback)
            if st.button("Change selections", key="dismiss_llm_fallback"):
                st.session_state.pop("pending_llm_fallback", None)
                st.session_state.pop("pending_llm_fallback_detail", None)
                st.experimental_rerun()  # the prompt above is already drawn; redraw without it

        listing_filter = st.session_state.get("listing_filter")
        if listing_filter:
            st.markdown(
                f"Practice questions for **{listing_filter['topic']}** "
                f"({listing_filter['experience_level']}) · "
                f"{listing_filter['question_type']}"
            )
            st.caption(f"Subtopics: {', '.join(listing_filter['sub_topics'])}")
            if st.session_state.get("status_error"):
                st.error(st.session_state["status_error"])
            if st.session_state.get("refine_error"):
                st.error(st.session_state["refine_error"])

            listing, listing_error = fetch_saved_questions_page(
                listing_filter, st.session_state.get("listing_page", 1)
            )
            if listing_error:
                st.error(listing_error)
            elif listing["total"] == 0:
                st.info("No saved questions match this selection yet.")
            else:
                st.session_state["listing_page"] = listing["page"]  # the API clamps past-end pages
                st.progress(listing["checked_total"] / listing["total"])
                st.caption(
                    f"{listing['checked_total']} of {listing['total']} saved questions checked"
                )
                if listing["total_pages"] > 1:
                    render_pager(listing, "top")

                for question in listing["questions"]:
                    render_question_card(question, key_prefix="interview_")

                if listing["total_pages"] > 1:
                    st.markdown("---")
                    render_pager(listing, "bottom")
        else:
            st.info("Set your interview preferences on the left to get started.")


    with ask_tab:
        st.subheader("Ask a question")
        st.caption(
            "Ask any technical or interview question and get a short, easy answer. "
            "With Auto or Own knowledge base the AI agent picks the model; otherwise "
            "it uses the model chosen on the right. Answers are saved in My questions."
        )
        with st.form("ask_form", clear_on_submit=True):
            asked_question = st.text_area(
                "Your question",
                height=110,
                placeholder="e.g. What is the difference between a process and a thread?",
            )
            ask_submitted = st.form_submit_button("Ask")
        if ask_submitted:
            if len(asked_question.strip()) < 3:
                st.error("Type a question first.")
            elif model is None:
                st.error("Select a configured model.")
            else:
                with st.spinner("The AI agent is answering…"):
                    ask_result, ask_error = ask_agent(asked_question.strip(), model)
                if ask_error:
                    st.error(ask_error)
                else:
                    st.session_state["asked_in_session"] = True
                    st.session_state["my_questions_page"] = 1
                    st.session_state["process_info"] = ask_result["process_info"]
                    render_current_metrics(
                        current_metrics_placeholder, ask_result["process_info"]
                    )
                    updated_metrics, metrics_error = fetch_runtime_metrics(API_URL)
                    render_overall_metrics(
                        runtime_metrics_placeholder, updated_metrics, metrics_error
                    )

        if st.session_state.get("asked_in_session"):
            # The newest saved question, loaded fresh so status / Refine changes show.
            latest, latest_error = fetch_user_questions_page(1, page_size=1)
            if latest_error:
                st.error(latest_error)
            elif latest["questions"]:
                st.markdown("#### Latest answer")
                if st.session_state.get("status_error"):
                    st.error(st.session_state["status_error"])
                if st.session_state.get("refine_error"):
                    st.error(st.session_state["refine_error"])
                render_question_card(
                    latest["questions"][0], answer_label="Answer", key_prefix="ask_"
                )

    with my_questions_tab:
        st.subheader("My questions")
        my_listing, my_listing_error = fetch_user_questions_page(
            st.session_state.get("my_questions_page", 1)
        )
        if my_listing_error:
            st.error(my_listing_error)
        elif my_listing["total"] == 0:
            st.info("You haven't asked any questions yet. Use the Ask a question tab.")
        else:
            st.session_state["my_questions_page"] = my_listing["page"]
            st.progress(my_listing["checked_total"] / my_listing["total"])
            st.caption(
                f"{my_listing['checked_total']} of {my_listing['total']} "
                "of your questions checked"
            )
            if my_listing["total_pages"] > 1:
                render_pager(my_listing, "top", state_key="my_questions_page")
            for question in my_listing["questions"]:
                render_question_card(question, answer_label="Answer", key_prefix="my_")
            if my_listing["total_pages"] > 1:
                st.markdown("---")
                render_pager(my_listing, "bottom", state_key="my_questions_page")

    with quick_topic_tab:
        st.subheader("Quick Topic")
        st.caption(
            "Type a topic, e.g. \"What is RAG?\". The AI agent creates a box with its "
            "methodologies, types, other technical details and a small example, all as "
            "bullet points. Boxes are saved; the newest is first."
        )
        with st.form("quick_topic_form", clear_on_submit=True):
            topic_text = st.text_input("Topic", placeholder="e.g. What is RAG?")
            topic_submitted = st.form_submit_button("Create box")
        if topic_submitted:
            if len(topic_text.strip()) < 2:
                st.error("Type a topic first.")
            elif model is None:
                st.error("Select a configured model.")
            else:
                with st.spinner("The AI agent is building the topic box…"):
                    topic_result, topic_error = create_quick_topic(topic_text.strip(), model)
                if topic_error:
                    st.error(topic_error)
                else:
                    st.session_state["quick_topics_page"] = 1
                    st.session_state["process_info"] = topic_result["process_info"]
                    render_current_metrics(
                        current_metrics_placeholder, topic_result["process_info"]
                    )
                    updated_metrics, metrics_error = fetch_runtime_metrics(API_URL)
                    render_overall_metrics(
                        runtime_metrics_placeholder, updated_metrics, metrics_error
                    )

        topics_page, topics_error = fetch_quick_topics_page(
            st.session_state.get("quick_topics_page", 1)
        )
        if topics_error:
            st.error(topics_error)
        elif topics_page["total"] == 0:
            st.info("No topic boxes yet. Type a topic above to create one.")
        else:
            st.session_state["quick_topics_page"] = topics_page["page"]
            pager_options = {
                "state_key": "quick_topics_page",
                "items_key": "topics",
                "noun": "topic",
            }
            if topics_page["total_pages"] > 1:
                render_pager(topics_page, "top", **pager_options)
            for topic in topics_page["topics"]:
                render_topic_box(topic)
            if topics_page["total_pages"] > 1:
                st.markdown("---")
                render_pager(topics_page, "bottom", **pager_options)

