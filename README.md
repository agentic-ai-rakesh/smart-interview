# Smart Interview

A small interview-practice app with a FastAPI backend and a Streamlit frontend. The backend uses a LangGraph agent with LangChain models (Gemini, GPT and Claude, all through OpenRouter) to generate topic-aware interview questions and suggested answers, and stores results in MongoDB.

## Highlights

- **Every question and answer is stored in MongoDB.** Each generated set, each *Ask a question*
  answer and each *Quick Topic* box is saved as a document in one collection, so it can be
  revisited later with **no LLM call** (no wait, no token cost).
- **Auto model selection.** With *Auto*, the LangGraph agent picks the best-suited model for each
  request (Gemini Flash-Lite, GPT-4o mini or Claude Sonnet, all through OpenRouter), with a
  rule-based fallback.
- **Interview practice tools:** topic, subtopic, level and type selection; checked/unchecked status
  per question; refine a question or answer in simpler words; per-request and per-step metrics.

## Storage (MongoDB)

All results go to the collection named by `MONGODB_COLLECTION` in the database `MONGODB_DATABASE`
(defaults in `.env.example`: `interviewdb` / `interview_responses`).

| What is saved | Document | Revisit without an LLM call |
|---|---|---|
| Generated interview questions and answers | `topic`, `sub_topics`, `experience_level`, `question_type`, `questions[]` (each with `question_id`, text, answer, `status`), `process_info` | *Own knowledge base* model option, and the paginated question list |
| *Ask a question* answers | `kind: "user_question"` | *My questions* tab |
| *Quick Topic* boxes | `kind: "quick_topic"` | *Quick Topic* list |

- **Reuse instead of regenerate:** choose **Own knowledge base** as the model. The API reads saved
  questions that match the topic, subtopics, level and type, removes duplicate texts, and returns a
  random set of them. No model is called and nothing is saved again. If there are not enough
  matches, it says so, and you can generate more with a model.
- **Updates in place:** *Mark as checked* and *Refine* update the saved document (`status`,
  `refined_text`, `refined_answer`), so your progress and refined versions are there next time.
- **Lists are history reads:** the question list, *My questions* and *Quick Topic* lists are paged
  MongoDB reads (newest first, 5 per page).

## Auto model selection

Models are listed in `data/models.json`, each with a plain-language `description` of what it is
good at.

1. **You pick a model:** that model is always used.
2. **Auto (or no model):** the agent sends the request details (topic, subtopics, level, question
   type, count) and the model descriptions to the agent model, which returns its choice and a
   short reason. For example, conceptual Beginner questions go to the fast, low-cost Gemini
   Flash-Lite, and Advanced code or scenario problems go to Claude Sonnet. Only models whose API
   key is set are offered.
3. **Fallback:** if that choice fails or is unusable, `data/model_routing.json` maps the question
   type to a model (General → Gemini, Code and Scenario problems → Claude Sonnet).

*Ask a question* and *Quick Topic* use the same idea on the question text, with the default model as
the fallback. The chosen model and the reason appear under *Current Metrics* in the app.

## Project structure

```text
smart_interview/
├── api/
│   ├── __init__.py
│   ├── agent_interview.py
│   ├── main.py
│   ├── model_selection.py
│   ├── progress_info.py
│   ├── prompt_interview.py
│   └── schemas.py
├── data/
│   ├── models.json
│   ├── model_routing.json
│   └── topics.json
├── .env.example
├── frontend/
│   ├── app.py
│   └── progress_info.py
├── scripts/
│   ├── add_question_status.py
│   └── migrate_experience_levels.py
├── tests/
│   └── test_main.py
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

## Setup

Prerequisites: Python 3.9 or newer, and a running MongoDB server (local or remote).

```bash
git clone https://github.com/rakeshrex0505/smart-interview.git
cd smart-interview
make install            # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp -n .env.example .env # then fill in your own values
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The requirements pin Altair to version 4 because the available Streamlit version
uses Altair's `vegalite.v4` API.

## Configure model providers

Create `.env` from `.env.example` if you do not already have one. If `.env`
already exists, keep it and add or update the key there:

```bash
cp -n .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The current Gemini Flash-Lite, OpenAI GPT-4o
mini, and Claude Sonnet 4.6 choices are routed through OpenRouter, so the same
key can be used for each. Available models and their defaults are configured in
`data/models.json`, with automatic question-type routing in
`data/model_routing.json`. Keep `.env` private; it is excluded from Git.
Set `MONGODB_URI`, `MONGODB_DATABASE`, and `MONGODB_COLLECTION` in `.env` to
configure response storage. Each successfully generated response is saved as a
MongoDB document with its complete API response, including questions and
`process_info`. The endpoint returns an explicit service error if MongoDB is
not configured or the response cannot be saved.
For a MongoDB server with authentication enabled, set `MONGODB_URI` to include
the MongoDB username and password, for example
`mongodb://USERNAME:PASSWORD@localhost:27017/?authSource=admin`. Replace the
placeholders with your credentials, set `authSource` to the database where the
MongoDB user was created, and URL-encode reserved characters in the credentials.
Keep credentials only in the private `.env` file.
Question generation reports an explicit API error if the key is missing or
generation fails.

Processing failures are logged with their operation and returned as readable
API error details, including the underlying exception message when available.
The response keeps an appropriate HTTP status; API keys and MongoDB URI
credentials are redacted from error details. The Streamlit app displays the
API's `detail` message for failed requests. Each decorated processing step logs
its completion or failure and elapsed time. `GET /api/metrics` and the Streamlit
Overall Metrics panel include per-step run, success, failure, and duration
aggregates for the lifetime of the API worker.
Explanatory captions use the same theme accent color as the app's buttons and
selected items.

To add another model from an already supported provider, add an entry to
`data/models.json` with a unique `id`, display `name`, provider, model name,
API-key environment variable, structured-output method, temperature, and
default setting. To add a new provider, add its factory in
`api/model_selection.py` and allow its provider identifier in `ModelConfig`.

## Run the app

From this folder:

```bash
make backend    # API on http://127.0.0.1:8000 (auto-reload)
make frontend   # Streamlit on http://localhost:8501 (second terminal)
make run        # both in one terminal; Ctrl+C stops both
```

From any other folder, point `make` at this one:

```bash
make -C /path/to/smart-interview run
```

Other commands: `make install` (create `.venv` and install requirements), `make test`, and
`make help`. Ports can be changed, e.g. `make run BACKEND_PORT=8002`. The commands always run
inside this folder with its own `.venv`, so no environment needs to be activated first.

The API is available at `http://127.0.0.1:8000`; its interactive documentation is at `http://127.0.0.1:8000/docs`. The Streamlit app uses that API by default. To use a different API address, set `SMART_INTERVIEW_API_URL` before starting Streamlit.

## Question status

Every question has a permanent `question_id` and a `status`: `Unchecked` (default, shown as a
yellow pill with a "🟡 Mark as checked" button) or `Checked` (green pill, "✅ Checked" button that
switches it back). Clicking the button saves the status in MongoDB, so it is kept when the same
question is served again from saved history. A progress bar shows how many are checked.

The API endpoint is `PATCH /api/questions/{question_id}/status` with `{"status": "Checked"}` or
`{"status": "Unchecked"}`. Questions saved before this feature get their `question_id` and default
status with `make migrate-status ARGS=--apply` (already run once; safe to run again).

## AI agent (LangGraph)

Every LLM task runs through one LangGraph agent in `api/agent_interview.py`; all of its prompts
are in `api/prompt_interview.py` (shared scope and style rules in `AGENT_SCOPE`).

```
START -> route_task -+- generate -> select_model -> generate -> validate -+-> END
                     |                                  ^                 |
                     |                                  +---- retry ------+  (invalid output, once)
                     +- ask ---------> select_ask_model -> answer -> END
                     +- quick_topic -> select_ask_model -> topic_card -> END
                     +- refine_question / refine_answer -> refine -> END
```

- **Model selection**: with *Auto* (or no model), the agent's LLM picks the model from the
  question parameters (type, level, topic, count) using each model's `description` in
  `data/models.json`. If its pick is unusable, `data/model_routing.json` is the fallback. A model
  you pick yourself is always used. The choice and reason show under *Current Metrics*.
- **Generation**: questions match the selected level, use clean and easy language, and are
  practical, in the style of screening rounds at large IT companies (TCS, Infosys, HCL, Wipro,
  Accenture, ...). Answers are short bullet points; examples or diagrams only when they help.
  Trends come from the model's training knowledge; the agent does not browse the web.
- **Validation**: wrong question count or subtopics -> one retry with feedback, then an error.
- **Refine**: *✨ Refine question* rewrites a question in simpler words; *✨ Refine answer*
  explains the answer again in a different, easier way. Each click gives a new version. Results
  are saved as `refined_text` / `refined_answer`; the originals are kept and still shown.
  API: `POST /api/questions/{question_id}/refine` with `{"target": "question"}` or `"answer"`.
- **Ask a question** (middle-panel tab): type any technical or interview question. With *Auto*
  or *Own knowledge base* the agent's LLM picks the model for the question (default model as the
  fallback); otherwise the model chosen on the right is used. Answers are short, easy bullet
  points aimed at the question, with methods/types or an example only when needed. Each question
  and answer is saved in MongoDB (`kind: "user_question"`, kept apart from interview lists) and has
  the same status pill and *Refine* buttons. API: `POST /api/ask` with `{"question": ..., "model": ...}`.
- **My questions** (middle-panel tab): every question you asked, newest first, 5 per page with
  *⬅ Previous* / *Next ➡*. API: `GET /api/user-questions?page=1&page_size=5`.
- **Quick Topic** (middle-panel tab): type a topic such as "What is RAG?" and the agent builds a
  box with **Overview**, **Methodologies**, **Types**, **Other technical details** and a small
  **Example**, all as one-line bullets (ambiguous acronyms take their software/AI meaning). The
  model is chosen like for *Ask a question*. Boxes are saved (`kind: "quick_topic"`) and listed
  newest first, 5 per page. API: `POST /api/quick-topics` with `{"topic": ..., "model": ...}`,
  `GET /api/quick-topics?page=1&page_size=5`.
- **Agent model**: choosing models and refining use the default model in `data/models.json`, or
  the one named by `AGENT_MODEL_ID` in `.env`.

## Question list and pagination

After you generate (with a model or from saved history), the list shows every saved question that
matches the selected topic, subtopics, experience level and question type, newest first, 5 per page.
Use **⬅ Previous** (left) and **Next ➡** (right), shown above and below the list, to move between pages; newly generated questions are
on page 1. The progress bar counts checked questions across all pages.

API: `POST /api/questions/saved` with `topic`, `sub_topics`, `experience_level`,
`question_type`, `page` (from 1) and `page_size` (default 5, max 50). It returns `total`,
`checked_total`, `page`, `page_size`, `total_pages` and that page's `questions`; a page past
the end returns the last page.

## API

- `GET /health` — API health check.
- `GET /api/topics` — return the topic-to-subtopic mapping from `data/topics.json`.
- `GET /api/models` — return the model choices configured in `data/models.json`.
- `GET /api/metrics` — return process-local runtime request, error, concurrency,
  throughput, and latency observations, plus per-step run, success, failure,
  and duration aggregates. The metrics endpoint itself is excluded from request
  measurements.
- `POST /api/interview/questions` — generate up to five questions and suggested
  answers. The Streamlit model selector defaults to `knowledge_base`; choose
  `auto` or a configured model to generate new questions. `model` can be
  `auto`, a configured model ID, or `knowledge_base`. When omitted or set to
  `auto`, the agent chooses the model (see *Auto model selection*), falling back
  to `data/model_routing.json` (General → Gemini Flash-Lite, Code and Scenario
  problems → Claude Sonnet). Selecting a specific model overrides automatic
  routing. The `knowledge_base` option
  retrieves saved questions matching the selected topic, experience level,
  question type, and subtopics without making an LLM call, selecting a random
  set of unique matches on each request; it returns an error if the history
  does not contain enough matching unique questions. In the
  Streamlit app, the user can change the selections or explicitly confirm an
  automatically routed LLM call to generate new questions. Responses retrieved
  from history are not saved again. The optional `temperature` must be between
  `0` and `2`; when omitted, the selected model's configured temperature is
  used.

Successful question-generation responses also include `process_info` with
reported token counts, request-validation, model-setup, prompt-preparation,
LLM-call/structured-parsing, and response-assembly timings, plus total API
handler time and non-LLM API time. It also includes output tokens per second and
provider-reported cost when available. Token counts and cost are `null` if the
selected provider does not report them. Timings cover the API handler through
response assembly; they exclude network transfer and machine-wide OS time.
These per-request timings are not a scalability benchmark.
The Streamlit model panel also shows API runtime observations: request and
server-error counts since process start, current and peak observed concurrency,
average/p95 latency from the latest 1,000 completed requests, and throughput
over the last 60 seconds. These counters are in-memory and local to one API
worker; they reset on restart and do not indicate maximum capacity or
multi-worker behavior. Use a controlled concurrent load test to evaluate
scalability.

Example request:

```json
{
  "topic": "Python",
  "sub_topics": [
    "Object-oriented programming",
    "Testing with pytest"
  ],
  "experience_level": "Intermediate",
  "question_type": "Code problem",
  "question_count": 5
}
```

Choose a required `topic` and one or more of its `sub_topics` from `data/topics.json`, a required `experience_level` from `Beginner`, `Intermediate`, `Proficient`, `Advanced`, or `Expert` (the UI defaults to `Advanced`), and a required `question_type` from `General`, `Code problem`, or `Scenario problem`. `question_count` is the total number of questions and must be between 1 and 5. All of these selections are included in the LangChain prompt. Questions are kept to 2–3 concise lines, and suggested answers are high-level summaries. Code problem questions include a concrete coding task; their answers include a language-labeled code solution and a brief summary.

### Editing topics and subtopics

`data/topics.json` is a JSON object whose keys are topic names and whose values are arrays of that topic's subtopics. Add, edit, or remove a topic or subtopic in this file; the frontend and API read it dynamically. Keep each topic's subtopic array non-empty and use unique, non-empty strings.

## Run tests

```bash
make test
```

The tests mock the model and MongoDB, so they need no API key or database. Use `make test`
rather than a bare `pytest`: it exposes this folder as the `smart_interview` package the tests import.
