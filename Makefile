# Smart Interview commands.
# Run from this folder:      make backend
# or from anywhere:          make -C /path/to/smart_interview backend
#
# Commands always run inside this folder with its own .venv: running from the parent folder
# would let a sibling folder named `langchain` shadow the real `langchain` library.

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(ROOT)/.venv/bin

BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 8501
export SMART_INTERVIEW_API_URL ?= http://127.0.0.1:$(BACKEND_PORT)

# Tests import the project as the `smart_interview` package; expose only this folder under
# that name (outside the project, so file watchers never see a symlink loop).
TEST_PKG_DIR := $(or $(TMPDIR),/tmp)/smart_interview_pkg

.PHONY: help install backend frontend run test migrate-levels migrate-status

help:
	@echo "make install         create .venv and install requirements"
	@echo "make backend         FastAPI on http://127.0.0.1:$(BACKEND_PORT) (auto-reload)"
	@echo "make frontend        Streamlit on http://localhost:$(FRONTEND_PORT)"
	@echo "make run             backend + frontend together (Ctrl+C stops both)"
	@echo "make test            run the test suite"
	@echo "make migrate-levels  dry run of relabeling saved year ranges (add ARGS=--apply)"
	@echo "make migrate-status  dry run of adding question_id/status to saved questions (ARGS=--apply)"

install:
	cd "$(ROOT)" && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

backend:
	cd "$(ROOT)" && "$(VENV)/python" -m uvicorn api.main:app --reload --host 127.0.0.1 --port $(BACKEND_PORT)

frontend:
	cd "$(ROOT)" && "$(VENV)/python" -m streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

run:
	cd "$(ROOT)" && \
	"$(VENV)/python" -m uvicorn api.main:app --reload --host 127.0.0.1 --port $(BACKEND_PORT) & \
	BACKEND_PID=$$!; \
	trap 'kill $$BACKEND_PID 2>/dev/null; wait $$BACKEND_PID 2>/dev/null' EXIT INT TERM; \
	sleep 2; \
	cd "$(ROOT)" && "$(VENV)/python" -m streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

test:
	mkdir -p "$(TEST_PKG_DIR)" && ln -sfn "$(ROOT)" "$(TEST_PKG_DIR)/smart_interview"
	cd "$(ROOT)" && PYTHONPATH="$(TEST_PKG_DIR)" "$(VENV)/python" -m pytest -q tests

migrate-levels:
	cd "$(ROOT)" && "$(VENV)/python" scripts/migrate_experience_levels.py $(ARGS)

migrate-status:
	cd "$(ROOT)" && "$(VENV)/python" scripts/add_question_status.py $(ARGS)
