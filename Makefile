SHELL := /bin/zsh

PYTHON ?= .venv/bin/python
PIP ?= $(PYTHON) -m pip

.PHONY: help install test qa run-api run-dashboard run clean-logs

help:
	@echo "Available targets:"
	@echo "  make install        Install dependencies"
	@echo "  make test           Run test suite"
	@echo "  make qa             Run one-command QA readiness"
	@echo "  make run-api        Start FastAPI server"
	@echo "  make run-dashboard  Start Streamlit dashboard"
	@echo "  make run            Start dashboard (default app UX)"
	@echo "  make clean-logs     Remove rotated log files"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

qa:
	$(PYTHON) scripts/qa_readiness.py

run-api:
	$(PYTHON) -m uvicorn api:app --reload

run-dashboard:
	$(PYTHON) -m streamlit run streamlit_app.py

run: run-dashboard

clean-logs:
	rm -f output/logs/app.log*
