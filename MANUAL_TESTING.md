# 🕵️ SYNAPSE: Manual Testing Guide

This document provides a step-by-step guide for manually verifying the core functionality of the SYNAPSE project. Follow these sections to ensure the system is operating as expected.

---

## 🏗 1. Environment & Setup Verification

Ensure the environment is correctly configured before starting functional tests.

- [ ] **Check Python Version**: Run `python --version`. It should be `3.10.x`.
- [ ] **Dependency Audit**: Run `make install`. Ensure no installation errors occur.
- [ ] **Environment Variables**: Copy `.env.example` to `.env` and verify `X_BEARER_TOKEN` is present.
- [ ] **Directory Readiness**: Verify that `data/`, `output/`, and their subdirectories are created automatically upon first run.

---

## ⌨️ 2. CLI Pipeline Testing (Phases 1-4)

Test the underlying data pipeline using the command-line interface.

### A. Dataset Generation (Phase 1-3)
- [ ] **Command**: `python main.py sample_user --max-tweets 10 --ai-posts 10`
- [ ] **Expectation**: 
    - Console output shows "Dataset saved to: data/processed/dataset.csv".
    - Feature matrix shape is printed (e.g., `(20, 400+)`).
    - CSV files appear in `data/processed/`.

### B. Model Training (Phase 4)
- [ ] **Command**: `python main.py sample_user --train-phase4 --plot-metrics`
- [ ] **Expectation**:
    - Comparison table of RF, SVM, and XGBoost results printed.
    - "Saved best model to: data/processed/best_persona_model.pkl".
    - Visualization files (confusion matrix) generated in `data/processed/`.

### C. Prediction Mode
- [ ] **Command**: `python main.py --predict-features-csv data/processed/X_features.csv`
- [ ] **Expectation**: 
    - A table showing `username`, `classification`, and `synthetic_score` is displayed in the terminal.

---

## 🖥 3. Streamlit Dashboard Testing

Test the primary user interface and its interactive features.

### A. Launch & Authentication
- [ ] **Action**: Run `make run-dashboard`.
- [ ] **Action**: If `DASHBOARD_AUTH_USERNAME` is set in `.env`, verify the login screen appears.
- [ ] **Action**: Enter valid/invalid credentials to test the gate.

### B. Investigation Flow
- [ ] **Action**: Enter a username (e.g., `NASA`) in the command bar and press Enter or click "Analyze".
- [ ] **Expectation**: 
    - "Analyzing @NASA..." spinner appears.
    - Results render including Risk Level, Synthetic Score, and Confidence description.

### C. Visual Insights
- [ ] **Verify**: Activity heatmaps (Hourly/Daily) render correctly in the "Behavioral" tab.
- [ ] **Verify**: Word count distribution or stylometric markers render in the "Stylometric" tab.
- [ ] **Verify**: Network graph renders in the "Attribution" tab and allows zooming/panning.

### D. Export & History
- [ ] **Action**: Click "Download PDF Report".
- [ ] **Expectation**: Browser downloads a `.pdf` file. Open it and verify the SYNAPSE header and analysis results are present.
- [ ] **Action**: Run a second analysis for another user.
- [ ] **Verify**: The "Analysis History" table at the bottom of the page updates with both users.
- [ ] **Action**: Select both users in the "Side-by-Side Comparison" section.
- [ ] **Verify**: Comparative panels and the "Score Delta" appear.

---

## 🔌 4. API Endpoint Testing

Verify the FastAPI backend routes.

- [ ] **Action**: Run `make run-api`.
- [ ] **Test `/health`**: Visit `http://127.0.0.1:8000/health`. Should return `{"status": "ok"}`.
- [ ] **Test `/analyze`**: Run `curl "http://127.0.0.1:8000/analyze?username=test_user"`.
    - **Expectation**: Returns valid JSON with Persona Detection schema.
- [ ] **Test `/history`**: Run `curl "http://127.0.0.1:8000/history"`.
    - **Expectation**: Returns a list of previously analyzed accounts.

---

## 🚩 5. Edge Cases & Error Handling

- [ ] **Empty User**: Try analyzing a user with no tweets. Verify the system falls back gracefully to AI generation or shows a "No content" warning.
- [ ] **Invalid User**: Enter a non-existent X handle. Verify the UI handles the error without crashing.
- [ ] **Network Failure**: Simulate no internet connection and verify that relevant error messages are shown for scraping failures.

---
*End of Manual Testing Guide.*
