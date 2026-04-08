# ⚡ SYNAPSE: AI Persona Detection & Attribution

**SYNAPSE** is a high-performance Open Source Intelligence (OSINT) pipeline designed to detect, analyze, and attribute digital personas. Leveraging advanced machine learning, stylometric analysis, and graph-based clustering, SYNAPSE distinguishes between organic human activity and synthetic, AI-generated content on social media platforms.

> [!IMPORTANT]
> This project is designed for investigative researchers and OSINT analysts to identify coordinated inauthentic behavior (CIB) and artificial persona networks.

---

## 🚀 Key Features

- **Multi-Phase Pipeline**: A modular approach from data acquisition to attribution.
- **Deep Stylometric Analysis**: Extracting linguistic patterns, vocabulary richness, and sentence structure.
- **Behavioral Fingerprinting**: Analyzing activity timestamps, engagement patterns, and interaction frequency.
- **AI Persona Identification**: Specialized classifiers (Random Forest, XGBoost) trained to detect LLM-generated nuances.
- **Attribution Clustering**: Grouping personas based on high-dimensional similarity and network graphs.
- **Premium Dashboard**: A state-of-the-art Streamlit interface for investigation and visualization.
- **API-First Design**: FastAPI backend for seamless integration with other intelligence tools.
- **Comprehensive Reporting**: Automated PDF investigation reports with automated insights.

---

## 🛠 Project Structure

```text
persona-detection/
├── app/                        # Core Application Logic
│   ├── api/                    # FastAPI routes & authentication
│   ├── data_acquisition/       # Scrapers (X-API) & AI generation
│   ├── ui/                     # Streamlit dashboard components
│   ├── attribution_clustering.py # Attribution & graph-based logic
│   ├── config.py               # Global settings & directory management
│   ├── feature_extraction.py    # NLP & behavioral feature engineering
│   ├── integration_service.py   # Orchestrator & report generator
│   ├── persona_detection.py     # Machine Learning pipeline (Phase 4)
│   └── utils/                  # Shared utilities (logging, etc.)
├── data/                       # Data Storage
│   ├── raw/                    # Raw source data from scrapers
│   └── processed/              # Cleaned datasets & trained models
├── output/                     # Analysis Results
│   ├── reports/                # Generated PDF investigation reports
│   ├── exports/                # CSV/JSON data exports
│   ├── snapshots/              # Visualization snapshots
│   └── logs/                   # Application logs
├── scripts/                    # Automation & QA scripts
├── tests/                      # Automated test suite
├── main.py                     # CLI entry point for the pipeline
├── streamlit_app.py            # Streamlit dashboard entry point
├── api.py                      # FastAPI server entry point
├── Makefile                    # Development & operational shortcuts
└── requirements.txt            # Project dependencies
```

---

## 📂 The 5-Phase Pipeline

### Phase 1: Data Acquisition
Collecting ground-truth data. This involves scraping real tweets via the X API and generating synthetic datasets using LLM patterns to create a balanced, labeled corpus.

### Phase 2: Preprocessing
Cleaning, normalizing, and tokenizing raw text data. This phase ensures data consistency across both human and synthetic sources.

### Phase 3: Feature Extraction
The heart of the analysis. We extract over 400+ features including:
- **Stylometrics**: Word count, character count, punctuation density, stop-word frequency.
- **Behavioral**: Hourly activity heatmaps, daily frequency, response latencies.
- **TF-IDF Vectorization**: Capture latent linguistic kernels and unique vocabulary signatures.

### Phase 4: Persona Detection
Training and evaluating sophisticated classifiers. We compare multiple models (Random Forest, SVM, XGBoost) and select the champion based on F1-score to handle real-world classification tasks.

### Phase 5: Attribution & Clustering
Grouping personas into "clusters" using DBSCAN and NetworkX. This phase identifies potential "botnets" or coordinated networks by analyzing high-dimensional similarity in behavioral and stylometric space.

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10.x
- X API Bearer Token (for data acquisition)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Craftzmen/persona-detection.git
cd persona-detection

# Install dependencies via Makefile
make install
```

### 3. Environment Setup
Create a `.env` file from the template:
```bash
cp .env.example .env
```
Fill in your `X_BEARER_TOKEN` and optional authentication credentials.

---

## 🖥 Dashboard Usage

The interactive dashboard is the primary tool for analysts:

```bash
make run-dashboard
```

**Features included:**
- **Search & Analyze**: Enter any X username to perform an end-to-end investigation.
- **Risk Scoring**: Visual indicators for `Low`, `Medium`, and `High` synthetic risk.
- **Interactive Graphs**: Explore the attribution network and linked personas.
- **History & Comparison**: Compare two personas side-by-side to find behavioral overrides.
- **Report Export**: Download a professional PDF report of your clinical findings.

---

## 🔌 API Integration

Start the FastAPI server for programmatic access:

```bash
make run-api
```

### Key Endpoints:
- `GET /health`: System health check.
- `GET /analyze?username=NASA`: Run full persona investigation.
- `GET /history`: Retrieve persistent analysis logs.

---

## 🧪 Testing & Quality Assurance

Maintain high reliability using the automated test suite:

```bash
# Run all tests
make test

# Run QA readiness check (Smoke tests + performance benchmarks)
make qa
```

---

## 🛠 Makefile Shortcuts

| Command | Description |
| :--- | :--- |
| `make install` | Safely install all project dependencies |
| `make run` | Launch the Streamlit investigation console |
| `make run-api` | Start the FastAPI production server |
| `make test` | Execute the unit and integration test suite |
| `make qa` | Run the comprehensive QA readiness workflow |
| `make clean-logs` | Flush application log files |

---

## 📄 License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

*Built with ❤️ by the Craftzmen
