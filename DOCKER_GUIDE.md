# CareerPilot — Docker Deployment & Containerization Guide

This guide describes how to run the complete **CareerPilot** Agentic AI system using Docker and Docker Compose.

---

## 1. Prerequisites

- **Docker Desktop** (Engine v20+ / Compose v2.20+)
- **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/))

---

## 2. Architecture Overview

The system runs as two orchestrated services connected on an internal Docker bridge network:

```
                  ┌──────────────────────────────┐
                  │    Host Browser / Client     │
                  └──────┬────────────────┬──────┘
                         │                │
                Port 8501│                │Port 8000
                         ▼                ▼
       ┌────────────────────────┐  ┌────────────────────────┐
       │   careerpilot-frontend │  │   careerpilot-backend  │
       │   (Streamlit Dashboard)│  │   (FastAPI + LangGraph)│
       └───────────┬────────────┘  └───────────┬────────────┘
                   │                           │
                   │    HTTP /api requests     │
                   └───────────────────────────┘
                                 │
                   ┌─────────────┴──────────────┐
                   │    careerpilot_storage     │
                   │    (Named Docker Volume)   │
                   │  - careerpilot.db (SQLite) │
                   │  - chroma_db/ (Vectorstore)│
                   └────────────────────────────┘
```

- **`backend` (FastAPI + LangGraph):**
  - Serves REST endpoints for agents, RAG, adaptive learning loops, placement tests, and mock interviews.
  - Automatically initializes SQLite tables and ChromaDB vector store.
  - Includes automated health check probe: `curl -f http://localhost:8000/health`.

- **`frontend` (Streamlit):**
  - Interactive student UI for roadmap tracking, study documents, mock interviews, and readiness reports.
  - Waits for backend healthcheck to turn green before accepting traffic.

- **`careerpilot_storage` (Shared Named Volume):**
  - Persists student preparation history, questions, evaluations, and vector embeddings across container restarts.

---

## 3. Quickstart: Launch with Docker Compose

### Step 1: Configure Environment Variables
Set your Gemini API key in your terminal or create/verify your `.env` file:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key-here"

# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### Step 2: Build and Start Containers
```bash
docker compose up --build -d
```

### Step 3: Verify Running Services
```bash
docker compose ps
```
Both `careerpilot-backend` and `careerpilot-frontend` will show `Up` and `healthy`.

---

## 4. Accessing Applications

| Service | URL | Description |
| :--- | :--- | :--- |
| **Streamlit Web UI** | `http://localhost:8501` | Candidate preparation dashboard & mock interviews |
| **FastAPI Interactive Docs** | `http://localhost:8000/docs` | Swagger UI for all REST endpoints |
| **Backend Health Check** | `http://localhost:8000/health` | Service uptime and version probe |

---

## 5. Viewing Logs

Stream live logs from both services or an individual service:

```bash
# All services
docker compose logs -f

# Backend service only
docker compose logs -f backend

# Frontend service only
docker compose logs -f frontend
```

---

## 6. Stopping and Cleanup

To stop containers while **preserving** persistent data:
```bash
docker compose down
```

To stop containers and **wipe** all stored data volumes (fresh start):
```bash
docker compose down -v
```
