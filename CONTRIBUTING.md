# Contributing to CareerPilot

Thank you for contributing to **CareerPilot: An Agentic AI for Personalized Placement Preparation**! We welcome contributions to enhance tutor agents, RAG retrieval quality, testing coverage, and candidate experience.

---

## 🛠️ Local Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/katurikeerthi123-sys/CareerPilot.git
   cd CareerPilot
   ```

2. **Set Up Python Environment (Python 3.11 or 3.12 recommended)**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\Activate.ps1
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY into .env
   ```

---

## 🧪 Running Tests

Before submitting code, ensure all automated tests pass:

```bash
# Run all tests (fast unit & integration)
pytest -q -m "not live"

# Run with full verbosity
pytest -v

# Run live tests (requires valid GEMINI_API_KEY in .env)
pytest -m "live"
```

---

## 🐳 Docker Verification

Verify that container images build and launch cleanly:

```bash
# Build multi-service images
docker compose build

# Launch services in background
docker compose up -d

# Check service health
docker compose ps
```

---

## 📐 Code Style & Best Practices

- Follow PEP 8 standards with type hints where feasible.
- Maintain documentation integrity and preserve docstrings across agent nodes.
- Keep agent prompt templates well-structured with clear fallback handling.
- Never commit secrets, `.env`, or database files (`*.db`, `chroma_db/`).

---

## 🐛 Reporting Issues & Submitting PRs

1. Create a feature or bugfix branch (`git checkout -b feature/amazing-feature`).
2. Commit your changes with concise, descriptive commit messages.
3. Open a Pull Request against `main` explaining the problem solved and testing instructions.
