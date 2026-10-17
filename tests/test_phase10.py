"""
CareerPilot — Phase 10 Tests (Docker Containerization & Deployment)

Covers:
  - Dockerfile presence, multi-service structure, and python base image
  - docker-compose.yml YAML syntax validation & service configurations
  - .dockerignore exclusions for secrets, local databases, and caches
  - Port mapping and environment variable orchestration
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestDockerConfiguration:
    def test_dockerfile_exists_and_valid(self):
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile must exist at project root"

        content = dockerfile.read_text(encoding="utf-8")
        assert "FROM python:3.13-slim" in content, "Dockerfile should use python:3.13-slim base image"
        assert "requirements.txt" in content, "Dockerfile must copy and install requirements.txt"
        assert "EXPOSE 8000 8501" in content or ("8000" in content and "8501" in content)
        assert "backend.main:app" in content

    def test_dockerignore_exists_and_protects_secrets(self):
        dockerignore = PROJECT_ROOT / ".dockerignore"
        assert dockerignore.exists(), ".dockerignore must exist at project root"

        lines = [line.strip() for line in dockerignore.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert ".env" in lines or any(".env" in l for l in lines), ".dockerignore must exclude .env secrets"
        assert "*.db" in lines or any(".db" in l for l in lines), ".dockerignore must exclude local database files"
        assert "venv/" in lines or "venv" in lines, ".dockerignore must exclude virtual environment"
        assert "__pycache__/" in lines or "__pycache__" in lines

    def test_docker_compose_syntax_and_services(self):
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml must exist at project root"

        with open(compose_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "services" in data, "docker-compose.yml must define services"
        services = data["services"]

        # Backend service checks
        assert "backend" in services, "Must declare backend service"
        backend = services["backend"]
        assert "ports" in backend
        assert any("8000" in str(p) for p in backend["ports"]), "Backend must expose port 8000"
        assert "healthcheck" in backend, "Backend must have healthcheck defined"

        # Frontend service checks
        assert "frontend" in services, "Must declare frontend service"
        frontend = services["frontend"]
        assert "ports" in frontend
        assert any("8501" in str(p) for p in frontend["ports"]), "Frontend must expose port 8501"
        assert "depends_on" in frontend, "Frontend must depend on backend"

        # Volume checks
        assert "volumes" in data, "Must declare persistent data volumes"
        assert "careerpilot_storage" in data["volumes"]
