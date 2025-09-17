# Setting Up a Similar Testing Architecture

## Overview
This guide shows how to adapt the LeRobot testing architecture for your own project with different requirements and dependencies.

## Step 1: Project Structure Setup

Create the following directory structure:
```
your-project/
├── .github/
│   └── workflows/
│       ├── full_tests.yml
│       └── fast_tests.yml
├── docker/
│   ├── Dockerfile.internal
│   └── Dockerfile.user
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── Makefile
├── requirements.txt
├── pyproject.toml (or package.json, etc.)
└── README.md
```

## Step 2: GitHub Actions CI Setup

### 2.1 Create `.github/workflows/full_tests.yml`

Key sections to customize:

#### **Triggers** (Customize based on your needs):
```yaml
on:
  workflow_dispatch:  # Manual trigger
  pull_request_review:
    types: [submitted]
  push:
    branches: [main, develop]  # Your main branches
    paths:
      - "src/**"      # Your source code paths
      - "tests/**"
      - ".github/workflows/**"
      - "requirements.txt"  # Your dependency files
```

#### **Environment Variables** (Adjust for your project):
```yaml
env:
  PYTHON_VERSION: "3.10"  # Your Python version
  NODE_VERSION: "18"      # If using Node.js
  DOCKER_IMAGE_NAME: your-org/your-project  # Your Docker image
```

#### **System Dependencies** (Install what you need):
```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update && sudo apt-get install -y \
    build-essential git curl \
    # Add your specific dependencies here
    # For ML: libglib2.0-0 libegl1-mesa-dev
    # For web: nginx apache2
    # For databases: postgresql-client mysql-client
```

### 2.2 Job Structure (Adapt to your needs):

#### **Job 1: Basic Tests**
```yaml
basic-tests:
  name: Basic Tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/ -v
```

#### **Job 2: Docker Build** (If you need containers):
```yaml
build-docker:
  name: Build Docker
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -f docker/Dockerfile.user -t ${{ env.DOCKER_IMAGE_NAME }} .
```

#### **Job 3: Integration Tests** (If you need advanced testing):
```yaml
integration-tests:
  name: Integration Tests
  runs-on: ubuntu-latest
  needs: [build-docker]
  steps:
    - name: Run integration tests
      run: make test-integration
```

## Step 3: Docker Setup

### 3.1 Create `docker/Dockerfile.user` (Development)

```dockerfile
# Base image - choose what fits your project
FROM python:3.10-slim  # or node:18, golang:1.21, etc.

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/app/.venv/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    # Add your specific system dependencies
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app_user \
    && usermod -aG sudo app_user

# Set working directory
WORKDIR /app
RUN chown -R app_user:app_user /app

# Switch to non-root user
USER app_user

# Install your dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY . .

# Default command
CMD ["/bin/bash"]
```

### 3.2 Create `docker/Dockerfile.internal` (CI/Production)

```dockerfile
# For ML/AI projects, use GPU base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# For web projects, use standard base
# FROM ubuntu:22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/app/.venv/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    # Add your specific dependencies
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app_user

# Set working directory
WORKDIR /app
RUN chown -R app_user:app_user /app

# Switch to non-root user
USER app_user

# Install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY . .

# Default command
CMD ["/bin/bash"]
```

## Step 4: Makefile Setup

Create a `Makefile` with your test targets:

```makefile
.PHONY: test test-unit test-integration test-e2e

# Default test target
test: test-unit test-integration

# Unit tests
test-unit:
	pytest tests/unit/ -v

# Integration tests
test-integration:
	pytest tests/integration/ -v

# End-to-end tests (customize based on your needs)
test-e2e:
	# Example for a web API:
	# Start your service
	# Run API tests
	# Stop service
	
	# Example for ML model:
	# Train minimal model
	# Evaluate model
	# Test deployment

# Development setup
setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Docker commands
docker-build:
	docker build -f docker/Dockerfile.user -t your-project .

docker-run:
	docker run -it --rm your-project

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
```

## Step 5: Dependency Management

### 5.1 Create `requirements.txt`
```
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0

# Testing
pytest>=6.0.0
pytest-cov>=2.12.0
pytest-timeout>=1.4.0

# Development
black>=21.0.0
flake8>=3.9.0
mypy>=0.910

# Add your specific dependencies here
```

### 5.2 Create `requirements-dev.txt`
```
-r requirements.txt

# Development tools
jupyter>=1.0.0
ipython>=7.0.0
pre-commit>=2.15.0

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
```

## Step 6: Test Structure

### 6.1 Unit Tests (`tests/unit/`)
```python
# tests/unit/test_core.py
import pytest
from your_project.core import YourClass

def test_basic_functionality():
    obj = YourClass()
    assert obj.method() == expected_result

def test_edge_cases():
    # Test edge cases
    pass
```

### 6.2 Integration Tests (`tests/integration/`)
```python
# tests/integration/test_api.py
import pytest
import requests

def test_api_endpoint():
    response = requests.get("http://localhost:8000/api/test")
    assert response.status_code == 200
```

### 6.3 End-to-End Tests (`tests/e2e/`)
```python
# tests/e2e/test_full_workflow.py
import pytest

def test_complete_workflow():
    # Test your complete application workflow
    # 1. Setup
    # 2. Execute main functionality
    # 3. Verify results
    # 4. Cleanup
    pass
```

## Step 7: Configuration Files

### 7.1 Create `pytest.ini`
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
```

### 7.2 Create `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 21.9b0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/flake8
    rev: 3.9.2
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.0.1
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

## Step 8: Customization for Different Project Types

### 8.1 Web Application
- Add nginx/apache configuration
- Include database setup in Docker
- Add API testing with requests/httpx
- Include frontend testing if applicable

### 8.2 Machine Learning Project
- Use GPU-enabled base images
- Add model training/evaluation tests
- Include data validation tests
- Add performance benchmarking

### 8.3 Microservices
- Create separate Dockerfiles for each service
- Add service discovery testing
- Include load testing
- Add health check endpoints

### 8.4 Data Pipeline
- Add data validation tests
- Include ETL process testing
- Add monitoring and alerting tests
- Include data quality checks

## Step 9: Advanced Features

### 9.1 Parallel Testing
```yaml
# In GitHub Actions
strategy:
  matrix:
    python-version: [3.9, 3.10, 3.11]
    os: [ubuntu-latest, windows-latest, macos-latest]
```

### 9.2 Caching
```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### 9.3 Security Scanning
```yaml
- name: Run security scan
  uses: securecodewarrior/github-action-add-sarif@v1
  with:
    sarif-file: security-scan-results.sarif
```

## Step 10: Monitoring and Notifications

### 10.1 Slack Notifications
```yaml
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 10.2 Test Coverage Reports
```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Best Practices

1. **Start Simple**: Begin with basic unit tests and gradually add complexity
2. **Incremental Adoption**: Don't try to implement everything at once
3. **Document Everything**: Keep your testing strategy documented
4. **Regular Maintenance**: Update dependencies and test configurations regularly
5. **Monitor Performance**: Track test execution times and optimize slow tests
6. **Security First**: Always use non-root users in containers
7. **Resource Management**: Clean up temporary resources and images

## Common Pitfalls to Avoid

1. **Over-engineering**: Don't create complex testing infrastructure for simple projects
2. **Ignoring Local Development**: Ensure tests can run locally without CI
3. **Hardcoded Values**: Use environment variables for configuration
4. **Missing Cleanup**: Always clean up resources in tests
5. **Inconsistent Environments**: Ensure CI and local environments match
6. **Slow Tests**: Optimize test execution time
7. **Poor Error Messages**: Make test failures easy to debug

This guide provides a solid foundation for setting up a robust testing architecture similar to LeRobot's, but adapted to your specific project needs.
