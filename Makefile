# Terminal++ Development Makefile
# =============================================================================

.PHONY: help install dev build test clean docker setup lint format security docs deploy

# Default target
.DEFAULT_GOAL := help

# Variables
COMPOSE_FILE := docker-compose.yml
COMPOSE_DEV_FILE := docker-compose.dev.yml
COMPOSE_PROD_FILE := docker-compose.prod.yml
PROJECT_NAME := terminal-plus-plus

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(GREEN)Terminal++ Development Commands$(NC)"
	@echo "================================="
	@echo ""
	@echo "$(YELLOW)Setup & Installation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(setup|install)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(dev|build|test)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(docker|up|down)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Code Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(lint|format|security)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Other:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE "(setup|install|dev|build|test|docker|up|down|lint|format|security)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Setup & Installation
# =============================================================================

setup: ## Initial project setup (run this first)
	@echo "$(GREEN)Setting up Terminal++ development environment...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(RED)⚠️  Please edit .env file with your configuration$(NC)"; \
	fi
	@if [ ! -f frontend/.env.local ]; then \
		echo "$(YELLOW)Creating frontend/.env.local...$(NC)"; \
		echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local; \
		echo "NEXT_PUBLIC_WS_URL=ws://localhost:8000" >> frontend/.env.local; \
	fi
	@echo "$(GREEN)✅ Setup complete! Run 'make install' to install dependencies$(NC)"

install: ## Install all dependencies (frontend & backend)
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	cd backend && python -m venv venv && \
		source venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt
	@echo "$(YELLOW)Installing frontend dependencies...$(NC)"
	cd frontend && npm ci
	@echo "$(GREEN)✅ Dependencies installed successfully$(NC)"

install-backend: ## Install backend dependencies only
	@echo "$(GREEN)Installing backend dependencies...$(NC)"
	cd backend && python -m venv venv && \
		source venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements.txt

install-frontend: ## Install frontend dependencies only
	@echo "$(GREEN)Installing frontend dependencies...$(NC)"
	cd frontend && npm ci

# =============================================================================
# Development
# =============================================================================

dev: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker-compose -f $(COMPOSE_DEV_FILE) up --build -d
	@echo "$(GREEN)✅ Development environment started$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Backend API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API Docs: http://localhost:8000/docs$(NC)"

dev-local: ## Start development servers locally (without Docker)
	@echo "$(GREEN)Starting local development servers...$(NC)"
	@echo "$(YELLOW)Starting database and Redis...$(NC)"
	docker-compose up -d postgres redis
	@echo "$(YELLOW)Starting backend server...$(NC)"
	cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "$(YELLOW)Starting frontend server...$(NC)"
	cd frontend && npm run dev &
	@echo "$(GREEN)✅ Local development servers started$(NC)"

dev-backend: ## Start only backend services
	@echo "$(GREEN)Starting backend development environment...$(NC)"
	docker-compose up -d postgres redis
	cd backend && source venv/bin/activate && uvicorn app.main:app --reload

dev-frontend: ## Start only frontend development
	@echo "$(GREEN)Starting frontend development server...$(NC)"
	cd frontend && npm run dev

build: ## Build all services
	@echo "$(GREEN)Building all services...$(NC)"
	docker-compose build
	cd frontend && npm run build

build-backend: ## Build backend service only
	@echo "$(GREEN)Building backend service...$(NC)"
	docker-compose build backend

build-frontend: ## Build frontend service only
	@echo "$(GREEN)Building frontend service...$(NC)"
	cd frontend && npm run build

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	make test-backend
	make test-frontend

test-backend: ## Run backend tests
	@echo "$(GREEN)Running backend tests...$(NC)"
	cd backend && source venv/bin/activate && pytest -v --cov=app --cov-report=html --cov-report=term

test-frontend: ## Run frontend tests
	@echo "$(GREEN)Running frontend tests...$(NC)"
	cd frontend && npm run test

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	docker-compose -f $(COMPOSE_FILE) run --rm backend pytest tests/integration/ -v

test-watch: ## Run tests in watch mode
	@echo "$(GREEN)Running tests in watch mode...$(NC)"
	cd backend && source venv/bin/activate && pytest-watch &
	cd frontend && npm run test:watch &

coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	cd backend && source venv/bin/activate && pytest --cov=app --cov-report=html
	cd frontend && npm run test:coverage
	@echo "$(GREEN)Coverage reports generated$(NC)"
	@echo "$(YELLOW)Backend: backend/htmlcov/index.html$(NC)"
	@echo "$(YELLOW)Frontend: frontend/coverage/lcov-report/index.html$(NC)"

# =============================================================================
# Docker Commands
# =============================================================================

docker-up: ## Start all services with Docker
	@echo "$(GREEN)Starting Docker services...$(NC)"
	docker-compose up -d

docker-down: ## Stop all Docker services
	@echo "$(GREEN)Stopping Docker services...$(NC)"
	docker-compose down

docker-restart: ## Restart all Docker services
	@echo "$(GREEN)Restarting Docker services...$(NC)"
	docker-compose restart

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-ps: ## Show running containers
	docker-compose ps

docker-shell-backend: ## Open shell in backend container
	docker-compose exec backend bash

docker-shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend sh

docker-clean: ## Clean up Docker resources
	@echo "$(GREEN)Cleaning up Docker resources...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

# =============================================================================
# Database Operations
# =============================================================================

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	cd backend && source venv/bin/activate && alembic upgrade head

db-migration: ## Create new database migration
	@echo "$(GREEN)Creating new migration...$(NC)"
	@read -p "Migration name: " name; \
	cd backend && source venv/bin/activate && alembic revision --autogenerate -m "$$name"

db-rollback: ## Rollback last migration
	@echo "$(GREEN)Rolling back last migration...$(NC)"
	cd backend && source venv/bin/activate && alembic downgrade -1

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)⚠️  This will destroy all database data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker-compose down postgres; \
		docker volume rm terminal-plus-plus_postgres_data; \
		docker-compose up -d postgres; \
		sleep 5; \
		make db-migrate; \
	fi

db-shell: ## Open database shell
	docker-compose exec postgres psql -U postgres -d terminal_plus_plus

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting for all code
	@echo "$(GREEN)Running linting...$(NC)"
	make lint-backend
	make lint-frontend

lint-backend: ## Run backend linting
	@echo "$(GREEN)Running backend linting...$(NC)"
	cd backend && source venv/bin/activate && \
		flake8 app/ && \
		mypy app/ && \
		black --check app/

lint-frontend: ## Run frontend linting
	@echo "$(GREEN)Running frontend linting...$(NC)"
	cd frontend && npm run lint

format: ## Format all code
	@echo "$(GREEN)Formatting code...$(NC)"
	make format-backend
	make format-frontend

format-backend: ## Format backend code
	@echo "$(GREEN)Formatting backend code...$(NC)"
	cd backend && source venv/bin/activate && \
		black app/ && \
		isort app/

format-frontend: ## Format frontend code
	@echo "$(GREEN)Formatting frontend code...$(NC)"
	cd frontend && npm run lint:fix

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	make security-backend
	make security-frontend

security-backend: ## Run backend security checks
	@echo "$(GREEN)Running backend security checks...$(NC)"
	cd backend && source venv/bin/activate && \
		bandit -r app/ && \
		safety check

security-frontend: ## Run frontend security checks
	@echo "$(GREEN)Running frontend security checks...$(NC)"
	cd frontend && npm audit --audit-level=high

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	cd backend && source venv/bin/activate && \
		sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)Documentation generated at backend/docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	cd backend/docs/_build/html && python -m http.server 8080

# =============================================================================
# Production & Deployment
# =============================================================================

deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(NC)"
	docker-compose -f $(COMPOSE_PROD_FILE) up -d --build
	@echo "$(GREEN)✅ Staging deployment complete$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(RED)⚠️  Production deployment$(NC)"
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$(GREEN)Deploying to production...$(NC)"; \
		docker-compose -f $(COMPOSE_PROD_FILE) up -d --build; \
		echo "$(GREEN)✅ Production deployment complete$(NC)"; \
	fi

backup-db: ## Backup production database
	@echo "$(GREEN)Creating database backup...$(NC)"
	docker-compose exec postgres pg_dump -U postgres terminal_plus_plus > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Database backup created$(NC)"

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean up development environment
	@echo "$(GREEN)Cleaning up development environment...$(NC)"
	docker-compose down -v
	cd backend && rm -rf __pycache__ .pytest_cache .coverage htmlcov/
	cd frontend && rm -rf .next node_modules/.cache coverage/
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

logs: ## Show application logs
	docker-compose logs -f backend frontend

status: ## Show system status
	@echo "$(GREEN)System Status$(NC)"
	@echo "=============="
	@echo "$(YELLOW)Docker Services:$(NC)"
	@docker-compose ps
	@echo ""
	@echo "$(YELLOW)Container Resources:$(NC)"
	@docker stats --no-stream
	@echo ""
	@echo "$(YELLOW)Disk Usage:$(NC)"
	@docker system df

monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Starting monitoring services...$(NC)"
	docker-compose up -d prometheus grafana
	@echo "$(GREEN)✅ Monitoring started$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3001 (admin/admin)$(NC)"

# =============================================================================
# Quick Commands
# =============================================================================

start: dev ## Alias for dev
stop: docker-down ## Stop all services
restart: docker-restart ## Restart all services
shell: docker-shell-backend ## Open backend shell