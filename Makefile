.PHONY: help install data features train index app test lint format clean reproduce

PYTHON := python
PIP := pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data: ## Download / prepare raw data
	$(PYTHON) -m immopilot.data.load_zurich_open
	$(PYTHON) -m immopilot.data.load_listings
	@echo " Raw data ready in data/raw/"

features: ## Build processed feature tables
	$(PYTHON) -m immopilot.features.build_features
	@echo " Processed features in data/processed/"

train: train-numeric train-cv ## Train all models

train-numeric: ## Train numeric models (Linear, RF, XGB, MLP)
	$(PYTHON) -m immopilot.models.train_baseline
	$(PYTHON) -m immopilot.models.train_rf
	$(PYTHON) -m immopilot.models.train_xgb
	$(PYTHON) -m immopilot.models.train_nn
	@echo " Numeric models in models/"

train-cv: ## Train CV condition classifier
	$(PYTHON) -m immopilot.cv.train_classifier
	@echo " CV models in models/"

index: ## Build RAG vector store
	$(PYTHON) -m immopilot.nlp.build_index
	@echo " FAISS index in models/rag/"

app: ## Launch Gradio app on http://localhost:7860
	$(PYTHON) app/app.py

test: ## Run pytest suite
	pytest tests/ -v

lint: ## Run linters
	ruff check src/ app/ tests/

format: ## Auto-format code
	ruff format src/ app/ tests/
	ruff check --fix src/ app/ tests/

clean: ## Remove generated artifacts (keeps raw data)
	rm -rf data/interim/* data/processed/* models/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

reproduce: data features train index ## Full pipeline from scratch
	@echo ""
	@echo ""
	@echo "   Full pipeline reproduced."
	@echo "  Run 'make app' to launch the UI."
	@echo ""
