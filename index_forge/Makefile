.PHONY: install run test clean help full-run

# Determine the python interpreter
PYTHON = python3
VENV = .venv
BIN = $(VENV)/bin

help:
	@echo "IndexForge Development CLI"
	@echo "--------------------------"
	@echo "make install    - Setup virtual environment and install dependencies"
	@echo "make full-run   - Execute the entire pipeline (Setup -> Ingest -> Engine -> Backtest)"
	@echo "make test       - Run the pytest suite"
	@echo "make viz        - Generate only the performance chart"
	@echo "make clean      - Remove temporary database, logs, and cache"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install pytest pytest-cov tabulate matplotlib seaborn

full-run:
	$(BIN)/python main.py --full-run

viz:
	$(BIN)/python main.py --visualize

test:
	$(BIN)/pytest tests/ --cov=index_math --cov=backtesting

clean:
	rm -rf logs/
	rm -rf assets/
	rm -f indexforge.db
	rm -f indexforge_report.md
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
