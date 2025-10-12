VENV := .virtual_environment

all: help

help:
	@echo
	@echo "Targets:"
	@echo "install                     - Install environment necessary to support this project."
	@echo "install-deb                 - Install OS packages necessary to support this project. Assumes apt/dpkg package management system."
	@echo "install-pip                 - Install Python packages necessary to support this project."
	@echo "install-mac                 - Install packages for macOS using Homebrew."
	@echo "generate-data               - Generate employee and patient test data for scheduler."
	@echo "agent                       - Run the nursing scheduler agent with example query."
	@echo "test-unit                   - Run unit tests with mocked dependencies (fast)."
	@echo "test-integration            - Run integration tests with mocked LLM calls."
	@echo "test-all                    - Run all tests (unit + integration)."
	@echo "visualize                   - Visualize the most recent schedule (requires schedule file)."
	@echo "visualize-html              - Create HTML visualization of the most recent schedule."
	@echo "clean-all                   - Clean everything including logs and visualizations."
	@echo "clean                       - Clean generated data files and Python cache."
	@echo

$(VENV):
	python3.12 -m venv $(VENV)

install: install-deb install-pip

install-deb:
	@echo python3.12-venv is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python3.12-venv ffmpeg; do \
		dpkg -l | egrep '^ii *'$${package}' ' 2>&1 > /dev/null || sudo apt install $${package}; \
	done

install-pip: $(VENV)
	source $(VENV)/bin/activate; pip3 install --upgrade -r requirements.txt

install-mac: install-deb-mac install-pip
	
install-deb-mac:
	@echo python@3.12 is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python@3.12 ffmpeg; do \
		brew list --versions $${package} 2>&1 > /dev/null || brew install $${package}; \
	done

agent:
	source $(VENV)/bin/activate; python -m src.agent

generate-data:
	source $(VENV)/bin/activate; python -m scripts.generate_data

test-unit:
	source $(VENV)/bin/activate; python tests/run_tests.py --unit

test-integration:
	source $(VENV)/bin/activate; python tests/run_tests.py --integration

test-all:
	source $(VENV)/bin/activate; python tests/run_tests.py

visualize:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make visualize FILE=path/to/schedule.json"; \
		echo "Example: make visualize FILE=schedules/schedule_20250108_143022.json"; \
	else \
		source $(VENV)/bin/activate; python scripts/visualize_schedule.py $(FILE); \
	fi

visualize-html:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make visualize-html FILE=path/to/schedule.json"; \
		echo "Example: make visualize-html FILE=schedules/schedule_20250108_143022.json"; \
	else \
		source $(VENV)/bin/activate; python scripts/visualize_schedule.py $(FILE) --html; \
	fi

clean-all: clean
	@echo "Cleaning logs and visualizations..."
	rm -rf logs/ visualizations/ schedules/
	@echo "Deep cleanup complete"

clean:
	@echo "Cleaning generated files..."
	rm -f data/employees.json data/patients.json data/metadata.json
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete"