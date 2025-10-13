# Nursing Scheduler Agent

An AI-powered scheduling agent that generates optimal nursing schedules for healthcare facilities using constraint-based optimization and iterative refinement.

**Key Capabilities:**
- Generates initial schedules using intelligent heuristics
- Validates against hard constraints (patient safety) and soft constraints (employee preferences)  
- Iteratively refines schedules to minimize violations
- Provides actionable staffing recommendations

**Scheduling Rules:** Each patient requires 1-3 nurses per shift with specific skill levels. Each nurse can only care for one patient per shift. The agent prioritizes patient safety while optimizing for cost, continuity of care, and employee satisfaction.

## 🚀 Quick Start

## Prerequisites

- **Python 3.12+**  
  Make sure you are using Python 3.12 or later.

```bash
python3 --version
```

- **API Keys (stored in `.env`)**  
  You will need two sets of keys:

1. **Gemini API key**  
   - Create via [Google AI Studio](https://aistudio.google.com/) → "Get API key"  
   - Save in `.env` as:  
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

2. **LangSmith API Key**  
   - Create via [LangSmith](https://smith.langchain.com/)  
   - Save in `.env` as:  
     ```
     LANGCHAIN_API_KEY=your_api_key
     ```


### Generate Test Data
```bash
make generate-data
```

This creates random employee and patient data files. You can modify the numbers in `scripts/generate_data.py` (default: 80 employees, 8 patients).

### Run the Scheduler
```bash
make agent
```

### 💡 Example Usage

```bash
# 1. Generate test data (80 employees, 8 patients)
make generate-data

# 2. Run the scheduling agent
make agent

# 3. Visualize the generated schedule
make visualize FILE=schedules/schedule_20251011_025615.json
```

The agent will interactively ask what you want to optimize and generate schedules accordingly.

## Run Tests
```bash
# Fast unit tests with mocked dependencies
make test-unit

# Integration tests for agent workflow
make test-integration

# All tests (unit + integration)
make test-all

```

## Available Make Commands
```bash
# Environment Setup
make install-mac        # Install on macOS with Homebrew
make install           # Install on Linux/Ubuntu with apt
make generate-data     # Generate test employee/patient data

# Running the System
make agent             # Run interactive scheduling agent

# Testing (Fast, No API Calls)
make test-unit         # Unit tests with mocked dependencies
make test-integration  # Integration tests with mocked LLM
make test-all          # All unit + integration tests

# Visualization
make visualize FILE=path/to/schedule.json      # Text visualization
make visualize-html FILE=path/to/schedule.json # HTML visualization

# Maintenance
make clean             # Clean generated files and cache
make clean-all         # Deep clean including logs/schedules
```

See `src/config.py` for complete configuration options.

## 🏗️ Architecture

### 5-Tool System

The system uses a streamlined 5-tool architecture:

1. **Constraint Validator** - Validates patient safety and coverage requirements
2. **Schedule Scorer** - Multi-objective optimization (cost, continuity, fairness)  
3. **Schedule Generator** - Creates schedules using greedy, iterative, or random strategies
4. **Staffing Analyzer** - Provides hiring/firing recommendations and utilization analysis
5. **Schedule Comparator** - LLM-based schedule comparison and improvement suggestions

### Core Agent (`src/agent.py`)
- **Google Gemini Integration**: Uses Gemini 2.5 Flash with LangChain
- **Tool Orchestration**: Manages the 5 specialized tools
- **Rate Limiting**: Prevents API quota exhaustion (9 calls/minute)
- **Token Optimization**: Efficient API usage
- **Session Management**: Handles complex multi-step workflows

## 🧪 Testing

#### Unit Tests
```bash
make test-unit          # Fast execution with mocked dependencies
```

#### Integration Tests
```bash
make test-integration   # Agent workflow with mocked LLM calls
```

#### All Tests
```bash
make test-all          # Complete test suite (32 tests)
```

## 📈 Visualization

After running the agent, you can generate a basic text or HTML visualization of the schedule as described below.

### Generate Schedule Visualization
```bash
# Text-based visualization
make visualize FILE=schedules/schedule_20250108_143022.json

# HTML visualization  
make visualize-html FILE=schedules/schedule_20250108_143022.json
```

## File Structure
```
cs6300-hw4/
│
├── 📋 Core Files
│   ├── README.md          # Project documentation
│   ├── Makefile           # Commands (test-unit, test-all, agent, etc.)
│   ├── requirements.txt   # Python dependencies
│   └── .env               # API keys
│
├── 🧠 Source Code (src/)
│   ├── agent.py          # Main AI agent
│   ├── tools.py          # 5 scheduling tools
│   ├── config.py         # Configuration
│   └── monitoring.py     # Performance tracking
│
├── 🧪 Tests (tests/)
│   ├── fixtures.py       # Test data
│   ├── run_tests.py      # Test runner
│   └── test_*.py         # 32 tests (5 files)
│
├── 🛠️ Scripts (scripts/)
│   ├── generate_data.py       # Create test data
│   └── visualize_schedule.py  # Make charts
│
├── 📊 Data
│   ├── data/             # employees.json, patients.json
│   ├── schedules/        # 57 generated schedules
│   ├── visualizations/   # 68 charts (HTML + text)
│   └── logs/             # 119 execution logs
│
└── ⚙️ Config
    └── config/           # Configuration files
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: Advanced AI capabilities for intelligent scheduling
- **LangChain**: Framework for AI agent development
- **Claude AI**: Code generation