# Nursing Facility Schedule Optimizer

An intelligent AI-powered system for optimizing nursing shift schedules using Google Gemini and advanced constraint-based algorithms. Features a streamlined 5-tool architecture with comprehensive testing and monitoring capabilities.

## 🌟 Features

### Core Capabilities
- **AI-Powered Optimization**: Uses Google Gemini 2.5 Flash for intelligent decision-making
- **5-Tool Architecture**: Streamlined system with Constraint Validator, Schedule Scorer, Schedule Generator, Staffing Analyzer, and Schedule Comparator
- **Constraint-Based Scheduling**: Handles hard constraints (coverage, skills, levels) and soft optimization (cost, fairness)
- **Comprehensive Testing**: Unit tests with mocked dependencies plus integration tests for full workflow validation
- **Skill Efficiency Optimization**: 8-tier employee priority system with team-aware skill waste reduction
- **Rate-Limited API Calls**: Prevents quota exhaustion with intelligent call management
- **Real-time Performance Monitoring**: Tracks optimization progress and system metrics

### Advanced Features
- **Schedule Comparison Tool**: Objective LLM-based decision making for schedule improvements
- **Token Optimization**: Efficient API usage to stay within budget constraints
- **Iterative Improvement**: Continuously refines schedules based on constraint violations
- **Comprehensive Validation**: Separates critical patient violations from employee optimization
- **Visualization Tools**: Text and HTML schedule visualization with detailed metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Google Gemini API key
- Virtual environment support

### Installation

```bash
# Clone repository
git clone <repository-url>
cd cs6300-hw4

# Install environment (macOS)
make install-mac

# Install environment (Linux/Ubuntu)  
make install

# Set up API key
export GEMINI_API_KEY="your-api-key-here"
```

### Generate Test Data
```bash
make generate-data
```

### Run the Scheduler
```bash
make agent
```

### Run Tests
```bash
# Fast unit tests with mocked dependencies
make test-unit

# Integration tests for agent workflow
make test-integration

# All tests (unit + integration)
make test-all

# End-to-end tests with real LLM calls
make tests
```
```bash
make agent
```

## 📋 Usage Examples

### Basic Schedule Generation
```python
from src.agent import NursingSchedulerAgent

agent = NursingSchedulerAgent()
result = agent.run("Generate an optimized schedule for this week")
```

### Advanced Configuration
```python
from src.config import load_config
from src.monitoring import PerformanceMonitor

# Load custom configuration
config = load_config("config/custom_config.json")

# Initialize with monitoring
monitor = PerformanceMonitor()
agent = NursingSchedulerAgent(config=config, monitor=monitor)
```

## 🔧 Configuration

The system supports comprehensive configuration through JSON files and environment variables:

### Core Settings
- **Scheduling**: Max nurses per patient, consecutive shift limits, skill efficiency weights
- **Scoring**: Cost, continuity, fairness, and overtime optimization weights  
- **API**: Model selection, rate limiting, token optimization
- **Generation**: Iteration limits, convergence criteria, stopping conditions

### Environment Variables
```bash
GEMINI_API_KEY=your-api-key
SCHEDULER_MODEL=gemini-2.5-flash
SCHEDULER_RATE_LIMIT=9
SCHEDULER_MAX_ITERATIONS=10
SCHEDULER_LOG_DIR=logs
```

### Available Make Commands
```bash
# Environment Setup
make install-mac        # Install on macOS with Homebrew
make install           # Install on Linux/Ubuntu with apt
make generate-data     # Generate test employee/patient data

# Running the System
make agent             # Run interactive scheduling agent
make tests             # Run end-to-end tests with real LLM calls

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

The system uses a streamlined 5-tool architecture for optimal performance:

1. **Constraint Validator** (`src/tools.py`)
   - Validates hard constraints (coverage, skills, availability)
   - Separates critical patient violations from employee optimizations
   - Tracks violation types for targeted improvements
   - Returns detailed violation categorization

2. **Schedule Scorer** (`src/tools.py`)
   - Multi-objective optimization (cost, continuity, fairness, overtime)
   - Configurable weight system for different priorities
   - Normalization for fair comparison across schedules
   - Detailed component scoring breakdown

3. **Schedule Generator** (`src/tools.py`)
   - **Greedy Strategy**: Efficient initial schedule creation
   - **Iterative Strategy**: Targeted improvement based on violations
   - **Random Strategy**: Exploration for optimization
   - 8-tier employee priority system with skill efficiency
   - Team-aware skill waste and duplication penalties

4. **Staffing Analyzer** (`src/tools.py`)
   - Analyzes staffing needs based on violations and utilization
   - Provides hiring/firing recommendations
   - Tracks employee utilization patterns
   - Identifies skill gaps and overstaffing

5. **Schedule Comparator** (`src/tools.py`)
   - Objective LLM-based schedule comparison
   - Multi-criteria decision making
   - Explains improvement rationale
   - Supports iterative optimization

### Core Agent (`src/agent.py`)
- **Google Gemini Integration**: Uses Gemini 2.5 Flash with LangChain
- **Tool Orchestration**: Manages the 5 specialized tools
- **Rate Limiting**: Prevents API quota exhaustion (9 calls/minute)
- **Token Optimization**: Efficient API usage
- **Session Management**: Handles complex multi-step workflows

### Key Algorithms

#### Skill Efficiency Optimization
- **Team-Aware Priority Scoring**: Considers skill redundancy when building teams
- **Waste Penalty**: Penalizes unused skills to improve efficiency
- **Duplication Penalty**: Prevents skill overlap in patient teams
- **Load Balancing**: Prioritizes underutilized employees

#### Constraint Handling
- **Priority Separation**: Patient violations (CRITICAL) vs employee violations (SOFT)
- **Iterative Resolution**: Focuses on critical constraints first
- **Graceful Degradation**: Accepts employee constraint violations for patient safety

#### Multi-Strategy Generation
- **Greedy**: Fast, efficient initial solutions
- **Iterative**: Targeted violation fixing with seed schedules
- **Random**: Exploration for local optima escape

## 📊 Performance Metrics

The system tracks comprehensive metrics across multiple dimensions:

### Quality Metrics
- **Patient Violations**: Must be 0 for valid schedules
- **Employee Violations**: Soft constraints for optimization
- **Schedule Score Breakdown**: Cost, continuity, fairness, overtime components
- **Improvement Trends**: Quality progression over iterations

### Performance Metrics  
- **Generation Time**: Average time per schedule creation
- **API Efficiency**: Calls per minute and rate limiting effectiveness
- **Token Usage**: Optimization for cost control
- **Convergence Analysis**: Iteration effectiveness and stopping criteria

### Tool-Specific Metrics
- **Constraint Validator**: Violation detection accuracy and categorization
- **Schedule Scorer**: Component scoring consistency and normalization
- **Schedule Generator**: Strategy effectiveness (greedy vs iterative vs random)
- **Staffing Analyzer**: Utilization analysis and recommendation quality
- **Schedule Comparator**: Decision accuracy and explanation quality

### Testing Performance
- **Unit Tests**: 29 tests in ~0.001 seconds (mocked dependencies)
- **Integration Tests**: 3 tests in ~0.441 seconds (mocked LLM calls)
- **End-to-End Tests**: Variable timing based on real API calls
- **Test Coverage**: Comprehensive coverage of all 5 tools plus agent

## 🧪 Testing

### Comprehensive Test Suite

The system includes a robust testing framework with multiple test types:

#### Unit Tests (29 tests)
```bash
make test-unit          # Fast execution with mocked dependencies
```

**Coverage:**
- **ConstraintValidator**: Schedule validation, violation detection, consecutive shift checking
- **ScheduleScorer**: Cost calculation, continuity/fairness penalties, custom weights
- **ScheduleGenerator**: All 3 strategies (greedy/iterative/random), violation parsing
- **StaffingAnalyzer**: Basic analysis functionality, utilization calculations

#### Integration Tests (3 tests)  
```bash
make test-integration   # Agent workflow with mocked LLM calls
```

**Coverage:**
- Agent initialization and tool creation
- Mocked LLM interaction workflow
- 5-tool system verification

#### End-to-End Tests
```bash
make tests             # Real LLM calls for system validation
```

**Coverage:**
- Complete workflow scenarios
- Real Google Gemini API integration
- Performance benchmarking

#### All Tests
```bash
make test-all          # Complete test suite (32 tests)
```

### Test Architecture

- **Mocked Dependencies**: Unit tests use `unittest.mock` for external dependencies
- **Test Fixtures**: Realistic sample data in `tests/fixtures.py`
- **Custom Test Runner**: `tests/run_tests.py` with flexible execution options
- **CI/CD Ready**: Fast unit tests suitable for continuous integration

### Test Categories
- **Unit Tests**: Individual component validation with mocks
- **Integration Tests**: Agent workflow without external API calls  
- **Performance Tests**: Rate limiting and optimization validation
- **Constraint Tests**: Edge cases and violation handling scenarios

## 📈 Visualization

### Generate Schedule Visualization
```bash
# Text-based visualization
make visualize FILE=schedules/schedule_20250108_143022.json

# HTML visualization  
make visualize-html FILE=schedules/schedule_20250108_143022.json
```

## 🔍 Monitoring & Debugging

### Real-time Monitoring
The system provides live performance feedback:
```
🔍 PERFORMANCE MONITOR
============================================================
Session Duration: 45.2s
API Calls: 23 (30.5/min)
Schedules Generated: 5
Best Results:
  Patient Violations: 0
  Employee Violations: 3
  Total Score: 1,247.50
Efficiency:
  Schedules/min: 6.6
============================================================
```

### Log Analysis
Check `logs/` directory for:
- Session performance logs
- Quality improvement history
- Detailed iteration tracking

## 🚀 Production Deployment

### Key Production Features
- **Rate Limiting**: Prevents API quota exhaustion (9 calls/minute default)
- **Token Optimization**: Minimizes API costs through efficient prompting
- **Error Handling**: Graceful degradation on API failures
- **Comprehensive Testing**: Fast unit tests for CI/CD + integration tests for validation
- **Monitoring**: Performance tracking and quality metrics
- **Configuration**: Environment-based settings for different deployment stages

### Recommended Settings
```json
{
  "api": {
    "calls_per_minute": 9,
    "enable_token_optimization": true,
    "max_retries": 3
  },
  "generation": {
    "max_iterations": 10,
    "target_patient_violations": 0,
    "max_generation_time_seconds": 300
  },
  "testing": {
    "enable_unit_tests_in_ci": true,
    "run_integration_tests": false,
    "mock_external_dependencies": true
  }
}
```

### Development Workflow
1. **Fast Feedback**: `make test-unit` for immediate validation (0.001s)
2. **Integration Check**: `make test-integration` for workflow validation (0.441s)
3. **Full Validation**: `make test-all` for complete coverage (0.005s)
4. **System Testing**: `make tests` for end-to-end validation with real APIs

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    - name: Install dependencies
      run: make install
    - name: Run unit tests
      run: make test-unit
    - name: Run integration tests  
      run: make test-integration
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass:
   ```bash
   make test-unit        # Must pass for CI/CD
   make test-integration # Must pass for workflow validation
   make test-all         # Complete test suite
   ```
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

### Development Guidelines

- **Write Tests First**: Add unit tests for new components in `tests/`
- **Use Mocks**: Keep unit tests fast with mocked external dependencies
- **Follow Architecture**: Maintain the 5-tool system separation of concerns
- **Document Changes**: Update README for new features or architectural changes
- **Test Integration**: Ensure new tools integrate properly with the agent

### File Structure
```
cs6300-hw4/
├── src/
│   ├── agent.py          # Main AI agent with 5-tool orchestration
│   ├── tools.py          # 5 core tools implementation
│   └── config.py         # Configuration management
├── tests/
│   ├── fixtures.py       # Test data and utilities
│   ├── test_*.py         # Unit tests for each tool
│   └── run_tests.py      # Custom test runner
├── scripts/
│   ├── generate_data.py  # Test data generation
│   ├── run_tests.py      # End-to-end system tests
│   └── visualize_schedule.py  # Schedule visualization
├── data/                 # Generated employee/patient data
├── schedules/           # Generated schedule outputs
└── Makefile            # Build and test automation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: Advanced AI capabilities for intelligent scheduling
- **LangChain**: Framework for AI agent development
- **Constraint Programming**: Academic research in nurse scheduling optimization