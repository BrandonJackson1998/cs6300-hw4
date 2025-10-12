"""
Configuration Management for Nursing Scheduler

Centralized configuration with validation and environment support.
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SchedulingConfig:
    """Core scheduling algorithm configuration"""
    max_nurses_per_patient: int = 3
    max_consecutive_shifts: int = 3
    default_shift_duration_hours: int = 4
    
    # Skill efficiency optimization
    enable_skill_efficiency: bool = True
    skill_waste_penalty_weight: float = 2.0
    skill_duplication_penalty_weight: float = 3.0
    
    # Load balancing
    utilization_warning_threshold: float = 0.6
    utilization_danger_threshold: float = 0.8
    underutilization_bonus_weight: float = 1.5
    
    # Emergency override for critical coverage
    enable_emergency_override: bool = True


@dataclass  
class ScoringConfig:
    """Schedule scoring weights and thresholds"""
    cost_weight: float = 1.0
    continuity_weight: float = 2.0
    fairness_weight: float = 1.0
    overtime_weight: float = 1.5
    
    # Normalization factors
    cost_normalization_factor: float = 1000.0
    ideal_nurses_per_patient: float = 3.0
    overtime_threshold_hours: float = 40.0


@dataclass
class APIConfig:
    """API and rate limiting configuration"""
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    calls_per_minute: int = 9
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Token optimization
    enable_token_optimization: bool = True
    max_violations_in_prompt: int = 10


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    enable_monitoring: bool = True
    log_directory: str = "logs"
    save_session_logs: bool = True
    print_realtime_status: bool = True
    quality_tracking_enabled: bool = True


@dataclass
class GenerationConfig:
    """Schedule generation strategy configuration"""
    default_strategy: str = "greedy"
    max_iterations: int = 10
    convergence_patience: int = 3  # Iterations without improvement
    
    # Stopping criteria
    target_patient_violations: int = 0
    acceptable_employee_violations: int = 5
    max_generation_time_seconds: float = 300.0  # 5 minutes


@dataclass
class NursingSchedulerConfig:
    """Complete configuration for nursing scheduler"""
    scheduling: SchedulingConfig
    scoring: ScoringConfig  
    api: APIConfig
    monitoring: MonitoringConfig
    generation: GenerationConfig
    
    # File paths
    employees_file: str = "data/employees.json"
    patients_file: str = "data/patients.json"
    config_file: str = "config/scheduler_config.json"


class ConfigManager:
    """Manages configuration loading, validation, and environment overrides"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/scheduler_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> NursingSchedulerConfig:
        """Load configuration from file with environment overrides"""
        # Start with defaults
        config_dict = self._get_default_config()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                config_dict = self._merge_configs(config_dict, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        # Apply environment overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Validate and create config object
        return self._dict_to_config(config_dict)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        default_config = NursingSchedulerConfig(
            scheduling=SchedulingConfig(),
            scoring=ScoringConfig(),
            api=APIConfig(),
            monitoring=MonitoringConfig(),
            generation=GenerationConfig()
        )
        return asdict(default_config)
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config_dict: Dict) -> Dict:
        """Apply environment variable overrides"""
        # API configuration from environment
        if "GEMINI_API_KEY" in os.environ:
            config_dict.setdefault("api", {})
        
        if "SCHEDULER_MODEL" in os.environ:
            config_dict.setdefault("api", {})["model_name"] = os.environ["SCHEDULER_MODEL"]
        
        if "SCHEDULER_RATE_LIMIT" in os.environ:
            config_dict.setdefault("api", {})["calls_per_minute"] = int(os.environ["SCHEDULER_RATE_LIMIT"])
        
        # Monitoring configuration
        if "SCHEDULER_DISABLE_MONITORING" in os.environ:
            config_dict.setdefault("monitoring", {})["enable_monitoring"] = False
        
        if "SCHEDULER_LOG_DIR" in os.environ:
            config_dict.setdefault("monitoring", {})["log_directory"] = os.environ["SCHEDULER_LOG_DIR"]
        
        # Generation configuration
        if "SCHEDULER_MAX_ITERATIONS" in os.environ:
            config_dict.setdefault("generation", {})["max_iterations"] = int(os.environ["SCHEDULER_MAX_ITERATIONS"])
        
        if "SCHEDULER_TARGET_PATIENT_VIOLATIONS" in os.environ:
            config_dict.setdefault("generation", {})["target_patient_violations"] = int(os.environ["SCHEDULER_TARGET_PATIENT_VIOLATIONS"])
        
        # File paths
        if "SCHEDULER_EMPLOYEES_FILE" in os.environ:
            config_dict["employees_file"] = os.environ["SCHEDULER_EMPLOYEES_FILE"]
        
        if "SCHEDULER_PATIENTS_FILE" in os.environ:
            config_dict["patients_file"] = os.environ["SCHEDULER_PATIENTS_FILE"]
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict) -> NursingSchedulerConfig:
        """Convert dictionary to typed configuration object"""
        try:
            return NursingSchedulerConfig(
                scheduling=SchedulingConfig(**config_dict.get("scheduling", {})),
                scoring=ScoringConfig(**config_dict.get("scoring", {})),
                api=APIConfig(**config_dict.get("api", {})),
                monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
                generation=GenerationConfig(**config_dict.get("generation", {})),
                employees_file=config_dict.get("employees_file", "data/employees.json"),
                patients_file=config_dict.get("patients_file", "data/patients.json"),
                config_file=config_dict.get("config_file", "config/scheduler_config.json")
            )
        except TypeError as e:
            raise ValueError(f"Invalid configuration: {e}")
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        file_path = config_file or self.config_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to dictionary and save
        config_dict = asdict(self.config)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {file_path}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate file paths
        if not os.path.exists(self.config.employees_file):
            issues.append(f"Employees file not found: {self.config.employees_file}")
        
        if not os.path.exists(self.config.patients_file):
            issues.append(f"Patients file not found: {self.config.patients_file}")
        
        # Validate API configuration
        if not os.getenv("GEMINI_API_KEY"):
            issues.append("GEMINI_API_KEY environment variable not set")
        
        if self.config.api.calls_per_minute <= 0:
            issues.append("API calls_per_minute must be positive")
        
        if self.config.api.temperature < 0 or self.config.api.temperature > 1:
            issues.append("API temperature must be between 0 and 1")
        
        # Validate scheduling configuration
        if self.config.scheduling.max_nurses_per_patient <= 0:
            issues.append("max_nurses_per_patient must be positive")
        
        if self.config.scheduling.max_consecutive_shifts <= 0:
            issues.append("max_consecutive_shifts must be positive")
        
        # Validate scoring weights
        scoring = self.config.scoring
        if any(w < 0 for w in [scoring.cost_weight, scoring.continuity_weight, 
                              scoring.fairness_weight, scoring.overtime_weight]):
            issues.append("All scoring weights must be non-negative")
        
        # Validate generation configuration
        if self.config.generation.max_iterations <= 0:
            issues.append("max_iterations must be positive")
        
        if self.config.generation.convergence_patience <= 0:
            issues.append("convergence_patience must be positive")
        
        return issues
    
    def print_config_summary(self):
        """Print human-readable configuration summary"""
        print("\n" + "="*60)
        print("🔧 SCHEDULER CONFIGURATION")
        print("="*60)
        
        print(f"\n📋 SCHEDULING:")
        print(f"  Max nurses per patient: {self.config.scheduling.max_nurses_per_patient}")
        print(f"  Max consecutive shifts: {self.config.scheduling.max_consecutive_shifts}")
        print(f"  Skill efficiency enabled: {self.config.scheduling.enable_skill_efficiency}")
        
        print(f"\n📊 SCORING WEIGHTS:")
        print(f"  Cost: {self.config.scoring.cost_weight}")
        print(f"  Continuity: {self.config.scoring.continuity_weight}")
        print(f"  Fairness: {self.config.scoring.fairness_weight}")
        print(f"  Overtime: {self.config.scoring.overtime_weight}")
        
        print(f"\n🔌 API:")
        print(f"  Model: {self.config.api.model_name}")
        print(f"  Rate limit: {self.config.api.calls_per_minute}/min")
        print(f"  Temperature: {self.config.api.temperature}")
        
        print(f"\n⚡ GENERATION:")
        print(f"  Strategy: {self.config.generation.default_strategy}")
        print(f"  Max iterations: {self.config.generation.max_iterations}")
        print(f"  Target patient violations: {self.config.generation.target_patient_violations}")
        
        print(f"\n📁 FILES:")
        print(f"  Employees: {self.config.employees_file}")
        print(f"  Patients: {self.config.patients_file}")
        
        # Validation
        issues = self.validate_config()
        if issues:
            print(f"\n⚠️  CONFIGURATION ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ Configuration is valid")
        
        print("="*60)


# Convenience function for easy access
def load_config(config_file: Optional[str] = None) -> NursingSchedulerConfig:
    """Load and return scheduler configuration"""
    manager = ConfigManager(config_file)
    return manager.config


# Example usage
if __name__ == "__main__":
    # Create and save default configuration
    manager = ConfigManager()
    manager.print_config_summary()
    
    # Save default config for reference
    os.makedirs("config", exist_ok=True)
    manager.save_config("config/default_config.json")