"""
Test fixtures and utilities for nursing scheduler tests.
"""

import json
from typing import Dict, List


# Test data fixtures
SAMPLE_EMPLOYEES = [
    {
        "employee_id": "E001",
        "name": "Alice Johnson",
        "level": 2,
        "skills": ["Medication Administration", "Wound Care"],
        "hourly_pay_rate": 32.0,
        "max_hours_per_week": 40,
        "min_hours_per_week": 20,
        "available_shifts": ["D01S0", "D01S1", "D02S0", "D02S1"],
        "excluded_patients": [],
        "excluded_employees": []
    },
    {
        "employee_id": "E002", 
        "name": "Bob Smith",
        "level": 1,
        "skills": ["Medication Administration"],
        "hourly_pay_rate": 28.0,
        "max_hours_per_week": 32,
        "min_hours_per_week": 16,
        "available_shifts": ["D01S0", "D01S2", "D02S0", "D02S2"],
        "excluded_patients": [],
        "excluded_employees": []
    },
    {
        "employee_id": "E003",
        "name": "Carol Davis",
        "level": 3,
        "skills": ["Medication Administration", "Wound Care", "IV Therapy"],
        "hourly_pay_rate": 38.0,
        "max_hours_per_week": 40,
        "min_hours_per_week": 32,
        "available_shifts": ["D01S1", "D01S2", "D02S1", "D02S2"],
        "excluded_patients": [],
        "excluded_employees": []
    }
]

SAMPLE_PATIENTS = [
    {
        "patient_id": "P001",
        "name": "John Doe",
        "age": 65,
        "min_level": 2,
        "required_skills": ["Medication Administration"],
        "nurses_needed": 2,
        "care_shifts": ["D01S0", "D01S1", "D02S0", "D02S1"],
        "excluded_employees": []
    },
    {
        "patient_id": "P002",
        "name": "Jane Smith", 
        "age": 78,
        "min_level": 1,
        "required_skills": ["Wound Care"],
        "nurses_needed": 1,
        "care_shifts": ["D01S0", "D02S0"],
        "excluded_employees": []
    }
]

SAMPLE_SCHEDULE = {
    "D01S0": {
        "P001": ["E001", "E002"],
        "P002": ["E003"]
    },
    "D01S1": {
        "P001": ["E001", "E003"]
    },
    "D02S0": {
        "P001": ["E002", "E003"],
        "P002": ["E001"]
    },
    "D02S1": {
        "P001": ["E001", "E003"]
    }
}

INVALID_SCHEDULE = {
    "D01S0": {
        "P001": ["E001"],  # Need 2 nurses but only 1 assigned
        "P002": []         # Need 1 nurse but none assigned
    }
}


def create_test_config():
    """Create a test configuration object."""
    return {
        "max_consecutive_shifts": 3,
        "enable_budget_mode": False,
        "optimization_weights": {
            "cost": 1.0,
            "continuity": 2.0,
            "fairness": 1.0,
            "overtime": 1.5
        }
    }


def save_test_data(employees: List[Dict], patients: List[Dict], base_path: str = "tests/data"):
    """Save test data to JSON files for testing."""
    import os
    os.makedirs(base_path, exist_ok=True)
    
    with open(f"{base_path}/test_employees.json", "w") as f:
        json.dump(employees, f, indent=2)
    
    with open(f"{base_path}/test_patients.json", "w") as f:
        json.dump(patients, f, indent=2)


def mock_llm_response(content: str = "Mocked LLM response"):
    """Create a mock LLM response object."""
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    return MockResponse(content)