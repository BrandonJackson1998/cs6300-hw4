"""
Tests for the ScheduleGenerator class.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools import ScheduleGenerator
from tests.fixtures import SAMPLE_EMPLOYEES, SAMPLE_PATIENTS, SAMPLE_SCHEDULE, create_test_config


class TestScheduleGenerator(unittest.TestCase):
    """Test cases for ScheduleGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_test_config()
        self.generator = ScheduleGenerator(SAMPLE_EMPLOYEES, SAMPLE_PATIENTS, self.config)
    
    def test_init(self):
        """Test generator initialization."""
        self.assertEqual(len(self.generator.employees), 3)
        self.assertEqual(len(self.generator.patients), 2)
        self.assertEqual(self.generator.config, self.config)
    
    def test_generate_greedy_strategy(self):
        """Test greedy schedule generation."""
        result = self.generator.generate(strategy="greedy")
        
        self.assertIsInstance(result, dict)
        self.assertIn('schedule', result)
        self.assertIn('metadata', result)
        
        metadata = result['metadata']
        self.assertEqual(metadata['strategy'], 'greedy')
        self.assertIn('num_assignments', metadata)
        self.assertFalse(metadata['seed_used'])
    
    def test_generate_iterative_strategy(self):
        """Test iterative schedule generation."""
        violations = ["P001 needs care during D01S0 but no nurses assigned"]
        result = self.generator.generate(
            strategy="iterative", 
            seed_schedule=SAMPLE_SCHEDULE,
            previous_violations=violations
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('schedule', result)
        self.assertIn('metadata', result)
        
        metadata = result['metadata']
        self.assertEqual(metadata['strategy'], 'iterative')
        self.assertTrue(metadata['seed_used'])
        self.assertEqual(metadata['improved_from_violations'], 1)
    
    def test_generate_random_strategy(self):
        """Test random schedule generation."""
        result = self.generator.generate(strategy="random")
        
        self.assertIsInstance(result, dict)
        self.assertIn('schedule', result)
        self.assertEqual(result['metadata']['strategy'], 'random')
    
    def test_count_assignments(self):
        """Test assignment counting."""
        count = self.generator._count_assignments(SAMPLE_SCHEDULE)
        expected = 10  # Count assignments in SAMPLE_SCHEDULE
        self.assertEqual(count, expected)
    
    def test_count_assignments_empty(self):
        """Test counting assignments in empty schedule."""
        count = self.generator._count_assignments({})
        self.assertEqual(count, 0)
    
    def test_get_week_number(self):
        """Test week number extraction."""
        week = self.generator._get_week_number("D01S0")
        self.assertEqual(week, 0)  # Day 1 is in week 0
        
        week = self.generator._get_week_number("D08S2")
        self.assertEqual(week, 1)  # Day 8 is in week 1
    
    def test_parse_uncovered_violation(self):
        """Test parsing uncovered violation strings."""
        violation = "P001 needs care during D01S0 but no nurses assigned"
        patient_id, shift_id = self.generator._parse_uncovered_violation(violation)
        
        self.assertEqual(patient_id, "P001")
        self.assertEqual(shift_id, "D01S0")
    
    def test_parse_insufficient_violation(self):
        """Test parsing insufficient violation strings."""
        violation = "P001 needs 2 nurses during D01S0 but only 1 assigned"
        patient_id, shift_id, needed = self.generator._parse_insufficient_violation(
            violation, {"D01S0": {"P001": ["E001"]}}
        )
        
        self.assertEqual(patient_id, "P001")
        self.assertEqual(shift_id, "D01S0")
        self.assertEqual(needed, 1)  # 2 needed - 1 assigned = 1 needed
    
    def test_parse_skill_gap_violation(self):
        """Test parsing skill gap violation strings."""
        violation = "P001 requires skills {Wound Care} during D01S0 but team lacks them"
        patient_id, shift_id = self.generator._parse_skill_gap_violation(violation)
        
        self.assertEqual(patient_id, "P001")
        self.assertEqual(shift_id, "D01S0")
    
    def test_parse_level_violation(self):
        """Test parsing level violation strings."""
        violation = "P001 requires level 3 during D01S0 but highest assigned is 2"
        patient_id, shift_id = self.generator._parse_level_violation(violation)
        
        self.assertEqual(patient_id, "P001")
        self.assertEqual(shift_id, "D01S0")


if __name__ == '__main__':
    unittest.main()