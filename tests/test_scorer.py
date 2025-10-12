"""
Tests for the ScheduleScorer class.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tools import ScheduleScorer
from tests.fixtures import SAMPLE_EMPLOYEES, SAMPLE_PATIENTS, SAMPLE_SCHEDULE


class TestScheduleScorer(unittest.TestCase):
    """Test cases for ScheduleScorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ScheduleScorer(SAMPLE_EMPLOYEES, SAMPLE_PATIENTS)
    
    def test_init(self):
        """Test scorer initialization."""
        self.assertEqual(len(self.scorer.employees), 3)
        self.assertEqual(len(self.scorer.patients), 2)
    
    def test_score_schedule_default_weights(self):
        """Test scoring with default weights."""
        result = self.scorer.score(SAMPLE_SCHEDULE)
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_score', result)
        self.assertIn('breakdown', result)
        
        breakdown = result['breakdown']
        self.assertIn('total_cost', breakdown)
        self.assertIn('continuity_penalty', breakdown)
        self.assertIn('fairness_penalty', breakdown)
        self.assertIn('overtime_penalty', breakdown)
    
    def test_score_schedule_custom_weights(self):
        """Test scoring with custom weights."""
        custom_weights = {
            'cost': 2.0,
            'continuity': 1.0,
            'fairness': 1.5,
            'overtime': 0.5
        }
        
        result = self.scorer.score(SAMPLE_SCHEDULE, custom_weights)
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_score', result)
        self.assertGreater(result['total_score'], 0)
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        cost = self.scorer._calculate_cost(SAMPLE_SCHEDULE)
        
        self.assertIsInstance(cost, (int, float))
        self.assertGreater(cost, 0)  # Should have some cost
    
    def test_calculate_continuity_penalty(self):
        """Test continuity penalty calculation."""
        penalty = self.scorer._calculate_continuity_penalty(SAMPLE_SCHEDULE)
        
        self.assertIsInstance(penalty, (int, float))
        self.assertGreaterEqual(penalty, 0)  # Penalty should be non-negative
    
    def test_calculate_fairness_penalty(self):
        """Test fairness penalty calculation."""
        penalty = self.scorer._calculate_fairness_penalty(SAMPLE_SCHEDULE)
        
        self.assertIsInstance(penalty, (int, float))
        self.assertGreaterEqual(penalty, 0)  # Penalty should be non-negative
    
    def test_calculate_overtime_penalty(self):
        """Test overtime penalty calculation."""
        penalty = self.scorer._calculate_overtime_penalty(SAMPLE_SCHEDULE)
        
        self.assertIsInstance(penalty, (int, float))
        self.assertGreaterEqual(penalty, 0)  # Penalty should be non-negative
    
    def test_empty_schedule_scoring(self):
        """Test scoring an empty schedule."""
        result = self.scorer.score({})
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['breakdown']['total_cost'], 0)


if __name__ == '__main__':
    unittest.main()