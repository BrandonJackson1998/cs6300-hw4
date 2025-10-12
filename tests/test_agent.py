"""
Integration tests for the NursingSchedulerAgent.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.fixtures import create_test_config


class TestNursingSchedulerAgentBasic(unittest.TestCase):
    """Basic integration tests for agent initialization."""
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('src.agent.ChatGoogleGenerativeAI')
    @patch('src.agent.create_tool_calling_agent')
    @patch('src.agent.AgentExecutor')
    def test_agent_creation(self, mock_executor, mock_agent_creator, mock_chat, mock_json, mock_open):
        """Test that agent can be created with mocked dependencies."""
        from src.agent import NursingSchedulerAgent
        
        # Mock JSON loading to return minimal data
        mock_json.side_effect = [
            [{"employee_id": "E001", "name": "Test Nurse"}],  # employees
            [{"patient_id": "P001", "name": "Test Patient"}]   # patients
        ]
        
        # Create agent
        agent = NursingSchedulerAgent()
        
        # Verify it was created
        self.assertIsNotNone(agent)
        
    @patch('builtins.open')
    @patch('json.load')
    @patch('src.agent.ChatGoogleGenerativeAI')
    @patch('src.agent.create_tool_calling_agent')
    @patch('src.agent.AgentExecutor')
    def test_tools_creation(self, mock_executor, mock_agent_creator, mock_chat, mock_json, mock_open):
        """Test that agent creates the expected 5 tools."""
        from src.agent import NursingSchedulerAgent
        
        # Mock JSON loading
        mock_json.side_effect = [
            [{"employee_id": "E001", "name": "Test Nurse"}],
            [{"patient_id": "P001", "name": "Test Patient"}]
        ]
        
        agent = NursingSchedulerAgent()
        tools = agent._create_tools()
        
        # Verify 5 tools are created
        self.assertEqual(len(tools), 5)
        
        # Verify tool names
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            'validate_schedule',
            'score_schedule', 
            'generate_schedule',
            'analyze_staffing',
            'compare_schedules'
        ]
        
        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names)


class TestAgentMockIntegration(unittest.TestCase):
    """Test agent with fully mocked workflow."""
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('src.agent.ChatGoogleGenerativeAI')
    @patch('src.agent.create_tool_calling_agent')
    @patch('src.agent.AgentExecutor')
    def test_run_method_with_mocks(self, mock_executor, mock_agent_creator, mock_chat, mock_json, mock_open):
        """Test agent run method with mocked LLM response."""
        from src.agent import NursingSchedulerAgent
        
        # Mock JSON loading
        mock_json.side_effect = [
            [{"employee_id": "E001", "name": "Test Nurse"}],
            [{"patient_id": "P001", "name": "Test Patient"}]
        ]
        
        # Mock agent executor response
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance.invoke.return_value = {
            'output': 'Successfully generated a nursing schedule with 95% coverage.'
        }
        
        agent = NursingSchedulerAgent()
        result = agent.run("Create a schedule for next week")
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('output', result)
        
        # Verify executor was called
        mock_executor_instance.invoke.assert_called_once()


if __name__ == '__main__':
    unittest.main()