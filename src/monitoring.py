"""
Performance Monitoring for Nursing Scheduler

Tracks key metrics for schedule quality and system performance.
"""

import time
import json
import os
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime


class PerformanceMonitor:
    """Monitors and logs performance metrics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_start = datetime.now()
        self.metrics = {
            'generation_times': [],
            'validation_times': [],
            'api_calls': 0,
            'schedules_generated': 0,
            'best_patient_violations': float('inf'),
            'best_employee_violations': float('inf'),
            'best_total_score': float('inf'),
            'iterations': 0
        }
    
    def start_timer(self) -> float:
        """Start timing an operation"""
        return time.time()
    
    def end_timer(self, start_time: float, operation: str) -> float:
        """End timing and record duration"""
        duration = time.time() - start_time
        if operation == 'generation':
            self.metrics['generation_times'].append(duration)
        elif operation == 'validation':
            self.metrics['validation_times'].append(duration)
        return duration
    
    def record_api_call(self):
        """Record an API call"""
        self.metrics['api_calls'] += 1
    
    def record_schedule_generated(self, validation_result: Dict):
        """Record a generated schedule and its quality"""
        self.metrics['schedules_generated'] += 1
        self.metrics['iterations'] += 1
        
        patient_violations = validation_result.get('patient_violations', 0)
        employee_violations = validation_result.get('employee_violations', 0)
        
        # Track best results
        if patient_violations < self.metrics['best_patient_violations']:
            self.metrics['best_patient_violations'] = patient_violations
        if employee_violations < self.metrics['best_employee_violations']:
            self.metrics['best_employee_violations'] = employee_violations
    
    def record_score(self, score_result: Dict):
        """Record schedule score"""
        total_score = score_result.get('total_score', float('inf'))
        if total_score < self.metrics['best_total_score']:
            self.metrics['best_total_score'] = total_score
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        now = datetime.now()
        session_duration = (now - self.session_start).total_seconds()
        
        avg_generation_time = (
            sum(self.metrics['generation_times']) / len(self.metrics['generation_times'])
            if self.metrics['generation_times'] else 0
        )
        
        avg_validation_time = (
            sum(self.metrics['validation_times']) / len(self.metrics['validation_times'])
            if self.metrics['validation_times'] else 0
        )
        
        return {
            'session_duration_seconds': round(session_duration, 2),
            'api_calls_total': self.metrics['api_calls'],
            'api_calls_per_minute': round(self.metrics['api_calls'] / (session_duration / 60), 2),
            'schedules_generated': self.metrics['schedules_generated'],
            'iterations': self.metrics['iterations'],
            'avg_generation_time_seconds': round(avg_generation_time, 3),
            'avg_validation_time_seconds': round(avg_validation_time, 3),
            'best_patient_violations': self.metrics['best_patient_violations'],
            'best_employee_violations': self.metrics['best_employee_violations'],
            'best_total_score': round(self.metrics['best_total_score'], 2),
            'efficiency_metrics': {
                'schedules_per_minute': round(self.metrics['schedules_generated'] / (session_duration / 60), 2),
                'avg_time_per_schedule': round(session_duration / max(1, self.metrics['schedules_generated']), 2)
            }
        }
    
    def save_session_log(self):
        """Save session metrics to file"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"performance_{timestamp}.json")
        
        summary = self.get_summary()
        summary['session_start'] = self.session_start.isoformat()
        summary['session_end'] = datetime.now().isoformat()
        
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return log_file
    
    def print_realtime_status(self):
        """Print current performance status"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("🔍 PERFORMANCE MONITOR")
        print("="*60)
        print(f"Session Duration: {summary['session_duration_seconds']}s")
        print(f"API Calls: {summary['api_calls_total']} ({summary['api_calls_per_minute']}/min)")
        print(f"Schedules Generated: {summary['schedules_generated']}")
        print(f"Iterations: {summary['iterations']}")
        print(f"\n📊 BEST RESULTS:")
        print(f"  Patient Violations: {summary['best_patient_violations']}")
        print(f"  Employee Violations: {summary['best_employee_violations']}")
        print(f"  Total Score: {summary['best_total_score']}")
        print(f"\n⚡ EFFICIENCY:")
        print(f"  Avg Generation Time: {summary['avg_generation_time_seconds']}s")
        print(f"  Schedules/min: {summary['efficiency_metrics']['schedules_per_minute']}")
        print("="*60)


class ScheduleQualityTracker:
    """Tracks schedule quality over time"""
    
    def __init__(self):
        self.history = []
    
    def add_result(self, iteration: int, validation_result: Dict, score_result: Dict = None):
        """Add a schedule result to history"""
        record = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'patient_violations': validation_result.get('patient_violations', 0),
            'employee_violations': validation_result.get('employee_violations', 0),
            'total_violations': validation_result.get('total_violations', 0),
            'valid': validation_result.get('valid', False)
        }
        
        if score_result:
            record.update({
                'total_score': score_result.get('total_score', 0),
                'cost': score_result.get('breakdown', {}).get('total_cost', 0),
                'continuity_penalty': score_result.get('breakdown', {}).get('continuity_penalty', 0),
                'fairness_penalty': score_result.get('breakdown', {}).get('fairness_penalty', 0),
                'overtime_penalty': score_result.get('breakdown', {}).get('overtime_penalty', 0)
            })
        
        self.history.append(record)
    
    def get_improvement_trend(self) -> Dict:
        """Analyze improvement trend"""
        if len(self.history) < 2:
            return {"status": "insufficient_data"}
        
        # Compare first and last 3 results
        first_batch = self.history[:3]
        last_batch = self.history[-3:]
        
        avg_patient_early = sum(r['patient_violations'] for r in first_batch) / len(first_batch)
        avg_patient_late = sum(r['patient_violations'] for r in last_batch) / len(last_batch)
        
        avg_employee_early = sum(r['employee_violations'] for r in first_batch) / len(first_batch)
        avg_employee_late = sum(r['employee_violations'] for r in last_batch) / len(last_batch)
        
        return {
            'status': 'analyzed',
            'patient_violations_trend': avg_patient_late - avg_patient_early,
            'employee_violations_trend': avg_employee_late - avg_employee_early,
            'total_iterations': len(self.history),
            'best_iteration': min(self.history, key=lambda x: (x['patient_violations'], x['employee_violations']))['iteration'],
            'convergence_status': 'improving' if avg_patient_late < avg_patient_early else 'stable' if avg_patient_late == avg_patient_early else 'degrading'
        }
    
    def save_history(self, filename: str = None):
        """Save quality history to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/quality_history_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump({
                'history': self.history,
                'trend_analysis': self.get_improvement_trend(),
                'summary': {
                    'total_iterations': len(self.history),
                    'best_patient_violations': min(r['patient_violations'] for r in self.history),
                    'best_employee_violations': min(r['employee_violations'] for r in self.history),
                    'final_patient_violations': self.history[-1]['patient_violations'] if self.history else None,
                    'final_employee_violations': self.history[-1]['employee_violations'] if self.history else None
                }
            }, f, indent=2)
        
        return filename