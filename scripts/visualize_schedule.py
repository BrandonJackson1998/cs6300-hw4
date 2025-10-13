"""
Schedule Visualization Tool

Creates visual representations of nursing schedules.
Run: python scripts/visualize_schedule.py <schedule_file.json> [--html]
"""

import json
import sys
import re
from collections import defaultdict
import os

def visualize_schedule_text(schedule_file: str, output_file: str = None):
    """Create a text-based visualization of the schedule"""
    
    with open(schedule_file, 'r') as f:
        data = json.load(f)
    
    schedule = data.get('schedule', data)
    
    try:
        with open('data/employees.json', 'r') as f:
            employees = {e['employee_id']: e for e in json.load(f)}
        with open('data/patients.json', 'r') as f:
            patients = {p['patient_id']: p for p in json.load(f)}
    except:
        employees = {}
        patients = {}
    
    output = []
    output.append("="*100)
    output.append("NURSING FACILITY SCHEDULE")
    output.append("="*100)
    output.append("")
    
    days = defaultdict(lambda: defaultdict(dict))
    for shift_id, patient_assignments in schedule.items():
        day = int(shift_id[1:3])
        shift = int(shift_id[4:])
        days[day][shift] = patient_assignments
    
    shift_times = {0: "12am-4am", 1: "4am-8am", 2: "8am-12pm", 3: "12pm-4pm", 4: "4pm-8pm", 5: "8pm-12am"}
    
    for day in sorted(days.keys()):
        output.append(f"\n{'='*100}")
        output.append(f"DAY {day:02d}")
        output.append(f"{'='*100}")
        
        for shift in sorted(days[day].keys()):
            shift_time = shift_times.get(shift, f"Shift {shift}")
            output.append(f"\n  {shift_time}:")
            output.append(f"  {'-'*96}")
            
            patient_assignments = days[day][shift]
            if not patient_assignments:
                output.append("    (No assignments)")
                continue
            
            for patient_id, employee_ids in sorted(patient_assignments.items()):
                patient_name = patients.get(patient_id, {}).get('name', patient_id)
                output.append(f"    Patient {patient_id} ({patient_name}):")
                
                for emp_id in employee_ids:
                    emp_name = employees.get(emp_id, {}).get('name', emp_id)
                    emp_level = employees.get(emp_id, {}).get('level', '?')
                    output.append(f"      → {emp_id} ({emp_name}, Level {emp_level})")
    
    employee_shifts = defaultdict(int)
    patient_nurses = defaultdict(set)
    
    for shift_id, patient_assignments in schedule.items():
        for patient_id, employee_ids in patient_assignments.items():
            for emp_id in employee_ids:
                employee_shifts[emp_id] += 1
                patient_nurses[patient_id].add(emp_id)
    
    output.append("\n\n" + "="*100)
    output.append("SUMMARY STATISTICS")
    output.append("="*100)
    output.append(f"\nTotal shifts scheduled: {sum(employee_shifts.values())}")
    output.append(f"Total days covered: {len(days)}")
    output.append(f"\nEmployee Workload:")
    for emp_id, count in sorted(employee_shifts.items(), key=lambda x: -x[1]):
        emp_name = employees.get(emp_id, {}).get('name', emp_id)
        hours = count * 4
        output.append(f"  {emp_id} ({emp_name}): {count} shifts ({hours}h)")
    
    output.append(f"\nPatient Continuity (unique nurses per patient):")
    for patient_id, nurses in sorted(patient_nurses.items()):
        patient_name = patients.get(patient_id, {}).get('name', patient_id)
        output.append(f"  {patient_id} ({patient_name}): {len(nurses)} unique nurses")
    
    result = "\n".join(output)
    print(result)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"\n✓ Visualization saved to: {output_file}")
    
    return result


def create_html_visualization(schedule_file: str, output_file: str = "schedule_visualization.html"):
    """Create an interactive HTML visualization with tabs and search"""
    
    with open(schedule_file, 'r') as f:
        data = json.load(f)
    
    schedule = data.get('schedule', data)
    
    staffing_narrative = data.get('staffing_narrative', None)
    if staffing_narrative:
        print(f"✓ Found AI staffing narrative in schedule file ({len(staffing_narrative)} chars)")
    else:
        print(f"ℹ No AI staffing narrative found (older schedule or generation failed)")
    
    validation_data = data.get('validation', {})
    patient_violation_count = validation_data.get('patient_violations', 0)
    employee_violation_count = validation_data.get('employee_violations', 0)
    
    saved_violations = []
    if 'violations' in validation_data:
        saved_violations = validation_data['violations']
        print(f"✓ Loaded {len(saved_violations)} violations from agent validation")
        print(f"  🚨 Patient violations (CRITICAL): {patient_violation_count}")
        print(f"  ⚠️  Employee violations (SOFT): {employee_violation_count}")
    
    try:
        with open('data/employees.json', 'r') as f:
            employees = {e['employee_id']: e for e in json.load(f)}
        with open('data/patients.json', 'r') as f:
            patients = {p['patient_id']: p for p in json.load(f)}
    except:
        employees = {}
        patients = {}
    
    days = defaultdict(lambda: defaultdict(dict))
    employee_shifts = defaultdict(int)
    patient_nurses = defaultdict(set)
    
    for shift_id, patient_assignments in schedule.items():
        day = int(shift_id[1:3])
        shift = int(shift_id[4:])
        days[day][shift] = patient_assignments
        
        for patient_id, employee_ids in patient_assignments.items():
            for emp_id in employee_ids:
                employee_shifts[emp_id] += 1
                patient_nurses[patient_id].add(emp_id)
    
    shift_times = {0: "12am-4am", 1: "4am-8am", 2: "8am-12pm", 3: "12pm-4pm", 4: "4pm-8pm", 5: "8pm-12am"}
    
    if saved_violations:
        violations = []
        for v_str in saved_violations:
            if "needs care during" in v_str and "no nurses assigned" in v_str:
                patient_id = None
                for pid in patients.keys():
                    if pid in v_str:
                        patient_id = pid
                        break
                
                violations.append({
                    'type': 'no_coverage',
                    'patient_id': patient_id or 'Unknown',
                    'patient_name': patients.get(patient_id, {}).get('name', 'Unknown') if patient_id else 'Unknown',
                    'shift_id': v_str.split()[-1] if v_str.split()[-1].startswith('D') else 'Unknown',
                    'message': v_str
                })
            elif "needs" in v_str and "nurses but" in v_str:
                patient_id = None
                for pid in patients.keys():
                    if pid in v_str:
                        patient_id = pid
                        break
                
                violations.append({
                    'type': 'insufficient_nurses',
                    'patient_id': patient_id or 'Unknown',
                    'patient_name': patients.get(patient_id, {}).get('name', 'Unknown') if patient_id else 'Unknown',
                    'shift_id': 'Multiple',
                    'message': v_str
                })
            else:
                violations.append({
                    'type': 'other',
                    'patient_id': 'N/A',
                    'patient_name': 'N/A',
                    'shift_id': 'N/A',
                    'message': v_str
                })
    else:
        violations = []
        for patient_id, patient in patients.items():
            for shift_id in patient.get('care_shifts', []):
                if shift_id not in schedule or patient_id not in schedule[shift_id]:
                    violations.append({
                        'type': 'no_coverage',
                        'patient_id': patient_id,
                        'patient_name': patient.get('name', patient_id),
                        'shift_id': shift_id,
                        'message': f"Patient {patient_id} ({patient.get('name', '')}) needs care during {shift_id} but no nurses assigned"
                    })
                elif len(schedule[shift_id][patient_id]) < patient.get('nurses_needed', 1):
                    violations.append({
                        'type': 'insufficient_nurses',
                        'patient_id': patient_id,
                        'patient_name': patient.get('name', patient_id),
                        'shift_id': shift_id,
                        'message': f"Patient {patient_id} needs {patient.get('nurses_needed', 1)} nurses but only {len(schedule[shift_id][patient_id])} assigned for {shift_id}"
                    })
    
    uncovered_shifts = defaultdict(list)
    skill_gaps = defaultdict(int)
    level_needs = defaultdict(int)
    
    for v in violations:
        if v['type'] in ['no_coverage', 'insufficient_nurses']:
            patient_id = v['patient_id']
            if patient_id in patients:
                uncovered_shifts[patient_id].append(v['shift_id'])
                patient = patients[patient_id]
                for skill in patient.get('required_skills', []):
                    skill_gaps[skill] += 1
                level_needs[patient.get('min_level', 1)] += 1
    
    overworked = []
    underutilized = []
    
    for emp_id, emp in employees.items():
        shifts_worked = employee_shifts.get(emp_id, 0)
        total_hours = shifts_worked * 4
        avg_weekly = total_hours / 4
        max_hours = emp.get('max_hours_per_week', 40)
        expected_hours = emp.get('expected_hours_per_week', 40)
        
        if avg_weekly > expected_hours:  # Anyone above expected hours is "overworked"
            overworked.append({
                'id': emp_id,
                'name': emp.get('name', emp_id),
                'hours': avg_weekly,
                'max': max_hours,
                'expected': expected_hours,  # Store expected hours for overtime calculation
                'utilization': (avg_weekly / max_hours * 100)
            })
        elif avg_weekly < expected_hours * 0.3:  # Use expected hours for underutilized detection too
            underutilized.append({
                'id': emp_id,
                'name': emp.get('name', emp_id),
                'hours': avg_weekly,
                'max': max_hours,
                'utilization': (avg_weekly / max_hours * 100) if max_hours > 0 else 0
            })
    
    total_shifts = sum(employee_shifts.values())
    
    html_parts = []
    
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Nursing Schedule Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; }
        h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; font-size: 2.5em; }
        .tabs { display: flex; background: #f8f9fa; border-bottom: 2px solid #dee2e6; flex-wrap: wrap; }
        .tab { flex: 1; min-width: 120px; padding: 20px 10px; text-align: center; cursor: pointer; background: #f8f9fa; border: none; font-size: 15px; font-weight: 600; color: #6c757d; transition: all 0.3s; }
        .tab:hover { background: #e9ecef; }
        .tab.active { background: white; color: #667eea; border-bottom: 3px solid #667eea; }
        .tab-content { display: none; padding: 30px; max-height: 80vh; overflow-y: auto; }
        .tab-content.active { display: block; }
        .search-bar { width: 100%; padding: 15px 20px; font-size: 16px; border: 2px solid #dee2e6; border-radius: 8px; margin: 20px 0; }
        .search-bar:focus { outline: none; border-color: #667eea; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 2.5em; font-weight: bold; }
        .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; }
        .list-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .list-item { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }
        .list-item:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.15); }
        .list-item-header { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }
        .list-item-detail { margin: 8px 0; color: #7f8c8d; font-size: 14px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin: 2px; }
        .badge-level-1 { background: #ecf0f1; color: #7f8c8d; }
        .badge-level-2 { background: #d6eaf8; color: #2980b9; }
        .badge-level-3 { background: #ebdef0; color: #8e44ad; }
        .badge-skill { background: #d5f4e6; color: #27ae60; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        .badge-success { background: #d4edda; color: #155724; }
        .violation-item { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .violation-critical { background: #f8d7da; border-left-color: #dc3545; }
        .recommendation { background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .recommendation h4 { color: #0c5460; margin-bottom: 10px; }
        .day { background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .day-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 20px; margin: -20px -20px 20px -20px; border-radius: 10px 10px 0 0; font-size: 20px; font-weight: bold; }
        .shift { margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 5px; }
        .shift-time { font-weight: bold; color: #667eea; margin-bottom: 15px; font-size: 18px; }
        .patient { margin: 15px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .patient-name { font-weight: bold; color: #e74c3c; margin-bottom: 10px; font-size: 16px; }
        .nurse { display: inline-block; margin: 5px 5px 5px 0; padding: 8px 15px; background: #d5f4e6; border-left: 4px solid #27ae60; border-radius: 5px; }
        .nurse.level-1 { background: #e8e8e8; border-left-color: #95a5a6; }
        .nurse.level-2 { background: #d6eaf8; border-left-color: #3498db; }
        .nurse.level-3 { background: #ebdef0; border-left-color: #9b59b6; }
        .no-results { text-align: center; padding: 40px; color: #7f8c8d; font-size: 18px; display: none; }
        .expandable { cursor: pointer; color: #667eea; text-decoration: underline; font-style: italic; }
        .expandable:hover { color: #764ba2; }
        .expanded-content { display: none; margin-top: 5px; }
        .expanded-content.show { display: block; }
        .violation-list { margin-top: 8px; padding: 10px; background: #fff3cd; border-left: 3px solid #ffc107; border-radius: 4px; }
        .violation-list-item { margin: 5px 0; font-size: 13px; color: #856404; }
        .calendar-table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .calendar-table th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 10px; font-weight: 600; border: 1px solid #5568d3; position: sticky; top: 0; z-index: 10; }
        .calendar-table td { border: 1px solid #dee2e6; padding: 8px; vertical-align: top; min-width: 120px; font-size: 12px; }
        .calendar-table .shift-label { background: #f8f9fa; font-weight: bold; color: #667eea; white-space: nowrap; }
        .calendar-assignment { border-left: 3px solid; padding: 4px 8px; margin: 2px 0; border-radius: 3px; font-size: 11px; cursor: pointer; transition: all 0.2s; }
        .calendar-assignment:hover { opacity: 0.8; transform: scale(1.02); }
        .calendar-assignment.highlighted { box-shadow: 0 0 0 3px #ffc107; outline: 2px solid #ffc107; z-index: 100; position: relative; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Nursing Facility Schedule</h1>
        <div class="tabs">
            <button class="tab active" onclick="showTab('employee-schedule')">👤 Employee Hub</button>
            <button class="tab" onclick="showTab('schedule')">📅 Full Schedule</button>
            <button class="tab" onclick="showTab('employees')">👥 All Employees</button>
            <button class="tab" onclick="showTab('patients')">🛏️ Patients</button>
            <button class="tab" onclick="showTab('violations')">⚠️ Analysis</button>
        </div>
        
        <div id="employee-schedule" class="tab-content active">
            <h2 style="color: #2c3e50; margin-bottom: 20px;">Select an Employee</h2>
            <select id="employeeSelector" class="search-bar" onchange="showEmployeeSchedule()" style="cursor: pointer;">
                <option value="">-- Choose Employee --</option>
""")
    
    for emp_id, emp in sorted(employees.items(), key=lambda x: x[1].get('name', '')):
        html_parts.append(f'                <option value="{emp_id}">{emp.get("name", emp_id)} (Level {emp.get("level", "?")})</option>\n')
    
    html_parts.append("""
            </select>
            
            <div id="employeeScheduleContent" style="display: none; margin-top: 30px;">
                <!-- Content will be populated by JavaScript -->
            </div>
        </div>
        
        <div id="schedule" class="tab-content">
            <input type="text" class="search-bar" id="scheduleSearch" placeholder="🔍 Search by day, patient, or employee..." onkeyup="searchSchedule()">
            
            <div style="margin: 20px 0; text-align: center;">
                <button onclick="changeWeek(-1)" style="padding: 10px 20px; margin: 0 10px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: 600;">← Previous Week</button>
                <span id="weekDisplay" style="font-size: 18px; font-weight: bold; color: #2c3e50;">Week 1 (Days 1-7)</span>
                <button onclick="changeWeek(1)" style="padding: 10px 20px; margin: 0 10px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: 600;">Next Week →</button>
            </div>
            
            <div id="calendarView" style="overflow-x: auto;">
                <!-- Calendar will be populated by JavaScript -->
            </div>
        </div>
        
        <div id="employees" class="tab-content">
            <input type="text" class="search-bar" id="employeeSearch" placeholder="🔍 Search employees..." onkeyup="searchEmployees()">
            <div class="list-grid" id="employeeList">
""")
    
    for emp_id, emp in sorted(employees.items(), key=lambda x: x[1].get('name', '')):
        skills_html = "".join([f'<span class="badge badge-skill">{s}</span>' for s in emp.get('skills', [])])
        emp_type = "Full-time" if emp.get('max_hours_per_week', 0) >= 40 else "Part-time"
        shifts_worked = employee_shifts.get(emp_id, 0)
        total_hours = shifts_worked * 4
        avg_weekly = total_hours / 4
        utilization = (avg_weekly / emp.get('max_hours_per_week', 1) * 100) if emp.get('max_hours_per_week', 0) > 0 else 0
        
        excluded_patients_list = emp.get('excluded_patients', [])
        excluded_employees_list = emp.get('excluded_employees', [])
        
        emp_violations = [v for v in saved_violations if emp_id in v and emp.get('name', '') in v]
        violation_shifts = set()
        for v_str in emp_violations:
            shift_matches = re.findall(r'D\d{2}S\d', v_str)
            violation_shifts.update(shift_matches)
        
        available_shifts = emp.get('available_shifts', [])
        days_available = defaultdict(list)
        for shift_id in available_shifts:
            if len(shift_id) >= 5:
                day = int(shift_id[1:3])
                shift_num = int(shift_id[4:])
                days_available[day].append((shift_num, shift_id))
        
        all_availability_html = []
        
        for day in sorted(days_available.keys()):
            shifts = sorted(days_available[day])
            shift_nums = [s[0] for s in shifts]
            
            if len(shift_nums) == 6:
                day_text = f"Day {day}: 24/7"
            elif shift_nums:
                time_ranges = []
                i = 0
                while i < len(shift_nums):
                    start = shift_nums[i]
                    end = start
                    
                    while i + 1 < len(shift_nums) and shift_nums[i + 1] == shift_nums[i] + 1:
                        i += 1
                        end = shift_nums[i]
                    
                    if start == end:
                        time_ranges.append(shift_times[start])
                    else:
                        start_time = shift_times[start].split('-')[0]
                        end_time = shift_times[end].split('-')[1]
                        time_ranges.append(f"{start_time}-{end_time}")
                    
                    i += 1
                
                day_text = f"Day {day}: {', '.join(time_ranges)}"
            else:
                day_text = f"Day {day}: No shifts"
            
            all_availability_html.append(day_text)
        
        total_days_available = len(days_available)
        expand_id = f"expand_{emp_id}"
        
        if total_days_available > 0:
            all_days_text = '<br>'.join(all_availability_html)
            availability_html = f'<span class="expandable" onclick="toggleAvailability(\'{expand_id}\', this)">Show</span>'
            availability_html += f'<div id="{expand_id}" class="expanded-content">{all_days_text}</div>'
        else:
            availability_html = '<em>No availability</em>'
        
        violation_html = ""
        if emp_violations:
            violation_count = len(emp_violations)
            violations_readable = []
            
            for v_str in emp_violations:
                readable_v = v_str
                shift_pattern = re.compile(r'D(\d{2})S(\d)')
                
                def replace_shift(match):
                    day = int(match.group(1))
                    shift = int(match.group(2))
                    time = shift_times.get(shift, f"Shift {shift}")
                    return f"Day {day}, {time}"
                
                readable_v = shift_pattern.sub(replace_shift, readable_v)
                violations_readable.append(readable_v)
            
            violations_list_html = '<br>'.join([f'• {v}' for v in violations_readable])
            violation_html = f'<div class="violation-list"><strong>⚠️ {violation_count} Violation(s):</strong><br>{violations_list_html}</div>'
        
        exclusions_html = ""
        if excluded_patients_list:
            patient_names = [patients.get(pid, {}).get('name', pid) for pid in excluded_patients_list]
            exclusions_html += f'<div class="list-item-detail" style="color: #dc3545; margin-top: 8px;">🚫 Cannot work with patients: <strong>{", ".join(patient_names)}</strong></div>'
        if excluded_employees_list:
            emp_names = [employees.get(eid, {}).get('name', eid) for eid in excluded_employees_list]
            exclusions_html += f'<div class="list-item-detail" style="color: #dc3545;">🚫 Cannot work with employees: <strong>{", ".join(emp_names)}</strong></div>'
        
        html_parts.append(f"""
                <div class="list-item searchable-employee" data-name="{emp.get('name', '').lower()}">
                    <div class="list-item-header">{emp.get('name', emp_id)}</div>
                    <div class="list-item-detail">
                        <span class="badge badge-level-{emp.get('level', 1)}">Level {emp.get('level', '?')}</span>
                        <span class="badge {'badge-warning' if emp.get('max_hours_per_week', 0) >= 40 else 'badge-success'}">{emp_type}</span>
                    </div>
                    <div class="list-item-detail">💰 ${emp.get('hourly_pay_rate', 0):.2f}/hr | ⏰ {avg_weekly:.1f}h/week ({utilization:.0f}% capacity)</div>
                    <div class="list-item-detail">📅 Worked {shifts_worked} shifts ({total_hours}h total)</div>
                    <div class="list-item-detail" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px;">{skills_html}</div>
                    <div class="list-item-detail" style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 13px; line-height: 1.6;">
                        <strong>📅 Available:</strong> {availability_html}
                    </div>
                    {violation_html}
                    {exclusions_html}
                </div>
""")
    
    html_parts.append("""
            </div>
            <div id="employeeNoResults" class="no-results">No employees found</div>
        </div>
        
        <div id="patients" class="tab-content">
            <input type="text" class="search-bar" id="patientSearch" placeholder="🔍 Search patients..." onkeyup="searchPatients()">
            <div class="list-grid" id="patientList">
""")
    
    for patient_id, patient in sorted(patients.items(), key=lambda x: x[1].get('name', '')):
        skills_html = "".join([f'<span class="badge badge-skill">{s}</span>' for s in patient.get('required_skills', [])])
        care_shifts_count = len(patient.get('care_shifts', []))
        
        care_shifts_list = patient.get('care_shifts', [])
        shift_times_used = set()
        for shift_id in care_shifts_list:
            if len(shift_id) >= 5:
                shift_num = int(shift_id[4:])
                shift_times_used.add(shift_num)
        
        if len(shift_times_used) == 6:
            care_schedule = "24/7 Care"
        elif shift_times_used:
            sorted_shifts = sorted(shift_times_used)
            start_shift = sorted_shifts[0]
            end_shift = sorted_shifts[-1]
            
            is_continuous = all(i in shift_times_used for i in range(start_shift, end_shift + 1))
            
            if is_continuous:
                start_time = shift_times[start_shift].split('-')[0]
                end_time = shift_times[end_shift].split('-')[1]
                care_schedule = f"{start_time}-{end_time}"
            else:
                time_ranges = []
                for shift in sorted_shifts:
                    time_ranges.append(shift_times[shift])
                care_schedule = ", ".join(time_ranges)
        else:
            care_schedule = "No scheduled care"
        
        excluded_employees_list = patient.get('excluded_employees', [])
        
        exclusions_html = ""
        if excluded_employees_list:
            emp_names = [employees.get(eid, {}).get('name', eid) for eid in excluded_employees_list]
            exclusions_html = f'<div class="list-item-detail" style="color: #dc3545; margin-top: 10px;">🚫 Cannot work with: <strong>{", ".join(emp_names)}</strong></div>'
        
        html_parts.append(f"""
                <div class="list-item searchable-patient" data-name="{patient.get('name', '').lower()}">
                    <div class="list-item-header">{patient.get('name', patient_id)}</div>
                    <div class="list-item-detail">
                        <span class="badge badge-level-{patient.get('min_level', 1)}">Needs Level {patient.get('min_level', '?')}+</span>
                        <span class="badge badge-skill">{patient.get('nurses_needed', 1)} nurses/shift</span>
                    </div>
                    <div class="list-item-detail">⏰ Care Needed: {care_schedule}</div>
                    <div class="list-item-detail">📅 {care_shifts_count} shifts/month | 👥 {len(patient_nurses.get(patient_id, set()))} unique nurses assigned</div>
                    <div class="list-item-detail" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px;">{skills_html if skills_html else '<span style="color: #95a5a6;">No special skills required</span>'}</div>
                    {exclusions_html}
                </div>
""")
    
    html_parts.append("""
            </div>
            <div id="patientNoResults" class="no-results">No patients found</div>
        </div>
        
        <div id="violations" class="tab-content">
            <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Schedule Analysis & Staffing Recommendations</h2>
            
            <div class="summary" style="margin-bottom: 30px;">
                <div class="stat"><div class="stat-value">""" + str(len(violations)) + """</div><div class="stat-label">Total Violations</div></div>
                <div class="stat"><div class="stat-value">""" + str(len(set(v['patient_id'] for v in violations if v['patient_id'] != 'N/A'))) + """</div><div class="stat-label">Patients Affected</div></div>
                <div class="stat"><div class="stat-value">""" + str(len(overworked)) + """</div><div class="stat-label">Overworked Staff</div></div>
                <div class="stat"><div class="stat-value">""" + str(len(underutilized)) + """</div><div class="stat-label">Underutilized Staff</div></div>
            </div>
""")
    
    if staffing_narrative:
        html_parts.append(f"""
            <div class="recommendation" style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #667eea; padding: 25px; margin: 30px 0;">
                <h3 style="color: #667eea; margin-bottom: 15px; display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">🤖</span>
                    AI Staffing Analysis & Recommendations
                </h3>
                <div style="color: #2c3e50; line-height: 1.8; font-size: 15px; white-space: pre-wrap;">{staffing_narrative}</div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 13px; font-style: italic;">
                    ✨ Generated by AI during schedule optimization • Review recommendations with clinical leadership
                </div>
            </div>
""")
    else:
        html_parts.append("""
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; margin: 20px 0;">
                <p style="color: #856404; margin: 0;">
                    ℹ️ <strong>AI Analysis Not Available</strong><br>
                    This schedule was generated without AI staffing recommendations. 
                    Re-run the agent with GEMINI_API_KEY configured to generate a new schedule with AI analysis.
                </p>
            </div>
""")
    
    if skill_gaps or level_needs:
        html_parts.append("""
            <h3 style="color: #2c3e50; margin: 20px 0;">💼 Hiring Recommendations</h3>
""")
        
        if skill_gaps:
            top_skills = sorted(skill_gaps.items(), key=lambda x: -x[1])[:5]
            html_parts.append('<div class="recommendation"><h4>🎯 Critical Skills Needed</h4>')
            for skill, count in top_skills:
                urgency = "🔴 HIGH" if count > 50 else "🟡 MEDIUM" if count > 20 else "🟢 LOW"
                html_parts.append(f'<div class="list-item-detail">• <strong>{skill}</strong>: {count} uncovered shifts - {urgency} priority</div>')
            html_parts.append('<p style="margin-top: 15px; color: #0c5460; font-weight: bold;">💡 Recommendation: Hire nurses with these skills to eliminate coverage gaps.</p></div>')
        
        if level_needs:
            html_parts.append('<div class="recommendation"><h4>📊 Experience Level Requirements</h4>')
            for level, count in sorted(level_needs.items(), key=lambda x: -x[1]):
                urgency = "🔴 URGENT" if count > 30 else "🟡 NEEDED" if count > 10 else "🟢 OPTIONAL"
                html_parts.append(f'<div class="list-item-detail">• <strong>Level {level} nurses</strong>: {count} uncovered shifts - {urgency}</div>')
            html_parts.append('<p style="margin-top: 15px; color: #0c5460; font-weight: bold;">💡 Recommendation: Prioritize hiring experienced nurses at these levels.</p></div>')
    
    if overworked:
        html_parts.append(f'<h3 style="color: #2c3e50; margin: 30px 0 20px 0;">🔥 Overworked Staff ({len(overworked)} employees)</h3>')
        html_parts.append('<div class="recommendation" style="background: #fff3cd; border-left-color: #ffc107;"><h4 style="color: #856404;">⚠️ Working Over Maximum Capacity</h4>')
        for emp in sorted(overworked, key=lambda x: -x['hours']):
            overtime = max(0, emp['hours'] - emp.get('expected', 40))
            html_parts.append(f'<div class="list-item-detail">• <strong>{emp["name"]}</strong>: {emp["hours"]:.1f}h/week (Expected: {emp.get("expected", 40):.0f}h, Max: {emp.get("max", 40):.0f}h) - <span style="color: #dc3545; font-weight: bold;">+{overtime:.1f}h over Preference</span></div>')
        html_parts.append('<p style="margin-top: 15px; color: #856404; font-weight: bold;">⚠️ Risk: These employees are at or exceeding their maximum capacity. Monitor for burnout and ensure proper compensation.</p></div>')
    
    if underutilized:
        html_parts.append(f'<h3 style="color: #2c3e50; margin: 30px 0 20px 0;">📉 Underutilized Staff ({len(underutilized)} employees)</h3>')
        html_parts.append('<div class="recommendation" style="background: #d1ecf1; border-left-color: #17a2b8;"><h4 style="color: #0c5460;">💡 Optimization</h4>')
        for emp in sorted(underutilized, key=lambda x: x['utilization']):
            html_parts.append(f'<div class="list-item-detail">• <strong>{emp["name"]}</strong>: {emp["hours"]:.1f}h/week ({emp["utilization"]:.0f}% capacity)</div>')
        html_parts.append('<p style="margin-top: 15px; color: #0c5460; font-weight: bold;">💡 Options: Increase hours, reassign, or consider separation.</p></div>')
    
    html_parts.append(f"""
            <h3 style="color: #2c3e50; margin: 30px 0 20px 0;">⚠️ Detailed Violations ({len(violations)} total)</h3>
            <p style="color: #6c757d; margin-bottom: 15px;">{'✓ From agent validation' if saved_violations else '⚠️ Calculated (may be incomplete)'}</p>
""")
    
    if violations:
        violation_types = defaultdict(list)
        for v in violations:
            violation_types[v['type']].append(v)
        
        for vtype, vlist in violation_types.items():
            type_label = {'no_coverage': '❌ No Coverage', 'insufficient_nurses': '⚠️ Insufficient Nurses', 'other': '🔧 Other'}.get(vtype, vtype)
            html_parts.append(f'<h4 style="color: #495057; margin: 20px 0 10px 0;">{type_label} ({len(vlist)})</h4>')
            
            for v in vlist:
                violation_class = 'violation-critical' if v['type'] == 'no_coverage' else 'violation-item'
                
                readable_message = v["message"]
                shift_pattern = re.compile(r'D(\d{2})S(\d)')
                
                def replace_shift(match):
                    day = int(match.group(1))
                    shift = int(match.group(2))
                    time = shift_times.get(shift, f"Shift {shift}")
                    return f"<strong>Day {day}, {time}</strong>"
                
                readable_message = shift_pattern.sub(replace_shift, readable_message)
                
                html_parts.append(f'<div class="{violation_class}">{readable_message}</div>\n')
    else:
        html_parts.append('<div class="recommendation" style="background: #d4edda;"><h4 style="color: #155724;">✅ Perfect Schedule!</h4></div>')
    
    html_parts.append("""
        </div>
    </div>
    
    <script>
        const employeesData = """ + json.dumps(employees) + """;
        const patientsData = """ + json.dumps(patients) + """;
        const scheduleData = """ + json.dumps(schedule) + """;
        const violationsData = """ + json.dumps([v['message'] for v in violations]) + """;
        const shiftTimes = {0: "12am-4am", 1: "4am-8am", 2: "8am-12pm", 3: "12pm-4pm", 4: "4pm-8pm", 5: "8pm-12am"};
        
        let currentWeek = 0;
        const totalDays = """ + str(max(days.keys()) if days else 28) + """;
        
        // Generate unique colors for each employee
        const employeeColors = {};
        const colorPalette = [
            '#FFE5E5', '#E5F5FF', '#E5FFE5', '#FFF5E5', '#F5E5FF', '#FFE5F5',
            '#E5FFFF', '#FFFFE5', '#E5E5FF', '#FFE5EE', '#E5FFF5', '#F5FFE5',
            '#FFE5FB', '#E5F5F5', '#FFF5F5', '#F5E5E5', '#E5FBE5', '#FFEEE5',
            '#E5EEFF', '#FFE5F0', '#E5FFE9', '#F0E5FF', '#FFEBE5', '#E5F9FF',
            '#FFFFE9', '#E9FFE5', '#FFE5EA', '#E5E9FF', '#F9FFE5', '#FFE8E5',
            '#E5FCFF', '#FFF9E5', '#E5FFD5', '#FFD5E5', '#D5E5FF', '#E5FFD9',
            '#FFE5D5', '#E5D5FF', '#D5FFE5', '#FFD9E5', '#E5FFE8', '#F0FFE5',
            '#E5F0FF', '#FFE5F8', '#F8E5FF', '#E5FFF8', '#FFE5E8', '#E8FFE5',
            '#E5E8FF', '#FFEDE5', '#E5FFED', '#EDFFE5', '#E5EDFF', '#FFE5ED',
            '#EDE5FF', '#D9FFE5', '#E5D9FF', '#FFE5D9', '#E5FFD1', '#D1E5FF',
            '#FFD1E5', '#E5FFDC', '#DCE5FF', '#FFDCE5', '#E5DCFF', '#DCFFE5',
            '#E5FFCC', '#CCE5FF', '#FFCCE5', '#E5CCFF', '#CCFFE5', '#E5FFC8',
            '#C8E5FF', '#FFC8E5', '#E5C8FF', '#C8FFE5', '#E5FFFA', '#FAE5FF',
            '#FFFACE5', '#E5FAFF'
        ];
        
        function getEmployeeColor(empId) {
            if (!employeeColors[empId]) {
                const keys = Object.keys(employeeColors);
                employeeColors[empId] = colorPalette[keys.length % colorPalette.length];
            }
            return employeeColors[empId];
        }
        
        function getDarkerShade(hex) {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            
            const darker = (val) => Math.max(0, val - 40);
            
            return `#${darker(r).toString(16).padStart(2, '0')}${darker(g).toString(16).padStart(2, '0')}${darker(b).toString(16).padStart(2, '0')}`;
        }
        
        let currentHighlightedEmployee = null;
        
        function highlightEmployee(empId, event) {
            if (event) event.stopPropagation();
            
            document.querySelectorAll('.calendar-assignment.highlighted').forEach(el => {
                el.classList.remove('highlighted');
            });
            
            if (currentHighlightedEmployee === empId) {
                currentHighlightedEmployee = null;
                return;
            }
            
            currentHighlightedEmployee = empId;
            document.querySelectorAll(`.emp-${empId}`).forEach(el => {
                el.classList.add('highlighted');
            });
        }
        
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.calendar-assignment')) {
                document.querySelectorAll('.calendar-assignment.highlighted').forEach(el => {
                    el.classList.remove('highlighted');
                });
                currentHighlightedEmployee = null;
            }
        });
        
        function renderCalendar(startDay) {
            const endDay = Math.min(startDay + 6, totalDays);
            const daysToShow = [];
            for (let d = startDay + 1; d <= endDay + 1; d++) {
                daysToShow.push(d);
            }
            
            let html = '<table class="calendar-table">';
            html += '<thead><tr><th style="min-width: 100px;">Shift</th>';
            daysToShow.forEach(day => {
                html += `<th>Day ${day}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            for (let shift = 0; shift < 6; shift++) {
                const timeRange = shiftTimes[shift];
                html += '<tr>';
                html += `<td class="shift-label">${timeRange}</td>`;
                
                daysToShow.forEach(day => {
                    const shiftId = `D${String(day).padStart(2, '0')}S${shift}`;
                    const assignments = scheduleData[shiftId] || {};
                    
                    html += '<td>';
                    
                    const sortedPatients = Object.entries(assignments).sort((a, b) => {
                        return a[0].localeCompare(b[0]);
                    });
                    
                    sortedPatients.forEach(([patientId, employeeIds]) => {
                        const patient = patientsData[patientId];
                        const patientName = patient ? patient.name : patientId;
                        
                        employeeIds.forEach(empId => {
                            const emp = employeesData[empId];
                            const empName = emp ? emp.name : empId;
                            const color = getEmployeeColor(empId);
                            html += `<div class="calendar-assignment emp-${empId}" style="background: ${color}; border-left-color: ${getDarkerShade(color)};" title="${patientName} - ${empName}" onclick="highlightEmployee('${empId}', event)">`;
                            html += `<strong>${patientName}</strong>: ${empName}`;
                            html += `</div>`;
                        });
                    });
                    
                    html += '</td>';
                });
                
                html += '</tr>';
            }
            
            html += '</tbody></table>';
            document.getElementById('calendarView').innerHTML = html;
            
            const weekNum = Math.floor(startDay / 7) + 1;
            document.getElementById('weekDisplay').textContent = `Week ${weekNum} (Days ${startDay + 1}-${endDay + 1})`;
        }
        
        function changeWeek(direction) {
            currentWeek += direction;
            const startDay = currentWeek * 7;
            
            if (startDay < 0) {
                currentWeek = 0;
                return;
            }
            if (startDay >= totalDays) {
                currentWeek = Math.floor((totalDays - 1) / 7);
                return;
            }
            
            renderCalendar(startDay);
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            renderCalendar(0);
        });
        
        function formatShiftReadable(shiftId) {
            if (!shiftId || shiftId.length < 5) return shiftId;
            const day = parseInt(shiftId.substring(1, 3));
            const shift = parseInt(shiftId.substring(4));
            const timeRange = shiftTimes[shift] || `Shift ${shift}`;
            return `Day ${day}, ${timeRange}`;
        }
        
        function showEmployeeSchedule() {
            const empId = document.getElementById('employeeSelector').value;
            const content = document.getElementById('employeeScheduleContent');
            
            if (!empId) {
                content.style.display = 'none';
                return;
            }
            
            const emp = employeesData[empId];
            if (!emp) return;
            
            const assignedShifts = [];
            const patientsAssigned = new Set();
            
            for (const [shiftId, patientAssignments] of Object.entries(scheduleData)) {
                for (const [patientId, employeeIds] of Object.entries(patientAssignments)) {
                    if (employeeIds.includes(empId)) {
                        assignedShifts.push({shiftId, patientId});
                        patientsAssigned.add(patientId);
                    }
                }
            }
            
            const empViolations = violationsData.filter(v => 
                v.includes(empId) || v.includes(emp.name)
            );
            
            const totalHours = assignedShifts.length * 4;
            const avgWeekly = totalHours / 4;
            const utilization = emp.max_hours_per_week > 0 ? (avgWeekly / emp.max_hours_per_week * 100) : 0;
            
            const availableShifts = emp.available_shifts || [];
            const daysAvailable = {};
            availableShifts.forEach(shiftId => {
                if (shiftId.length >= 5) {
                    const day = parseInt(shiftId.substring(1, 3));
                    const shiftNum = parseInt(shiftId.substring(4));
                    if (!daysAvailable[day]) daysAvailable[day] = [];
                    daysAvailable[day].push(shiftNum);
                }
            });
            
            const allAvailabilityLines = [];
            Object.keys(daysAvailable).sort((a, b) => parseInt(a) - parseInt(b)).forEach(day => {
                const shiftNums = daysAvailable[day].sort((a, b) => a - b);
                
                if (shiftNums.length === 6) {
                    allAvailabilityLines.push(`Day ${day}: 24/7`);
                } else if (shiftNums.length > 0) {
                    const timeRanges = [];
                    let i = 0;
                    while (i < shiftNums.length) {
                        const start = shiftNums[i];
                        let end = start;
                        
                        while (i + 1 < shiftNums.length && shiftNums[i + 1] === shiftNums[i] + 1) {
                            i++;
                            end = shiftNums[i];
                        }
                        
                        if (start === end) {
                            timeRanges.push(shiftTimes[start]);
                        } else {
                            const startTime = shiftTimes[start].split('-')[0];
                            const endTime = shiftTimes[end].split('-')[1];
                            timeRanges.push(`${startTime}-${endTime}`);
                        }
                        
                        i++;
                    }
                    
                    allAvailabilityLines.push(`Day ${day}: ${timeRanges.join(', ')}`);
                }
            });
            
            const availExpandId = `avail_${empId}`;
            const availabilityHtml = allAvailabilityLines.length > 0 
                ? `<span class="expandable" onclick="toggleAvailability('${availExpandId}', this)">Show</span><div id="${availExpandId}" class="expanded-content">${allAvailabilityLines.join('<br>')}</div>`
                : '<em>No availability</em>';
            
            let html = '<div class="list-item" style="max-width: 100%;">';
            html += `<div class="list-item-header" style="font-size: 24px;">${emp.name}</div>`;
            
            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">';
            html += `<div class="stat" style="padding: 15px;"><div class="stat-value" style="font-size: 1.8em;">${assignedShifts.length}</div><div class="stat-label">Shifts Assigned</div></div>`;
            html += `<div class="stat" style="padding: 15px;"><div class="stat-value" style="font-size: 1.8em;">${assignedShifts.length * 4}h</div><div class="stat-label">Total Hours</div></div>`;
            html += `<div class="stat" style="padding: 15px;"><div class="stat-value" style="font-size: 1.8em;">${patientsAssigned.size}</div><div class="stat-label">Patients</div></div>`;
            html += `<div class="stat" style="padding: 15px;"><div class="stat-value" style="font-size: 1.8em;">${empViolations.length}</div><div class="stat-label">Violations</div></div>`;
            html += '</div>';
            
            const skills = emp.skills.map(s => `<span class="badge badge-skill">${s}</span>`).join('');
            const empType = emp.max_hours_per_week >= 40 ? "Full-time" : "Part-time";
            html += '<div class="list-item-detail" style="margin: 15px 0;">';
            html += `<span class="badge badge-level-${emp.level}">Level ${emp.level}</span>`;
            html += `<span class="badge ${emp.max_hours_per_week >= 40 ? 'badge-warning' : 'badge-success'}">${empType}</span>`;
            html += `<span class="badge badge-warning">$${emp.hourly_pay_rate}/hr</span>`;
            html += `<span class="badge badge-skill">${emp.max_hours_per_week}h/week max</span>`;
            html += '</div>';
            html += `<div style="margin: 15px 0; display: flex; flex-wrap: wrap; gap: 5px;">${skills}</div>`;
            
            html += `<div class="list-item-detail" style="margin: 15px 0; font-size: 15px;">💰 $${emp.hourly_pay_rate}/hr | ⏰ ${avgWeekly.toFixed(1)}h/week (${utilization.toFixed(0)}% capacity)</div>`;
            
            html += `<div class="list-item-detail" style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 13px; line-height: 1.6;">`;
            html += `<strong>📅 Available:</strong> ${availabilityHtml}`;
            html += `</div>`;
            
            const excludedPatients = emp.excluded_patients || [];
            const excludedEmployees = emp.excluded_employees || [];
            
            if (excludedPatients.length > 0) {
                const patientNames = excludedPatients.map(pid => patientsData[pid]?.name || pid).join(', ');
                html += `<div class="list-item-detail" style="color: #dc3545; margin-top: 10px;">🚫 Cannot work with patients: <strong>${patientNames}</strong></div>`;
            }
            if (excludedEmployees.length > 0) {
                const empNames = excludedEmployees.map(eid => employeesData[eid]?.name || eid).join(', ');
                html += `<div class="list-item-detail" style="color: #dc3545; margin-top: 8px;">🚫 Cannot work with employees: <strong>${empNames}</strong></div>`;
            }
            
            if (empViolations.length > 0) {
                html += '<div class="violation-list" style="margin: 20px 0;">';
                html += `<strong style="font-size: 16px;">⚠️ ${empViolations.length} Violation(s):</strong><br><br>`;
                empViolations.forEach(v => {
                    const readable = v.replace(/D(\d{2})S(\d)/g, (match, day, shift) => {
                        return `Day ${parseInt(day)}, ${shiftTimes[parseInt(shift)]}`;
                    });
                    html += `<div class="violation-list-item">• ${readable}</div>`;
                });
                html += '</div>';
            }
            
            if (patientsAssigned.size > 0) {
                html += '<h3 style="color: #2c3e50; margin: 30px 0 15px 0;">🛏️ Patients Assigned</h3>';
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">';
                patientsAssigned.forEach(patientId => {
                    const patient = patientsData[patientId];
                    if (patient) {
                        const patientSkills = patient.required_skills || [];
                        const skillBadges = patientSkills.map(s => `<span class="badge badge-skill" style="font-size: 11px;">${s}</span>`).join('');
                        
                        html += `<div class="list-item" style="margin: 0;">`;
                        html += `<div class="list-item-header" style="font-size: 16px;">${patient.name}</div>`;
                        html += `<div class="list-item-detail">`;
                        html += `<span class="badge badge-level-${patient.min_level}">Needs Level ${patient.min_level}+</span>`;
                        html += `<span class="badge badge-warning">${patient.nurses_needed} nurse(s)/shift</span>`;
                        html += `</div>`;
                        if (skillBadges) {
                            html += `<div class="list-item-detail" style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px;">${skillBadges}</div>`;
                        }
                        html += `</div>`;
                    }
                });
                html += '</div>';
            }
            
            html += '<h3 style="color: #2c3e50; margin: 30px 0 15px 0;">📅 Schedule</h3>';

            // Add week navigation
            html += '<div style="margin: 20px 0; text-align: center;">';
            html += '<button onclick="changeEmployeeWeek(-1)" style="padding: 10px 20px; margin: 0 10px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: 600;">← Previous Week</button>';
            html += '<span id="employeeWeekDisplay" style="font-size: 18px; font-weight: bold; color: #2c3e50;">Week 1 (Days 1-7)</span>';
            html += '<button onclick="changeEmployeeWeek(1)" style="padding: 10px 20px; margin: 0 10px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: 600;">Next Week →</button>';
            html += '</div>';

            html += '<div id="employeeCalendarView"></div>';
            
            html += '</div>';
            
            content.innerHTML = html;
            content.style.display = 'block';

            // Store current employee and reset week
            currentEmployeeId = empId;
            currentEmployeeWeek = 0;
            // Render the calendar
            renderEmployeeCalendar(empId, 0);
        }

        let currentEmployeeWeek = 0;
        let currentEmployeeId = null;

        function changeEmployeeWeek(direction) {
            currentEmployeeWeek += direction;
            const startDay = currentEmployeeWeek * 7;
            
            if (startDay < 0) {
                currentEmployeeWeek = 0;
                return;
            }
            if (startDay >= totalDays) {
                currentEmployeeWeek = Math.floor((totalDays - 1) / 7);
                return;
            }
            
            renderEmployeeCalendar(currentEmployeeId, startDay);
        }

        function renderEmployeeCalendar(empId, startDay) {
            const endDay = Math.min(startDay + 6, totalDays);
            const daysToShow = [];
            for (let d = startDay + 1; d <= endDay + 1; d++) {
                daysToShow.push(d);
            }
            
            // Get employee's assigned shifts
            const assignedShifts = [];
            for (const [shiftId, patientAssignments] of Object.entries(scheduleData)) {
                for (const [patientId, employeeIds] of Object.entries(patientAssignments)) {
                    if (employeeIds.includes(empId)) {
                        assignedShifts.push({shiftId, patientId});
                    }
                }
            }
            
            // Build day schedule
            const daySchedule = {};
            assignedShifts.forEach(({shiftId, patientId}) => {
                const day = parseInt(shiftId.substring(1, 3));
                if (!daySchedule[day]) daySchedule[day] = {};
                const shift = parseInt(shiftId.substring(4));
                if (!daySchedule[day][shift]) daySchedule[day][shift] = [];
                daySchedule[day][shift].push(patientId);
            });
            
            let html = '<table class="calendar-table">';
            html += '<thead><tr><th style="min-width: 100px;">Shift</th>';
            daysToShow.forEach(day => {
                html += `<th>Day ${day}</th>`;
            });
            html += '</tr></thead><tbody>';
            
            for (let shift = 0; shift < 6; shift++) {
                const timeRange = shiftTimes[shift];
                html += '<tr>';
                html += `<td class="shift-label">${timeRange}</td>`;
                
                daysToShow.forEach(day => {
                    const shiftId = `D${String(day).padStart(2, '0')}S${shift}`;
                    const emp = employeesData[empId];
                    const isAvailable = emp.available_shifts.includes(shiftId);
                    
                    // Add red background if unavailable
                    const bgStyle = !isAvailable ? ' style="background-color: #ffe0e0;"' : '';
                    html += `<td${bgStyle}>`;
                    
                    if (daySchedule[day] && daySchedule[day][shift]) {
                        const sortedPatients = daySchedule[day][shift].sort((a, b) => a.localeCompare(b));
                        
                        sortedPatients.forEach(patientId => {
                            const patient = patientsData[patientId];
                            const patientName = patient ? patient.name : patientId;
                            const color = getEmployeeColor(empId);
                            
                            const availabilityNote = !isAvailable ? ' ⚠️' : '';
                            html += `<div class="calendar-assignment" style="background: ${color}; border-left-color: ${getDarkerShade(color)};" title="${patientName}${!isAvailable ? ' - NOT AVAILABLE' : ''}">`;
                            html += `<strong>${patientName}</strong>${availabilityNote}`;
                            html += `</div>`;
                        });
                    } else if (!isAvailable) {
                        // Show unavailable indicator even if no assignment
                        html += `<div style="font-size: 10px; color: #999; padding: 4px; font-style: italic;">Unavailable</div>`;
                    }
                    
                    html += '</td>';
                });
                
                html += '</tr>';
            }
            
            html += '</tbody></table>';
            document.getElementById('employeeCalendarView').innerHTML = html;
            
            const weekNum = Math.floor(startDay / 7) + 1;
            document.getElementById('employeeWeekDisplay').textContent = `Week ${weekNum} (Days ${startDay + 1}-${endDay + 1})`;
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        function searchSchedule() {
            const search = document.getElementById('scheduleSearch').value.toLowerCase();
            if (!search) {
                document.querySelectorAll('.calendar-assignment').forEach(a => a.style.display = 'block');
                return;
            }
            
            document.querySelectorAll('.calendar-assignment').forEach(assignment => {
                const text = assignment.textContent.toLowerCase();
                assignment.style.display = text.includes(search) ? 'block' : 'none';
            });
        }
        
        function searchEmployees() {
            const search = document.getElementById('employeeSearch').value.toLowerCase();
            let count = 0;
            document.querySelectorAll('.searchable-employee').forEach(emp => {
                if (emp.dataset.name.includes(search)) {
                    emp.style.display = 'block';
                    count++;
                } else {
                    emp.style.display = 'none';
                }
            });
            document.getElementById('employeeNoResults').style.display = count === 0 ? 'block' : 'none';
            document.getElementById('employeeList').style.display = count === 0 ? 'none' : 'grid';
        }
        
        function searchPatients() {
            const search = document.getElementById('patientSearch').value.toLowerCase();
            let count = 0;
            document.querySelectorAll('.searchable-patient').forEach(pat => {
                if (pat.dataset.name.includes(search)) {
                    pat.style.display = 'block';
                    count++;
                } else {
                    pat.style.display = 'none';
                }
            });
            document.getElementById('patientNoResults').style.display = count === 0 ? 'block' : 'none';
            document.getElementById('patientList').style.display = count === 0 ? 'none' : 'grid';
        }
        
        function toggleAvailability(elementId, linkElement) {
            const element = document.getElementById(elementId);
            const isShowing = element.classList.contains('show');
            
            element.classList.toggle('show');
            
            if (linkElement) {
                linkElement.textContent = isShowing ? 'Show' : 'Hide';
            }
        }
    </script>
</body>
</html>
""")

    html = ''.join(html_parts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Interactive HTML visualization created: {output_file}")
    print(f"  Open in browser: file://{os.path.abspath(output_file)}")
    
    return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_schedule.py <schedule_file.json> [--html]")
        sys.exit(1)
    
    schedule_file = sys.argv[1]
    
    if not os.path.exists(schedule_file):
        print(f"Error: File not found: {schedule_file}")
        sys.exit(1)
    
    os.makedirs("visualizations", exist_ok=True)
    
    timestamp = schedule_file.replace('.json', '').split('/')[-1]
    text_output = f"visualizations/{timestamp}_schedule.txt"
    visualize_schedule_text(schedule_file, text_output)
    
    if '--html' in sys.argv or '-h' in sys.argv:
        html_output = f"visualizations/{timestamp}_schedule.html"
        create_html_visualization(schedule_file, html_output)


if __name__ == "__main__":
    main()