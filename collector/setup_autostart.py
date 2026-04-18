"""
setup_autostart.py — Auto-start the collector when Windows boots
================================================================
Run this ONCE as Administrator:
    python setup_autostart.py

It creates a Windows Task Scheduler task that:
  - Starts data_collector.py automatically when you log in
  - Runs silently in the background (no window)
  - Restarts automatically if it crashes
  - Only runs when you're logged in (not as a system service)

To REMOVE the auto-start:
    python setup_autostart.py --remove
"""

import os
import sys
import subprocess
import argparse

TASK_NAME   = "MentalHealthTracker"
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_collector.py')
PYTHON_PATH = sys.executable   # path to current Python interpreter


def create_task():
    """Register the collector as a Windows scheduled task."""

    # Build the command that Task Scheduler will run
    # pythonw.exe = runs Python without a console window
    pythonw = PYTHON_PATH.replace('python.exe', 'pythonw.exe')
    if not os.path.exists(pythonw):
        pythonw = PYTHON_PATH   # fallback to python.exe

    xml_content = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Mental Health Tracker - Background Data Collector</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>10</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{pythonw}</Command>
      <Arguments>"{SCRIPT_PATH}"</Arguments>
      <WorkingDirectory>{os.path.dirname(SCRIPT_PATH)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""

    # Write XML to temp file
    xml_path = os.path.join(os.environ.get('TEMP', '.'), f'{TASK_NAME}.xml')
    with open(xml_path, 'w', encoding='utf-16') as f:
        f.write(xml_content)

    # Register with Task Scheduler
    result = subprocess.run(
        ['schtasks', '/Create', '/F', '/TN', TASK_NAME, '/XML', xml_path],
        capture_output=True, text=True
    )
    os.remove(xml_path)

    if result.returncode == 0:
        print(f"[OK] Task '{TASK_NAME}' created successfully.")
        print(f"     The collector will now start automatically when you log in.")
        print(f"     To start it NOW without rebooting, run:")
        print(f"     schtasks /Run /TN {TASK_NAME}")
    else:
        print(f"[ERROR] Failed to create task:")
        print(result.stderr)
        print("\nTry running this script as Administrator.")


def remove_task():
    """Remove the scheduled task."""
    result = subprocess.run(
        ['schtasks', '/Delete', '/F', '/TN', TASK_NAME],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[OK] Task '{TASK_NAME}' removed.")
    else:
        print(f"[ERROR] {result.stderr}")


def start_now():
    """Start the task immediately."""
    result = subprocess.run(
        ['schtasks', '/Run', '/TN', TASK_NAME],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[OK] Collector started.")
    else:
        print(f"[ERROR] {result.stderr}")


def check_status():
    """Check if the task is running."""
    result = subprocess.run(
        ['schtasks', '/Query', '/TN', TASK_NAME, '/FO', 'LIST'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Task '{TASK_NAME}' not found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup auto-start for the data collector')
    parser.add_argument('--remove', action='store_true', help='Remove the scheduled task')
    parser.add_argument('--start',  action='store_true', help='Start the task immediately')
    parser.add_argument('--status', action='store_true', help='Check task status')
    args = parser.parse_args()

    if args.remove:
        remove_task()
    elif args.start:
        start_now()
    elif args.status:
        check_status()
    else:
        create_task()
        print("\nWould you like to start the collector right now? (y/n): ", end='')
        if input().strip().lower() == 'y':
            start_now()
