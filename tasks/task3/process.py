#!/usr/bin/env python3
# Forithmus process entrypoint for task3.
# Kept tiny so the real task code remains in task3_report_to_ct.py.
import runpy, sys
sys.argv = ["task3_report_to_ct.py", "--mode", "forithmus"]
runpy.run_path("/opt/app/task3_report_to_ct.py", run_name="__main__")
