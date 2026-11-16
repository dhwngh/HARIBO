"""Streamlit entrypoint compatibility file.

The previous GitHub repository showed that APP/main.py had been deleted.
If the existing Streamlit deployment still points to APP/main.py, this file
keeps that entrypoint working by loading the current APP/app.py application.
"""
import runpy
from pathlib import Path

app_path = Path(__file__).resolve().with_name("app.py")
runpy.run_path(str(app_path), run_name="__main__")
