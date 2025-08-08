import sys
import os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning
build_exe_options = {
    "packages": ["os", "sys", "streamlit", "pandas", "google.generativeai", "dotenv", 
                 "protobuf", "typing_extensions", "pydantic", "model", "streamlit_dir", "utils"],
    "excludes": ["tkinter"],
    "include_files": [],
    "include_msvcr": True,
}

# GUI applications require a different base on Windows (the default is for a console application)
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="PromptPilot",
    version="0.1.0",
    description="PromptPilot Application",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "streamlit_app.py",
            base=base,
            target_name="PromptPilot",
            icon=None,  # You can add an .ico file here if you have one
        )
    ],
    python_requires=">=3.8",
)