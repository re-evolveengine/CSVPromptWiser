# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

block_cipher = None

# Get the base directory - use the current working directory
base_dir = os.getcwd()

# Step 1 & 2: Include metadata from packages using importlib.metadata
datas = []
for pkg in ["streamlit", "pydantic", "google-api-core"]:
    try:
        datas += copy_metadata(pkg)
    except Exception as e:
        print(f"Warning: Could not copy metadata for {pkg}: {e}")

a = Analysis(
    ['streamlit_dir/streamlit_app.py'],
    pathex=[base_dir],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'streamlit',
        'pandas',
        'google.generativeai',
        'google.ai.generativelanguage',
        'google.api_core',
        'google.api',
        'google.auth',
        'google.oauth2',
        'pydantic',
        'pydantic_core',
        'pydantic.json',
        'pydantic.typing',
        'pydantic.fields',
        'pydantic.main',
        'pydantic.networks',
        'pydantic.types',
        'pydantic.utils',
        'pydantic.validators',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add data files from streamlit_dir if it exists
streamlit_dir = os.path.join(base_dir, 'streamlit_dir')
if os.path.exists(streamlit_dir):
    for root, _, files in os.walk(streamlit_dir):
        for f in files:
            src_path = os.path.join(root, f)
            dest_path = os.path.relpath(root, base_dir)
            a.datas.append((src_path, dest_path, 'DATA'))

# Collect data files from packages
for package in ['streamlit', 'google', 'pydantic']:
    try:
        for src_path, dest_dir in collect_data_files(package):
            a.datas.append((src_path, dest_dir, 'DATA'))
    except Exception as e:
        print(f"Warning: Could not collect data files for {package}: {e}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ChunkwisePrompter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='ChunkwisePrompter',
)
