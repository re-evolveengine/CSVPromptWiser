# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['cli/cli_app.py'],
    pathex=['.'],  # Add the project root to the Python path
    binaries=[],
    datas=[
    ('.env', '.'),  # Load API key
    ],
    hiddenimports=[
        'cli.cli_flow_controller',
        # Add any other modules that might be imported dynamically
    ],
    hookspath=[],
    hooksconfig={},
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PromptPilot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False if you don't want a console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True
)
