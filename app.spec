# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = str(Path(SPECPATH).resolve().parents[1])
hiddenimports = [
    name for name in collect_submodules('sklearn')
    if '.tests' not in name and not name.endswith('.conftest')
] + [
    'maldi_imm.SpectrumObject',
    'maldi_imm.preprocessing',
]


a = Analysis(
    ['app.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        ('Images', 'Images'),
        ('screens/PCA', 'screens/PCA'),
        ('screens/fig_threshold', 'screens/fig_threshold'),
        ('screens/fig_rf_local', 'screens/fig_rf_local'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'numpy.tests',
        'pandas.tests',
        'scipy.tests',
        'sklearn.tests',
        'plotly.tests',
        'torch',
        'torch.testing',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MarIA',
    icon='Images\\maria_logo.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
