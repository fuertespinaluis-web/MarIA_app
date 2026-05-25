# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


project_root = str(Path(SPECPATH).resolve())
hiddenimports = [
    name for name in collect_submodules('sklearn')
    if '.tests' not in name and not name.endswith('.conftest')
] + [
    'application.maldi_imm.SpectrumObject',
    'application.maldi_imm.preprocessing',
]


a = Analysis(
    ['app.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        ('application/assets', 'application/assets'),
        ('application/screens/PCA', 'application/screens/PCA'),
        ('application/screens/fig_threshold', 'application/screens/fig_threshold'),
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
    icon='application\\assets\\maria_logo.ico',
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
