from setuptools import setup

setup(
    name='solvent_namd',
    version='0.0.0',
    description='ML-Accelerated Non-Adiabatic Molecular Dynamics',
    author='Noah Shinn, Sulin Liu',
    packages=['solvent_namd'],
    entry_points={
        "console_scripts": [
            'solvent_namd = solvent_namd.scripts.run_namd:main'
        ]
    },
    install_requires=[
        'joblib',
        'numpy',
        'pyyaml',
        'nequip==0.5.5'
    ]
)
