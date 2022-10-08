# SolventDynamics
SolventDynamics is an open-source code for computing accelerated non-adiabatic molecular dynamics by utilizing machine learning inferred energy and force vectors.

**PLEASE NOTE:** the Solvent Dynamics code is still under active development.

## Installation
Python >= 3.7
torch >= 1.12
CUDA >= 11.6 (for gpu-accelerated inference)

To install:
  * Install torch
  ```
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  ```
  * or torch with CUDA (optional)
  ```
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  ```

  * Install joblib for multiprocessing
  ```
  pip install joblib
  ```

  * Install SolventDynamics (dev)
    from source:
  ```
  git clone https://github.com/noahshinn024/solvent-namd.git
  cd solvent_namd
  python setup.py develop
  ```

## Authors
* Noah Shinn
* Sulin Liu 
