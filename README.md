# SolventDynamics
SolventDynamics is an open-source code for computing accelerated non-adiabatic molecular dynamics by utilizing machine learning inferred energy and force vectors.

**PLEASE NOTE:** the Solvent Dynamics code is still under active development. Single-state dynamics can be performed with this current version. The FSSH algorithm is in dev status.

## Installation
Python >= 3.7
torch >= 1.12
CUDA >= 11.6 (for gpu-accelerated inference)

To install:
  * Create virtual environment
  ```
  python -m venv ./solvent_venv
  source ./solvent_venv/bin/activate
  ```
  * Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 11.6
  ```
  wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
  sudo rpm -i cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
  sudo yum clean all
  sudo yum -y install nvidia-driver-latest-dkms cuda
  sudo yum -y install cuda-drivers
  ```
  * Install [torch](https://pytorch.org/) with CUDA
  ```
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  ```
  * Install [torch](https://pytorch.org/) - nightly build for vmap - with CUDA (optional)
  ```
  pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
  ```
  * Install [joblib](https://joblib.readthedocs.io/en/latest/installing.html) for multiprocessing
  ```
  pip install joblib
  ```
  * Install [solvent](https://github.com/noahshinn024/solvent) from source:
  ```
  git clone https://github.com/noahshinn024/solvent.git
  cd solvent
  python setup.py develop
  ```
  * Install solvent_namd from source by running:
  ```
  git clone https://github.com/noahshinn024/solvent-namd.git
  cd solvent_namd
  python setup.py develop
  ```

Several demo scripts are included in `./demo`:

## Authors
* Noah Shinn
* Sulin Liu 
