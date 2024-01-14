# SolventDynamics

SolventDynamics is an open-source code for computing accelerated non-adiabatic molecular dynamics by utilizing machine learning inferred energy and force vectors.

## Installation

Python >= 3.7
torch >= 1.12
CUDA >= 11.6

To install:

- Create virtual environment

```bash
python -m venv ./solvent_venv
source ./solvent_venv/bin/activate
```

- Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 11.6

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
sudo yum clean all
sudo yum -y install nvidia-driver-latest-dkms cuda
sudo yum -y install cuda-drivers
```

- Install [torch](https://pytorch.org/) with CUDA

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
```

- Install [torch](https://pytorch.org/) - nightly build for vmap - with CUDA (optional)

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

- Install [solvent](https://github.com/noahshinn/solvent) from source:

```bash
git clone https://github.com/noahshinn/solvent.git
cd solvent
python setup.py develop
```

- Install solvent_namd from source by running:

```bash
git clone https://github.com/noahshinn/solvent-namd.git
cd solvent_namd
python setup.py develop
```

- Install the rest of the dependencies

```bash
pip install -r ./requirements.txt
```
