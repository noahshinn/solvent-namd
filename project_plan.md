# Project Plan

## Checkpoints

### Refactor to pair with NequIP
  - config
  - load from json, yaml
  - model loading
  - dynamically set natoms, nstates
  - NequIP calculator
  - success metrics:
    - successful loading, model evaluation, force grads

### Fix logger
  - log config
  - md.energy file
  - units
  - success metrics:
    - run sample run for 10fs and check for proper logging
    - md.xyz, md.energy, initial conditions

### Single-state dynamics test run (100fs)
  - ex. start on s1 and don't hop
  - success metrics:
    - correct units used throughout propagation step + logging
    - no more than 10% of trajectories with early termination

### Single or multi-model options
  - single: one model trained to predict multiple energy states
  - multiple: N models trained to predict single energy states where N is the number of energy states
  - success metrics:
    - ability to specify model(s) in yaml or json file

### NAC model load option
  - fix FSSH approximation
  - fix Zhu-Nakamura for fallback surface hopping prediction
  - success metrics:
    - dimension fixes

### Train NequIP model to learn NAC vector
  - if successful, then adapt dynamics to use coordinate, species -> NAC vector
  - if not successful, try to learn coordinate, species -> vector after energy scale
    - adapt dynamics to use ML-energies for NAC vector predictions
  - if not successful, try to use multiple NequIP models to learn coordinate, species -> NAC vector (in specific energy difference regions)
    - adapt dynamics to use multiple NAC models
  - ideally, adapt dynamics to use a variety of strategies
  - success metrics:
    - error within 5% near conical intersections
    - MAE within 0.1 in near 0 regions

### Multi-state dynamics test run (1ps)
  - start on s1 and allow hopping
  - success metrics:
    - no more than 25% trajectories with early termination
    - quantum yield check

### Multi-state dynamics test run (1ns):
  - start on s1 and allow hopping
  - success metrics:
    - no more than 50% trajectories with early termination
    - quantum yield check
