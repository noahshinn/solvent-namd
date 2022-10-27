from ase.io import read

atoms = read('./traj_0.md.xyz', index=':3')
pos = [a.positions for a in atoms]
