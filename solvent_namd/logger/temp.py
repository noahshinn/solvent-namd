import numpy as np

itr = 4
size = 100
nstate = 3
atoms = np.random.rand(51, 1)
grads = np.random.rand(3, 51, 3)

def print_coord(xyz):
    ## This function convert a numpy array of coordinates to a formatted string
    coord = ''
    for line in xyz:
        e, x, y, z = line
        coord += '%-5s%24.16f%24.16f%24.16f\n' % (e, float(x), float(y), float(z))

    return coord

log_info = ''
for n in range(nstate):
    grad = grads[n]
    log_info += """
&gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (n + 1, print_coord(np.concatenate((atoms, grad), axis=1)))

print(log_info)
