import torch
from e3nn import o3
import time

from solvent import models 

_MODEL_FILE = 'cyp-15-training-params.pt'
_JIT_SAVE_FILE = 'cyp-15-trained-model.pt'


model = models.SolventModel(
    irreps_in='3x0e',
    hidden_sizes=[125, 40, 25, 15],
    irreps_out='3x0e',
    irreps_node_attr=None,
    irreps_edge_attr=o3.Irreps.spherical_harmonics(3), # type: ignore
    nlayers=3,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=2,
    nradial_neurons=128,
    navg_neighbors=16.0
)
model.load_state_dict(torch.load(_MODEL_FILE, map_location='cpu')['model'])

model2 = torch.jit.load(_JIT_SAVE_FILE)

x = {
    'x': torch.rand(51, 3),
    'pos': torch.rand(51, 3),
    'z': torch.rand(51)
}

# traced_model = torch.jit.trace(model, x)
# torch.jit.save(traced_model, _JIT_SAVE_FILE)

def _ml_forces(energies: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    pos.requires_grad = True
    nstates = energies.size(dim=0)
    forces = []
    for i in range(nstates):
        f = torch.autograd.grad(
            -energies[i],
            pos,
            create_graph=True,
            retain_graph=True,
        )[0]
        forces.append(f)
    forces = torch.stack(forces, dim=0)

    return forces

x['pos'].requires_grad = True
# y = model(x).squeeze()
y = model(x)
# y2 = model2(x)

f = _ml_forces(y, x['pos'])
print(f.shape)
# f2 = y2[0].backward(gradient=x['pos'], create_graph=True)[0]
# print(f2)

srt = time.perf_counter()
for i in range(10):
    y = model(x)
    f = _ml_forces(y, x['pos'])
print('model:', time.perf_counter() - srt)

# srt2 = time.perf_counter()
# for i in range(10):
    # y2 = model2(x)
# print('model2:', time.perf_counter() - srt2)

# y = model(x)
# y2 = model2(x)

# print(y)
# print(y2)
