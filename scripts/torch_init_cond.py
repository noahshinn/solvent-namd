import os
import sys
import heapq
import torch

from typing import Tuple, List

_NAME = 'cyp-s1qd-wigner'
_DIR = '/scratch/adrion.d/photoclick-chemistry/cyp-test/solvent/initcond-gen/wigner-s1'
_NTRAJ = 500
_SAVE_FILE = 'cyp-15-init-cond.pt'

class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


def dist(atom1: torch.Tensor, atom2: torch.Tensor) -> torch.Tensor:
    return (atom1 - atom2).pow(2).sum().sqrt()

def load_coords(f: str) -> torch.Tensor:
    coords = []
    with open(f, 'r') as file:
        data = file.readlines()[2:]
        for l in data:
            l_data = [float(i) for i in l.split()[1:]]
            coords.append(torch.FloatTensor(l_data))
    coords = torch.stack(coords, dim=0) 
    com = coords.mean(dim=0) 
    return coords - com

def load_velo(f: str) -> torch.Tensor:
    with open(f, 'r') as file:
        data = file.readlines()
        velo = []
        for l in data:
            l_data = [float(i) for i in l.split()]
            velo.append(torch.FloatTensor(l_data))
    return torch.stack(velo, dim=0)

def indices(coords: torch.Tensor) -> List[int]:
    center = torch.mean(coords[:6], dim=0)
    dist_queue = PriorityQueue()
    for i, c in enumerate(coords[6::3]):
        dist_queue.push(i, dist(c, center))
    idxs = []
    for _ in range(15):
        idxs.append(dist_queue.pop()) 
    return idxs

def get_new_data(idxs: List[int], coords: torch.Tensor, velo: torch.Tensor) -> torch.Tensor:
    new_coords = [coords[:3], coords[3:6]]
    new_velo = [velo[:3], velo[3:6]]
    for idx in idxs:
        new_coords.append(coords[idx:idx + 3])
        new_velo.append(velo[idx:idx + 3])
    return torch.stack((torch.cat(new_coords, dim=0), torch.cat(new_velo, dim=0)), dim=0)

def run() -> None:
    data = []
    for i in range(1, _NTRAJ + 1):
        dir_ = os.path.join(_DIR, f'{_NAME}-{i}')
        xyz_file = os.path.join(dir_, f'{_NAME}-{i}.xyz')
        velo_file = os.path.join(dir_, f'{_NAME}-{i}.velocity.xyz')
        coords = load_coords(xyz_file)
        velo = load_velo(velo_file)
        idxs = indices(coords)
        traj_data = get_new_data(idxs, coords, velo)
        data.append(traj_data) 
    
    save_data = torch.stack(data, dim=0)
    torch.save(save_data, _SAVE_FILE)
    print('finished:', save_data.shape)
    print(f'saved to {_SAVE_FILE}')


if __name__ == '__main__':
    run()
