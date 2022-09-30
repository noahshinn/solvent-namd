"""
STATUS: DEV

"""

import torch
from torch_geometric.data.data import Data

from solvent_namd import computer
from solvent_namd.trajectory import TrajectoryHistory, Snapshot

from solvent_namd.logger import Logger
from typing import Dict, NamedTuple


class TerminationStatus(NamedTuple):
    should_terminate: bool
    exit_code: int


class TrajectoryPropagator:
    def __init__(
            self,
            logger: Logger,
            model: torch.nn.Module,
            init_state: int,
            nstates: int,
            state_mult: torch.Tensor,
            natoms: int,
            mass: torch.Tensor,
            atom_types: torch.Tensor,
            one_hot_key: Dict,
            init_coords: torch.Tensor,
            init_velo: torch.Tensor,
            init_forces: torch.Tensor,
            init_energies: torch.Tensor,
            init_a: torch.Tensor,
            init_h: torch.Tensor,
            init_d: torch.Tensor,
            delta_t: float,
            ic_e_thresh: float,
            isc_e_thresh: float,
            max_hop: int
        ) -> None:
        """
        Initializes a trajectory propagator.

        Args:
            model (torch.nn.Module): A trained and loaded neural network
                model.
            res_model (torch.nn.Module): A trained and loaded residual
                block placed on top of the outputs of the standard model.
            state (int): The current electronic state that is currently
                being populated.
            mass (torch.Tensor): Atomic mass tensor of size (N) where N is
                the number of atoms.
            atom_types (torch.Tensor): A one-hot tensor representing atom
                types.
            one_hot_key (torch.Tensor): The key to the one-hot tensor.
            init_coords (torch.Tensor): Starting conditions: coordinates.
            init_velo (torch.Tensor): Starting conditions: velocities.
            init_forces (torch.Tensor): Starting conditions: forces.
            init_energies (torch.Tensor): Starting conditions: energies.
            init_a (torch.Tensor): Starting conditions: state-density matrix.
            init_h (torch.Tensor): Starting conditions: energy matrix.
            init_d (torch.Tensor): Starting conditions: non-adiabatic matrix.
            delta_t (float): The change in time from the previous snapshot
                to this snapshot in atomic units of time, au.

        Returns:
            None

        """
        self._logger = logger
        self._model = model

        self._iter = 0
        self._traj = TrajectoryHistory()
        self._mass = mass
        self._atom_types = atom_types
        self._natoms = natoms
        self._nstates = nstates
        self._state_mult = state_mult
        self._one_hot_key = one_hot_key
        self._cur_state = self._prev_state = init_state
        self._cur_coords = init_coords
        self._cur_velo = init_velo
        self._cur_forces = init_forces
        self._cur_energies = init_energies
        self._cur_a = init_a
        self._cur_h = init_h
        self._cur_d = init_d

        self._prev_coords = self._prev_prev_coords = torch.zeros_like(init_coords)
        self._prev_velo = self._prev_prev_velo = torch.zeros_like(init_velo)
        self._prev_forces = self._prev_prev_forces = torch.zeros_like(init_forces)
        self._prev_energies = self._prev_prev_energies = torch.zeros_like(init_energies)
        self._prev_a = self._prev_prev_a = torch.zeros_like(init_a)
        self._prev_h = self._prev_prev_h = torch.zeros_like(init_h)
        self._prev_d = self._prev_prev_d = torch.zeros_like(init_d)
        self._has_hopped = 'NO HOP'
        self._ic_e_thresh = ic_e_thresh
        self._isc_e_thresh = isc_e_thresh
        self._max_hop = max_hop

        self._kinetic_energy = torch.Tensor(0.0)

        self._delta_t = delta_t

    def propagate(self) -> None:
        """
        Propagates a trajectory by one step, by one snapshot.

        """
        self._save_snapshot()
        self._shift(mode='NUCLEAR')
        self._nuclear()
        self._shift(mode='ELECTRONIC')
        self._surface_hopping()
        self._reset_velo()

    def _nuclear(self) -> None:
        """
        Propagates a trajectory by one step, by one snapshot.

        """
        # FIXME: check if needed
        if self._iter == 0:
            self._scale_kinetic_energy()
            return

        self._cur_coords = computer.verlet_coords(
            state=self._cur_state,
            coords=self._cur_coords,
            mass=self._mass,
            velo=self._cur_velo,
            forces=self._cur_forces.clone(),
            delta_t=self._delta_t
        )

        self._cur_energies, self._cur_forces = computer.ml_energies_forces(
            model=self._model,
            structure=Data(
                x=self._atom_types,
                pos=self._cur_coords,
                z=self._mass
            )
        )

        self._cur_velo = computer.verlet_velo(
            state=self._cur_state,
            coords=self._cur_coords,
            mass=self._mass,
            velo=self._cur_velo,
            forces=self._cur_forces.clone(),
            forces_prev=self._prev_forces.clone(),
            delta_t=self._delta_t
        )

        self._kinetic_energy = computer.ke(
            mass=self._mass,
            velo=self._cur_velo
        )

    def _surface_hopping(self) -> None:
        a, h, d, v, has_hopped, state = computer.surface_hopping(
            state=self._cur_state,
            state_mult=self._state_mult,
            mass=self._mass,
            coord=self._cur_coords,
            coord_prev=self._prev_coords,
            coord_prev_prev=self._prev_prev_coords,
            velo=self._cur_velo,
            energies=self._cur_energies,
            energies_prev=self._prev_energies,
            energies_prev_prev=self._prev_prev_energies,
            forces=self._cur_forces,
            forces_prev=self._prev_forces,
            forces_prev_prev=self._prev_prev_forces,
            ke=self._kinetic_energy,
            ic_e_thresh=self._ic_e_thresh,
            isc_e_thresh=self._isc_e_thresh,
            max_hop=self._max_hop
        )
        self._cur_a = a
        self._cur_h = h
        self._cur_d = d
        self._cur_velo = v
        self._has_hopped = has_hopped 
        self._cur_state = state
 
    def log_step(self) -> None:
        """
        Logs the current trajectory step.

        """
        NotImplemented()

    # FIXME: check termination somewhere
    def status(self) -> TerminationStatus:
        """
        Determines if the current trajectory should be propagated further.

        Args:
            None

        Returns:
            (bool)

        """
        should_terminate = ...
        exit_code = ...
        return TerminationStatus(should_terminate, exit_code) # type: ignore

    # FIXME: check if needed
    def _scale_kinetic_energy(self) -> None:
        """
        Add or scale kinetic energy on initial step.

        """
        NotImplemented()

    # FIXME: check if needed     
    def _reset_velo(self) -> None:
        self._cur_velo = computer.reset_velo(
            mass=self._mass,
            coords=self._cur_coords,
            velo=self._cur_velo
        )

    # FIXME: find better way to decode one-hot
    def _save_snapshot(self) -> None:
        """
        Saves the current molecular system data to the running
        history.

        Args:
            None

        Returns:
            None

        """
        snapshot = Snapshot(
            iteration=self._iter,
            state=self._cur_state,
            one_hot=self._atom_types,
            one_hot_key=self._one_hot_key,
            coords=self._cur_coords.clone(),
            energy=self._cur_energies[self._cur_state],
            forces=self._cur_forces[self._cur_state]
        )
        self._traj.add(snapshot)

    def _shift(self, mode: str) -> None:
        """
        Shifts prev -> prev-prev, cur -> prev

        Args:
            mode (str): one of "NUCLEAR" | "ELECTRONIC"

        Returns:
            None

        """
        if mode == 'NUCLEAR':
            self._prev_prev_coords = self._prev_coords.clone()
            self._prev_prev_velo = self._prev_velo.clone()
            self._prev_prev_forces = self._prev_forces.clone()
            self._prev_prev_energies = self._prev_energies.clone()

            self._prev_coords = self._cur_coords.clone()
            self._prev_velo = self._cur_velo.clone()
            self._prev_forces = self._cur_forces.clone()
            self._prev_energies = self._cur_energies.clone()
        else:
            self._prev_prev_a = self._prev_a.clone()
            self._prev_prev_h = self._prev_h.clone()
            self._prev_prev_d = self._prev_d.clone()
            self._prev_state = self._cur_state
