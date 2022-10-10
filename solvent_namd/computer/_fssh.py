"""
STATUS: DEV

"""

import math
import torch

from typing import List, Tuple
from solvent_namd.types import SurfaceHoppingMetrics

_CUTOFF = 1e-16


def _avoid_singularity(
        e_i: torch.Tensor,
        e_j: torch.Tensor,
        state_i: int,
        state_j: int
    ) -> float:
    e_gap = (e_i - e_j).abs().item()
    s = math.copysign(1, state_i - state_j)
    return s * max(e_gap, _CUTOFF)

def _ktdc(
        state_i: int,
        state_j: int,
        cur_energies: torch.Tensor,
        prev_energies: torch.Tensor,
        prev_prev_energies: torch.Tensor,
        delta_t: float
    ) -> float:
    """
    Curvature-driven time-dependent coupling.
    *Truhlar et al J. Chem. Theory Comput. 2022 DOI:10.1021/acs.jctc.1c01080
    
    Args:

    Returns:

    """
    d_vt = _avoid_singularity(
        e_i=cur_energies[state_i],
        e_j=cur_energies[state_j],
        state_i=state_i,
        state_j=state_j
    )
    d_vt_d_t = _avoid_singularity(
        e_i=prev_energies[state_i],
        e_j=prev_energies[state_j],
        state_i=state_i,
        state_j=state_j
    )
    d_v_2d_t = _avoid_singularity(
        e_i=prev_prev_energies[state_i],
        e_j=prev_prev_energies[state_j],
        state_i=state_i,
        state_j=state_j
    )
    d2_v_d_2t = (d_vt - 2 * d_vt_d_t + d_v_2d_t) / delta_t ** 2

    d_ = d2_v_d_2t / d_vt > 0
    if d_ > 0:
        return d_ ** 0.5 / 2
    return 0.0

def _d_p_d_t(
        nstates: int,
        a: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
    """ John C. Tully, J. Chem. Phys. 93, 1061 (1990) """
    return torch.stack([a[k, j] * (-1j * h[i, k] - d[i, k])
        - a[i, k] * (-1j * h[k, j] - d[k, j]) for k in range(nstates)
        for j in range(nstates) for i in range(nstates)], dim=0)

def _b_matrix(
        nstates: int,
        a: torch.Tensor,
        h: torch.Tensor,
        d: torch.Tensor
    ) -> torch.Tensor:
    """ John C. Tully, J. Chem. Phys. 93, 1061 (1990) """
    return torch.stack([2 * torch.imag(torch.conj(a[i, j]) * h[i, j]) -2
        * torch.real(torch.conj(a[i, j]) * d[i, j]) for j in range(nstates)
        for i in range(nstates)], dim=0)

def _get_nac(
        natoms: int,
        cur_state: int,
        next_state: int,
        nac_coupling: List[Tuple[int]],
        nacs: torch.Tensor
    ) -> torch.Tensor:
    nac_pair = sorted((cur_state - 1, next_state - 1))
    if nac_pair in nac_coupling and nacs.size(dim=0) > 0:
        nac_pos = nac_coupling.index(nac_pair)
        return nacs[nac_pos]
    return torch.ones(natoms, 3)

def _fssh_log_info() -> None:
    NotImplemented()

def fssh(
        prev_a: torch.Tensor,
        prev_h: torch.Tensor,
        prev_d: torch.Tensor,
        nacs: torch.Tensor,
        socs: torch.Tensor,
        nsubsteps: int,
        step_size: float,
        iteration: int,
        nstates: int,
        cur_state: int,
        nmax_hop: int,
        e_deco: float,
        adjust: int,         # FIXME: better name and type
        reflect: int,        # FIXME: better name and type
        nacs_type: str,
        mass: torch.Tensor,
        cur_velo: torch.Tensor,
        cur_energies: torch.Tensor,
        prev_energies: torch.Tensor,
        prev_prev_energies: torch.Tensor,
        ke: torch.Tensor,
        nac_coupling: List[Tuple[int]],
        soc_coupling: List[Tuple[int]],
        state_mult: List[int]
    ) -> SurfaceHoppingMetrics:

    ## initialize nac matrix
    if iteration > 2:
        for n, pair in enumerate(nac_coupling):
            s1, s2 = pair
            if nacs_type == 'nac':
                nacme = torch.sum(velo * N[n]) / _avoid_singularity(E[s1], E[s2], s1, s2)
            elif nacs_type == 'ktdc':
                nacme = _ktdc(s1, s2, E, Ep, Epp, step_size * substep)
            Dt[s1, s2] = nacme
            Dt[s2, s1] = -Dt[s1, s2]

    ## initialize soc matrix
    for n, pair in enumerate(soc_coupling):
        s1, s2 = pair
        socme = S[n] / 219474.6  # convert cm-1 to Hartree
        Ht[s1, s2] = socme
        Ht[s2, s1] = socme

    ## initialize cur_state index and order
    stateindex = torch.argsort(E)
    stateorder = torch.argsort(E).argsort()

    ## start fssh calculation
    if iteration < 4:
        At[cur_state - 1, cur_state - 1] = 1
        Vt = velo
        info = '  No surface hopping is performed'
    else:
        dHdt = (Ht - h) / substep
        dDdt = (Dt - d) / substep
        nhop = 0
        
        if verbose >= 2:
            print('-------------- TEST ----------------')
            print('Iter: %s' % (iteration))
            print('Previous Population')
            print(a)
            print('Previous Hamiltonian')
            print(h)
            print('Previous NAC')
            print(d)
            print('Current Hamiltonian')
            print(Ht)
            print('Current NAC')
            print(Dt)
            print('One step population gradient')
            print('dPdt')
            print(dPdt(a,h,d))
            print('_b_matrix')
            print(_b_matrix(a+dPdt(a,h,d)*step_size*substep,h,d)*step_size*substep)
            print('Integration start')

        for i in range(substep):
            if integrate == 0:
                B = torch.zeros((nstates, nstates))
            g = torch.zeros(nstates)
            event = 0
            frustrated=0

            h += dHdt
            d += dDdt

            dAdt = dPdt(a, h, d)
            dAdt *= step_size
            a += dAdt
            dB = _b_matrix(a, h, d)
            B += dB

            exceed = torch.diag(torch.real(a)) - 1
            deplet = 0 - torch.diag(torch.real(a))
            rstate = [torch.argmax(exceed), torch.argmax(deplet)][torch.argmax([torch.amax(exceed), torch.amax(deplet)])]
            revert = torch.amax([exceed[rstate], deplet[rstate]])
            if revert > 0:
                a -= dAdt * torch.abs(revert / torch.real(dAdt)[rstate, rstate])  # revert a
                B -= dB * torch.abs(revert / torch.real(dAdt)[rstate, rstate])
                stop = 1 # stop if population exceed 1 or less than 0

            for j in range(nstates):
                if j != cur_state - 1:
                    g[j] += torch.amax([0, B[j, cur_state - 1] * step_size / torch.real(a[cur_state - 1, cur_state - 1])])

            z = torch.random.uniform(0, 1)

            gsum = 0
            for j in range(nstates):
                gsum += g[stateindex[j]]
                nhop = torch.abs(stateindex[j] - cur_state + 1)
                if gsum > z and 0 < nhop <= maxhop:
                    new_state = stateindex[j] + 1
                    event = 1
                    hop_g = torch.copy(g)
                    hop_gsum = gsum
                    hop_z = z
                    break

            if verbose > 2:
                print('\nSubIter: %5d' % (i+1))
                print('d nac matrix')
                print(d)
                print('a population matrix')
                print(a)
                print('B transition matrix')
                print(B)
                print('Probabality')
                print(' '.join(['%12.8f' % (x) for x in g]))
                print('Population')
                print(' '.join(['%12.8f' % (torch.real(x)) for x in torch.diag(a)]))
                print('Random: %s' % (z))
                print('old cur_state/new cur_state: %s / %s' % (cur_state, new_state))

            ## detect frustrated hopping and adjust velocity
            if event == 1:
                NAC = _get_nac(cur_state, new_state, nac_coupling, N, len(velo))
                Vt, frustrated = adjust_velo(E[cur_state - 1], E[new_state - 1], velo, M, NAC, adjust, reflect)
                if frustrated == 0:
                    cur_state = new_state

            ## decoherence of the propagation
            if usedeco != 'OFF':
                deco = float(usedeco)
                tau = torch.zeros(nstates)

                ## matrix tau
                for k in range(nstates):
                    if k != cur_state-1:
                        tau[k] = torch.abs( 1 / _avoid_singularity(
                            torch.real(h[cur_state - 1, cur_state - 1]), 
                            torch.real(h[k, k]),
                            cur_state - 1,
                            k)) * (1 + deco / Ekin) 

                ## update diagonal of a except for current cur_state
                for k in range(nstates):
                    for j in range(nstates):
                        if k != cur_state - 1 and j != cur_state - 1:
                            a[k, j] *= torch.exp(-step_size / tau[k]) * torch.exp(-step_size / tau[j])

                ## update diagonal of a for current cur_state
                Asum = 0.0
                for k in range(nstates):
                    if k != cur_state - 1:
                        Asum += torch.real(a[k, k])
                Amm = torch.real(a[cur_state - 1, cur_state - 1])
                a[cur_state - 1, cur_state - 1] = 1 - Asum

                ## update off-diagonal of a
                for k in range(nstates):
                    for j in range(nstates):
                        if   k == cur_state - 1 and j != cur_state - 1:
                            a[k, j] *= torch.exp(-step_size / tau[j]) * (torch.real(a[cur_state - 1, cur_state - 1]) / Amm)**0.5
                        elif k != cur_state - 1 and j == cur_state - 1:
                            a[k, j] *= torch.exp(-step_size / tau[k]) * (torch.real(a[cur_state - 1, cur_state - 1]) / Amm)**0.5

            if stop == 1:
                break

        ## final decision on velocity
        if cur_state == old_state:   # not hoped
            Vt = velo               # revert scaled velocity
            hoped = 0
        else:
            NAC = _get_nac(cur_state, new_state, nac_coupling, N, len(velo))
            Vt, frustrated = adjust_velo(E[old_state - 1], E[cur_state - 1], velo, M, NAC, adjust, reflect)
            if frustrated == 0:  # hoped
                hoped = 1
            else:                # frustrated hopping
                hoped = 2

        At = a

        if len(hop_g) == 0:
            hop_g = g
            hop_z = z
            hop_gsum = gsum

        summary = ''
        for n in range(nstates):
            summary += '    %-5s %-5s %-5s %12.8f\n' % (n + 1, state_mult[n], stateorder[n] + 1, hop_g[n])

        info = """
    Random number:           %12.8f
    Accumulated probability: %12.8f
    cur_state mult  level   probability 
%s
    """ % (hop_z, hop_gsum, summary)

    return SurfaceHoppingMetrics(a, h, d, velo, hop_type, cur_state, log_info)
