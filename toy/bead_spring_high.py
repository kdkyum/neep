import os.path as osp
import pickle

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

CURRENT_DIR = osp.abspath(osp.dirname(__file__))

# Spring and Stokes friction coefficient
k, e = 1, 1

# The leftmost (rightmost) temperature T1 (T2)
T1 = {8 : 0.416997, 16: 0.20768, 32: 0.10358, 48: 0.06899, 64: 0.05171, 80: 0.04136, 128: 0.02583}
T2 = 10

with open(osp.join(CURRENT_DIR, "covariance.pkl"), "rb") as f:
    cov_dict = pickle.load(f)

allow_num_beads = cov_dict.keys()
print(allow_num_beads)

def sampling(num_beads, num_trjs):
    """Sampling the states of beads in steady state.
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.
        T1 : Leftmost temperature
        T2 : Rightmost temperature
        num_trjs : Number of trajectories you want. default = 1000.

    Returns:
        Sampled states from the probability density in steady state. 
    """
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    cov = cov_dict[num_beads]

    N = MultivariateNormal(torch.zeros(num_beads), torch.from_numpy(cov).float())
    positions = N.sample((num_trjs,))

    return positions


def simulation(num_trjs, trj_len, num_beads, dt, device='cpu', seed=0):
    """Simulation of a bead-spring 2d-lattice model
    
    Args:
        num_beads : Number of beads for each row
        T1 : LeftUp-most temperature
        T2 : RighDown-tmost temperature
        dt : time step 
        trj_len : length of trajectories
        seed : seed of random generator
        num_trjs : Number of trajectories you want. default = 1000.

    Returns:
        trajectories of a bead-spring 2d-lattice model
    """
    T = torch.linspace(T1[num_beads], T2, num_beads).to(device)
    dt2 = torch.sqrt(torch.tensor(dt).float())
    
    trj = torch.zeros(num_trjs, trj_len, num_beads).to(device)
    Drift = torch.zeros(num_beads, num_beads).to(device)
    position = sampling(num_beads, num_trjs).to(device)
    
    for i in range(num_beads):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < num_beads - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e

    rfc = torch.zeros(num_beads).to(device)
    for i in range(num_beads):
        rfc[i] = torch.sqrt(2 * e * T[i])

    torch.manual_seed(seed)
            
    for it in range(trj_len):
        RanForce = torch.randn(num_trjs, num_beads, device=device)
        RanForce *= rfc

        DriftForce = torch.einsum('ij,aj->ai', Drift, position)

        position += (DriftForce * dt + RanForce * dt2) / e

        trj[:, it, :] = position

    return trj


def del_shannon_etpy(trj):
    """Shannon entropy (or system entropy) difference for trajectories.
    
    Args:
        trj : Trajectories of multiple model for trj_len.
            So, its shape must be (number of trajectory, trj_len, number of beads).

    Returns:
        Shannon entropy difference for each step of 'trj'. 
        So, its shape is (number of trajectory, trj_len)
    """
    num_beads = trj.shape[-1]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"
    cov = cov_dict[num_beads]
    cov = torch.from_numpy(cov).float()
    etpy = torch.sum(trj @ torch.inverse(cov)* trj, axis=2)/2

    return etpy[:, 1:] - etpy[:, :-1]


def del_medium_etpy(trj):
    """Medium entropy (or system entropy) difference for trajectories.
    
    Args:
        trj : Trajectories of multiple model for trj_len. 
            So, its shape must be (number of trajectory, trj_len, number of beads).

    Returns:
        Medium entropy difference for each step of 'trj'. 
        So, its shape is (number of trajectory, trj_len) or (number of trajectory, trj_len)
    """
    num_beads = trj.shape[-1]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    Drift = torch.zeros(num_beads, num_beads).to(trj.device)
    T = torch.linspace(T1[num_beads], T2, num_beads).to(trj.device)

    for i in range(num_beads):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < num_beads - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e

    x_prev = trj[:, :-1, :]
    x_next = trj[:, 1:, :]
    dx = x_next - x_prev

    Fx = ((x_next + x_prev)/2) @ Drift

    dQ = Fx * dx
    etpy = torch.sum(dQ / T, dim=2)

    return etpy
