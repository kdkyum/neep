import os.path as osp
import pickle

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

CURRENT_DIR = osp.abspath(osp.dirname(__file__))

# Spring and Stokes friction coefficient
k, e = 1, 1

# The leftmost (rightmost) temperature T1 (T2)
T1 = {16: 0.20768, 32: 0.10358, 64: 0.05171, 128: 0.02583}
T2 = 10

with open(osp.join(CURRENT_DIR, "cov_1.pkl"), "rb") as f:
    cov_dict = pickle.load(f)

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

    allow_num_beads = [16, 32, 64, 128]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    if num_beads == 16:
        cov = cov_dict[16]

    elif num_beads == 32:
        cov = cov_dict[32]

    elif num_beads == 64:
        cov = cov_dict[64]

    elif num_beads == 128:
        cov = cov_dict[128]

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


# def sampling(num_beads, num_trjs):
#     """Sampling the states of beads in steady state.
    
#     Args:
#         num_beads : Number of beads. Here, we allow only 2 and 5.
#         T1 : Leftmost temperature
#         T2 : Rightmost temperature
#         num_trjs : Number of trajectories you want. default = 1000.

#     Returns:
#         Sampled states from the probability density in steady state. 
#     """

#     allow_num_beads = [16, 32, 64]
#     assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

#     if num_beads == 16:
#         cov = cov_dict[16]

#     elif num_beads == 32:
#         cov = cov_dict[32]

#     elif num_beads == 64:
#         cov = cov_dict[64]

#     positions = np.random.multivariate_normal(np.zeros((num_beads,)), cov, num_trjs)

#     return positions


# def simulation(num_trjs, trj_len, num_beads, T1, T2, dt, seed=0):
#     """Simulation of a bead-spring model
   
#     Args:
#         num_beads : Number of beads.
#         dt : time step 
#         trj_len : length of trajectories
#         seed : seed of random generator
#         num_trjs : Number of trajectories you want. default = 1000.

#     Returns:
#         trajectories of a bead-spring model
#     """
#     T = np.linspace(T1, T2, num_beads)  # Temperatures linearly varies.
#     Drift = np.zeros((num_beads, num_beads))
#     for i in range(num_beads):
#         if i > 0:
#             Drift[i][i - 1] = k / e
#         if i < num_beads - 1:
#             Drift[i][i + 1] = k / e
#         Drift[i][i] = -2 * k / e

#     dt2 = np.sqrt(dt)

#     rfc = np.zeros((num_beads,))
#     for i in range(num_beads):
#         rfc[i] = np.sqrt(2 * e * T[i])

#     np.random.seed(seed)

#     trj = np.zeros((num_trjs, num_beads, trj_len))
#     position = sampling(num_beads, num_trjs)

#     for it in range(trj_len):
#         RanForce = np.random.normal(0, 1.0, (num_trjs, num_beads))
#         RanForce *= rfc

#         DriftForce = np.dot(position, Drift)

#         position += (DriftForce * dt + RanForce * dt2) / e

#         trj[:, :, it] = position

#     return trj.transpose(0, 2, 1)


def p_ss(num_beads, x):
    """Probability density function in steady state.
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.
        x : states of beads for multiple models. 
            So, its shape must be (number of trajectory, trj_len, number of beads)
        T1 : Leftmost temperature
        T2 : Rightmost temperature

    Returns:
        Probability at x.
    """
    allow_num_beads = [16, 32, 64]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    cov = cov_dict[num_beads]

    return torch.exp(-np.sum(np.dot(x, np.linalg.inv(cov))*x, axis=2)/2)


def cov(num_beads):
    return cov_dict[num_beads]


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
    allow_num_beads = [16, 32, 64, 128]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    etpy_prev = -torch.log(p_ss(num_beads, trj[:, :-1, :]))
    etpy_next = -torch.log(p_ss(num_beads, trj[:, 1:, :]))

    return (etpy_next - etpy_prev).astype(np.float32)


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
    allow_num_beads = [16, 32, 64, 128]
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

