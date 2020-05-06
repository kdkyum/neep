import numpy as np
import pickle

# Spring and Stokes friction coefficient
k, e = 1, 1

# The leftmost (rightmost) temperature T1 (T2)
T1, T2 = 1, 10


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
    with open("covariance.pkl", "rb") as f:
        cov_dict = pickle.load(f)
    
    allow_num_beads = [8, 16, 32, 64, 128]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    if num_beads == 8:
        cov = cov_dict[8]

    elif num_beads == 16:
        cov = cov_dict[16]

    elif num_beads == 64:
        cov = cov_dict[32]

    elif num_beads == 128:
        cov = cov_dict[64]

    positions = np.random.multivariate_normal(np.zeros((num_beads,)), cov, num_trjs)

    return positions


def analytic_etpy(num_beads):
    """Analytic entropy production rate for a bead-spring model. 
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.

    Returns:
        Analytic entropy production rate for bead-spring model in steady state.
    """
    allow_num_beads = [8, 16, 32, 64, 128]
    assert num_beads in allow_num_beads, "'num_beads' must be 8, 16, 32, 64, or 128"

    if num_beads == 8:
        return 0.384426

    elif num_beads == 16:
        return 0.197442

    elif num_beads == 32:
        return 0.10423

    elif num_beads == 64:
        return 0.0550592

    elif num_beads == 128:
        return 0.0288033
