import numpy as np

# Spring and Stokes friction coefficient
k, e = 1, 1


def sampling(num_beads, T1, T2, num_trjs):
    """Sampling the states of beads in steady state.
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.
        T1 : Leftmost temperature
        T2 : Rightmost temperature
        num_trjs : Number of trajectories you want. default = 1000.

    Returns:
        Sampled states from the probability density in steady state. 
    """
    assert (num_beads == 2) or (num_beads == 5), "'num_beads' must be 2 or 5"

    if num_beads == 2:
        cov = [
            [(7 * T1 + T2) / (12.0 * k), (T1 + T2) / (6.0 * k)],
            [(T1 + T2) / (6.0 * k), (T1 + 7 * T2) / (12.0 * k)],
        ]
        positions = np.random.multivariate_normal(np.zeros((2,)), cov, num_trjs)

    elif num_beads == 5:
        cov = [
            [
                (1477 * T1 + 173 * T2) / (1980.0 * k),
                (487 * T1 + 173 * T2) / (990.0 * k),
                (619 * T1 + 371 * T2) / (1980.0 * k),
                (361 * T1 + 299 * T2) / (1980.0 * k),
                (T1 + T2) / (12.0 * k),
            ],
            [
                (487 * T1 + 173 * T2) / (990.0 * k),
                (2 * (15 * T1 + 7 * T2)) / (33.0 * k),
                (1141 * T1 + 839 * T2) / (1980.0 * k),
                (T1 + T2) / (3.0 * k),
                (299 * T1 + 361 * T2) / (1980.0 * k),
            ],
            [
                (619 * T1 + 371 * T2) / (1980.0 * k),
                (1141 * T1 + 839 * T2) / (1980.0 * k),
                (3 * (T1 + T2)) / (4.0 * k),
                (839 * T1 + 1141 * T2) / (1980.0 * k),
                (371 * T1 + 619 * T2) / (1980.0 * k),
            ],
            [
                (361 * T1 + 299 * T2) / (1980.0 * k),
                (T1 + T2) / (3.0 * k),
                (839 * T1 + 1141 * T2) / (1980.0 * k),
                (2 * (7 * T1 + 15 * T2)) / (33.0 * k),
                (173 * T1 + 487 * T2) / (990.0 * k),
            ],
            [
                (T1 + T2) / (12.0 * k),
                (299 * T1 + 361 * T2) / (1980.0 * k),
                (371 * T1 + 619 * T2) / (1980.0 * k),
                (173 * T1 + 487 * T2) / (990.0 * k),
                (173 * T1 + 1477 * T2) / (1980.0 * k),
            ],
        ]

        positions = np.random.multivariate_normal(np.zeros((5,)), cov, num_trjs)

    return positions


def simulation(num_trjs, trj_len, num_beads, T1, T2, dt, seed=0):
    """Simulation of a bead-spring model
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.
        T1 : Leftmost temperature
        T2 : Rightmost temperature
        dt : time step 
        trj_len : length of trajectories
        seed : seed of random generator
        num_trjs : Number of trajectories you want. default = 1000.

    Returns:
        trajectories of a bead-spring model
    """
    T = np.linspace(T1, T2, num_beads)  # Temperatures linearly varies.
    print(T)
    Drift = np.zeros((num_beads, num_beads))
    for i in range(num_beads):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < num_beads - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e

    dt2 = np.sqrt(dt)

    rfc = np.zeros((num_beads,))
    for i in range(num_beads):
        rfc[i] = np.sqrt(2 * e * T[i])

    np.random.seed(seed)

    print("MultiBead Simulation Starts with seed=", seed)

    trj = np.zeros((num_trjs, num_beads, trj_len))
    position = sampling(num_beads, T1, T2, num_trjs)

    for it in range(trj_len):
        RanForce = np.random.normal(0, 1.0, (num_trjs, num_beads))
        RanForce *= rfc

        DriftForce = np.dot(position, Drift)

        position += (DriftForce * dt + RanForce * dt2) / e

        trj[:, :, it] = position

    return trj


def p_ss(num_beads, x, T1, T2):
    """Probability density function in steady state.
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.
        x : states of beads for multiple models. 
            So, its shape must be (number of model, number of beads)
        T1 : Leftmost temperature
        T2 : Rightmost temperature

    Returns:
        Probability at x.
    """
    assert (num_beads == 2) or (num_beads == 5), "'num_beads' must be 2 or 5"

    if num_beads == 2:
        x1, x2 = x[:, 0], x[:, 1]
        return np.exp(
            -0.5
            * (
                4
                * k
                * (
                    T2 * (7 * x1 ** 2 - 4 * x1 * x2 + x2 ** 2)
                    + T1 * (x1 ** 2 - 4 * x1 * x2 + 7 * x2 ** 2)
                )
            )
            / (T1 ** 2 + 14 * T1 * T2 + T2 ** 2)
        )

    elif num_beads == 5:
        x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        return np.exp(
            -0.5
            * (
                (
                    4
                    * k
                    * (
                        12
                        * T2
                        * T1 ** 3
                        * (
                            568423 * x1 ** 2
                            + 743354 * x2 ** 2
                            + 1117575 * x3 ** 2
                            - 1528831 * x3 * x4
                            + 1837478 * x4 ** 2
                            + 23794 * x3 * x5
                            - 2009111 * x4 * x5
                            + 2044727 * x5 ** 2
                            - x1 * (612169 * x2 - 2306 * x3 + 6680 * x4 + 450 * x5)
                            - x2 * (862049 * x3 + 32768 * x4 + 1000 * x5)
                        )
                        + 12
                        * T2 ** 3
                        * T1
                        * (
                            2044727 * x1 ** 2
                            + 1837478 * x2 ** 2
                            + 1117575 * x3 ** 2
                            - 862049 * x3 * x4
                            + 743354 * x4 ** 2
                            + 2306 * x3 * x5
                            - 612169 * x4 * x5
                            + 568423 * x5 ** 2
                            - x1 * (2009111 * x2 - 23794 * x3 + 1000 * x4 + 450 * x5)
                            - x2 * (1528831 * x3 + 32768 * x4 + 6680 * x5)
                        )
                        + T2 ** 4
                        * (
                            5832179 * x1 ** 2
                            + 1277280 * x2 ** 2
                            + 374363 * x3 ** 2
                            - 274278 * x3 * x4
                            + 238824 * x4 ** 2
                            + 1562 * x3 * x5
                            - 194694 * x4 * x5
                            + 180563 * x5 ** 2
                            + x2 * (-376146 * x3 + 20448 * x4 + 4896 * x5)
                            - 2
                            * x1
                            * (2171241 * x2 + 190403 * x3 + 46272 * x4 + 10537 * x5)
                        )
                        + T1 ** 4
                        * (
                            180563 * x1 ** 2
                            + 238824 * x2 ** 2
                            + 374363 * x3 ** 2
                            - 376146 * x3 * x4
                            + 1277280 * x4 ** 2
                            - 380806 * x3 * x5
                            - 4342482 * x4 * x5
                            + 5832179 * x5 ** 2
                            - 2 * x1 * (97347 * x2 - 781 * x3 - 2448 * x4 + 10537 * x5)
                            - 6 * x2 * (45713 * x3 - 3408 * x4 + 15424 * x5)
                        )
                        + 2
                        * T2 ** 2
                        * T1 ** 2
                        * (
                            12677929 * x1 ** 2
                            + 15120156 * x2 ** 2
                            + 17577937 * x3 ** 2
                            - 16692708 * x3 * x4
                            + 15120156 * x4 ** 2
                            + 33022 * x3 * x5
                            - 13366932 * x4 * x5
                            + 12677929 * x5 ** 2
                            + x1
                            * (-13366932 * x2 + 33022 * x3 + 89904 * x4 + 26474 * x5)
                            + x2 * (-16692708 * x3 + 372768 * x4 + 89904 * x5)
                        )
                    )
                )
                / (
                    3.0
                    * (
                        115491 * T2 ** 5
                        + 4368823 * T2 ** 4 * T1
                        + 16424486 * T2 ** 3 * T1 ** 2
                        + 16424486 * T2 ** 2 * T1 ** 3
                        + 4368823 * T2 * T1 ** 4
                        + 115491 * T1 ** 5
                    )
                )
            )
        )

    return -1


def del_shannon_etpy(trj, T1, T2):
    """Shannon entropy (or system entropy) difference for trajectories.
    
    Args:
        trj : Trajectories of multiple model for trj_len.
            So, its shape must be (number of model, trj_len, number of beads).
        T1 : Leftmost temperature
        T2 : Rightmost temperature

    Returns:
        Shannon entropy difference for each step of 'trj'. 
        So, its shape is (number of model, trj_len) or (number of model, trj_len)
    """

    num_trjs = trj.shape[0]
    num_beads = trj.shape[1]
    assert (num_beads == 2) or (
        num_beads == 5
    ), "'number of beads' in 'trj' must be 2 or 5"
    trj_len = trj.shape[-1]
    etpy_avg = np.zeros((num_trjs, trj_len), dtype=np.float32)

    p0 = -np.log(p_ss(num_beads, trj[:, :, 0], T1, T2))
    for it in range(trj_len):
        pt = -np.log(p_ss(num_beads, trj[:, :, it], T1, T2))
        etpy_avg[:, it] = pt - p0
        p0 = pt

    return etpy_avg[:, 1:]


def del_medium_etpy(trj, T1, T2):
    """Medium entropy (or system entropy) difference for trajectories.
    
    Args:
        trj : Trajectories of multiple model for trj_len. 
            So, its shape must be (number of model, trj_len, number of beads).
        T1 : Leftmost temperature
        T2 : Rightmost temperature

    Returns:
        Medium entropy difference for each step of 'trj'. 
        So, its shape is (number of model, trj_len) or (number of model, trj_len)
    """

    num_trjs = trj.shape[0]
    num_beads = trj.shape[1]
    assert (num_beads == 2) or (
        num_beads == 5
    ), "'number of beads' in 'trj' must be 2 or 5"
    trj_len = trj.shape[-1]
    etpy = np.zeros((num_trjs, trj_len), dtype=np.float32)

    dQ = np.zeros((num_trjs, num_beads))

    Drift = np.zeros((num_trjs, num_beads, num_beads))
    T = np.linspace(T1, T2, num_beads)

    for j in range(num_trjs):
        for i in range(num_beads):
            if i > 0:
                Drift[j][i][i - 1] = k / e
            if i < num_beads - 1:
                Drift[j][i][i + 1] = k / e
            Drift[j][i][i] = -2 * k / e

    x = trj[:, :, 0]
    for it in range(trj_len):
        dx = trj[:, :, it] - x

        Fx = np.zeros(x.shape)
        for i in range(num_trjs):
            Fx[i] = np.dot(Drift[i], (x + (dx / 2))[i])

        dQ = Fx * dx
        etpy[:, it] = np.sum(dQ / T, axis=1)

        x = trj[:, :, it]

    return etpy[:, 1:]


def analytic_etpy(num_beads, T1, T2):
    """Analytic entropy production rate for a bead-spring model. 
    
    Args:
        num_beads : Number of beads. Here, we allow only 2 and 5.
        T1 : Leftmost temperature
        T2 : Rightmost temperature

    Returns:
        Analytic entropy production rate for bead-spring model in steady state.
    """
    assert (num_beads == 2) or (num_beads == 5), "'num_beads' must be 2 or 5"

    if num_beads == 2:
        return k * ((T1 - T2) ** 2) / (4 * e * T1 * T2)

    elif num_beads == 5:
        return (
            k
            * (T1 - T2) ** 2
            * (111 * T1 ** 2 + 430 * T1 * T2 + 111 * T2 ** 2)
            / (495 * T1 * T2 * (3 * T1 + T2) * (T1 + 3 * T2) * e)
        )
