import numpy as np
import random
from itertools import product


class CartesianSampler:
    """Random subset sampling from {0, 1, ..., M-1} X {0, 1, ..., L-2} where X is Cartesian product.

    Attributes:
        M: number of trajectories
        L: trajectory length
        batch_size: input batch size for training the NEEP

    Examples::

        >>> # 10 trajectories, trajectory length 100, batch size 32
        >>> sampler = CartesianSampler(10, 100, 32) 
        >>> batch, next_batch = next(sampler)
    """

    def __init__(self, M, L, batch_size):
        self.cartesian_set = [[i, t] for i, t in product(range(0, M), range(0, L - 1))]
        self.batch_size = batch_size

    def __next__(self):
        batch = random.choices(self.cartesian_set, k=self.batch_size)
        next_batch = list(map(lambda x: [x[0], x[1] + 1], batch))
        batch = np.transpose(batch)
        next_batch = np.transpose(next_batch)
        return batch, next_batch
