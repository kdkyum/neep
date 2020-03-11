import random
from itertools import product

import numpy as np


class CartesianSampler:
    """Random subset sampling from {0, 1, ..., M-1} X {0, 1, ..., L-2} where X is Cartesian product.

    Attributes:
        M: number of trajectories.
        L: trajectory length.
        batch_size: input batch size for training the NEEP.
        train: if True randomly sample a subset else ordered sample. (default: True)

    Examples::

        >>> # 10 trajectories, trajectory length 100, batch size 32 for training
        >>> sampler = CartesianSampler(10, 100, 32) 
        >>> batch, next_batch = next(sampler)

        >>> # 5 trajectories, trajectory length 50, batch size 32 for test
        >>> test_sampler = CartesianSampler(5, 50, 32, train=False) 
        >>> batch, next_batch = next(test_sampler)
        >>> for batch, next_batch in test_sampler:
        >>>     print(batch, next_batch)
    """

    def __init__(self, M, L, batch_size, train=True):
        self.cartesian_set = [[i, t] for i, t in product(range(0, M), range(0, L - 1))]
        self.size = len(self.cartesian_set)
        self.batch_size = batch_size
        self.training = train
        self.index = 0

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.training:
            batch = random.choices(self.cartesian_set, k=self.batch_size)
            next_batch = list(map(lambda x: [x[0], x[1] + 1], batch))
            batch = np.transpose(batch)
            next_batch = np.transpose(next_batch)
            return (batch[0], batch[1]), (next_batch[0], next_batch[1])
        else:
            prev_idx = self.index * self.batch_size
            next_idx = (self.index + 1) * self.batch_size
            if prev_idx >= self.size:
                raise StopIteration
            batch = self.cartesian_set[prev_idx:next_idx]
            next_batch = list(map(lambda x: [x[0], x[1] + 1], batch))
            self.index += 1
            batch = np.transpose(batch)
            next_batch = np.transpose(next_batch)
            return (batch[0], batch[1]), (next_batch[0], next_batch[1])
