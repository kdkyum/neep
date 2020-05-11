import random
from itertools import product

import torch


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

    def __init__(self, M, L, batch_size, device="cpu", train=True):
        self.size = M * (L - 1)
        self.M = M
        self.L = L
        self.batch_size = batch_size
        self.device = device
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
            ens_idx = torch.randint(self.M, (self.batch_size,), device=self.device)
            traj_idx = torch.randint(
                0, (self.L - 1), (self.batch_size,), device=self.device
            )
            batch = (ens_idx, traj_idx)
            next_batch = (ens_idx, traj_idx + 1)
            return batch, next_batch
        else:
            prev_idx = self.index * self.batch_size
            next_idx = (self.index + 1) * self.batch_size
            if prev_idx >= self.size:
                raise StopIteration
            elif next_idx >= self.size:
                next_idx = self.size
            ens_idx = torch.arange(prev_idx, next_idx, device=self.device) // (
                self.L - 1
            )
            traj_idx = torch.arange(prev_idx, next_idx, device=self.device) % (
                self.L - 1
            )
            self.index += 1
            batch = (ens_idx, traj_idx)
            next_batch = (ens_idx, traj_idx + 1)
            return batch, next_batch


class CartesianSeqSampler(CartesianSampler):
    def __init__(self, M, L, n, batch_size, device="cpu", train=True):
        """Random subset with sequence length of n sampling from 
           {0, 1, ..., M-1} X {0, 1, ..., L-n} where X is Cartesian product.

        Attributes:
            M: number of trajectories.
            L: trajectory length.
            n: sequence length.
            batch_size: input batch size for training the RNEEP.
            train: if True randomly sample a subset else ordered sample. (default: True)

        Examples::

            >>> # 10 trajectories, trajectory length 100, sequence length 64, batch size 32 for training
            >>> sampler = CartesianSeqSampler(10, 100, 64, 32) 
            >>> batch = next(sampler)

            >>> # 5 trajectories, trajectory length 50, sequence length 32, batch size 64 for test
            >>> test_sampler = CartesianSeqSampler(5, 50, 32, 64, train=False) 
            >>> batch = next(test_sampler)
            >>> for batch in test_sampler:
            >>>     print(batch)
        """

        super(CartesianSeqSampler, self).__init__(
            M, L, batch_size, device=device, train=train
        )
        self.size = M * (L - n + 1)
        self.dropL = torch.arange(0, L, n - 1)[-1].item()
        self.test_size = self.dropL * M
        self.seq_length = n
        self.trj_idx = torch.ones(
            self.seq_length, self.batch_size, device=self.device, dtype=torch.long
        )
        self.trj_idx = torch.cumsum(self.trj_idx, 0) - 1

    def __next__(self):
        if self.training:
            ens_idx = torch.randint(self.M, (self.batch_size,), device=self.device)
            traj_idx = torch.randint(
                0,
                (self.L - self.seq_length + 1),
                (self.batch_size,),
                device=self.device,
            )
            traj_idx = self.trj_idx + traj_idx
            return (ens_idx, traj_idx)
        else:
            prev_idx = self.index * self.batch_size * (self.seq_length - 1)
            if prev_idx >= self.test_size:
                raise StopIteration

            next_idx = prev_idx + self.batch_size * (self.seq_length - 1)
            isLast = next_idx > self.test_size
            if isLast:
                batch_size = (self.test_size - prev_idx) // (self.seq_length - 1)
                next_idx = prev_idx + batch_size * (self.seq_length - 1)

            ens_idx = (
                torch.arange(
                    prev_idx, next_idx, self.seq_length - 1, device=self.device
                )
                // self.dropL
            )
            traj_idx = (
                torch.arange(
                    prev_idx, next_idx, self.seq_length - 1, device=self.device
                )
                % self.dropL
            )

            if isLast:
                traj_idx = self.trj_idx[:, : len(traj_idx)] + traj_idx
            else:
                traj_idx = self.trj_idx + traj_idx
            self.index += 1
            batch = (ens_idx, traj_idx)
            return batch
