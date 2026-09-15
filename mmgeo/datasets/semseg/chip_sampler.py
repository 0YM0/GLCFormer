# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class ChipSampler(Sampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True,
                 num_chips_per_tile=1,
                 ) -> None:
        """
        num_chips_per_tile: repeat the number of this value. [1,2] -> [1,1,2,2]
        """
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size 

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        self.num_chips_per_tile = num_chips_per_tile

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) * self.num_chips_per_tile / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) * self.num_chips_per_tile - rank) / world_size)
            self.total_size = len(self.dataset) * self.num_chips_per_tile
         

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)  
            #indices = torch.concat([indices, indices-len(indices)])
            #indices = indices.tolist() 
            indices = indices.repeat_interleave(self.num_chips_per_tile).tolist() #CHICHA 
        else:
            indices = torch.arange(len(self.dataset))
            indices = indices.repeat_interleave(self.num_chips_per_tile).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch