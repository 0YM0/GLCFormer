import math
import torch
from typing import Optional, Iterator, Sized
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class DistributedSubsetSampler(Sampler):
    """
    DDP 환경에서 전체 데이터셋 중 일부 비율만 사용하는 Sampler

    Args:
        dataset (Sized): 전체 데이터셋
        subset_ratio (float): 사용할 subset 비율 (예: 0.1은 10%)
        shuffle (bool): 셔플 여부
        seed (int, optional): 랜덤 시드
        round_up (bool): world_size로 나누어 떨어지도록 extra 샘플 추가 여부
    """

    def __init__(
        self,
        dataset: Sized,
        subset_ratio: float = 0.1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True
    ):
        assert 0 < subset_ratio <= 1.0, "subset_ratio는 0과 1 사이여야 합니다."
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.subset_ratio = subset_ratio
        self.shuffle = shuffle
        self.round_up = round_up

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        # 전체 길이 기준으로 서브셋 인덱스 수 설정
        self.subset_len = int(self.dataset_len * self.subset_ratio)

        if self.round_up:
            self.num_samples = math.ceil(self.subset_len / world_size)
            self.total_size = self.num_samples * world_size
        else:
            self.num_samples = math.ceil((self.subset_len - rank) / world_size)
            self.total_size = self.subset_len

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 전체 인덱스 중 subset만 샘플링
        all_indices = torch.randperm(self.dataset_len, generator=g).tolist()
        indices = all_indices[:self.subset_len]

        # 셔플 적용
        if self.shuffle:
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        # round_up일 경우 padding 추가
        if self.round_up:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size // len(indices)) + 1))[:padding_size]

        # 각 rank별로 나눠줌
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
