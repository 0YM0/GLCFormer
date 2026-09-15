from mmengine.hooks import Hook
import numpy as np

class RandomSeedHook(Hook):
    def before_train_epoch(self, runner):
        seed = runner.epoch + runner.seed  # 예: epoch마다 다른 seed
        np.random.seed(seed)
