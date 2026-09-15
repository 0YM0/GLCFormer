# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from mmengine.registry import HOOKS 
from mmengine.version import __version__
from mmengine.hooks import Hook 
from mmengine.logging import print_log
import wandb 

@HOOKS.register_module()
class WandbSweepHook(Hook): #for parameter tuning 
    """Custom your hook for sweep parameters.
    Re-assign the sweep parameter into mmengine Config
    """

    priority = 'VERY_LOW'  

    def _reset(self, runner) -> None:
        #get config
        cfg = runner.cfg  
        train_dataloader=cfg.get('train_dataloader')
        val_dataloader=cfg.get('val_dataloader')
        test_dataloader=cfg.get('test_dataloader')
        train_cfg=cfg.get('train_cfg')
        val_cfg=cfg.get('val_cfg')
        test_cfg=cfg.get('test_cfg')
        auto_scale_lr=cfg.get('auto_scale_lr')
        optim_wrapper=cfg.get('optim_wrapper')
        param_scheduler=cfg.get('param_scheduler')
        val_evaluator=cfg.get('val_evaluator')
        test_evaluator=cfg.get('test_evaluator')   
        env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))) 
        randomness=cfg.get('randomness', dict(seed=None)) 

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper] 
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        runner._train_dataloader = train_dataloader
        runner._train_loop = train_cfg
 
        runner.optim_wrapper = optim_wrapper

        runner.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and runner.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. 
        runner._check_scheduler_cfg(param_scheduler)
        runner.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        runner._val_dataloader = val_dataloader
        runner._val_loop = val_cfg
        runner._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        runner._test_dataloader = test_dataloader
        runner._test_loop = test_cfg
        runner._test_evaluator = test_evaluator
 
        # environment.
        runner.setup_env(env_cfg) 
        runner._randomness_cfg = randomness
        runner.set_randomness(**randomness)   

    def _recursive_set(self, d, keys, new_v):
        keys = keys[1:] 
        if len(keys)<2:
            last_key = keys[-1] 
            d[last_key] = new_v 

            return   
        if isinstance(d, list):
            if 'arr_' in keys[0]:
                i = int(keys[0].split('arr_')[1])
                d[i] = new_v
                return 
            for in_d in d:
                type_value = keys[0].split("type_")[1]
                if in_d['type'] == type_value: 
                    keys = keys[1:]
                    if len(keys)<2:
                        in_d[keys[0]] = new_v 
                        return 
                    else: 
                        self._recursive_set(in_d[keys[0]], keys, new_v)
        else:
            self._recursive_set(d[keys[0]], keys, new_v)   

    def after_build_visualizer(self, runner) -> None : 
        """Initialize wandb

        Args:
            runner (Runner): The runner of the training process.
        """     
        print('HI!!!')
        self.wandb = runner.visualizer.get_backend('WandbVisBackend').experiment  
        if self.wandb.run is None:
            print("re-initialze@@")
            self.wandb = wandb.init()    
        elif runner.cfg.get('wandb_runname') is not None: 
            self.wandb.run.name = runner.cfg.get('wandb_runname') + "_" + self.wandb.run.name 
        self.wandb_config = self.wandb.config 
        
    
        for wandb_key, new_value in self.wandb_config.items():
            keys = wandb_key.split("-")
            if keys[0] != 'sweep':
                continue  
            self._recursive_set(runner.cfg, keys, new_value) 

        """
  
        for wandb_key in self.wandb_config.keys(): 
            if "sweep-" not in wandb_key:
                continue   
            keys = wandb_key.split("-")[1:]    
 
            if len(keys) == 2:
                if runner.cfg.get(keys[0]) is None:
                    runner.cfg.setdefault(keys[0], {keys[1]: self.wandb_config[wandb_key]}) 
                else:
                    if runner.cfg[keys[0]].get(keys[1]) is None:
                        runner.cfg[keys[0]].setdefault(keys[1], self.wandb_config[wandb_key])
                    else:
                        runner.cfg[keys[0]][keys[1]] = self.wandb_config[wandb_key]
            elif len(keys) == 3:  
                if runner.cfg.get(keys[0]) is None:
                    runner.cfg.setdefault(keys[0], {keys[1]: {keys[2]: self.wandb_config[wandb_key]}}) 
                else:
                    if runner.cfg[keys[0]].get(keys[1]) is None:
                        runner.cfg[keys[0]].setdefault(keys[1], {keys[2]: self.wandb_config[wandb_key]})
                    else:
                        if runner.cfg[keys[0]][keys[1]].get(keys[2]) is None:
                            runner.cfg[keys[0]][keys[1]].setdefault(keys[2], self.wandb_config[wandb_key])
                        else:
                            runner.cfg[keys[0]][keys[1]][keys[2]] = self.wandb_config[wandb_key]
            elif len(keys) == 4: 
                #only for param_schduler 
                assert 'array' in keys
                arr_idx = keys.index('array')   
                for idx, scheduler in enumerate(runner.cfg[keys[arr_idx-1]]):   
                    if scheduler.type == keys[arr_idx+1]:   
                        runner.cfg[keys[arr_idx-1]][idx][keys[arr_idx+2]] = self.wandb_config[wandb_key]
            elif len(keys) == 5:
                #only for param_scheduler
                assert 'array' in keys
                arr_idx = keys.index('array')   
                pass_idx = keys.index('pass') + 1
                for idx, scheduler in enumerate(runner.cfg[keys[arr_idx-1]]):   
                    if scheduler.type == keys[pass_idx]:
                        continue 
                    runner.cfg[keys[arr_idx-1]][idx][keys[arr_idx+1]] = self.wandb_config[wandb_key] 
                    param_scheduler = self.wandb_config[wandb_key] 
                    if param_scheduler != 'PolyLR':
                        for idx, scheduler in enumerate(runner.cfg.param_scheduler):   
                            try:
                                delattr(runner.cfg.param_scheduler[idx], 'eta_min')
                                delattr(runner.cfg.param_scheduler[idx], 'power')
                            except:
                                print('\033[93m pass--> wandb_hook.py -->\033[0m', scheduler) 

        """
        try:
            if runner.cfg.optim_wrapper.optimizer.type != 'SGD': 
                delattr(runner.cfg.optim_wrapper.optimizer, 'momentum')  
        except:
            pass  
  
        self._reset(runner)  