# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

import numpy as np
@HOOKS.register_module()
class PatchSegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False, 
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def _after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        total_curr_iter = runner.iter + batch_idx 
 
        if total_curr_iter % self.interval == 0:  
            sample_idx = 0  
            data_sample = data_batch['data_samples'][sample_idx] 
            img_path = data_sample.img_path
            window_name = f'train_{osp.basename(img_path)}'  
  
            #img: tensor.float32 / [0,255] 
            #img = data_batch['inputs'][sample_idx].detach().cpu().numpy().transpose(1,2,0).astype(np.uint8) 
            #img = np.ascontiguousarray(img) #메모리 연속성 보장 

            bbox = data_sample.read_bbox 
            if bbox is None or len(bbox) != 4:
                crop_size = runner.model.data_preprocessor.size
                bbox  =  (0, 0, crop_size[0], crop_size[1])
 
            self._visualizer.add_datasample(
                window_name,
                img_path,
                data_sample=data_sample,
                bbox = bbox, 
                show=self.show,
                wait_time=self.wait_time,
                draw_original = True , 
                step=total_curr_iter) 

    def after_val_slide(self, runner): 
        ... 


    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        #img_bytes = get(img_path, backend_args=self.backend_args)
        #img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}' 



        data_sample = outputs[0]
        bbox = data_sample.read_bbox

        if bbox is None: 
            crop_size = runner.model.data_preprocessor.size
            bbox  =  (0, 0, crop_size[0], crop_size[1]) 

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img_path,
                data_sample=outputs[0],
                bbox = bbox, 
                show=self.show,
                wait_time=self.wait_time,
                draw_original=True, 
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None: 
        
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        #img_bytes = get(img_path, backend_args=self.backend_args)
        #img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}' 



        data_sample = outputs[0]
        bbox = data_sample.read_bbox

        if bbox is None: 
            crop_size = runner.model.data_preprocessor.size
            bbox  =  (4000, 4000, crop_size[0], crop_size[1]) 

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img_path,
                data_sample=outputs[0],
                bbox = bbox, 
                show=self.show,
                wait_time=self.wait_time,
                draw_original=True, 
                step=total_curr_iter)

        #raise NotImplementedError()