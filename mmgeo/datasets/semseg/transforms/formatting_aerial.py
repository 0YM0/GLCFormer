# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmengine.registry import TRANSFORMS
from mmseg.structures import SegDataSample


import logging 
from mmengine.logging import print_log
import torch

@TRANSFORMS.register_module()
class CustomPackSegInputs(BaseTransform): 
    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data))) 

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        if 'gt_heat_map' in results:
            gt_heat_map = dict(
                data=to_tensor(results['gt_heat_map'][None, ...]))
            data_sample.set_data(dict(gt_heat_map=PixelData(**gt_heat_map)))

        #print(data_sample.gt_sem_seg.data.shape)
        #print(">>>SEM")
        #print(data_sample.gt_heat_map.data.shape)
        #print(">>>HEAT")
        #raise ValueError() 

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        if 'landcover_graph' in results:
            img_meta['landcover_graph'] = results['landcover_graph']
        if 'graph_path' in results:
            img_meta['graph_path'] = results['graph_path']

        
        data_sample.set_metainfo(img_meta) 
        packed_results['data_samples'] = data_sample
 

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackSegInputsWithGraph(CustomPackSegInputs):
    """Pack segmentation inputs and keep land-cover graph metadata."""

    pass
