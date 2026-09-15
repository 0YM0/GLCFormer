# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmengine.registry import DATASETS
import os.path as osp
from typing import List

import mmengine.fileio as fileio

@DATASETS.register_module()
class KoreabarnDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'barn'),
        palette=[[64,93,114], [255,248,243], ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_bin_labelIds.tif', 
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
 