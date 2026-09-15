# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmengine.registry import DATASETS
import os.path as osp
import copy 
from typing import List, Optional, Union, Callable, Sequence
import mmengine
from mmengine.dataset import BaseDataset, Compose
import mmengine.fileio as fileio
from mmengine.logging import print_log
import logging 
import pandas as pd 
import os 

@DATASETS.register_module()
class KoreabarnListDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'barn'),
        palette=[[64,93,114], [255,248,243], ]
    )
    """


    """
    def __init__(self,
                data_root = 'data/', 
                data_list_path = '.csv', 
                img_dir = 'images/',
                mask_dir ='masks/',
                label_dir = 'labels/',
                use_png = True, 
                 img_suffix='.tif', 
                 seg_map_suffix='.tif', 
                 shuffle: bool = True, 
                 reduce_zero_label=False,
                 filter_cfg: Optional[dict] = None,
                 metainfo: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 backend_args: Optional[dict] = None,
                 subset: Optional[float] = None, 
                 debug = False, 
                 graph_dir: Optional[str] = None,
                 graph_path_col: Optional[str] = None,
                 graph_required: bool = False,
                 ) -> None:

        #custom 
        self.data_list_path = osp.join(data_root, data_list_path)  
        self.img_dir = osp.join(data_root, img_dir ) 
        self.mask_dir = osp.join(data_root, mask_dir) 
        self.label_dir = osp.join(data_root, label_dir)   
        self.use_png = use_png
        self.debug = debug
        self.graph_path_col = graph_path_col
        self.graph_required = graph_required
        if graph_dir is None:
            self.graph_dir = None
        elif osp.isabs(graph_dir):
            self.graph_dir = graph_dir
        else:
            self.graph_dir = osp.join(data_root, graph_dir)

        self.subset = subset
        #list shuffle 
        self.shuffle = shuffle 
        #basesegdataset
        self.img_suffix = img_suffix 
        self.seg_map_suffix = seg_map_suffix
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self._indices = indices 
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))
        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    def load_data_list(self) -> List[dict]:  
        data_df = pd.read_csv(self.data_list_path)
        if self.use_png:
            image_key = 'png_fn' 
        else:
            image_key = 'image_fn' 
        seg_map_key = 'mask_fn'
        label_key = 'label_fn'
        data_list = [] 

        if self.debug:
            data_df = data_df.iloc[:2] # 

        for idx, row in data_df.iterrows(): 
            if 'xmin' in row.keys():
                # bbox = (left, bottom, right, top)
                query = (row['xmin'], row['ymin'],
                        row['chip_size'], row['chip_size']
                ) 
            else:
                query = None

            graph_path = None
            if self.graph_path_col is not None and self.graph_path_col in row.keys():
                value = row[self.graph_path_col]
                if pd.notna(value):
                    graph_path = str(value)
                    if not osp.isabs(graph_path) and self.graph_dir is not None:
                        graph_path = osp.join(self.graph_dir, graph_path)
            elif self.graph_dir is not None and {'xmin', 'ymin', 'chip_size'}.issubset(set(row.keys())):
                image_base = osp.splitext(osp.basename(str(row[image_key])))[0]
                x = int(float(row['xmin']))
                y = int(float(row['ymin']))
                chip_size = int(float(row['chip_size']))
                graph_path = osp.join(self.graph_dir, f'{image_base}_x{x}_y{y}_s{chip_size}.pt')
 
            data_info = dict(
                img_path = osp.join(self.img_dir, row[image_key].replace(".tif", self.img_suffix)), 
                seg_map_path = osp.join(self.mask_dir, row[seg_map_key].replace(".tif", self.seg_map_suffix)),
                label_path = osp.join(self.label_dir, row[label_key]),
                label_map = self.label_map, 
                reduce_zero_label= self.reduce_zero_label, 
                seg_fields = [],  
                query = query,
                graph_path = graph_path,
                graph_dir = self.graph_dir,
                graph_required = self.graph_required,
            )  

            data_list.append(data_info) 
        data_list = sorted(data_list, key=lambda x: x['img_path']) 
        return data_list 
    

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
