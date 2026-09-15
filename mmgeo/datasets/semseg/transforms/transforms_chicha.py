# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmengine.registry import TRANSFORMS
import rasterio 
try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False 



@TRANSFORMS.register_module()
class LinearNormalization(BaseTransform): 
    def __init__(self, scale_factor: int = 255 ):
        self.scale_factor = scale_factor

    def transform(self, results: dict) -> dict:
        img = results['img']
        upper = np.max(img, axis=(0,1))
        lower = np.min(img, axis=(0,1))  
        img = (img - lower) / (upper-lower +   1e-7) 
        
        results['img'] = (img*self.scale_factor).astype(np.float32) # uint8  
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ 
        return repr_str



@TRANSFORMS.register_module()
class PercentileStretching(BaseTransform): 
    def __init__(self, scale_factor: int = 255 ):
        self.scale_factor = scale_factor

    def transform(self, results: dict) -> dict:
        img = results['img']
        upper = np.percentile(img, 99.9, axis=(0,1))
        lower = np.percentile(img, 0.1, axis=(0,1))  
        img = (img - lower) / (upper-lower +   1e-7) 
        
        #results['img'] = (img*self.scale_factor).astype(np.float32) # uint8  
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ 
        return repr_str


@TRANSFORMS.register_module()
class CustomRandomRotFlip(BaseTransform):
    """Rotate and flip the image & seg or just rotate the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - gt_seg_map

    Args:
        rotate_prob (float): The probability of rotate image.
        flip_prob (float): The probability of rotate&flip image.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
    """

    def __init__(self, probs=(0.5,0.5), degree=(-20, 20),
                 apply_prob: Union[int, float] = 0.5):
        self.rotate_prob = probs[0]
        self.flip_prob = probs[1]
        assert 0 <= self.rotate_prob <= 1 and 0 <= self.flip_prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.apply_prob = apply_prob 

    def random_rot_flip(self, results: dict) -> dict:
        k = np.random.randint(0, 4)
        results['img'] = np.rot90(results['img'], k)
        for key in results.get('seg_fields', []):
            results[key] = np.rot90(results[key], k)
        axis = np.random.randint(0, 2)
        results['img'] = np.flip(results['img'], axis=axis).copy()
        for key in results.get('seg_fields', []):
            results[key] = np.flip(results[key], axis=axis).copy()
        return results

    def random_rotate(self, results: dict) -> dict:
        angle = np.random.uniform(min(*self.degree), max(*self.degree))
        results['img'] = mmcv.imrotate(results['img'], angle=angle)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.imrotate(results[key], angle=angle)
        return results

    def transform(self, results: dict) -> dict:
        """Call function to rotate or rotate & flip image, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated or rotated & flipped results.
        """

        if random.random()  > self.apply_prob: 
            return results 
        
        rotate_flag = 0
        if random.random() < self.rotate_prob:
            results = self.random_rotate(results)
            rotate_flag = 1
        if random.random() < self.flip_prob and rotate_flag == 0:
            results = self.random_rot_flip(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_prob={self.rotate_prob}, ' \
                    f'flip_prob={self.flip_prob}, ' \
                    f'degree={self.degree})'
        return repr_str

@TRANSFORMS.register_module()
class CustomGenerateEdge(BaseTransform):
    """Generate Edge for CE2P approach.

    Edge will be used to calculate loss of
    `CE2P <https://arxiv.org/abs/1809.05996>`_.

    Modified from https://github.com/liutinglt/CE2P/blob/master/dataset/target_generation.py # noqa:E501

    Required Keys:

        - img_shape
        - gt_seg_map

    Added Keys:
        - gt_edge_map (np.ndarray, uint8): The edge annotation generated from the
            seg map by extracting border between different semantics.

    Args:
        edge_width (int): The width of edge. Default to 3.
        ignore_index (int): Index that will be ignored. Default to 255.
    """

    def __init__(self, edge_width: int = 3, ignore_index: int = 255,
                 switch:int=1,
                 apply_prob: Union[int, float]=0.5 ) -> None:
        super().__init__()
        self.edge_width = edge_width
        self.ignore_index = ignore_index
        self.switch = switch
        self.apply_prob = apply_prob

    def transform(self, results: Dict) -> Dict:
        """Call function to generate edge from segmentation map.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with edge mask.
        """
        if not self.switch: # if 0
            return results 
        if np.random.random() > self.apply_prob: 
            return results 
        
        h, w = results['img_shape']
        edge = np.zeros((h, w), dtype=np.uint8)
        seg_map = results['gt_seg_map']

        # down
        edge_down = edge[1:h, :]
        edge_down[(seg_map[1:h, :] != seg_map[:h - 1, :])
                  & (seg_map[1:h, :] != self.ignore_index) &
                  (seg_map[:h - 1, :] != self.ignore_index)] = 1
        # left
        edge_left = edge[:, :w - 1]
        edge_left[(seg_map[:, :w - 1] != seg_map[:, 1:w])
                  & (seg_map[:, :w - 1] != self.ignore_index) &
                  (seg_map[:, 1:w] != self.ignore_index)] = 1
        # up_left
        edge_upleft = edge[:h - 1, :w - 1]
        edge_upleft[(seg_map[:h - 1, :w - 1] != seg_map[1:h, 1:w])
                    & (seg_map[:h - 1, :w - 1] != self.ignore_index) &
                    (seg_map[1:h, 1:w] != self.ignore_index)] = 1
        # up_right
        edge_upright = edge[:h - 1, 1:w]
        edge_upright[(seg_map[:h - 1, 1:w] != seg_map[1:h, :w - 1])
                     & (seg_map[:h - 1, 1:w] != self.ignore_index) &
                     (seg_map[1:h, :w - 1] != self.ignore_index)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.edge_width, self.edge_width))
        edge = cv2.dilate(edge, kernel)

        results['gt_edge_map'] = edge
        results['edge_width'] = self.edge_width

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'edge_width={self.edge_width}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str

 

@TRANSFORMS.register_module()
class CustomCLAHE(BaseTransform):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, 
                 tile_grid_size=(8, 8),
                 switch: int=1 ,
                 apply_prob: Union[int,float] = 0.5):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size
        self.switch = switch 
        self.apply_prob = apply_prob 

    def transform(self, results: dict) -> dict:
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        if not self.switch: #if 0 
            return results
        if random.random()  <= self.apply_prob: 
            for i in range(results['img'].shape[2]):
                results['img'][:, :, i] = mmcv.clahe(
                    np.array(results['img'][:, :, i], dtype=np.uint8),
                    self.clip_limit, self.tile_grid_size)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, ' \
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str



@TRANSFORMS.register_module()
class CustomPhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: Sequence[float] = (0.5, 1.5),
                 saturation_range: Sequence[float] = (0.5, 1.5),
                 hue_delta: int = 18,
                 threshold: float = 0.5,
                 apply_prob: Union[int, float] = 0.5):
        self.computed = [] #CHICHA
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.threshold = threshold
        self.apply_prob = apply_prob 

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if random.random() < self.threshold:
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if random.random() < self.threshold:
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if random.random()  < self.threshold:
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if random.random()  < self.threshold:
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if random.random()  > self.apply_prob: 
            return results 
        sample_idx = results['sample_idx']
        if sample_idx in self.computed : #CHICHA
            img = results['img']
            # random brightness
            img = self.brightness(img)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                img = self.contrast(img)

            # random saturation
            img = self.saturation(img)

            # random hue
            img = self.hue(img)

            # random contrast
            if mode == 0:
                img = self.contrast(img)

            results['img'] = img
        self.computed.append(sample_idx)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class CustomRandomCrop(BaseTransform): 
    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255,
                 apply_prob: Union[int, float] = 0.5):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.apply_prob = apply_prob

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
 
        #chicha custom 
        if self.cat_max_ratio < 1.:
            for i in range(1e+7):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox) # crop 
                labels, label_cnt = np.unique(seg_temp, return_counts=True) # crop 
                total = np.sum(label_cnt)
                label_cnt = label_cnt[ labels == self.ignore_index]
                if isinstance(label_cnt, list):
                    label_cnt = label_cnt[0]  
                if label_cnt/total < self.cat_max_ratio:   
                    img = self.crop(img, crop_bbox)
                    _, img_cnt = np.unique(img, return_counts=True)
                    if isinstance(img_cnt, list):
                        img_cnt = img_cnt[0]  
                    if img_cnt / np.sum(img_cnt) < 0.3:
                        break  
                if i>=1e+7:
                    print("failed cropping......")
        return crop_bbox, img 

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img'] 
        #crop the image
        crop_bbox, img = self.crop_bbox(results)  # cropped img return...

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox) 
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
 