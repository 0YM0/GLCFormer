# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence 

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer

from mmengine.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.utils import get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer
import rasterio

try:
    from scipy.ndimage import rotate as scipy_rotate
except:
    scipy_rotate = None 
@VISUALIZERS.register_module()
class PatchSegLocalVisualizer(SegLocalVisualizer): 
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs) 


    def _rotate(self, patch_big, theta_deg, crop_size):  
        patch_rot = scipy_rotate(
            patch_big,
            angle=theta_deg,
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        ) 

        h, w = patch_rot.shape[:2] 
        crop_h, crop_w = crop_size
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        patch_cropped = patch_rot[start_y:start_y + crop_h, 
                            start_x:start_x + crop_w]     
        return patch_cropped

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      bbox: Sequence, 
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = True) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Default to True.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data 
        sem_seg = sem_seg[:, bbox[1]: bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        sem_seg = np.ascontiguousarray(sem_seg)

        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        if with_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # (0,1] to change the size of the text relative to the image
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            if image.shape[0] < 300 or image.shape[1] < 300:
                thickness = 1
                rectangleThickness = 1
            else:
                thickness = 2
                rectangleThickness = 2
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            #masks = masks.astype(np.uint8)
            mask = np.ascontiguousarray(mask, dtype=np.uint8)

            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
                text = classes[classes_id]
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness)
                
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     classes_color, -1)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     (0, 0, 0), rectangleThickness)
                mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                   font, fontScale, fontColor, thickness,
                                   lineType)
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg
    
    @master_only
    def add_datasample(
            self,
            name: str,
            img_path: str,
            data_sample: Optional[SegDataSample] = None,
            bbox: Sequence = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True,
            draw_original: bool = False ) -> None: 
 
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None) 

        gt_img_data = None
        pred_img_data = None  
        with rasterio.open(img_path) as src:  
            if bbox is None:
                raise NotImplementedError("PatchSegLocalVisualizer에서는 아직 bbox 없는 버전 구현안함")
            assert isinstance(bbox, Sequence) and len(bbox) == 4, "bbox must be (xmin, ymin, crop_size, crop_size for visualization in PatchSegVisualizer"

            minx, miny, bbox_w, bbox_h = bbox 
            pad = (src.width - bbox_w, src.height- bbox_h)   
            if minx > pad[0] : 
                minx = pad[0]
            if miny > pad[1]: 
                miny = pad[1]  

            bbox = (int(np.floor(minx)), int(np.floor(miny)), 
                    int(np.ceil(bbox_w)), int(np.ceil(bbox_h)))
            image = src.read(window=rasterio.windows.Window(*bbox), 
                                            boundless=True, fill_value=0) 
        image = image.transpose(1,2,0) 

        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.' 
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 bbox ,
                                                 classes, palette, with_labels) 

        if draw_pred and data_sample is not None:
            if 'pred_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.' 
                pred_img_data = self._draw_sem_seg(image,
                                                   data_sample.pred_sem_seg,
                                                   bbox, 
                                                   classes, palette,
                                                   with_labels)
 
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if draw_original:
            drawn_img = np.concatenate((image, drawn_img), axis=1)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)