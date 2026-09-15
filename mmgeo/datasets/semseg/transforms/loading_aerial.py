# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path

from typing import Dict, Optional, Union, Sequence
try:
    from typing import Literal
except: #python 3.7
    from typing_extensions import Literal
import mmcv
import logging 
from mmengine.logging import print_log
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmengine.registry import TRANSFORMS
from mmseg.utils import datafrombytes
 
import os
import os.path as osp
import geopandas as gpd   
from rasterio.features import rasterize
try:
    import rasterio  
    from rasterio.windows import Window 
except ImportError:
    rasterio = None   
try:
    from scipy.ndimage import rotate as scipy_rotate
except:
    scipy_rotate = None 
import torch   
import math  

import cv2 
from mmengine.dist import get_rank 
from mmengine.runner import set_random_seed 

import numpy as np
import cv2
import os

@TRANSFORMS.register_module()
class CustomLoadAnnotations(MMCV_LoadAnnotations): 
    def __init__(
        self,
        binary_mode: bool = True, 
        multi_channel : bool = True, 
        reduce_zero_label=None,  
        backend_args=None, 
        ignore_index =255, 
        crop_size: Optional[Sequence] = None,  
        translate:int = 0, 
        patch_type: Optional[Literal[  'fixed',  'affine']] = None
    ) -> None: 
        #custom  
        self.binary_mode = binary_mode
        self.multi_channel = multi_channel   
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index= ignore_index 
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')    

        self.translate = translate 
        self.patch_type = patch_type
        if self.patch_type is None:
            self.sampling_func = self.image_loading 
        else:
            self.crop_h, self.crop_w = crop_size 
            if self.patch_type == 'center':
                self.sampling_func = self.center_sampling
            elif self.patch_type == 'fixed':
                self.sampling_func = self.fixed_sampling
            elif self.patch_type == 'random':
                self.sampling_func = self.random_sampling
            elif self.patch_type == 'affine':
                if scipy_rotate is None:
                    raise NotImplementedError("scipy가 설치되어야 함 ")
                self.sampling_func = self.affine_sampling
                self.diag = int(np.ceil(np.sqrt(crop_size[0]**2 + crop_size[1]**2 )))
        """
        if self.patch_type is None:
            self.sampling_func = self.image_loading 
        else: 
            if self.patch_type == 'fixed':
                self.sampling_func = self.fixed_sampling 
            else: 
                self.sampling_func = self.affine_sampling
                """
        if rasterio is None:
            raise NotImplementedError("rasterio가 설치되어야 함") 

    def _load_data(self, filename, window ): 
        with rasterio.open(filename) as src: 
            img = src.read( window=window, 
                            boundless=True, 
                            fill_value = self.ignore_index, 
                            )   
        return img.squeeze() 

    def _get_filename(self, results):
        return results['seg_map_path']

    def fixed_sampling(self, results):  
        """
        bbox = (xmin, ymin, xmax, ymax)이 주어지면 
        그대로 바운딩박스대로 샘플링  
        """
        assert results['query'] !=  None 
        filename = self._get_filename(results)  
        window = Window(*results['query'] ) 
        img = self._load_data(filename, window)
        return img 
    

    def random_sampling(self, results):  
        """
        원래 주어진 bbox에서 이동을 적용하여 샘플링
        """
        filename = results['img_path']     
        with rasterio.open(filename) as src:
            width, height = src.width, src.height
            xmin = np.random.randint(0,width-self.crop_w) 
            ymin = np.random.randint(0,height - self.crop_h)
            window = Window(xmin,ymin, self.crop_w, self.crop_h) 
        img = self._load_data(filename, window ) 
        results['read_bbox'] = (xmin, ymin, self.crop_w, self.crop_h) 
        return img 

    def affine_sampling(self, results): 
        """
        원래 주어진 bbox에서 회전 및 이동을 적용하여 샘플링
        """
        assert results['query'] !=  None 
        filename = self._get_filename(results)
        minx, miny, bbox_w, bbox_h = results['query']  #query = (xmin, ymin, xmax, ymax)

        minx += torch.randint(-self.translate, self.translate, (1,)).item()  
        miny += torch.randint(-self.translate, self.translate, (1,)).item()

        # 원본 이미지 크기
        with rasterio.open(filename) as src:
            w, h = src.width, src.height

        minx = np.clip(minx, 0, w - bbox_w)
        miny = np.clip(miny, 0, h - bbox_h)
        maxx = minx + bbox_w
        maxy = miny + bbox_h
 
        minx = max(0, math.floor((minx+maxx)/2 - self.diag/2 )) 
        miny = max(0, math.floor((miny+maxy)/2 - self.diag/2 ))  

        window = Window(minx, miny, self.diag, self.diag) 
        
        patch_big = self._load_data(filename, window, )  #rotate 함수 사용 전 HWC로 만들어야 힘  
        theta_deg = torch.randint(0, 90, (1,)).item() #torch에서만 매 에포크 시드가 다름 
        #theta_deg *= -1
 

        patch_rot = scipy_rotate(
            patch_big,
            angle=theta_deg,
            reshape=False,
            order=0,
            mode='constant',
            cval=0 # self.ignore_index
        )

        h, w = patch_rot.shape[:2] 
        start_y = (h - self.crop_h) // 2
        start_x = (w - self.crop_w) // 2
        patch_cropped = patch_rot[start_y:start_y + self.crop_h, 
                            start_x:start_x + self.crop_w]   

        results['read_bbox'] = (minx, miny, self.diag, self.diag)
        results['rotate_deg'] = theta_deg 
        results['crop_size'] = (self.crop_h, self.crop_w)         
        return patch_cropped 

    def image_loading(self, results):  
        filename = self._get_filename(results)    
        return self._load_data(filename, window=None) 
 
    def _load_seg_map(self, results: dict) -> None:     
        results['read_bbox'] = results['query']  #image 읽어오면서 query가 바뀔 수 있으므로 변경된 bbox는 read_bbox에 담음 
        results['rotate_deg'] = None 
        results['crop_size'] = None 

        gt_semantic_seg = self.sampling_func(results)    
        #print("gt_semantic_seg.shape : ",gt_semantic_seg.shape)
        #logging.... 
        #from mmengine.logging import print_log
        #import logging 
        msg = f""">>> custom/datasets/transforms/loading_aerial>>> 
        seg map shape: {gt_semantic_seg.shape}
        >>> >>> 
        """
        #print_log(msg, logger = 'current', level=logging.DEBUG)
         
        if gt_semantic_seg.shape[0] == 0 :
            raise ValueError("gt 샘플링이 이상함ㅠ.ㅠ")
       
        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None: 
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
    
    def _draw_for_debug(self, results:dict) -> None: 
        #debugging 
        img = results['img'] 
        gt_sem_seg = results['gt_seg_map'] 
        idkey = np.random.randint(0,100)
        img_path = os.path.basename(results['img_path']) 
        filename = f'/data/yerang/projects/mmsegmentation/project/aerial_project/work_dirs/exp1_250520/debug_vis/{img_path}_{idkey}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True )
 
        drawn_img = np.repeat(gt_sem_seg[:,:, np.newaxis], 3, axis=2) #(512,512,3)
        drawn_img *= 255  
        print(gt_sem_seg.shape)
        print('max-->', np.max(gt_sem_seg, axis=(0,1)))
        print(np.min(gt_sem_seg, axis=(0,1)))
        drawn_img = np.concatenate([img, drawn_img], axis=1)
        cv2.imwrite(filename, drawn_img)
        print(f"{filename}에 저장됨")
        raise NotImplementedError 

    def is_valid(self, results:dict) -> bool:
        has_non_ignore = np.any(results['gt_seg_map'] != self.ignore_index)
        return has_non_ignore

    def transform(self, results: dict) -> dict:    
        if self.patch_type in ['affine', 'random']:
            max_trials = 100
            valid = False 
            for attempt in range(max_trials):   
                self._load_seg_map(results)    
                if self.is_valid(results):
                    valid = True 
                    break 
        else:
            self._load_seg_map(results)    
            valid = self.is_valid(results)
        if not valid:
            gt_seg_map = results['gt_seg_map']
            print_log(f"[SKIP] Only ignore_index in gt_seg_map: {np.unique(gt_seg_map)}", logger='current', level=logging.WARNING) ## 추가
            """
            print_log(f"{np.unique(gt_seg_map)}", logger = 'current', level=logging.WARNING)
            print_log(f"{results['seg_map_path']}", logger = 'current', level=logging.WARNING)
            print_log(f"{results['read_bbox']}", logger = 'current', level=logging.WARNING)
            """
            return None
        return results 

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '  
        return repr_str

@TRANSFORMS.register_module()
class CustomLoadHeatmaps(MMCV_LoadAnnotations): 
    def __init__(
        self,
        binary_mode: bool = True, 
        multi_channel : bool = True, 
        reduce_zero_label=None,  
        backend_args=None, 
        patch_type: Optional[Literal[  'fixed',  'affine']] = None
    ) -> None: 
        #custom  
        self.binary_mode = binary_mode
        self.multi_channel = multi_channel   
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized') 

 
        self.patch_type = patch_type
        if self.patch_type is None:
            self.sampling_func = self.image_loading 
        else: 
            if self.patch_type == 'fixed':
                self.sampling_func = self.fixed_sampling 
            else: 
                self.sampling_func = self.affine_sampling
        if rasterio is None:
            raise NotImplementedError("rasterio가 설치되어야 함") 

    def _load_data(self, filename, window):
        with rasterio.open(filename) as src: 
            img = src.read(1, window=window) 
        return img 

    def _get_filename(self, results):        
        filename = results['seg_map_path'].replace("masks/", "heatmaps/").replace("_binary.tif", "_heatmap_sig15.tif")
        return filename

    def fixed_sampling(self, results):  
        """
        bbox = (xmin, ymin, xmax, ymax)이 주어지면 
        그대로 바운딩박스대로 샘플링  
        """
        assert results['read_bbox'] !=  None 
        filename = self._get_filename(results)  
        window = Window(*results['read_bbox'] )         
        return self._load_data(filename, window) 

    def affine_sampling(self, results): 
        """
        원래 주어진 bbox에서 회전 및 이동을 적용하여 샘플링
        """ 
        assert results['read_bbox'] !=  None 
        filename = self._get_filename(results)         
        window = Window(*results['read_bbox'])   
        patch_big = self._load_data(filename, window)
        #torch.randint(0, 360, (1,)).item() #torch에서만 매 에포크 시드가 다름 
        raise NotImplementedError
        patch_rot = scipy_rotate(
            patch_big,
            angle=results['rotate_deg']  ,
            reshape=False,
            order=1,
            mode='constant',
            cval=-1 
        )

        h, w = patch_rot.shape[:2] 
        crop_h, crop_w = results['crop_size'] 
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        patch_cropped = patch_rot[start_y:start_y + crop_h, 
                            start_x:start_x + crop_w]    
        return patch_cropped  
    


    def image_loading(self, results):  
        return self._load_data(self._get_filename(results))
 
    def _load_heat_map(self, results: dict) -> None:    
        gt_heat_map = self.sampling_func(results)   

        #print(gt_heat_map.shape, np.max(gt_heat_map))
        #from mmengine.logging import print_log
        #import logging 
        msg = f""">>> custom/datasets/transforms/loading_aerial>>> 
        gt_heat_map map shape: {gt_heat_map.shape}
        >>> >>> 
        """
        #print_log(msg, logger = 'current', level=logging.DEBUG)
           
        results['gt_heat_map'] = gt_heat_map
        results['seg_fields'].append('gt_heat_map') #걍 뎁스로 기록... 
    
    def _draw_for_debug(self, results:dict) -> None: 
        #debugging 
        img = results['img'] 
        gt_sem_seg = results['gt_heat_map'] 
        idkey = np.random.randint(0,100)
        img_path = os.path.basename(results['img_path']) 
        filename = f'/data/yerang/projects/mmsegmentation/project/aerial_project/work_dirs/exp1_250520/debug_vis/{img_path}_{idkey}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True )
 
        drawn_img = np.repeat(gt_sem_seg[:,:, np.newaxis], 3, axis=2) #(512,512,3)
        drawn_img *= 255  
        print(gt_sem_seg.shape)
        print('max-->', np.max(gt_sem_seg, axis=(0,1)))
        print(np.min(gt_sem_seg, axis=(0,1)))
        drawn_img = np.concatenate([img, drawn_img], axis=1)
        cv2.imwrite(filename, drawn_img)
        print(f"{filename}에 저장됨")
        raise NotImplementedError 


    def transform(self, results: dict) -> dict:   
        self._load_heat_map(results)    
        return results 

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '  
        return repr_str

@TRANSFORMS.register_module()
class CustomLoadImageFromFile(BaseTransform): 
    def __init__(self, 
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 bands: Optional[Sequence] = None, 
                 backend_args: Optional[dict] = None,
                 patch_type: Optional[Literal['center', 
                                    'fixed',
                                    'random',
                                    'affine']] = None,
                ignore_index : int =255, 
                 ) -> None: 
        """
        패치 샘플링 방식들은 results에 bbox 키값이 있어야 함 
        """
        self.to_float32 = to_float32
        self.ignore_index= ignore_index 
        self.backend_args = backend_args.copy() if backend_args else None   
        self.bands = bands 
        self.patch_type = patch_type
        if self.patch_type is None:
            self.sampling_func = self.image_loading 
        else: 
            if self.patch_type == 'fixed':
                self.sampling_func = self.fixed_sampling 
            else: 
                self.sampling_func = self.affine_sampling
        if rasterio is None:
            raise NotImplementedError("rasterio가 설치되어야 함")


    def _load_data(self, filename, window, bands=None ):
        with rasterio.open(filename) as src: 
            if bands is None:
                bands = [x for x in range(1, src.count+1)]
            img = src.read(bands,
                            window=window, 
                            boundless=True, 
                            fill_value = self.ignore_index 
                            ) 
        return np.ascontiguousarray(img.transpose(1,2,0)) 
    

    def _get_filename(self, results):
        return results['img_path']


    def center_sampling(self, results):  
        """
        bbox = (l,b,r,t)이 주어지면 해당 바운드의 중심을 기준으로
        crop_size만큼 패치를 샘플링
        """
        raise NotImplementedError
        filename = results['img_path']     
        query = results['query']  
        xmin = (query[2]-query[0])/2 - self.crop_w/2 
        ymin = (query[3]-query[1])/2 - self.crop_h/2  
        xmin, ymin = max(0, xmin), max(0, ymin) 
        window = Window(xmin,ymin, crop_w, crop_h)
        results['read_bbox'] = (xmin, ymin, crop_w, crop_h) 

        img = self._load_data(filename, window, bands=self.bands ) 
        return np.ascontiguousarray(img.transpose(1,2,0)) 

    def fixed_sampling(self, results):  
        """
        bbox = (xmin, ymin, xmax, ymax)이 주어지면 
        그대로 바운딩박스대로 샘플링  
        """
        filename = self._get_filename(results)     
        window = Window(*results['read_bbox'] ) 

        img = self._load_data(filename, window, bands=self.bands ) 
        return img

    def random_sampling(self, results):  
        """
        원래 주어진 bbox에서 이동을 적용하여 샘플링
        """
        filename = results['img_path']     
        with rasterio.open(filename) as src:
            width, height = src.width, src.height
            xmin = np.random.randint(0,width-self.crop_w) 
            ymin = np.random.randint(0,height - self.crop_h)
            window = Window(xmin,ymin, self.crop_w, self.crop_h) 
            
        img = self._load_data(filename, window, bands=self.bands ) 
        results['read_bbox'] = (xmin, ymin, self.crop_w, self.crop_h) 
        return img


    def affine_sampling(self, results): 
        """
        원래 주어진 bbox에서 회전 및 이동을 적용하여 샘플링
        """ 
        filename = self._get_filename(results)     

        window = Window(*results['read_bbox']) 
        patch_big = self._load_data(filename, window,  bands=self.bands )  
        #torch.randint(0, 360, (1,)).item() #torch에서만 매 에포크 시드가 다름 
         
        patch_rot = scipy_rotate(
            patch_big,
            angle=results['rotate_deg']  ,
            reshape=False,
            order=1,         # 연속 보간 
            mode='constant',
            cval=self.ignore_index
        )

        h, w = patch_rot.shape[:2] 
        crop_h, crop_w = results['crop_size'] 
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        patch_cropped = patch_rot[start_y:start_y + crop_h, 
                            start_x:start_x + crop_w]    
        return patch_cropped  
    

    """ # 기존 코드
    def image_loading(self, results):  
        filename = results['img_path']     
        with rasterio.open(filename) as src:  
            if self.bands is not None:
                img =  src.read(self.bands) 
            else: 
                img =  src.read()  
        return img 
    """

    def image_loading(self, results): # 수정 코드
        """전체 도엽을 그대로 읽을 때도 (H, W, C) 로 맞춰 반환."""
        filename = results['img_path']
        with rasterio.open(filename) as src:
            img = src.read(self.bands) if self.bands is not None else src.read()  # (C,H,W)
            #print(src.width, src.height, src.count, src.bounds)
        img = img.transpose(1, 2, 0)
        #print("img.shape :", img.shape)
        return img 
    
    def debug_save_image(self, img, save_path):
        """이미지 디버그 저장 함수 (dtype/범위 자동 처리)."""
        arr = img
        if np.issubdtype(arr.dtype, np.floating):
            vmin, vmax = arr.min(), arr.max()
            if vmax <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype == np.uint16:
            arr = (arr / 256).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, arr)
        print(f"[DEBUG SAVE] {save_path}, shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

    

    def transform(self, results: Dict) -> Dict:  
        """
        미리 정의한 query 버전 
        """  
        transposed_img = self.sampling_func(results)  # fixed_sampling - (512,512,3) / image_loading - (3, 3999, 3301) -> (3999, 3301, 3)

        if self.to_float32:
            transposed_img = transposed_img.astype(np.float32) 

        #print(f"img shape in LOADING: {transposed_img.shape}")
        #print(f"[CustomLoadImageFromFile] img shape: {transposed_img.shape}, path: {results['img_path']}")
        results['img'] = np.copy(transposed_img)
        #results['img'] = transposed_img
        results['img_shape'] = transposed_img.shape[:2]
        results['ori_shape'] = transposed_img.shape[:2] 

        if results['img'].shape[:2] != results['gt_seg_map'].shape[:2]:
            raise ValueError(f"Image/Mask shape mismatch: {results['img'].shape} vs {results['gt_seg_map'].shape}")

        # 파일명 식별자 생성
        """
        base = os.path.basename(results['img_path']).replace('.tif', '')
        self.debug_save_image(transposed_img, f"work_dirs/loading_debug/debug_after_load_{base}.jpg")
        self.debug_save_image(results['img'], f"work_dirs/loading_debug/debug_before_return_{base}.jpg")
        self.debug_save_image(results['gt_seg_map'] * 255, f"work_dirs/loading_debug/debug_mask_{base}.jpg")
        #raise RuntimeError
        """
        return results


 
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', " 
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str
