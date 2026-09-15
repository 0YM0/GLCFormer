# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Sequence, Literal
 
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

# 참고 링크 : https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/loading.py

@TRANSFORMS.register_module()
class LoadRSAnnotations(BaseTransform):
    """ 
    """ 
    def __init__(
        self,
        with_bbox: bool = False,
        with_label: bool = False,
        with_seg: bool = True,
        with_keypoints: bool = False,
        binary_mode : bool = False ,  
        reduce_zero_label:bool=None, 
        reduct_dim :bool = False ,
        stretch_ch: bool = False ,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints 

        self.reduce_zero_label = reduce_zero_label
        self.binary_mode = binary_mode
        self.reduct_dim = reduct_dim 
        self.stretch_ch = stretch_ch
 
    def _load_bboxes(self, results: dict) -> None: 
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None: 
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64) 

    def _load_kps(self, results: dict) -> None: 
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict: 
        if self.with_bbox: #여기서는 사용하지 않음 
            self._load_bboxes(results)
        if self.with_label: #여기서는 사용하지 않음 
            self._load_labels(results)
        if self.with_seg: 
            self._load_seg_map(results)
        if self.with_keypoints: #여기서는 사용하지 않음 
            self._load_kps(results)
        return results  
    def _load_seg_map(self, results: dict) -> None: 
        with rasterio.open(results['seg_map_path']) as f: # 파일 열기
            gt_semantic_seg = f.read().squeeze().astype(np.uint8) 
        
             
        if self.binary_mode:# binary_mode면 실행
            gt_semantic_seg = gt_semantic_seg.squeeze() 
            mask = np.where(gt_semantic_seg>0) # 0보다 큰 픽셀의 위치 찾음
            gt_semantic_seg[mask] = 1 # 찾은 픽셀의 값을 모두 1로 변환 

        #gt_semantic_seg = gt_semantic_seg.astype(np.float32)
        # 현재 사용 X
        # reduce zero_label
        # 0번 라벨을 모델이 예측할 때 고려하지 않게 만듬
        # 라벨 0,1,..,5 존재, reduce zero_label == True
        # 0번 -> 255번 변환, 나머지 1~5번 -> 0~4번 변환한 뒤 모델의 입력으로 들어감
        # 모델이 실제로는 1~5번의 선택지 중에서만 예측
        if self.reduce_zero_label is None: 
            self.reduce_zero_label = results['reduce_zero_label'] # 주로 라벨 0을 배경으로 사용, 0을 다른 값으로 변경할 지 여부 제어
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255 # 0인 값을 255로 변경
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        # 현재 사용 X
        # 'label_map' = 라벨 0번을 1번으로 바꾸는 매핑(예시)    
        # 'seg_fields' = 다른 세그멘테이션의 경우에 뎁스맵 같은 다른 맵도 들어가서 구분 짓기 위해서 입력하는 정보
        # modify if custom classes
        if results.get('label_map', None) is not None: # result가 label_map 포함 시, 기존 라벨을 새로운 라벨로 매핑
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')   

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadRSImage(BaseTransform): 
    def __init__(self, 
            train_ratio: Optional[float] = None,
            test_ratio: Optional[float] = None, 
            patch_size: Union[tuple, int, list] = 1024, 
            bands: Optional[Sequence] = None, 
            to_float32: bool = True,
            to_8bit: bool = False,
            test_mode: bool = False,  
            stretch_ch: bool = False ,
            ):
        self.train_ratio = train_ratio 
        self.test_ratio = test_ratio 
        self.to_float32 = to_float32
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size) 
        self.patch_size = patch_size 
        self.test_mode = test_mode  
        self.to_8bit = to_8bit
        self.bands = bands 
 
        self.stretch_ch = stretch_ch


        if rasterio is None:
            raise RuntimeError('rasterio is not installed')

    def transform(self, results: Dict) -> Dict:
        """Fu
        그냥 이미지 열기만 함 
        """ 
        filename = results['img_path']   
 
        with rasterio.open(filename) as f:
            img = f.read(self.bands) 
        if self.to_float32:
            img = img.astype(np.float32)


        if self.to_8bit:
            img *= 255.0 # 0~255 범위로 변환

        #img = img.transpose(1,2,0) #CHW -> HWC 
        img = np.einsum('ijk->jki', img)

        results['img'] = img # (1024, 1024, 3)
        results['img_shape'] = img.shape[:2] # (1024, 1024)
        results['ori_shape'] = img.shape[:2] 
        return results

    def __repr__(self): # 객체를 문자열로 표현
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str
 


#============
@TRANSFORMS.register_module()
class LoadLargeAnnotations(BaseTransform):
    """Load annotations for semantic segmentation provided by dataset.
        MMCV의 LoadAnnotations에서 가져와서 수정함 
        LoadSegMap만 사용함 

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file. -> mask 경로

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8) -> mask map

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """ 
    def __init__(
        self,
        with_bbox: bool = False,
        with_label: bool = False,
        with_seg: bool = True,
        with_keypoints: bool = False,
        binary_mode : bool = False ,  
        reduce_zero_label:bool=None, 
        reduct_dim :bool = False ,
        stretch_ch: bool = False ,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints 

        self.reduce_zero_label = reduce_zero_label
        self.binary_mode = binary_mode
        self.reduct_dim = reduct_dim 
        self.stretch_ch = stretch_ch
 
    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64) 

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox: #여기서는 사용하지 않음 
            self._load_bboxes(results)
        if self.with_label: #여기서는 사용하지 않음 
            self._load_labels(results)
        if self.with_seg: 
            self._load_seg_map(results)
        if self.with_keypoints: #여기서는 사용하지 않음 
            self._load_kps(results)
        return results 

#################################################################################################

    #### _load_seg_map()만 사용
    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        # segmentation map 읽기
        with rasterio.open(results['seg_map_path']) as f: # 파일 열기
            # 래스터 데이터에서 처리할 영역 정의, 'img_bbox'는 보통 (col_off, row_off, width, height)형태 튜플
            window = rasterio.windows.Window(*results['img_bbox']) 
            # 지정된 영역의 픽셀 데이터 읽어옴, numpy 배열로 반환, .squeeze() ex) (1,H,W)->(H,W), .astype(np.unit8) : 0~255 
            gt_semantic_seg = f.read(window=window).squeeze().astype(np.uint8) 
        
             
        if self.binary_mode:# binary_mode면 실행
            gt_semantic_seg = gt_semantic_seg.squeeze() 
            mask = np.where(gt_semantic_seg>0) # 0보다 큰 픽셀의 위치 찾음
            gt_semantic_seg[mask] = 1 # 찾은 픽셀의 값을 모두 1로 변환



        #gt_semantic_seg = gt_semantic_seg.astype(np.float32)
        # 현재 사용 X
        # reduce zero_label
        # 0번 라벨을 모델이 예측할 때 고려하지 않게 만듬
        # 라벨 0,1,..,5 존재, reduce zero_label == True
        # 0번 -> 255번 변환, 나머지 1~5번 -> 0~4번 변환한 뒤 모델의 입력으로 들어감
        # 모델이 실제로는 1~5번의 선택지 중에서만 예측
        if self.reduce_zero_label is None: 
            self.reduce_zero_label = results['reduce_zero_label'] # 주로 라벨 0을 배경으로 사용, 0을 다른 값으로 변경할 지 여부 제어
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255 # 0인 값을 255로 변경
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        # 현재 사용 X
        # 'label_map' = 라벨 0번을 1번으로 바꾸는 매핑(예시)    
        # 'seg_fields' = 다른 세그멘테이션의 경우에 뎁스맵 같은 다른 맵도 들어가서 구분 짓기 위해서 입력하는 정보
        # modify if custom classes
        if results.get('label_map', None) is not None: # result가 label_map 포함 시, 기존 라벨을 새로운 라벨로 매핑
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')  


#################################################################################################


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadLargeRSImage(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """ 


    def __init__(self, 
            train_ratio: Optional[float] = None,
            test_ratio: Optional[float] = None, 
            patch_size: Union[tuple, int, list] = 1024, 
            bands: Optional[Sequence] = None, 
            to_float32: bool = True,
            to_8bit: bool = False,
            test_mode: bool = False, 
            max_T: int = 100,  # 훈련 패치를 추출할 최대 시도 횟수
            stretch_ch: bool = False ,
            ):
        self.train_ratio = train_ratio 
        self.test_ratio = test_ratio 
        self.to_float32 = to_float32
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size) 
        self.patch_size = patch_size 
        self.test_mode = test_mode 
        self.max_T = max_T  # iteration 
        self.to_8bit = to_8bit
        self.bands = bands 
 
        self.stretch_ch = stretch_ch


        if rasterio is None:
            raise RuntimeError('rasterio is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        해당 방식은 매번 파일을 열어서 bbox를 계산해야 하기 때문에 
        시간이 불필요하게 걸릴 수 있다. 
        그렇지만 아직 실험 단계이므로 해당 방식으로 진행 중이다.... 
        최적화 시에 dataset 에서 iterable로 도는 방법을 추구해야 한다. 

        Args: -> 입력 값 설명(데이터 타입, 추가 설명, ...)
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        """
        COCO 형식 (x,y,w,h) ->(좌상단 x, 좌상단 y, 바운딩 박스의 W, 바운딩 박스의 H)
        """
        filename = results['img_path']   
 
        with rasterio.open(filename) as f:
            width = f.width
            height = f.height 

            if self.test_mode: 
                if self.train_ratio is None: #Test set 
                    test_h = int(self.test_ratio*height) # test set의 높이를 계산해 잘라냄
                    bounds = (0, height-test_h, width, test_h)  # (x,y,w,h) 
                else: # Validation Set / train, test 제외 남은 비율
                    val_h = int((1-self.train_ratio-self.test_ratio)*height)
                    bounds = (0, int(self.train_ratio*height),width, val_h) 
                bbox = bounds
                img = f.read(window=Window(*bbox)) 
                x_crop, y_crop = bbox[:2] # x, y 좌표를 크롭 위치로 설정
            else: #Train mode (random sampling)
                if self.train_ratio is None: # Raw tiles    
                    bounds = (0,0, width, height) #x,y,w,h 
                else: # 세로 크기의 일부만 훈련 데이터로 사용
                    bounds = (0,0,width, int(height*self.train_ratio))
    
                for _ in range(self.max_T):  # 랜덤한 좌표 선택
                    x_crop = np.random.randint(bounds[0], max(bounds[0]+bounds[2], bounds[0]+self.patch_size[0]))
                    y_crop = np.random.randint(bounds[1], max(bounds[1]+bounds[3], bounds[1]+self.patch_size[1]))
    
                    bbox = (x_crop, y_crop, self.patch_size[0], self.patch_size[1])
                    #윈도우가 이미지 범위를 넘어가면 자동으로 밖의 영역을 0으로 채워줌 
                    img = f.read(window=rasterio.windows.Window(*bbox)) 

                    if np.sum(img) >0: # 픽셀 값의 합이 0보다 큰 경우 반복문 종료(빈 영역이 아닌 유효한 데이터를 찾음)
                        break 
                
        if self.to_float32:
            img = img.astype(np.float32)


        if self.to_8bit:
            img *= 255.0 # 0~255 범위로 변환

        #img = img.transpose(1,2,0) #CHW -> HWC 
        img = np.einsum('ijk->jki', img)

        results['img'] = img # (1024, 1024, 3)
        results['img_shape'] = img.shape[:2] # (1024, 1024)
        results['ori_shape'] = img.shape[:2]
        #results['img_raw_size'] = (width, height) 
        results['img_bbox'] = bbox #x,y,w,h
        #results['img_bounds'] = bounds 
  
        return results

    def __repr__(self): # 객체를 문자열로 표현
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str
 


# 토지피복도 불러오기
@TRANSFORMS.register_module()
class LoadLUImage(BaseTransform):
    def __init__(self, 
                 mode: str = None, 
                 normalize: bool = False,
                 landcover_suffix: str = None):
        """
        Parameters:
            add_to (str): Where to add the LU data ('img' to concatenate with the image).
            normalize (bool): Whether to normalize LU data to 0~1.
            landcover_folder (str): Path to the folder containing landcover images.
            landcover_suffix (str): Suffix pattern for landcover images (e.g., 'landcover').
        """
        self.mode = mode 
        self.normalize = normalize
        self.landcover_suffix = landcover_suffix

    def transform(self, results: Dict) -> Dict:
        """
        Transform function to load and merge multiple LU images with the RGB image.

        Parameters:
            results (Dict): Input dictionary containing image metadata.

        Returns:
            Dict: Updated dictionary with merged LU data.
        """
        filename = results['img_path']    # /data/jym/mmsegmentation/data/experiment/images/3channel/normalized
        bbox = results['img_bbox']
        img = results['img']
        #print("[1] RGB Image Path : ",filename)
        landcover_folder = '/'.join(filename.split("/")[:-4]) + self.landcover_suffix

        # Replace the path to point to the landcover folder
        base_filename = filename.split("/")[-1].replace(".tif", "")  # Extract base filename
        landcover_pattern = f"{landcover_folder}/{base_filename}_landcover_*.tif"

        # Find all matching landcover files
        import glob
        landcover_files = sorted(glob.glob(landcover_pattern))
        #print("[2] Landcover Path : ",landcover_files)

        if not landcover_files:
            raise FileNotFoundError(f"No landcover files found matching pattern: {landcover_pattern}")

        # Initialize a list to store all LU data
        lu_list = []

        for lc_file in landcover_files:
            with rasterio.open(lc_file) as f:
                LU = f.read(window=rasterio.windows.Window(*bbox))  # Read specific bbox
                """
                if self.normalize:
                    LU = (LU / 255).astype(np.float32)  # Normalize to 0~1
                """

                lu_list.append(LU)

        # Stack all LU data along the channel dimension
        LU_combined = np.concatenate(lu_list, axis=0)  # Combine along channel axis (C, H, W) -> (10, 512, 512)
        #print("[3] Landcover Combine : ",LU_combined.shape)


        img_dtype = img.dtype
        if self.mode == 'concat': 
            # Convert from CHW to HWC
            LU_combined = np.einsum('ijk->jki', LU_combined)
            #if not LU_combined.shape[2] == 10:
                #raise Exception("Landcover count check failed: Expected 10 channels, but got {}".format(LU_combined.shape[2]))
            img = np.concatenate([img, LU_combined], axis=2).astype(img_dtype)  # Concatenate along channel axis

            results['img'] = img 

        else:
            assert False, "아직 코드 안짯어유"
         
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mode={self.mode}, '
                    f'normalize={self.normalize}, ')
        return repr_str




# 토지피복도 불러오기
@TRANSFORMS.register_module()
class LoadLUImageYerang(BaseTransform):
    def __init__(self,  
                 mode = 'cocnat', 
                landcover_dir = '/data/yerang/projects/mmsegmentation/project/KCC2025/temp/landcover',
                img_suffix = '_3C_nodata0_labeltile.tif', 
                lu_suffix = '_landcover_class3.tif',
                 masking = False, 
                 shp_path = 'landcover/shp/' ,  
                 shp_prefix = 'Merged_'  ,
                 shp_suffix = '.shp' ,  
    ):  
        self.mode = mode 
        #raster ver. -> fast 
        self.landcover_dir = landcover_dir
        self.img_suffix = img_suffix
        self.lu_suffix = lu_suffix 

        #vector ver. -> too slow 
        self.masking = masking 
        self.shp_path = shp_path
        self.shp_prefix = shp_prefix
        self.shp_suffix = shp_suffix
        self.L3_classes = dict(
            building = ['111','112',
                        '121',
                        '131', '132', 
                        '141',
                        '161', '162','163'], 
            road = ['154'], 
            forest = ['311','321','331'], 
            # facility_plantation=['231'],   
            # other_plantation = ['252'],  
        ) 

    def rasterizing(self, img_path, shp_path, height, width):
        with rasterio.open(img_path) as f:
            transform = f.transform 
            img_crs = f.crs 
            
        shp_gdf = gpd.read_file(shp_path) 
        shp_gdf.to_crs(img_crs, inplace=True) 

        grouped = shp_gdf.groupby('L3_CODE') 
        L3_code_geoms = {
            code: list(grouped.get_group(code).geometry.values)
            for code in shp_gdf['L3_CODE'].unique()
            if code in {v for values in self.L3_classes.values() for v in values}
        }
        lu_list = [] 
        

        for i, (key,value) in enumerate(self.L3_classes.items(), start=1):
            #key_shp_gdf = shp_gdf[shp_gdf['L3_CODE'].isin(value)] 
            
            # 해당 클래스 value 리스트에 포함된 geometry만 추출
            geometries = [geom for v in value for geom in L3_code_geoms.get(v, []) if geom is not None]
            if not geometries:
                lu_list.append(np.zeros((height, width), dtype='uint8'))
                continue

            #1 or 255 
            #shapes = ((geom, 1) for geom in key_shp_gdf.geometry)
            shapes = ((geom, 1) for geom in geometries)
 
            raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )  
            lu_list.append(raster)  
        
        LU_stack = np.stack(lu_list, axis=2)
        return LU_stack 


    def transform(self, results: Dict) -> Dict:
        """
        Transform function to load and merge multiple LU images with the RGB image.

        Parameters:
            results (Dict): Input dictionary containing image metadata.

        Returns:
            Dict: Updated dictionary with merged LU data.
        """
        img_path = results['img_path']    # /data/jym/mmsegmentation/data/experiment/images/3channel/normalized
        
        #if self.masking:
        #filename = "_".join(osp.basename(img_path).split("_")[:5])
        #landcover_path = osp.join(img_path.split('images')[0], self.shp_path, self.shp_prefix + filename + self.shp_suffix)
        filename = osp.basename(img_path)
        landcover_path = osp.join(self.landcover_dir, filename.replace(self.img_suffix, self.lu_suffix))

        if not osp.isfile(landcover_path):
            raise FileNotFoundError(f"No landcover files found matching pattern: {shp_path}")

        #if '.shp' in landcover_path:
        #    LU_stack = self.rasterizing(img_path, landcover_path, height, width)
            #minx, miny, width, height = results['img_bbox']
            #maxx, maxy = minx+width, miny+height 
        bbox = results['img_bbox']
        img = results['img']

        with rasterio.open(landcover_path) as f:
            LU_stack = f.read(window = Window(*bbox)) 
            LU_stack = np.einsum('ijk->jki', LU_stack) * 255 

        img_dtype = img.dtype
        if self.mode == 'concat': 
            img = np.concatenate([img, LU_stack] , axis=2).astype(img_dtype)  # Concatenate along channel axis
            results['img'] = img 
        elif self.mode == 'addkey':
            results['landcover'] = LU_stack 
        else:
            raise NotImplementedError('LoadLUImage에서는 concat 모드만 지원합니다.')
         

        #raise RuntimeError("debugging")
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mode={self.mode}, '
                    f'normalize={self.normalize}, ')
        return repr_str



@TRANSFORMS.register_module()
class CustomLoadAnnotations(MMCV_LoadAnnotations): 
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
            self.crop_h, self.crop_w = crop_size  
            if self.patch_type == 'fixed':
                self.sampling_func = self.fixed_sampling 
            else:
                if scipy_rotate is None:
                    raise NotImplementedError("scipy가 설치되어야 함 ")
                self.sampling_func = self.affine_sampling
                self.diag = int(np.ceil(np.sqrt(crop_size[0]**2 + crop_size[1]**2 )))
        if rasterio is None:
            raise NotImplementedError("rasterio가 설치되어야 함") 

    def fixed_sampling(self, results):  
        """
        bbox = (xmin, ymin, xmax, ymax)이 주어지면 
        그대로 바운딩박스대로 샘플링  
        """
        filename = results['img_path']      
        window = Window(*results['bbox'] )
        with rasterio.open(filename) as src: 
            img = src.read(1, window=window) 
        return img   

    def affine_sampling(self, results): 
        """
        원래 주어진 bbox에서 회전 및 이동을 적용하여 샘플링
        """
        filename = results['seg_map_path']     
        window = Window(**results['read_bbox']) 
        with rasterio.open(filename) as src: 
            patch_big = src.read(1, window=window)  #멀티일때 수정하기 
        theta_deg = torch.randint(0, 360, (1,)).item() #torch에서만 매 에포크 시드가 다름 
        patch_rot = scipy_rotate(
            patch_big,
            angle=results['rotate_deg']  ,
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )
        start_x, end_x, start_y, end_y = results['crop_bbox'] 
        patch_cropped = patch_rot[start_x:end_x, start_y:end_y]
        return patch_cropped  

    def image_loading(self, results):  
        filename = results['img_path']      
        with rasterio.open(filename) as src:   
            img =  src.read(1) #multi일 때 수정하기.... 
        return img 
 
    def _load_seg_map(self, results: dict) -> None:    
        filename = results['seg_map_path'] 
        bbox = results['bbox'] #LoadImage 에서 이미 bbox None을 없앴다고 가정 
        
        gt_semantic_seg = self.sampling_fuc(results) 
        print(gt_semantic_seg.shape)
        raise NotImplementedError("디버깅 지점") 
      
        """
        if self.multi_channel:
            gt_semantic_seg = np.zeros((h,w), dtype=np.uint8) 
            gt_semantic_seg = np.sum(mask, axis=0)
        else:
            gt_semantic_seg = mask.squeeze() 

        if self.binary_mode: 
            gt_semantic_seg[gt_semantic_seg>0] = 1  
        
        if self.padding:   
            pad_height = max(0,self.pad_size[0] - h)   
            pad_width = max(0, self.pad_size[1] - w)   
               
            gt_semantic_seg = np.pad(gt_semantic_seg, 
                    ((0, pad_height), (0,pad_width)), 
                    mode='constant', 
                    constant_values=0) 
        """ 
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
   
    def transform(self, results: dict) -> dict:   
        self._load_seg_map(results) 
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
                 crop_size: Optional[Sequence] = None,  
                 backend_args: Optional[dict] = None,
                 patch_type: Optional[Literal['center', 
                                    'fixed',
                                    'random',
                                    'affine']] = None
                 ) -> None: 
        """
        패치 샘플링 방식들은 results에 bbox 키값이 있어야 함 
        """
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None  
        
        self.bands = bands 
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
    
        if rasterio is None:
            raise NotImplementedError("rasterio가 설치되어야 함")

    def center_sampling(self, results):  
        """
        bbox = (l,b,r,t)이 주어지면 해당 바운드의 중심을 기준으로
        crop_size만큼 패치를 샘플링
        """
        filename = results['img_path']     
        bbox = results['bbox']  
        xmin = (bbox[2]-bbox[0])/2 - self.crop_w/2 
        ymin = (bbox[3]-bbox[1])/2 - self.crop_h/2  
        xmin, ymin = max(0, xmin), max(0, ymin) 
        window = Window(xmin,ymin, crop_w, crop_h)
        with rasterio.open(filename) as src:
            if self.bands is not None:
                img = src.read(self.bands, window=window)
            else:
                img = src.read(window=window) 
        return img.transpose(1,2,0)  

    def fixed_sampling(self, results):  
        """
        bbox = (xmin, ymin, xmax, ymax)이 주어지면 
        그대로 바운딩박스대로 샘플링  
        """
        filename = results['img_path']      
        window = Window(*results['bbox'] )
        with rasterio.open(filename) as src:
            if self.bands is not None:
                img = src.read(self.bands, window=window)
            else:
                img = src.read(window=window) 
        return img.transpose(1,2,0) 

    def random_sampling(self, results):  
        """
        원래 주어진 bbox에서 이동을 적용하여 샘플링
        """
        raise NotImplementedError("아직 구현안함 ")

    def affine_sampling(self, results): 
        """
        원래 주어진 bbox에서 회전 및 이동을 적용하여 샘플링
        """
        filename = results['img_path']     
        bbox = results['bbox']  
        xmin = max(0, math.floor((bbox[2]-bbox[0])/2 - self.diag/2 )) 
        ymin = max(0, math.floor((bbox[3]-bbox[1])/2 - self.diag/2 ))  

        window = Window(xmin, ymin, self.crop_w, self.crop_h) 
        with rasterio.open(filename) as src:
            if self.bands is not None:
                patch_big = src.read(self.bands, window=window)
            else:
                patch_big = src.read(window=window) 
        theta_deg = torch.randint(0, 360, (1,)).item() #torch에서만 매 에포크 시드가 다름 
        patch_rot = scipy_rotate(
            patch_big,
            angle=theta_deg,
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )
        h, w = patch_rot.shape[:2] 
        start_y = (h - self.crop_h) // 2
        start_x = (w - self.crop_w) // 2
        patch_cropped = patch_rot[start_y:start_y + self.crop_h, 
                            start_x:start_x + self.crop_w] 


        results['read_bbox'] = (xmin, ymin, self.crop_w, self.crop_h)
        results['rotate_deg'] = theta_deg
        results['crop_bbox'] = (start_x, start_x + self.crop_w,
                                start_y, start_y + self.crop_h,
                                    )
        return patch_cropped.transpose(1,2,0) 

    def image_loading(self, results):  
        filename = results['img_path']      
        with rasterio.open(filename) as src:    
            if self.bands is not None:
                img =  src.read(self.bands) 
            else: 
                img =  src.read() 
        return img.transpose(1,2,0) 


    def transform(self, results: Dict) -> Dict:  
        """
        미리 정의한 query 버전 
        """ 
        transposed_img = self.sampling_func(results)  
        if self.to_float32:
            transposed_img = transposed_img.astype(np.float32) 
 
        results['img'] = transposed_img
        results['img_shape'] = transposed_img.shape[:2]
        results['ori_shape'] = transposed_img.shape[:2] 
        return results
 
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', " 
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str
