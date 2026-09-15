# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from typing import Union, Callable
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmengine.model import is_model_wrapper
import logging 
from mmengine.logging import print_log 

from torch.serialization import safe_globals
import mmengine.logging.history_buffer
import torch 
import rasterio
from affine import Affine 
from datetime import datetime     
import numpy.core.multiarray
import numpy as np  

import os 
class CKPRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 


    def load_from(self, filename,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        print("===모델 체크포인트 불러옴===")
        print(f"파일 존재 여부: {os.path.isfile(filename)}")
        with safe_globals([mmengine.logging.history_buffer.HistoryBuffer, numpy.core.multiarray._reconstruct]):
            try:
                checkpoint = torch.load(filename, weights_only=False, map_location = map_location)['state_dict'] 
            except Exception as e:
                raise e 
                checkpoint = torch.load_checkpoint(filename , weights_only=True, map_location='cpu')    
        """        
        # revise_keys 적용
        new_state_dict = checkpoint.copy()
        for pattern, replacement in revise_keys:
            for key in list(new_state_dict.keys()):
                new_key = re.sub(pattern, replacement, key)
                if new_key != key:
                    new_state_dict[new_key] = new_state_dict.pop(key)
        """
        if is_model_wrapper(self.model):
            raise NotImplementedError
            self.model.module.load_state_dict(checkpoint)  
        else:
            self.model.load_state_dict(checkpoint)   
        print("====Completed===") 
        return 

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--pred_dir', type=str) 
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
 
def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    #cfg.load_from = args.checkpoint
 
    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # Runner 생성 및 초기화
    runner = CKPRunner.from_cfg(cfg)


    if args.pred_dir is None: 
        output_dir = cfg.work_dir + "/preds_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        output_dir = args.pred_dir 
    os.makedirs(output_dir, exist_ok=True ) 

    runner.load_from(args.checkpoint, )


    # dataloader 설정
    dataloader = runner.test_dataloader

    # Evaluator 빌드
    if args.skip_eval  : #store false 
        evaluator = None 
    else: 
        evaluator = runner.test_evaluator
        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            evaluator = runner.build_evaluator(evaluator)

        # metainfo 설정
        if hasattr(dataloader.dataset, 'metainfo'):
            evaluator.dataset_meta = dataloader.dataset.metainfo
            runner.visualizer.dataset_meta = dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING) 


    # 모델을 eval 모드로 전환
    runner.model.eval()

    results = []
    # 추론 실행
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            outputs = runner.model.test_step(data_batch)
            # test_step의 출력은 이미 후처리된 결과일 수 있음
            # 일반적으로는 다음과 같이 리스트 형태로 반환됨
            #results.extend(seg_logits)
            if evaluator is not None: 
                evaluator.process(data_samples = outputs, data_batch=data_batch)
            #
            #pred_seg = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
            for data_sample in outputs:
                img_path = data_sample.img_path
                pred_sem_seg = data_sample.pred_sem_seg.data
                with rasterio.open(img_path) as src:
                    profile = src.profile  
                    original_transform = src.transform
                    width, height = src.width, src.height 

                if data_sample.read_bbox is not None: 
                    x_min, y_min, width, height = data_sample.read_bbox 
                    new_transform = Affine.translation(x_min, y_min) * original_transform
                else: 
                    new_transform = original_transform

                # Profile 업데이트
                profile.update(
                    width=width,
                    height=height,
                    transform=new_transform,
                    count = 1 ,
                    dtype = np.uint8, 
                    nodata=0 
                )
                output_path = osp.join(output_dir, osp.basename(img_path)) 
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(pred_sem_seg.cpu().numpy().squeeze().astype(profile['dtype']),1)

    if evaluator is not None:
        metrics = evaluator.evaluate(len(dataloader.dataset))
        print(metrics)
    
        # out 경로에 결과 저장 (옵션)
        if args.out is not None:
            os.makedirs(args.out, exist_ok=True)
            filename = osp.join(args.out, 'results.pkl')
            import mmcv
            mmcv.dump(metrics, filename)
            print(f'Results saved to {filename}') 

if __name__ == '__main__':
    main()
 