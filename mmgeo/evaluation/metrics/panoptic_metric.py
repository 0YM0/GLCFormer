# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmengine.registry import METRICS 
from sklearn import metrics 

#custom 
from scipy.ndimage import label

@METRICS.register_module()
class PanopticMetric(BaseMetric): 
    def __init__(self,
                 ignore_index: int = 255,
                 combined_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 binary: bool = False, 
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)  
        self.ignore_index = ignore_index
        self.metrics = combined_metrics # 
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.binary = binary 


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        train, val 루프 내에서 매번 입력을 받으면 픽셀을 계산해서 결과에 넣어줌 
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            seg_logits_np = data_sample['seg_logits']['data'].squeeze().cpu().numpy() 

            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label) #학습 데이터와 동일한 디바이스로 이동 
                area_intersect, area_union, area_pred_label, area_label = self.intersect_and_union(pred_label, label, 
                                                num_classes, self.ignore_index) 
                #
                iious = self.instancewise_iou(seg_logits_np, label, threshold = 0.5 ) #list 형태 
                pq, _, _ = self.panoptic_quality(seg_logits_np, label, threshold=0.5)  #float 형태  
                self.results.append( #pred, gt를 입력으로 넣어서 메트릭 계산
                     [area_intersect,area_union, area_pred_label, area_label, iious, pq  ]
                 ) 
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename) 

 
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        test, val의 루프에서 추가된 결과들을 모두 합산하려는 것. 
        for e in range(epoch):
            out = process()
            results.append(out)
        compute_metrics(results) 
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) >= 4 # custom  

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])   

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect = total_area_intersect, 
            total_area_union = total_area_union, 
            total_area_pred_label = total_area_pred_label,
            total_area_label = total_area_label, 
            metrics = self.metrics, 
            nan_to_num = self.nan_to_num, 
            beta = self.beta,
            binary = self.binary)
         
        class_names = self.dataset_meta['classes']

        # summary table - binary : np.nanmean(ret_metric_value)
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(ret_metric_value[0] * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val
        
        #>>> custom >>>
        # 솔직히 flatten이 맞는지 모르겠음.. 
        #ious = [[], [], ...[0,0], [0.33], ... []]  
        if isinstance(self.metrics, str):
            metrics = [self.metrics]
        else:
            metrics = self.metrics   
        for metric in metrics: 
            if metric == 'iIoU':   
                metrics['iIoU'] = np.mean(np.concatenate(results[4]))  #custom  
                ret_metrics['iIoU'] = [np.round(np.array(metrics['iIoU'])* 100, 2)]
            if metric == 'PQ':  
                metrics['PQ'] = np.mean(results[5])  #custom  
                ret_metrics['PQ'] =[np.round(np.array(metrics['PQ'])* 100, 2)]
        #>>> >>> 

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        }) 
        ret_metrics_class.update({'Class': class_names}) #modify 
        ret_metrics_class.move_to_end('Class', last=False) 

        class_table_data = PrettyTable()  
        for key, val in ret_metrics_class.items():  
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics #metric이 local에 저장되고 ret_metrics가 pretty table로 나타남 

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int): 

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect 
        return area_intersect, area_union, area_pred_label, area_label


    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1,
                              binary: bool=False): 
        def f_score(precision, recall, beta=1): 
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall + 1e-6)
            return score 

        def _cohen_kappa_score(confusion: np.ndarray, 
                            n_classes: int = 2, 
                            weights: Optional[str] = None) -> str :
            """
            args:
                - confusion: [[TN FP], [FN TP]] 형태의 넘파이 어레이 
                - weights(optinoal): None이면 가중치를 두지 않음. linear나 quadratic 가중치를 둘 수 있음 
            return:
                kappa_value
            """
            sum0 = np.sum(confusion, axis=0) 
            sum1 = np.sum(confusion, axis=1)
            expected = np.outer(sum0, sum1) / np.sum(sum0) 

            if weights is None:
                w_mat = np.ones([n_classes, n_classes], dtype=int)
                w_mat.flat[:: n_classes + 1] = 0
            else:  # "linear" or "quadratic"
                w_mat = np.zeros([n_classes, n_classes], dtype=int)
                w_mat += np.arange(n_classes)
                if weights == "linear":
                    w_mat = np.abs(w_mat - w_mat.T)
                else:
                    w_mat = (w_mat - w_mat.T) ** 2 
            k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
            return 1-k
        
        def _get_confusion(n_classes, total_area_intersect, total_area_pred_label,
                            total_area_label, ):  
            TPs, FPs, FNs, TNs = [],[],[],[]
            for i in range(n_classes):
                TP = total_area_intersect[i] 
                FP = total_area_pred_label[i] - TP
                FN = total_area_label[i] - TP 
                TN = np.sum(total_area_label) - (TP+FP+FN) 
                TPs.appned(TP) 
                FPs.append(FP) 
                FNs.append(FN) 
                TNs.append(TN) 
            return TPs, FPs, FNs, TNs

        if isinstance(metrics, str):
            metrics = [metrics] 
        allowed_metrics = ['IoU', 'Dice', 'F1',
                        'F2', 'Recall', 'Precision', 'Kappa', 'Confusion'
                        'iIoU', 'PQ'
                        ]
 
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()  
        ret_metrics = OrderedDict({'aAcc': all_acc}) 
        if binary:
            for metric in metrics: 
                postive_area_intersect = total_area_intersect[1].item() #tensor
                pos_area_union = total_area_union[1].item()
                pos_area_pred_label = total_area_pred_label[1].item()
                pos_area_label = total_area_label[1].item() 

                TP = postive_area_intersect
                FP = pos_area_pred_label - TP
                FN = pos_area_label - TP  
                TN = pos_area_label - TP- FN -FP  ##  
                precision = postive_area_intersect / pos_area_pred_label 
                recall = postive_area_intersect / pos_area_label

                if metric == 'IoU': 
                    iou = postive_area_intersect / pos_area_union
                    acc = postive_area_intersect / pos_area_label
                    ret_metrics['IoU'] = iou
                    ret_metrics['Acc'] = acc 
                elif metric == 'Dice':
                    dice = 2 * postive_area_intersect / (
                        pos_area_pred_label + pos_area_label)
                    acc = postive_area_intersect / pos_area_label
                    ret_metrics['Dice'] = dice 
                elif metric == 'Precision':
                    ret_metrics['Precision'] = precision
                elif metric == 'Recall' : 
                    ret_metrics['Recall'] = recall
                elif metric == 'Confusion':   
                    ret_metrics['TP'] = torch.tensor([TP]) 
                    ret_metrics['FP'] = torch.tensor([FP])  
                    ret_metrics['FN'] = torch.tensor([FN])  
                    ret_metrics['TN'] = torch.tensor([TN])   
                elif metric == 'F1':  
                    ret_metrics['F1'] = torch.tensor([f_score(precision, recall, beta=1)])  
                    ret_metrics['F0.3'] = torch.tensor([f_score(precision, recall, beta=0.3)]) 
                elif metric == 'F2':   
                    ret_metrics['F2'] =  torch.tensor([f_score(precision, recall, beta=2)])  
                elif metric == 'Kappa':  
                    kappas = []  
                    confusion = np.array([[TN, FP], [FN, TP]])   
                    ret_metrics['Kappa'] =  _cohen_kappa_score(confusion, n_classes=2 )    
        else:   
            raise NotImplementedError()
            n_classes = len(total_area_intersect)   #custom 
            for metric in metrics:
                if metric == 'IoU': 
                    iou = total_area_intersect / total_area_union
                    acc = total_area_intersect / total_area_label
                    ret_metrics['IoU'] = iou
                    ret_metrics['Acc'] = acc
                elif metric == 'Dice':
                    dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                    acc = total_area_intersect / total_area_label
                    ret_metrics['Dice'] = dice 
                elif metric == 'Precision':
                    precision = total_area_intersect / total_area_pred_label
                elif metric == 'Recall' : 
                    recall = total_area_intersect / total_area_label
                elif metric == 'Confusion':  
                    ret_metrics['TP'], ret_metrics['FP'], ret_metrics['FN'], ret_metrics['TN'] =  \
                                _get_confusion(n_classes, total_area_intersect, total_area_pred_label, total_area_label,) 
                elif metric == 'F1':
                    if 'precision' not in locals():
                        precision = total_area_intersect / total_area_pred_label
                    if 'recall' not in locals(): 
                        recall = total_area_intersect / total_area_label

                    f_value = torch.tensor([
                        f_score(x[0], x[1], beta) for x in zip(precision, recall)
                    ])
                    ret_metrics['F1'] = f_value
                elif metric == 'F2': 
                    if 'precision' not in locals():
                        precision = total_area_intersect / total_area_pred_label
                    if 'recall' not in locals(): 
                        recall = total_area_intersect / total_area_label
                    f2_value = torch.tensor([
                        f_score(x[0], x[1], beta=2) for x in zip(precision, recall)
                    ])  
                    ret_metrics['F2'] = f2_value 
                elif metric == 'Kappa': 
                    if 'TP' in ret_metrics.keys():
                        TNs, FPs, FNs, TPs =  \
                                    _get_confusion(n_classes, total_area_intersect, total_area_pred_label, total_area_label,) 
                    kappas = [] 
                    for i in range(nclasses):
                        confusion = np.array([[TNs[i], FPs[i]], [FNs[i], TPs[i]]]) 
                        kappas[i] =  _cohen_kappa_score(confusion, n_classes=n_classes)   
                    ret_metrics['Kappa'] = kappas  

        ret_metrics = {} 
        for metric, value in ret_metrics.items():
            print(metric, value)
            ret_metrics[metric] = value.numpy() 
        
        #ret_metrics = {
        #    metric: value.numpy()
        #    for metric, value in ret_metrics.items()
        #}
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })

 
        return ret_metrics

    #instance-wise IoU (iIoU) 
    @staticmethod
    def instancewise_iou(pred_mask: Union[torch.tensor, np.ndarray], 
                        gt_mask: Union[torch.tensor, np.ndarray], 
                        threshold: float = 0.5):
        """
        예측 마스크와 GT 마스크 간 instance-wise IoU 계산

        Args:
            pred_mask (torch.tensor): 예측 마스크 (H, W)
            gt_mask (torch.tensor): 정답 마스크 (H, W)
            threshold (float): binary threshold

        Returns:
            ious (List[float]): GT 인스턴스마다의 IoU
        """
        def extract_instances_from_binary_mask(binary_mask: np.ndarray, threshold: float = 0.5):
            # normalized mask : [0,1]  
            if binary_mask.max() > 1:
                binary_mask = binary_mask / 255.0

            binarized = (binary_mask > threshold).astype(np.uint8)
            instance_mask, num_instances = label(binarized)
            return instance_mask, num_instances


        #tensor to numpy 
        gt_mask_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
        pred_mask_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask

        pred_instances, pred_n = extract_instances_from_binary_mask(pred_mask_np, threshold)
        gt_instances, gt_n = extract_instances_from_binary_mask(gt_mask_np, threshold)
 
        ious = []
        for gt_id in range(1, gt_n + 1):
            gt_inst = (gt_instances == gt_id)
            best_iou = 0.0
            for pred_id in range(1, pred_n + 1):
                pred_inst = (pred_instances == pred_id)
                intersection = np.logical_and(gt_inst, pred_inst).sum()
                union = np.logical_or(gt_inst, pred_inst).sum()
                iou = intersection / union if union > 0 else 0
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

        return ious



    #panoptic quality (PQ) 
    @staticmethod
    def panoptic_quality(
        pred_mask: Union[torch.tensor, np.ndarray], 
        gt_mask: Union[torch.tensor, np.ndarray], 
        threshold: float = 0.5
    ):
        def extract_instances_from_binary_mask(binary_mask: np.ndarray, threshold: float = 0.5):
            if binary_mask.max() > 1:
                binary_mask = binary_mask / 255.0
            binarized = (binary_mask > threshold).astype(np.uint8)
            instance_mask, num_instances = label(binarized)
            return instance_mask, num_instances

        gt_mask_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
        pred_mask_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask

        pred_instances, pred_n = extract_instances_from_binary_mask(pred_mask_np, threshold)
        gt_instances, gt_n = extract_instances_from_binary_mask(gt_mask_np, threshold)

        tp = 0
        sum_iou = 0.0
        matched_gt = set()

        for pred_id in range(1, pred_n + 1):
            pred_inst = (pred_instances == pred_id)
            best_iou = 0.0
            matched_gt_id = -1
            for gt_id in range(1, gt_n + 1):
                if gt_id in matched_gt:
                    continue
                gt_inst = (gt_instances == gt_id)
                intersection = np.logical_and(pred_inst, gt_inst).sum()
                union = np.logical_or(pred_inst, gt_inst).sum()
                iou = intersection / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_id = gt_id
            if best_iou > 0.5:
                tp += 1
                sum_iou += best_iou
                matched_gt.add(matched_gt_id)

        fp = pred_n - tp
        fn = gt_n - len(matched_gt)

        # Recall, Precision 정의 (instance가 없을 경우 포함)
        if gt_n == 0:
            recall = 1.0
        else:
            recall = tp / gt_n

        if pred_n == 0:
            precision = 1.0 if gt_n == 0 else 0.0
        else:
            precision = tp / pred_n

        denom = tp + 0.5 * fp + 0.5 * fn
        pq = sum_iou / denom if denom > 0 else (1.0 if gt_n == 0 and pred_n == 0 else 0.0)

        return pq, precision, recall