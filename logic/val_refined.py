import os
import math
import logging
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.box_utils import nms_3D
from utils.generate_annot_csv_from_series_list import generate_annot_csv
from evaluationScript.eval_original import nodule_evaluation
from utils.utils import get_progress_bar
from .utils import get_memory_format
logger = logging.getLogger(__name__)

def convert_to_standard_csv(csv_path: str, annot_save_path: str, series_uids_save_path: str, spacing):
    '''
    convert [seriesuid	coordX	coordY	coordZ	w	h	d] to 
    'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'
    spacing:[z, y, x]
    '''
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'nodule_type']
    gt_list = []
    csv_file = pd.read_csv(csv_path)
    seriesuid = csv_file['seriesuid']
    coordX, coordY, coordZ = csv_file['coordX'], csv_file['coordY'], csv_file['coordZ']
    w, h, d = csv_file['w'], csv_file['h'], csv_file['d']
    nodule_type = csv_file['nodule_type']
    
    clean_seriesuid = []
    for j in range(seriesuid.shape[0]):
        if seriesuid[j] not in clean_seriesuid: 
            clean_seriesuid.append(seriesuid[j])
        gt_list.append([seriesuid[j], coordX[j], coordY[j], coordZ[j], w[j]/spacing[2], h[j]/spacing[1], d[j]/spacing[0], nodule_type[j]])
    df = pd.DataFrame(gt_list, columns=column_order)
    df.to_csv(annot_save_path, index=False)
    df = pd.DataFrame(clean_seriesuid)
    df.to_csv(series_uids_save_path, index=False, header=None)

def convert_to_standard_output(output: np.ndarray, series_name: str) -> List[List[Any]]:
    """
    convert [id, prob, ctr_z, ctr_y, ctr_x, d, h, w] to
    ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    """
    preds = []
    for j in range(output.shape[0]):
        preds.append([series_name, output[j, 4], output[j, 3], output[j, 2], output[j, 1], output[j, 7], output[j, 6], output[j, 5]])
    return preds

def val(args,
        model: nn.Module,
        detection_postprocess,
        val_loader: DataLoader,
        device: torch.device,
        image_spacing: List[float],
        series_list_path: str,
        exp_folder: str,
        epoch: int = 0,
        batch_size: int = 16,
        nms_keep_top_k: int = 40,
        nodule_type_diameters: Dict[str, Tuple[float, float]] = None,
        min_d: int = 0,
        min_size: int = 0,
        nodule_size_mode: str = 'seg_size') -> Dict[str, float]:
    
    annot_dir = os.path.join(exp_folder, 'annotation')
    os.makedirs(annot_dir, exist_ok=True)
    state = 'val'
    origin_annot_path = os.path.join(annot_dir, 'origin_annotation_{}.csv'.format(state))
    annot_path = os.path.join(annot_dir, 'annotation_{}.csv'.format(state))
    series_uids_path = os.path.join(annot_dir, 'seriesuid_{}.csv'.format(state))
    if min_d != 0:
        logger.info('When validating, ignore nodules with depth less than {}'.format(min_d))
    generate_annot_csv(series_list_path, origin_annot_path, spacing=image_spacing, min_d=min_d, min_size=min_size, mode=nodule_size_mode, nodule_type_diameters=nodule_type_diameters)
    convert_to_standard_csv(csv_path = origin_annot_path, 
                            annot_save_path=annot_path,
                            series_uids_save_path=series_uids_path,
                            spacing = image_spacing)
    
    model.eval()
    split_comber = val_loader.dataset.splitcomb
    all_preds = []
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to validate')
        
    with get_progress_bar('Validation', len(val_loader)) as progress_bar:
        for sample in val_loader:
            crop_images = sample['crop_images'].to(device, non_blocking=True)
            crop_bb_mins = sample['crop_bb_mins']
            nodule_centers = sample['nodule_centers']
            nodule_shapes = sample['nodule_shapes']
            num_splits = sample['num_splits']
            series_names = sample['series_names']
            series_paths = sample['series_paths']
            outputlist = []
            
            for i in range(int(math.ceil(crop_images.size(0) / batch_size))):
                end = (i + 1) * batch_size
                if end > crop_images.size(0):
                    end = crop_images.size(0)
                input = crop_images[i * batch_size:end]
                if args.val_mixed_precision:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            output = model(input)
                            output = detection_postprocess(output, device=device)
                else:
                    with torch.no_grad():
                        output = model(input)
                        output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                outputlist.append(output.data.cpu().numpy())
            
            outputs = np.concatenate(outputlist, 0)
            
            start_idx = 0
            for i in range(len(num_splits)):
                n_split = num_splits[i]
                output = split_comber.combine(outputs[start_idx:start_idx + n_split], crop_bb_mins[i], nodule_centers[i], nodule_shapes[i])
                output = torch.from_numpy(output).view(-1, 8)
                # Remove the padding
                object_ids = output[:, 0] != -1.0
                output = output[object_ids]
                
                # NMS
                if len(output) > 0:
                    keep = nms_3D(output[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
                    output = output[keep]
                output = output.numpy()
            
                preds = convert_to_standard_output(output, series_names[i])  
                all_preds.extend(preds)
                start_idx += n_split
                
            progress_bar.update(1)
    # Save the results to csv
    output_dir = os.path.join(annot_dir, f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    header = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    df = pd.DataFrame(all_preds, columns=header)
    pred_results_path = os.path.join(output_dir, 'predict_epoch_{}.csv'.format(epoch))
    df.to_csv(pred_results_path, index=False)
    
    
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_out, fixed_out, (best_f1_score, best_f1_threshold) = nodule_evaluation(annot_path = annot_path,
                                                                                series_uids_path = series_uids_path, 
                                                                                pred_results_path = pred_results_path,
                                                                                output_dir = output_dir,
                                                                                iou_threshold = args.val_iou_threshold,
                                                                                fixed_prob_threshold=args.val_fixed_prob_threshold)
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out
    
    logger.info('==> Epoch: {}'.format(epoch))
    for i in range(len(sens_points)):
        logger.info('==> fps:{:.3f} iou 0.1 frocs:{:.4f}'.format(FP_ratios[i], sens_points[i]))
    logger.info('==> mean frocs:{:.4f}'.format(np.mean(np.array(sens_points))))
    
    fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fixed_out
    metrics = {'tp': fixed_tp,
                'fp': fixed_fp,
                'fn': fixed_fn,
                'recall': fixed_recall,
                'precision': fixed_precision,
                'f1_score': fixed_f1_score,
                'best_f1_score': best_f1_score,
                'best_f1_threshold': best_f1_threshold}
    mean_recall = np.mean(np.array(sens_points))
    metrics['froc_mean_recall'] = float(mean_recall)
    for fp_ratio, sens_p in zip(FP_ratios, sens_points):
        metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
    
    return metrics