import os
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from dataload.utils import load_label, gen_label_path, gen_dicom_path, ALL_CLS, ALL_RAD, ALL_LOC, ALL_PROB, load_image, load_series_list
from evaluationScript.nodule_finding import NoduleFinding
from visualize.draw import draw_bbox_on_image, draw_pseu_bbox_and_label_on_image
from visualize.convert import noduleFinding2cude, gtNoduleFinding2cube
from analysis.utils import pred2nodulefinding
from utils.utils import get_progress_bar

def write_csv(series_nodules: Dict[str, NoduleFinding], save_path: str):
    header = ['series_name', 'nodule_idx', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'probability']
    header = ','.join(header)
    lines = []
    for series_name, nodule_findings in series_nodules.items():
        for i, n in enumerate(nodule_findings):
            lines.append(f'{series_name},{i},{n.ctr_x},{n.ctr_y},{n.ctr_z},{n.w},{n.h},{n.d},{n.prob}')
    with open(save_path, 'w') as f:
        f.write(header + '\n')
        f.write('\n'.join(lines))    
            
def get_args():
    parser = argparse.ArgumentParser(description='Visualize Hard False Positive')
    parser.add_argument('--val_set', type=str, default='./data/all.txt')
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
   
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--bbox_offset', type=int, default=0)
    
    parser.add_argument('--no_fp', action='store_true', default=False)
    parser.add_argument('--tp', action='store_true', default=False)
    parser.add_argument('--fn', action='store_true', default=False)
    
    parser.add_argument('--hard_FP_thresh', type=float, default=0.7)
    parser.add_argument('--half_image', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    hard_FP_save_folder = os.path.join(args.save_folder, 'hard_FP')
    TP_save_folder = os.path.join(args.save_folder, 'TP')
    FN_save_folder = os.path.join(args.save_folder, 'FN')
    
    os.makedirs(hard_FP_save_folder, exist_ok=True)
    os.makedirs(TP_save_folder, exist_ok=True)
    os.makedirs(FN_save_folder, exist_ok=True)
    series_infos = load_series_list(args.val_set)
    
    # Get series name to folder mapping
    series_names_to_folder = {s[1]: s[0] for s in series_infos}
    series_names = [s[1] for s in series_infos]
    
    # Load all nodule findings
    with open(args.pred_path, 'r') as f:
        lines = f.readlines()[1:] # skip header
    nodules = [pred2nodulefinding(line) for line in lines]
    
    # Draw hard false positive nodules
    if not args.no_fp:
        hard_FP_nodules = defaultdict(list)
        for n in nodules:
            if n.prob >= args.hard_FP_thresh and not n.is_gt:
                hard_FP_nodules[n.series_name].append(n)
        with get_progress_bar('Visualizing Hard FP', len(hard_FP_nodules)) as pbar:
            for series_name, nodule_findings in hard_FP_nodules.items():
                img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
                image = (load_image(img_path) * 255).astype(np.uint8)
                for i, nodule in enumerate(nodule_findings):
                    save_path = os.path.join(hard_FP_save_folder, f'{series_name}_{i}.png')
                    bboxes = noduleFinding2cude([nodule], image.shape)
                    prob = nodule.prob
                    extra_sup_title = '{}, Prob: {:.2f}'.format(series_name, prob)
                    draw_bbox_on_image(image, bboxes, (255, 0, 0), half_image=args.half_image, save_path=save_path, extra_sup_title=extra_sup_title, offset=args.offset, bbox_offset=args.bbox_offset)
                pbar.update(1)
        write_csv(hard_FP_nodules, os.path.join(hard_FP_save_folder, 'hard_FP.csv'))
            
    # Draw TP nodules
    if args.tp:
        TP_nodules = defaultdict(list)
        for n in nodules:
            if n.is_gt and n.prob > 0:
                TP_nodules[n.series_name].append(n)
                
        with get_progress_bar('Visualizing TP', len(TP_nodules)) as pbar:
            for series_name, nodule_findings in TP_nodules.items():
                img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
                image = (load_image(img_path) * 255).astype(np.uint8)
                for i, nodule in enumerate(nodule_findings):
                    save_path = os.path.join(TP_save_folder, f'{series_name}_{i}.png')
                    pred_bboxes, gt_bboxes = gtNoduleFinding2cube([nodule], image.shape)
                    draw_pseu_bbox_and_label_on_image(image, gt_bboxes, pred_bboxes, save_path=save_path, half_image=args.half_image, extra_sup_title=series_name)
                pbar.update(1)
            
    # Draw FN nodules
    if args.fn:
        FN_nodules = defaultdict(list)
        for n in nodules:
            if n.is_gt and n.prob == -1:
                FN_nodules[n.series_name].append(n)
            
        with get_progress_bar('Visualizing FN', len(FN_nodules)) as pbar:
            for series_name, nodule_findings in FN_nodules.items():
                if series_name == 'CHESTCT_Test0814': # bad series
                    pbar.update(1)
                    continue
                img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
                image = (load_image(img_path) * 255).astype(np.uint8)
                for i, nodule in enumerate(nodule_findings):
                    save_path = os.path.join(FN_save_folder, f'{series_name}_{i}.png')
                    bboxes = noduleFinding2cude([nodule], image.shape)
                    draw_bbox_on_image(image, bboxes, (0, 255, 0), half_image=args.half_image, save_path=save_path, extra_sup_title=series_name, offset=args.offset, bbox_offset=args.bbox_offset)
                pbar.update(1)