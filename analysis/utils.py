from dataload.utils import load_series_list, gen_dicom_path, gen_label_path, ALL_CLS, ALL_RAD, ALL_LOC
from evaluationScript.nodule_finding import NoduleFinding
from collections import defaultdict

from typing import List, Dict, Tuple

def str2bool(v):
    return v.lower() in ('true', '1')

def pred2nodulefinding(line: str) -> NoduleFinding:
    pred = line.strip().split(',')
    if len(pred) == 11: # no ground truth
        series_name, x, y, z, w, h, d, prob, nodule_type, match_iou, is_gt = pred
        gt_x = None
    else:
        series_name, x, y, z, w, h, d, prob, nodule_type, match_iou, is_gt, gt_x, gt_y, gt_z, gt_w, gt_h, gt_d = pred
    is_gt = str2bool(is_gt)
    nodule = NoduleFinding(series_name, x, y, z, w, h, d, nodule_type, prob, is_gt)
    if gt_x is not None:
        gt_nodule = NoduleFinding(series_name, gt_x, gt_y, gt_z, gt_w, gt_h, gt_d, nodule_type, prob, is_gt)
        nodule.set_match(match_iou, gt_nodule)
    return nodule
        
def load_predictions(pred_path: str ,series_list = './data/all_client_test.txt') -> Tuple[Dict[str, List[NoduleFinding]], List[NoduleFinding]]:
    series_list = load_series_list(series_list)
    dicom_paths = dict()

    for folder, series_name in series_list:
        dicom_path = gen_dicom_path(folder, series_name)
        dicom_paths[series_name] = dicom_path

    with open(pred_path, 'r') as f:
        lines = f.readlines()[1:]

    series_predictions = defaultdict(list)
    nodules = []
    for line in lines:
        info = line.strip().split(',')
        series_name = info[0]
        
        nodule_finding = pred2nodulefinding(line)
        series_predictions[series_name].append(nodule_finding)
        nodules.append(nodule_finding)
        
    return series_predictions, nodules