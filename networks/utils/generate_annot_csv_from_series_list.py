import json
import os
import math
import logging
import numpy as np
from dataload.utils import load_series_list, load_label, gen_label_path, ALL_CLS, ALL_LOC, ALL_RAD, NODULE_SIZE
import argparse
from typing import List, Dict, Tuple
from evaluationScript.nodule_typer import NoduleTyper
logger = logging.getLogger(__name__)

def generate_series_uids_csv(series_list_path: str, save_path: str):
    series_infos = load_series_list(series_list_path)
    header = 'seriesuid\n'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(header)
        for folder, series_name in series_infos:
            f.write(series_name + '\n')

def generate_annot_csv(series_list_path: str,
                       save_path: str,
                       nodule_type_diameters : Dict[str, Tuple[float, float]] = None,
                       spacing: List[float] = None,
                       mode = 'seg_size',
                       min_d: int = 0,
                       min_size: int = 6):
    spacing = np.array(spacing)
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'nodule_type']
    
    nodule_typer = NoduleTyper(nodule_type_diameters, spacing)
    all_locs = []
    all_rads = []
    all_types = []
    series_infos = load_series_list(series_list_path)
    for series_info in series_infos:
        folder = series_info[0]
        series_name = series_info[1]
        
        label_path = gen_label_path(folder, series_name)
        label = load_label(label_path, spacing, min_d, min_size)
        all_locs.append(label[ALL_LOC].tolist())
        all_rads.append(label[ALL_RAD].tolist())
        
        if mode == 'seg_size':
            all_types.append([nodule_typer.get_nodule_type_by_seg_size(s) for s in label[NODULE_SIZE]])
        elif mode == 'dhw':
            dhws = label[ALL_RAD]
            if len(dhws) == 0:
                all_types.append([])
                continue
            dhws = np.array(dhws) / spacing
            all_types.append([nodule_typer.get_nodule_type_by_dhw(d, h, w) for d, h, w in dhws])
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(','.join(column_order) + '\n')
        for series_i in range(len(series_infos)):
            for loc, rad, nodule_type in zip(all_locs[series_i], all_rads[series_i], all_types[series_i]):
                z, y, x = loc
                d, h, w = rad
                series_name = series_infos[series_i][1]
                row = [series_name]
                
                for value in [x, y, z, w, h, d]:
                    row.append('{:.2f}'.format(value))
                
                row.append(nodule_type)
                f.write(','.join(row) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_list_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./annotation.csv')
    args = parser.parse_args()
    series_list_path = args.series_list_path
    save_path = args.save_path
    generate_annot_csv(series_list_path, save_path)
    