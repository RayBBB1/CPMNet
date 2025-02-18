# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging

from networks.ResNet_3D_CPM import Resnet18, DetectionPostprocess
### data ###
from dataload.dataset import DetDataset
from dataload.utils import get_image_padding_value
from dataload.collate import infer_collate_fn
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import transform as transform

from logic.val_original import val
from logic.utils import load_model, get_memory_format

from utils.logs import setup_logging
from utils.utils import init_seed, write_yaml
from config import SAVE_ROOT, DEFAULT_OVERLAP_RATIO, IMAGE_SPACING, NODULE_TYPE_DIAMETERS

logger = logging.getLogger(__name__)
def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 2)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[96, 96, 96], help='crop size')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO, help='overlap ratio')
    parser.add_argument('--model_path', type=str, default='')
    # data
    parser.add_argument('--val_set', type=str, default='./data/all_client_test.txt', help='val_list')
    parser.add_argument('--min_d', type=int, default=0, help="min depth of nodule, if some nodule's depth < min_d, it will be` ignored")
    parser.add_argument('--min_size', type=int, default=5, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    parser.add_argument('--data_norm_method', type=str, default='none', help='normalize method, mean_std or scale or none')
    parser.add_argument('--memory_format', type=str, default='channels_first')
    # hyper-parameters
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.65, help='fixed probability threshold for validation')
    # detection-hyper-parameters
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_threshold', type=float, default=0.15, help='detection threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    # other
    parser.add_argument('--nodule_size_mode', type=str, default='seg_size', help='nodule size mode, seg_size or dhw')
    parser.add_argument('--max_workers', type=int, default=4, help='max number of workers, num_workers = min(batch_size, max_workers)')
    args = parser.parse_args()
    return args

def prepare_validation(args, device):
    detection_postprocess = DetectionPostprocess(topk=args.det_topk, 
                                                 threshold=args.det_threshold, 
                                                 nms_threshold=args.det_nms_threshold,
                                                 nms_topk=args.det_nms_topk,
                                                 crop_size=args.crop_size)
    # load model
    logger.info('Load model from "{}"'.format(args.model_path))
    model = load_model(args.model_path)
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    model = model.to(device = device, memory_format=memory_format)
    return model, detection_postprocess

def val_data_prepare(args):
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * args.overlap_ratio) for i in range(len(crop_size))]
    pad_value = get_image_padding_value(args.data_norm_method, use_water=False)
    
    logger.info('Crop size: {}, overlap size: {}'.format(crop_size, overlap_size))
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value, do_padding=False)
    val_dataset = DetDataset(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(args.batch_size, args.max_workers) , collate_fn=infer_collate_fn, pin_memory=True)
    
    logger.info("There are {} samples in the val set".format(len(val_loader.dataset)))
    return val_loader

if __name__ == '__main__':
    args = get_args()
    
    base_exp_folder = os.path.join(os.path.dirname(args.model_path), 'val_results')
    setup_logging(log_file=os.path.join(base_exp_folder, 'val.log'))
    
    if '*' not in args.model_path: # validation all models in the folder
        model_paths = [args.model_path]
    else:
        model_folder = os.path.dirname(args.model_path)
        model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pth')]
        
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        exp_folder = os.path.join(base_exp_folder, model_name)
        args.model_path = model_path
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, detection_postprocess = prepare_validation(args, device)
        init_seed(args.seed)
        
        val_loader = val_data_prepare(args)
        
        write_yaml(os.path.join(exp_folder, 'val_config.yaml'), args)
        logger.info('Save validation results to "{}"'.format(exp_folder))
        logger.info('Val set: "{}"'.format(args.val_set))
        metrics = val(args = args,
                    model = model,
                    detection_postprocess=detection_postprocess,
                    val_loader = val_loader, 
                    device = device,
                    image_spacing = IMAGE_SPACING,
                    series_list_path=args.val_set,
                    exp_folder=exp_folder,
                    nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                    min_d=args.min_d,
                    min_size=args.min_size,
                    nodule_size_mode=args.nodule_size_mode)
        
        with open(os.path.join(exp_folder, 'val_metrics.txt'), 'w') as f:
            for k, v in metrics.items():
                f.write('{}: {}\n'.format(k, v))