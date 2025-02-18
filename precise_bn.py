# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,6,16]

import logging
import argparse
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch
import numpy as np
from torch import nn
import transform as transform
import torchvision

from config import IMAGE_SPACING, DEFAULT_OVERLAP_RATIO
from logic.utils import load_model, save_states
from dataload.crop_fast import InstanceCrop
from dataload.dataset import TrainDataset
from torch.utils.data import DataLoader
from dataload.utils import get_image_padding_value, load_series_list
from dataload.collate import train_collate_fn

from utils.utils import init_seed, get_progress_bar
from utils.logs import setup_logging

# pyre-fixme[9]: BN_MODULE_TYPES has type `Tuple[Type[Module]]`; used as
#  `Tuple[Type[BatchNorm1d], Type[BatchNorm2d], Type[BatchNorm3d],
#  Type[SyncBatchNorm]]`.
BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

logger = logging.getLogger(__name__)

class _PopulationVarianceEstimator:
    """
    Alternatively, one can estimate population variance by the sample variance
    of all batches combined. This needs to use the batch size of each batch
    in this function to undo the bessel-correction.
    This produces better estimation when each batch is small.
    See Appendix of the paper "Rethinking Batch in BatchNorm" for details.

    In this implementation, we also take into account varying batch sizes.
    A batch of N1 samples with a mean of M1 and a batch of N2 samples with a
    mean of M2 will produce a population mean of (N1M1+N2M2)/(N1+N2) instead
    of (M1+M2)/2.
    """

    def __init__(self, mean_buffer: torch.Tensor, var_buffer: torch.Tensor) -> None:
        self.pop_mean: torch.Tensor = torch.zeros_like(mean_buffer)
        self.pop_square_mean: torch.Tensor = torch.zeros_like(var_buffer)
        self.tot = 0

    def update(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_size: int
    ) -> None:
        self.tot += batch_size
        batch_square_mean = batch_mean.square() + batch_var * (
            (batch_size - 1) / batch_size
        )
        self.pop_mean += (batch_mean - self.pop_mean) * (batch_size / self.tot)
        self.pop_square_mean += (batch_square_mean - self.pop_square_mean) * (
            batch_size / self.tot
        )

    @property
    def pop_var(self) -> torch.Tensor:
        return self.pop_square_mean - self.pop_mean.square()

@torch.no_grad()
def update_bn_stats(
    model: nn.Module,
    data_loader: Iterable[Any],
    device,
    batch_size = 16) -> None:
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    See Sec. 3 of the paper "Rethinking Batch in BatchNorm" for details.

    Args:
        model (nn.Module): the model whose bn stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.

            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        progress: None or "tqdm". If set, use tqdm to report the progress.
    """
    model.train()
    bn_layers = get_bn_modules(model)
    if len(bn_layers) == 0:
        return
    logger.info(f"Computing precise BN statistics for {len(bn_layers)} BN layers ...")

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    batch_size_per_bn_layer: Dict[nn.Module, int] = {}

    def get_bn_batch_size_hook(
        module: nn.Module, input: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        assert (module not in batch_size_per_bn_layer), "Some BN layers are reused. This is not supported and probably not desired."
        x = input[0]
        assert isinstance(x, torch.Tensor), f"BN layer should take tensor as input. Got {input}"
        # consider spatial dimensions as batch as well
        batch_size = x.numel() // x.shape[1]
        batch_size_per_bn_layer[module] = batch_size
        return (x,)

    hooks_to_remove = [
        bn.register_forward_pre_hook(get_bn_batch_size_hook) for bn in bn_layers
    ]

    estimators = [
        _PopulationVarianceEstimator(bn.running_mean, bn.running_var)
        for bn in bn_layers
    ]

    with get_progress_bar('Precise BN', len(data_loader)) as progress_bar:
        for iter, sample in enumerate(data_loader):
            images = sample['image'] # z, y, x
            # labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
            for i in range(int(math.ceil(images.size(0) / batch_size))):
                batch_size_per_bn_layer.clear()
                end = (i + 1) * batch_size
                if end > images.size(0):
                    end = images.size(0)
                
                image = images[i * batch_size:end].to(device, non_blocking=True)
                model(image)
                for i, bn in enumerate(bn_layers):
                    # Accumulates the bn stats.
                    bs = batch_size_per_bn_layer.get(bn, None)
                    if bs is None:
                        continue  # the layer was unused in this forward
                    estimators[i].update(bn.running_mean, bn.running_var, bs)
            progress_bar.update(1)
            
    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = estimators[i].pop_mean
        bn.running_var = estimators[i].pop_var
        bn.momentum = momentum_actual[i]
        
    for hook in hooks_to_remove:
        hook.remove()

def get_bn_modules(model: nn.Module) -> List[nn.Module]:
    """
    Find all BatchNorm (BN) modules that are in training mode. See
    fvcore.precise_bn.BN_MODULE_TYPES for a list of all modules that are
    included in this search.

    Args:
        model (nn.Module): a model possibly containing BN modules.

    Returns:
        list[nn.Module]: all BN modules in the model.
    """
    # Finds all the bn layers.
    bn_layers = [m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)]
    return bn_layers

def build_train_augmentation(args, crop_size: Tuple[int, int, int], pad_value: int, blank_side: int):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
        
    # transform_list_train = [transform.RandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
    # transform_list_train.append(transform.RandomRotate90(p=0.5, rot_xy=True, rot_xz=rot_zx, rot_yz=rot_zy))
    transform_list_train = []
    if args.use_crop:
        transform_list_train.append(transform.RandomCrop(p=0.3, crop_ratio=0.95, ctr_margin=10, pad_value=pad_value))
        
    transform_list_train.append(transform.CoordToAnnot())
                            
    logger.info('Augmentation: random flip: True, random rotation: {}, random crop: {}'.format([True, rot_zy, rot_zx], args.use_crop))
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--train_set', type=str, default='./data/pretrained_train.txt', help='path to the training set list')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[96, 96, 96], help='crop size for training')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO, help='overlap ratio for cropping')
    
    parser.add_argument('--tp_ratio', type=float, default=0.75, help='ratio of positive samples in a crop')
    parser.add_argument('--tp_iou', type=float, default=0.7, help='IoU threshold for positive samples')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size for training')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples in a crop')
    
    parser.add_argument('--min_d', type=int, default=0, help='minimum distance between two instances')
    parser.add_argument('--min_size', type=int, default=27, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    parser.add_argument('--use_bg', action='store_true', help='use background samples')
    
    parser.add_argument('--data_norm_method', type=str, default='none', help='data normalization method')
    parser.add_argument('--rand_rot', nargs='+', type=int, default=[0, 0, 0], help='random rotate')
    parser.add_argument('--use_crop', action='store_true', help='use random crop augmentation')
    
    parser.add_argument('--max_workers', type=int, default=4, help='max number of workers, num_workers = min(batch_size, max_workers)')
    parser.add_argument('--model_path', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    setup_logging()
    
    args = get_args()
    crop_size = args.crop_size
    overlap_size = (np.array(crop_size) * args.overlap_ratio).astype(np.int32).tolist()
    pad_value = get_image_padding_value(args.data_norm_method)
    rand_trans = [int(s * 2/3) for s in overlap_size]
    
    logger.info('Crop size: {}, overlap size: {}, pad value: {}, tp_ratio: {:.3f}'.format(crop_size, overlap_size, pad_value, args.tp_ratio))
    train_transform = build_train_augmentation(args, crop_size, pad_value, blank_side = 0)
    
    crop_fn_train = InstanceCrop(crop_size=crop_size, overlap_ratio=args.overlap_ratio, tp_ratio=args.tp_ratio, rand_trans=rand_trans, rand_rot=args.rand_rot, sample_num=args.num_samples, instance_crop=True)
    
    train_dataset = TrainDataset(series_list_path = args.train_set, crop_fn = crop_fn_train, image_spacing=IMAGE_SPACING, transform_post = train_transform, 
                                 min_d=args.min_d, min_size = args.min_size, use_bg = args.use_bg, norm_method=args.data_norm_method)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_collate_fn, 
                              num_workers=min(args.batch_size, args.max_workers), pin_memory=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(args.model_path)
    model.detection_loss = None
    model = model.to(device)
    logger.info('Number of samples: {}'.format(len(train_dataset) * args.num_samples))
    update_bn_stats(model, train_loader, device)
    
    save_path = os.path.join(os.path.dirname(args.model_path), 'precise_bn.pth')
    save_states(save_path, model)