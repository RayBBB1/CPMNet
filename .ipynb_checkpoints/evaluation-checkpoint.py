from evaluationScript.eval_old import nodule_evaluation
import numpy as np
import argparse
import os
import logging
from utils.logs import setup_logging
from utils.generate_annot_csv_from_series_list import generate_annot_csv
from config import IMAGE_SPACING
import pandas as pd
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # Rraining settings
    # parser.add_argument('--annot_path', type=str, required=True, help='annotation_val.csv')
    parser.add_argument('--series_uids_path', type=str, required=True, help='seriesuid_val.csv')
    parser.add_argument('--pred_results_path', type=str, required=True, help='predict.csv')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fixed_prob_threshold', type=float, default=0.65)
    parser.add_argument('--min_d', type=int, default=0, help="min depth of nodule, if some nodule's depth < min_d, it will be` ignored")
    parser.add_argument('--nodule_size_mode', type=str, default='seg_size', help='nodule size mode, seg_size or dhw')
    parser.add_argument('--min_size', type=int, default=27, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    parser.add_argument('--val_set', type=str, default='E:/Jack/Me_dataset_dicom_resize_npy/test/data_list_crop.txt', help='val_list')
    parser.add_argument('--origanl_annotation_path', type=str)
    args = parser.parse_args()
    return args

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
        # gt_list.append([seriesuid[j], coordX[j], coordY[j], coordZ[j], w[j], h[j], d[j], nodule_type[j]])
    df = pd.DataFrame(gt_list, columns=column_order)
    df.to_csv(annot_save_path, index=False)
    df = pd.DataFrame(clean_seriesuid)
    df.to_csv(series_uids_save_path, index=False, header=None)

if __name__ == '__main__':
    args = get_args()
    exp_folder = args.output_dir
    exp_folder = os.path.join(exp_folder, 'val_temp')
    annot_dir = os.path.join(exp_folder, 'annotation')
    os.makedirs(annot_dir, exist_ok=True)
    state = 'val'
    # origin_annot_path = os.path.join(annot_dir, 'origin_annotation_{}.csv'.format(state))
    origin_annot_path = args.origanl_annotation_path
    annot_path = os.path.join(annot_dir, 'annotation_{}.csv'.format(state))
    series_uids_path = os.path.join(annot_dir, 'seriesuid_{}.csv'.format(state))
    if args.min_d != 0:
        logger.info('When validating, ignore nodules with depth less than {}'.format(args.min_d))
    if args.min_size != 0:
        logger.info('When validating, ignore nodules with size less than {}'.format(args.min_size))
    # generate_annot_csv(args.val_set, annot_path, spacing=IMAGE_SPACING, min_d=args.min_d, mid_size=args.min_size, mode=args.nodule_size_mode)
    # convert_to_standard_csv(csv_path = annot_path, 
    #                         annot_save_path=annot_path,
    #                         series_uids_save_path=series_uids_path,
    #                         spacing = IMAGE_SPACING)

    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8.]
    # annot_path = args.annot_path
    series_uids_path = args.series_uids_path
    pred_results_path = args.pred_results_path
    output_dir = args.output_dir
    setup_logging(level='info', log_file=os.path.join(output_dir, 'log.txt'))
    
    froc_out, fixed_out, (best_f1_score, best_f1_threshold) = nodule_evaluation(annot_path = annot_path,
                                                                                series_uids_path = series_uids_path, 
                                                                                pred_results_path = pred_results_path,
                                                                                output_dir = output_dir,
                                                                                iou_threshold = 0.1,
                                                                                fixed_prob_threshold=args.fixed_prob_threshold)
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out



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
