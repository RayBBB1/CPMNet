# coding:utf-8
import os
import logging
from typing import Tuple, List, Any, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from .nodule_finding_old import NoduleFinding
from .tools import csvTools

logger = logging.getLogger(__name__)

DYNAMIC_RATIO = [0.7, 1.0, 1.0, 0.7]
# Evaluation settings
PERFORMBOOTSTRAPPING = True
NUMBEROFBOOTSTRAPSAMPLES = 1000
BOTHERNODULESASIRRELEVANT = True
CONFIDENCE = 0.95

SERIESUID = 'seriesuid'
COORDX = 'coordX'
COORDY = 'coordY'
COORDZ = 'coordZ'
WW = 'w'
HH = 'h'
DD = 'd'
NODULE_TYPE = 'nodule_type'
CADProbability_label = 'probability'

# plot settings
FROC_MINX = 0.125 # Mininum value of x-axis of FROC curve
FROC_MAXX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = True

def box_iou_union_3d(boxes1: List[float], boxes2: List[float], eps: float = 0.001) -> float:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.
    
    Args:
        boxes1: boxes [x1, x2, y1, y2, z1, z2]
                boxes2: boxes [x1, x2, y1, y2, z1, z2]
        eps: optional small constant for numerical stability
    """
    vol1 = (boxes1[1] - boxes1[0]) * (boxes1[3] - boxes1[2]) * (boxes1[5] - boxes1[4])
    vol2 = (boxes2[1] - boxes2[0]) * (boxes2[3] - boxes2[2]) * (boxes2[5] - boxes2[4])

    x1 = max(boxes1[0], boxes2[0])
    x2 = min(boxes1[1], boxes2[1])
    y1 = max(boxes1[2], boxes2[2])
    y2 = min(boxes1[3], boxes2[3]) 
    z1 = max(boxes1[4], boxes2[4]) 
    z2 = min(boxes1[5], boxes2[5])

    inter = (max((x2 - x1), 0) * max((y2 - y1), 0) * max((z2 - z1), 0)) + eps
    union = (vol1 + vol2 - inter)
    return inter / union

def dynamic_threshold_wrapper(dynamic_ratio: List[float], fixed_prob_threshold: float) -> float:
    def dynamic_threshold(candidate: NoduleFinding) -> float:
        if candidate == None:
            return fixed_prob_threshold
        if candidate.nodule_type == 'benign':
            return fixed_prob_threshold * dynamic_ratio[0]
        elif candidate.nodule_type == 'probably_benign':
            return fixed_prob_threshold * dynamic_ratio[1]
        elif candidate.nodule_type == 'probably_suspicious':
            return fixed_prob_threshold * dynamic_ratio[2]
        elif candidate.nodule_type == 'suspicious':
            return fixed_prob_threshold * dynamic_ratio[3]
    return dynamic_threshold

def gen_bootstrap_set(scan_to_cands_dict: Dict[str, np.ndarray], seriesUIDs_np: np.ndarray) -> np.ndarray:
    """
    Generates bootstrapped version of set(bootstrapping is sampling method with replacement)
    """
    num_scans = seriesUIDs_np.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_indices = np.random.randint(num_scans, size=num_scans)
    seriesUIDs_rand = seriesUIDs_np[rand_indices]
    
    # get a new list of candidates
    candidatesExists = False
    for series_uid in seriesUIDs_rand:
        if series_uid not in scan_to_cands_dict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scan_to_cands_dict[series_uid])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates, scan_to_cands_dict[series_uid]),axis = 1)

    return candidates

def compute_FROC_bootstrap(FROC_gt_list: List[float],
                          FROC_prob_list: List[float],
                          FROC_series_uids: List[str],
                          seriesUIDs: List[str],
                          FROC_is_FN_list: List[bool],
                          numberOfBootstrapSamples: int = 1000, 
                          confidence = 0.95):
    
    set1 = np.concatenate(([FROC_gt_list], [FROC_prob_list], [FROC_is_FN_list]), axis=0) # 3 x N, N is the number of candidates
    fp_scans_list = []
    sens_list = []
    precision_list = []
    thresholds_list = []
    
    FROC_series_uids_np = np.asarray(FROC_series_uids)
    seriesUIDs_np = np.asarray(seriesUIDs)
    # Make a dict with all candidates of all scans
    scan_to_cands_dict = {}
    for i in range(len(FROC_series_uids_np)):
        series_uid = FROC_series_uids_np[i]
        candidate = set1[:, i:i+1]

        if series_uid not in scan_to_cands_dict:
            scan_to_cands_dict[series_uid] = np.copy(candidate)
        else:
            scan_to_cands_dict[series_uid] = np.concatenate((scan_to_cands_dict[series_uid],candidate),axis = 1)
    
    for i in range(numberOfBootstrapSamples):
        # Generate a bootstrapped set
        btpsamp = gen_bootstrap_set(scan_to_cands_dict, seriesUIDs_np)
        fp_scans, sens, precisions, thresholds = compute_FROC(btpsamp[0,:], btpsamp[1,:],len(seriesUIDs_np),btpsamp[2,:])
    
        fp_scans_list.append(fp_scans)
        sens_list.append(sens)
        precision_list.append(precisions)
        thresholds_list.append(thresholds)

    # compute statistic
    all_fp_scans = np.linspace(FROC_MINX, FROC_MAXX, num=10000) # shape (10000,)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples, len(all_fp_scans)), dtype = 'float32')
    interp_precisions = np.zeros((numberOfBootstrapSamples, len(all_fp_scans)), dtype = 'float32')
    interp_thresholds = np.zeros((numberOfBootstrapSamples, len(all_fp_scans)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fp_scans, fp_scans_list[i], sens_list[i])
        interp_precisions[i,:] = np.interp(all_fp_scans, fp_scans_list[i], precision_list[i])
        interp_thresholds[i,:] = np.interp(all_fp_scans, fp_scans_list[i], thresholds_list[i])
    # compute mean and CI
    sens_mean, sens_lb, sens_up = compute_mean_ci(interp_sens, confidence = confidence)
    prec_mean, prec_lb, prec_up = compute_mean_ci(interp_precisions, confidence = confidence)
    thresholds_mean, thresholds_lb, thresholds_up = compute_mean_ci(interp_thresholds, confidence = confidence)
    
    return (all_fp_scans, thresholds_mean), (sens_mean, sens_lb, sens_up), (prec_mean, prec_lb, prec_up)

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]), dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]), dtype = 'float32') # lower bound
    sens_up   = np.zeros((interp_sens.shape[1]), dtype = 'float32') # upper bound
    
    Pz = (1.0 - confidence) / 2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[int(np.floor(Pz*len(vec)))]
        sens_up[i] = vec[int(np.floor((1.0-Pz)*len(vec)))]

    return sens_mean, sens_lb, sens_up

def compute_FROC(FROC_is_pos_list: List[float], 
                FROC_prob_list: List[float], 
                total_num_of_series: int,
                FROC_is_FN_list: List[bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        FROC_is_pos_list:
            each element is 1 if the sample is positive, 0 otherwise
        FROC_prob_list: 
            each element is the probability of the corresponding sample
        total_num_of_series: 
            total number of series
        FROC_is_FN_list:
            each element is True if the sample is a false negative, False otherwise
    Returns:
        A tuple of (fp_per_scan, sens, precisions, thresholds)
    """
    # Remove FNs
    FROC_is_pos_list_local = []
    FROC_prob_list_local = []
    for i in range(len(FROC_is_FN_list)):
        if FROC_is_FN_list[i] == False:
            FROC_is_pos_list_local.append(FROC_is_pos_list[i])
            FROC_prob_list_local.append(FROC_prob_list[i])
    
    num_of_detected_pos = sum(FROC_is_pos_list_local)
    num_of_gt_pos = sum(FROC_is_pos_list)
    num_of_cand = len(FROC_prob_list_local)
    
    if num_of_detected_pos == 0:
        fp_ratio = np.zeros((5,), dtype=np.float32)
        tp_ratio = np.zeros((5,), dtype=np.float32)
        thresholds = np.array([np.inf, 0.8, 0.4, 0.2, 0.1])
    else:
        fp_ratio, tp_ratio, thresholds = skl_metrics.roc_curve(FROC_is_pos_list_local, FROC_prob_list_local)
    
    # Compute false positive per scan along different thresholds
    if sum(FROC_is_pos_list) == len(FROC_is_pos_list): #  Handle border case when there are no false positives and ROC analysis give nan values.
        fp_per_scans = np.zeros(len(fp_ratio))
    else:
        fp_per_scans = fp_ratio * (num_of_cand - num_of_detected_pos) / total_num_of_series # shape (len(fp_ratio),)
    
    sens = (tp_ratio * num_of_detected_pos) / num_of_gt_pos # sensitivity
    precisions = (tp_ratio * num_of_detected_pos) / np.maximum(1, tp_ratio * num_of_detected_pos + fp_ratio * (num_of_cand - num_of_detected_pos)) # precision
    return fp_per_scans, sens, precisions, thresholds

def evaluateCAD(seriesUIDs: List[str], 
                results_path: str,
                output_dir: str,
                all_gt_nodules: Dict[str, List[NoduleFinding]],
                max_num_of_nodule_candidate_in_series: int = -1,
                iou_threshold = 0.1,
                fixed_prob_threshold = 0.8):
    """
    function to evaluate a CAD algorithm
    """
    nodule_output_file = open(os.path.join(output_dir,'Analysis_{}.txt'.format(iou_threshold)),'w')

    pred_results = csvTools.readCSV(results_path)
    all_pred_cands = {}
    
    # collect candidates from prediction result file
    for series_uid in seriesUIDs:
        nodules = {}
        header = pred_results[0]
        i = 0
        for result in pred_results[1:]:
            nodule_seriesuid = result[header.index(SERIESUID)]
            
            if series_uid == nodule_seriesuid:
                nodule = get_nodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1
        
        # If number of candidate in a series of prediction is larger than max_num_of_nodule_candidate_in_series, keep 
        # the top max_num_of_nodule_candidate_in_series candidates
        if (max_num_of_nodule_candidate_in_series > 0) and (len(nodules.keys()) > max_num_of_nodule_candidate_in_series):
            # sort the candidates by their probability
            sorted_nodules = sorted(nodules.items(), key=lambda x: x[1].CADprobability, reverse=True) 
            
            keep_nodules = dict()
            for i in range(max_num_of_nodule_candidate_in_series):
                keep_nodules[sorted_nodules[i][0]] = sorted_nodules[i][1]
            
            nodules = keep_nodules  
        
        all_pred_cands[series_uid] = nodules  
        
    # open output files
    FN_list_file = open(os.path.join(output_dir, "FN_{}.csv".format(iou_threshold)), 'w')
    FN_list_file.write("seriesuid,coordX,coordY,coordZ,w,h,d\n")

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate
    # initialize some variables to be used in the loop
    tp_count, fp_count, fn_count, tn_count = 0, 0, 0, 0
    total_num_of_cands, total_num_of_nodules = 0, 0
    duplicated_detection_count = 0
    min_prob_value = -10000.0  # minimum value of a float
    FROC_is_pos_list = []
    FROC_prob_list = []
    FROC_series_uids = []
    FROC_is_FN_list = []
    FROC_nodule_list = []
    FROC_candidates_list = []
    # -- loop over the cases
    FN_diameter = []
    FN_seriesuid = []
    for series_uid in seriesUIDs:
        # get the candidates based on the seriesUID
        pred_cands = all_pred_cands.get(series_uid, dict()) # A dict of candidates with key as candidateID and value as NoduleFinding
        total_num_of_cands += len(pred_cands.keys())  
        pred_cands_copy = pred_cands.copy() # make a copy in which items will be deleted

        # get the nodule annotations for this case
        gt_nodules = all_gt_nodules.get(series_uid, list()) # A list of NoduleFinding

        # - loop over each nodule annotation and determine whether it is covered by a candidate
        for gt_nodule in gt_nodules:
            total_num_of_nodules += 1

            x, y, z = float(gt_nodule.coordX), float(gt_nodule.coordY), float(gt_nodule.coordZ)
            w, h, d = float(gt_nodule.w), float(gt_nodule.h), float(gt_nodule.d)
            half_w, half_h, half_d = w/2, h/2, d/2

            nodule_matches = []
            for cand_id, candidate in pred_cands.items():
                cand_x, cand_y, cand_z = float(candidate.coordX), float(candidate.coordY), float(candidate.coordZ)
                cand_w, cand_h, cand_d = float(candidate.w), float(candidate.h), float(candidate.d)
                cand_half_w, cand_half_h, cand_half_d = cand_w/2, cand_h/2, cand_d/2
                
                # [x1, x2, y1, y2, z1, z2]
                pred_box = [cand_x - cand_half_w, cand_x + cand_half_w, 
                            cand_y - cand_half_h, cand_y + cand_half_h, 
                            cand_z - cand_half_d, cand_z + cand_half_d]
                
                gt_box = [x - half_w, x + half_w, 
                          y - half_h, y + half_h, 
                          z - half_d, z + half_d]
                
                iou = box_iou_union_3d(pred_box, gt_box)
                if iou >= iou_threshold:
                    nodule_matches.append(candidate)  
                    if cand_id not in pred_cands_copy.keys():
                        logger.info('This is strange: There are two nodules overlapping with the same prediction, but one of them is already deleted')
                        logger.info('series_uid: %s, coordX: %s, coordY: %s, coordZ: %s, w: %.2f, h: %.2f, d: %.2f' % (series_uid, gt_nodule.coordX, gt_nodule.coordY, gt_nodule.coordZ, gt_nodule.w, gt_nodule.h, gt_nodule.d))
                        logger.info('cand_id: %s, coordX: %s, coordY: %s, coordZ: %s, w: %.2f, h: %.2f, d: %.2f' % (cand_id, candidate.coordX, candidate.coordY, candidate.coordZ, candidate.w, candidate.h, candidate.d))
                    else:
                        del pred_cands_copy[cand_id]
                        
            # There are multiple candidates that overlap with the same ground truth nodule
            if len(nodule_matches) >= 2:  
                duplicated_detection_count += (len(nodule_matches) - 1)  
            
            FROC_is_pos_list.append(1.0) 
            FROC_series_uids.append(series_uid)  
            FROC_nodule_list.append(gt_nodule)
            if len(nodule_matches) > 0: # at least one candidate overlaps with the ground truth nodule
                tp_count += 1  
                # append the sample with the highest probability for the FROC analysis
                max_idx = np.argmax([float(candidate.CADprobability) for candidate in nodule_matches])
                max_prob = nodule_matches[max_idx].CADprobability
                FROC_prob_list.append(float(max_prob))
                FROC_is_FN_list.append(False)
                FROC_candidates_list.append(nodule_matches[max_idx])
                
            else: # no candidate overlaps with the ground truth nodule
                fn_count += 1
                # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                FROC_prob_list.append(min_prob_value)  
                
                FROC_is_FN_list.append(True)  
                # For FN
                FN_list_file.write("%s, %s,%s,%s,%.1f,%.1f,%.1f\n" % (series_uid, gt_nodule.coordX, gt_nodule.coordY, gt_nodule.coordZ, float(gt_nodule.w), float(gt_nodule.h), float(gt_nodule.d)))
                FN_diameter.append([w, h, d])
                FN_seriesuid.append(series_uid)
                FROC_candidates_list.append(None)
        
        # add all false positives to the vectors
        for cand_id, candidate in pred_cands_copy.items(): # the remaining candidates are false positives
            fp_count += 1
            FROC_is_pos_list.append(0.0) 
            FROC_prob_list.append(float(candidate.CADprobability))
            FROC_series_uids.append(series_uid)
            FROC_is_FN_list.append(False)
            FROC_nodule_list.append(candidate)
            FROC_candidates_list.append(candidate)
    # Statistics that are computed
    nodule_output_file.write("Candidate detection results:\n")
    nodule_output_file.write("    True positives: %d\n" % tp_count)
    nodule_output_file.write("    False positives: %d\n" % fp_count)
    nodule_output_file.write("    False negatives: %d\n" % fn_count)
    nodule_output_file.write("    True negatives: %d\n" % tn_count)
    nodule_output_file.write("    Total number of candidates: %d\n" % total_num_of_cands)
    nodule_output_file.write("    Total number of nodules: %d\n" % total_num_of_nodules)

    nodule_output_file.write("    Total number of ignored candidates which were double detections on a nodule: %d\n" % duplicated_detection_count)
    if int(total_num_of_nodules) == 0:
        nodule_output_file.write("    Sensitivity: 0.0\n")
    else:
        nodule_output_file.write("    Sensitivity: %.3f\n" % (float(tp_count) / float(total_num_of_nodules)))
    nodule_output_file.write("    Average number of candidates per scan: %.9f\n" % (float(total_num_of_cands) / float(len(seriesUIDs))))
    nodule_output_file.write("    FN_diammeter:\n")
    for idx, whd in enumerate(FN_diameter):
        nodule_output_file.write("    FN_%d: w:%.1f, h:%.1f, d:%.1f sericeuid: %s\n" % (idx+1, whd[0], whd[1], whd[2], FN_seriesuid[idx]))
    
    fixed_tp, fixed_fp, fixed_fn = 0, 0, 0
    
    classified_metrics = {'benign': [0 ,0, 0],
                        'probably_benign': [0 ,0, 0],
                        'probably_suspicious': [0 ,0, 0],
                        'suspicious': [0 ,0, 0]}
    series_metric = dict()
    logger.info('Fixed threshold: {}'.format(fixed_prob_threshold))
    # dynamic_threshold = dynamic_threshold_wrapper(DYNAMIC_RATIO, fixed_prob_threshold)
    for is_pos, prob, is_fn, nodule, cand, series_uid in zip(FROC_is_pos_list, FROC_prob_list, FROC_is_FN_list, FROC_nodule_list, FROC_candidates_list, FROC_series_uids):
        if series_uid not in series_metric:
            series_metric[series_uid] = [0, 0, 0]
        
        # threshold = dynamic_threshold(cand)
        if is_fn or (is_pos == 1.0 and prob < fixed_prob_threshold):
            fixed_fn += 1
            classified_metrics[nodule.nodule_type][2] += 1
            series_metric[series_uid][2] += 1
        elif is_pos == 1.0 and prob >= fixed_prob_threshold:
            fixed_tp += 1
            classified_metrics[nodule.nodule_type][0] += 1
            series_metric[series_uid][0] += 1
        elif is_pos == 0.0 and prob >= fixed_prob_threshold:
            fixed_fp += 1
            classified_metrics[nodule.nodule_type][1] += 1
            series_metric[series_uid][1] += 1
    
    fixed_recall = fixed_tp / max(fixed_tp + fixed_fn, 1e-6)
    fixed_precision = fixed_tp / max(fixed_tp + fixed_fp, 1e-6)
    fixed_f1_score = (2 * fixed_precision * fixed_recall) / max(fixed_precision + fixed_recall, 1e-6)
    
    template = '{:20s}: Recall={:.3f}, Precision={:.3f}, F1={:.3f}, TP={:4d}, FP={:4d}, FN={:4d}'
    for nodule_type, metrics in classified_metrics.items():
        tp, fp, fn = metrics
        recall = tp / max(tp + fn, 1e-6)
        precision = tp / max(tp + fp, 1e-6)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
        logger.info(template.format(nodule_type, recall, precision, f1_score, tp, fp, fn))
    logger.info(template.format('All', fixed_recall, fixed_precision, fixed_f1_score, fixed_tp, fixed_fp, fixed_fn))
    # Compute metrics for each series
    recall_series = []
    for series_uid, metrics in series_metric.items():
        tp, fp, fn = metrics
        recall = tp / max(tp + fn, 1e-6)
        recall_series.append(recall)
    logger.info('Recall(series_based): {:.3f}'.format(np.mean(recall_series)))
    # compute FROC
    fps, sens, precisions, thresholds = compute_FROC(FROC_is_pos_list = FROC_is_pos_list, 
                                                    FROC_prob_list = FROC_prob_list, 
                                                    total_num_of_series = len(seriesUIDs), 
                                                    FROC_is_FN_list = FROC_is_FN_list)
    
    if PERFORMBOOTSTRAPPING:  # True
        (fps_bs_itp, thresholds_mean), senstitivity_info, precision_info = compute_FROC_bootstrap(FROC_gt_list = FROC_is_pos_list,
                                                                                                FROC_prob_list = FROC_prob_list,
                                                                                                FROC_series_uids = FROC_series_uids,
                                                                                                seriesUIDs = seriesUIDs,
                                                                                                FROC_is_FN_list = FROC_is_FN_list,
                                                                                                numberOfBootstrapSamples = NUMBEROFBOOTSTRAPSAMPLES, 
                                                                                                confidence = CONFIDENCE)
        sens_bs_mean, sens_bs_lb, sens_bs_up = senstitivity_info
        prec_bs_mean, prec_bs_lb, prec_bs_up = precision_info
        f1_score_mean = 2 * prec_bs_mean * sens_bs_mean / np.maximum(1e-6, prec_bs_mean + sens_bs_mean)
        
        best_f1_index = np.argmax(f1_score_mean)
        best_f1_threshold = thresholds_mean[best_f1_index]
        best_f1_sens = sens_bs_mean[best_f1_index]
        best_f1_prec = prec_bs_mean[best_f1_index]
        best_f1_score = f1_score_mean[best_f1_index]
        logger.info('Best F1 score: {:.4f} at threshold: {:.3f}, Sens: {:.3f}, Prec: {:.3f}'.format(best_f1_score, best_f1_threshold, best_f1_sens, best_f1_prec))
        # Write FROC curve
        with open(os.path.join(output_dir, "froc_{}.txt".format(iou_threshold)), 'w') as f:
            f.write("FPrate,Sensivity,Precision,f1_score,Threshold\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.5f,%.5f,%.5f,%.5f,%.5f\n" % (fps_bs_itp[i], sens_bs_mean[i], prec_bs_mean[i], f1_score_mean[i], thresholds_mean[i]))
    # Write FROC vectors to disk as well
    with open(os.path.join(output_dir, "froc_gt_prob_vectors_{}.csv".format(iou_threshold)), 'w') as f:
        f.write("is_pos, prob\n")
        for i in range(len(FROC_is_pos_list)):
            f.write("%d,%.4f\n" % (FROC_is_pos_list[i], FROC_prob_list[i]))

    fps_itp = np.linspace(FROC_MINX, FROC_MAXX, num=10001)
    
    sens_itp = np.interp(fps_itp, fps, sens)
    prec_itp = np.interp(fps_itp, fps, precisions)
    
    sens_points = []
    prec_points = []
    
    if PERFORMBOOTSTRAPPING: # True
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(output_dir, "froc_bootstrapping_{}.csv".format(iou_threshold)), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.5f,%.5f,%.5f,%.5f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
            FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            total_sens = 0
            nodule_output_file.write('-'*20 + '\n')
            nodule_output_file.write("FP/Scan, Sensitivity, Precision\n")
            for fp_point in FPS:
                index = np.argmin(abs(fps_bs_itp - fp_point))
                nodule_output_file.write('{:.3f}, {:.3f}, {:.3f}\n'.format(fp_point, sens_bs_mean[index], prec_bs_mean[index]))
                sens_points.append(sens_bs_mean[index])
                prec_points.append(prec_bs_mean[index])
                total_sens += sens_bs_mean[index]
            nodule_output_file.write("\n")
            nodule_output_file.write("Froc_mean = {:.2f}\n".format(total_sens / len(FPS)))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(total_num_of_nodules) > 0:
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, lw=2)
        if PERFORMBOOTSTRAPPING:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_MINX
        xmax = FROC_MAXX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.title('FROC performance')
        
        if bLogPlot:
            plt.xscale('log')
            ax.xaxis.set_major_locator(plt.FixedLocator([0.125,0.25,0.5,1,2,4,8]))
            ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "froc_{}.png".format(iou_threshold)), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points), (fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score), (best_f1_score, best_f1_threshold)
    
def get_nodule(annot: List[Any], 
               header: List[str]) -> NoduleFinding:
    nodule = NoduleFinding()
    nodule.coordX = annot[header.index(COORDX)]
    nodule.coordY = annot[header.index(COORDY)]
    nodule.coordZ = annot[header.index(COORDZ)]

    nodule.w = float(annot[header.index(WW)])
    nodule.h = float(annot[header.index(HH)])
    nodule.d = float(annot[header.index(DD)])
    
    if NODULE_TYPE in header:
        nodule.nodule_type = annot[header.index(NODULE_TYPE)]
    else:
        nodule.auto_nodule_type()
    if CADProbability_label in header:
        nodule.CADprobability = annot[header.index(CADProbability_label)]
    
    return nodule

def collect_nodule_annotations(annotations: List[List[Any]],
                               seriesUIDs: List[str]) -> Dict[str, List[NoduleFinding]]:
    """Collects all nodule annotations from the annotations file and returns them in a dictionary
    
    Args:
        annotations: list of annotations
        annotations_excluded: list of annotations that are excluded from analysis
        seriesUIDs: list of CT images in seriesuids
    Returns:
        Dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    """
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        nodules = []
        numberOfIncludedNodules = 0
        
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(SERIESUID)]
            
            if seriesuid == nodule_seriesuid:
                nodule = get_nodule(annotation, header)
                nodules.append(nodule)
                numberOfIncludedNodules += 1
            
        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)
    
    # logger.info('Total number of included nodule annotations: {}'.format(noduleCount))
    logger.info('Total number of nodule annotations: {}'.format(noduleCountTotal))
    return allNodules
    
def collect(annot_path: str, 
            seriesuids_path: str) -> Tuple[Dict[str, List[NoduleFinding]], List[str]]:
    """Collects all nodule annotations from the annotations file and returns them in a dictionary
    Args:
        annot_path: path to annotations file
        seriesuids_path: path to seriesuids file
    Returns:
        A tuple of:
        - Dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
        - List of seriesuids
    """
    annotations = csvTools.readCSV(annot_path) 
    seriesUIDs_csv = csvTools.readCSV(seriesuids_path)
    
    seriesUIDs = []
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])

    allNodules = collect_nodule_annotations(annotations, seriesUIDs)  
    return allNodules, seriesUIDs
    
def nodule_evaluation(annot_path: str,
                    series_uids_path: str,
                    pred_results_path: str,
                    output_dir: str,
                    iou_threshold: float,
                    fixed_prob_threshold: float = 0.8,
                    max_num_of_nodule_candidate_in_series: int = 100):
    """
    function to load annotations and evaluate a CAD algorithm
    Args:
        annot_path: path to annotations file
        seriesuids_path: path to seriesuids file
        pred_results_path: path to prediction results file
        output_dir: output directory
        iou_threshold: iou threshold
    """
    all_gt_nodules, seriesUIDs = collect(annot_path, series_uids_path)
    
    out, fixed_out, (best_f1_score, best_f1_threshold) = evaluateCAD(seriesUIDs = seriesUIDs, 
                                                                    results_path = pred_results_path, 
                                                                    output_dir = output_dir, 
                                                                    all_gt_nodules = all_gt_nodules,
                                                                    max_num_of_nodule_candidate_in_series = max_num_of_nodule_candidate_in_series, 
                                                                    fixed_prob_threshold=fixed_prob_threshold,
                                                                    iou_threshold = iou_threshold)
    return out, fixed_out, (best_f1_score, best_f1_threshold)