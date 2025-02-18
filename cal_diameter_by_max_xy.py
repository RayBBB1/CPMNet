from scipy.spatial.distance import pdist, squareform
import numpy as np
from dataload.utils import load_label, load_series_list, gen_label_path, ALL_RAD
import cc3d
import os
from multiprocessing import Pool
import pickle
def load_gt_mask_maps(mask_maps_path: str):
    gt_mask_maps = np.load(mask_maps_path)
    # npz
    if mask_maps_path.endswith('.npz'):
        gt_mask_maps = gt_mask_maps['image'] 

    # binarize
    bg_mask = (gt_mask_maps <= 125)
    gt_mask_maps[bg_mask] = 0
    gt_mask_maps[~bg_mask] = 1
    return gt_mask_maps.astype(np.uint8, copy=False)

def get_cc3d(mask_path: str):
    binary_mask_maps = load_gt_mask_maps(mask_path)
    labels = cc3d.connected_components(binary_mask_maps, out_dtype=np.uint32)
    num_nodule = np.unique(labels).shape[0] - 1
    return labels, num_nodule

def get_diameter(mask_path: str):
    cc3d_labels, num_nodule = get_cc3d(mask_path)
    diameters = []
    for i in range(1, num_nodule + 1):
        nodule_mask = (cc3d_labels == i)
        nonzero = np.count_nonzero(nodule_mask, axis=(0, 1))
        if np.sum(nonzero) < 5:
            continue
        max_xy_z = np.argmax(nonzero)
        nodule_mask = nodule_mask[:, :, max_xy_z]
        
        non_zero_coords = np.transpose(np.nonzero(nodule_mask))
        pairwise_distances = squareform(pdist(non_zero_coords))
        max_distance_index = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
        farthest_point_1 = non_zero_coords[max_distance_index[0]]
        farthest_point_2 = non_zero_coords[max_distance_index[1]]
        diameter = np.linalg.norm(farthest_point_1 - farthest_point_2) * 0.8
        
        diameters.append(diameter)
    return diameters

if __name__ == '__main__':
    IMAGE_SPACING = [1.0, 0.8, 0.8]
    series_infos = load_series_list('./data/all.txt')
    ldct_test_list = []
    me_list = []
    for info in series_infos:
        if 'Test' in info[1]:
            ldct_test_list.append(info)
        else:
            me_list.append(info)
            
    ldct_mask_paths = [os.path.join(info[0], 'mask', '{}_crop.npz'.format(info[1])) for info in ldct_test_list]
    me_mask_paths = [os.path.join(info[0], 'mask', '{}_crop.npz'.format(info[1])) for info in me_list]
            
    ldct_label_paths = [gen_label_path(info[0], info[1]) for info in ldct_test_list]
    me_label_paths = [gen_label_path(info[0], info[1]) for info in me_list]

    ldct_labels = [load_label(path, IMAGE_SPACING, min_size=5) for path in ldct_label_paths]
    me_labels = [load_label(path, IMAGE_SPACING, min_size=5) for path in me_label_paths]

    ldct_diameters = []
    me_diameters = []
    
    with Pool(os.cpu_count() // 2) as p:
        ldct_diameters = p.map(get_diameter, ldct_mask_paths)
        # me_diameters = p.map(get_diameter, me_mask_paths)
    
    with open('ldct_diameters.pkl', 'wb') as f:
        pickle.dump(ldct_diameters, f)