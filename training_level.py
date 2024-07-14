import numpy as np
from scipy.optimize import linear_sum_assignment

def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / union

if __name__ == "__main__":


    #load sample data
    
    alpha0 = 0.5
    alpha1 = 0.6
    alpha2 = 0.8

    import pickle
    with open("data/sample_data.pkl", 'rb') as f:
        data = pickle.load(f)

    img_box = data["img_box"]   # (N*4)  (y_min, x_min, y_max, x_max) 
    proj_lidar_box = data["proj_lidar_box"]  # (N*4) projected 2d box from 3d box   
    img_score = data["img_score"] # (N*1)   
    proj_lidar_score = data["proj_lidar_score"] # (N*1)

    ious = iou(proj_lidar_box, img_box)
    row_ind, col_ind = linear_sum_assignment(-ious)
        
    #(row_ind, col_ind) = match_projected_idx, match_img_idx
    new_col_ind = []
    for i in range(len(row_ind)):
        if ious[row_ind[i],col_ind[i]] > alpha0:
            new_col_ind.append(col_ind[i])

    index_list =[] 

    for i in range(len(row_ind)):
        if (proj_lidar_score[row_ind[i]] + img_score[col_ind[i]])/2. > alpha1:
            index_list.append(row_ind[i])

    #unmatched but with high confidence score
    for i in range(len(proj_lidar_score)):
        if i in index_list:
            continue
        if proj_lidar_score[i] > alpha2:
            index_list.append(i)

    #final kept index for lidar box
    print(index_list)
    
