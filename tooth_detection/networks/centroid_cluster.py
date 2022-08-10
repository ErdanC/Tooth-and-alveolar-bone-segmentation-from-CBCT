import numpy as np
import torch.nn.functional as F

def fast_clsuter(seg, off):
    """
    implementation of the paper 'Clustering by fast search and find of density peaks'
    Args:
    bn_seg: predicted binary segmentation results -> (batch_size, 2, 120, 120, 120)
    off: predicted offset of x. y, z -> (batch_size, 3, 120, 120, 120)
    Returns:
    The centroids obtained from the cluster algorithm
    """
    centroids = np.array([])
    
    seg = seg[0, 0, :, :, :].cpu().data.numpy()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    off = off[0, :, :, :, :].cpu().data.numpy()
    
    # generate the voting map based on the binary segmentation and offset
    voting_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))
    num_fg = coord.shape[1]
    if num_fg < 1e4:
        return centroids
    coord = coord + off[:, seg == 1]
    coord = coord.astype(np.int)
    coord, coord_count  = np.unique(coord, return_counts = True, axis = 1)
    np.clip(coord, 0, voting_map.shape[0] - 1, out = coord)
    voting_map[coord[0], coord[1], coord[2]] = coord_count
    
    # calculate the score and distance matrix; find the miniest distance of higher score point;
    cluster = np.zeros(voting_map.shape)
    index_pts = (voting_map > 20)
    coord = np.array(np.nonzero((index_pts == 1)))
    num_pts = coord.shape[1]
    if num_pts < 1e1:
        return centroids
    coord_dis_row = np.repeat(coord[:, np.newaxis, :], num_pts, axis = 1)
    coord_dis_col = np.repeat(coord[:, :, np.newaxis], num_pts, axis = 2)
    coord_dis = np.sqrt(np.sum((coord_dis_col - coord_dis_row) ** 2, axis=0))
    coord_score = voting_map[index_pts]
    coord_score_row = np.repeat(coord_score[np.newaxis, :], num_pts, axis = 0)
    coord_score_col = np.repeat(coord_score[:, np.newaxis], num_pts, axis = 1)
    coord_score = coord_score_col - coord_score_row
    
    coord_dis[coord_score > -0.5] = 1e10
    weight_dis = np.amin(coord_dis, axis = 1)
    weight_score = voting_map[index_pts]
    
    centroids = coord[:, (weight_dis > 3) * (weight_score > 30)].transpose((1, 0))
    return centroids

def centroids_distance(gt_centroids, pred_centroids):
    """
    calculate the distance between the gt and pred centroids
    Args:
    gt_centroids: (tooth_num, 3)
    pred_centroids: (pred_num, 3)
    return:
    the miniest distance and the mapping between gt and pred centroids
    """
    gt_tooth_num = gt_centroids.shape[0]
    pred_tooth_num = pred_centroids.shape[0]
    gt_centroids_matrix = np.repeat(gt_centroids[:, np.newaxis, :], pred_tooth_num, axis = 1)
    pred_centroids_matrix = np.repeat(pred_centroids[np.newaxis, :, :], gt_tooth_num, axis = 0)
    centroids_matrix = np.sqrt(np.sum((gt_centroids_matrix - pred_centroids_matrix) ** 2, axis=2))
    min_dis = np.amin(centroids_matrix, axis = 1)
    min_map = np.argmin(centroids_matrix, axis = 1)
    
    return min_dis, min_map


def jitter_gt_centroids(gt_centroids):
    """jitter the gt_centroids, before adding them into rois, to be more robust for seg
    Args:
    gt_centroids: -> (tooth_num, 3)
    """
    
    jittered_offset = np.random.randint(4, size=(gt_centroids.shape[0], gt_centroids.shape[1]))
    
    gt_centroids = gt_centroids + jittered_offset
    
    return gt_centroids



def generate_training_centroids(gt_centroids, bn_seg, off, training_flag):
    """
    generate a set of centroids (32) for the following network based one the gt instance (only training) and predicted centroids
    Args:
    gt_centroids: gt centroids -> (batch_size, tooth_num, 3)
    bn_seg: predicted binary segmentation results -> (batch_size, 2, 120, 120, 120)
    off: predicted offset of x. y, z -> (batch_size, 3, 120, 120, 120)
    Returns:
    centroids: generated centroids for training (tooth_num, 3)
    """
    pred_centroids = fast_clsuter(bn_seg, off)
    if training_flag == False:
        return pred_centroids
    gt_centroids = gt_centroids[0, :, :].cpu().data.numpy()
    gt_centroids = gt_centroids / 2
    if pred_centroids.shape[0] == 0:
        return gt_centroids
    else:
        min_dis, min_map = centroids_distance(gt_centroids, pred_centroids)
        gt_centroids = jitter_gt_centroids(gt_centroids)
        gt_centroids[min_dis < 10.0, :] = pred_centroids[min_map[min_dis < 10.0], :]
        return gt_centroids