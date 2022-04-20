import os
import sys
import argparse
import torch
import math
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from skimage import morphology
from scipy import ndimage
from networks.vnet import VNet




def cen_cluster(seg, off):
    
    #implementation of the paper 'Clustering by fast search and find of density peaks'
    #Args:
    #bn_seg: predicted binary segmentation results -> (batch_size, 1, 128, 128, 128)
    #off: predicted offset of x. y, z -> (batch_size, 3, 128, 128, 128)
    #Returns:
    #The centroids obtained from the cluster algorithm
    
    centroids = np.array([])

    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0
    # generate the voting map based on the binary segmentation and offset
    voting_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))
    num_fg = coord.shape[1]
    coord = coord + off[:, seg == 1]
    coord = coord.astype(np.int)
    coord, coord_count  = np.unique(coord, return_counts = True, axis = 1)
    np.clip(coord[0], 0, voting_map.shape[0] - 1, out = coord[0])
    np.clip(coord[1], 0, voting_map.shape[1] - 1, out = coord[1])
    np.clip(coord[2], 0, voting_map.shape[2] - 1, out = coord[2])
    voting_map[coord[0], coord[1], coord[2]] = coord_count
    
    # calculate the score and distance matrix; find the miniest distance of higher score point;
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
    
    coord_dis[coord_score > -0.5] = 1e10 # remove half distance of the dual distance matrix (only keep the negtive distance values)
    weight_dis = np.amin(coord_dis, axis = 1)
    weight_score = voting_map[index_pts]
    
    centroids = coord[:, (weight_dis > 5) * (weight_score > 100)]

    cnt_test = np.zeros(voting_map.shape)
    cnt_test[centroids[0, :], centroids[1, :], centroids[2, :]] = 1
    cnt_test = ndimage.grey_dilation(cnt_test, size= (2, 2, 2))


    return centroids


def map_cntToskl(centroids, seg, skl_off, cen_off):
    
    #Maping the index from centroids to skeleton
    
    
    # mapping process
    ins_skl_map = np.zeros(seg.shape)
    bin_skl_map = np.zeros(seg.shape)
    voting_skl_map = np.zeros(seg.shape)
    coord = np.array(np.nonzero((seg == 1)))
    coord_cnt = coord + cen_off[:, seg == 1]
    coord_cnt = coord_cnt.astype(np.int)
    coord_mat = np.repeat(coord_cnt[:, :, np.newaxis], centroids.shape[1], axis = 2)
    cnt_mat = np.repeat(centroids[:, np.newaxis, :], coord_cnt.shape[1], axis = 1)
    coord_cnt_dis_mat = np.sqrt(np.sum((coord_mat - cnt_mat) ** 2, axis=0))
    cnt_label = np.argmin(coord_cnt_dis_mat, axis = 1) + 1
    coord_skl = coord + skl_off[:, seg == 1]
    coord_skl = coord_skl.astype(np.int)
    np.clip(coord_skl[0], 0, seg.shape[0] - 1, out = coord_skl[0])
    np.clip(coord_skl[1], 0, seg.shape[1] - 1, out = coord_skl[1])
    np.clip(coord_skl[2], 0, seg.shape[2] - 1, out = coord_skl[2])
    
    ins_skl_map[coord_skl[0,:], coord_skl[1,:], coord_skl[2,:]] = cnt_label
    # filter operation
    coord_skl_uq, coord_skl_count  = np.unique(coord_skl, return_counts = True, axis = 1)
    voting_skl_map[coord_skl_uq[0], coord_skl_uq[1], coord_skl_uq[2]] = coord_skl_count
    bin_skl_map[ins_skl_map > 0.5] = 1
    voting_skl_map[voting_skl_map < 3.5] = 0
    voting_skl_map[voting_skl_map > 3.5] = 1
    bin_skl_map = bin_skl_map * voting_skl_map
    bin_skl_map = morphology.remove_small_objects(bin_skl_map.astype(bool), 50, connectivity=1)
    ins_skl_map = ins_skl_map * bin_skl_map
    ins_skl_map = ndimage.grey_dilation(ins_skl_map, size= (2, 2, 2))
    return ins_skl_map
    
    
    
def detect(net_cnt, net_skl, image, stride_xy, stride_z, patch_size):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape


    # run the cnt and skl network of the 1st stage by different patches
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((2, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    skl_off = np.zeros((3, ) + image.shape).astype(np.float32)
    skl_map = np.zeros(image.shape).astype(np.float32)
    skl_sep = np.zeros(image.shape).astype(np.float32)
    cnt_off = np.zeros((3, ) + image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch_cnt = torch.from_numpy(test_patch).cuda(1)
                test_patch_skl = torch.from_numpy(test_patch).cuda(2)
                # network inference
                cnt_off_patch, seg_cnt_patch = net_cnt(test_patch_cnt)
                skl_off_patch, seg_skl_patch = net_skl(test_patch_skl)
                
                # skl offset
                skl_off_patch = skl_off_patch.cpu().data.numpy()
                skl_off[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = skl_off[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + skl_off_patch[0,:,:,:,:]
                # cnt offset
                cnt_off_patch = cnt_off_patch.cpu().data.numpy()
                cnt_off[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = cnt_off[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + cnt_off_patch[0,:,:,:,:]
                # seg
                y = (F.softmax(seg_cnt_patch, dim=1).cpu().data.numpy() + F.softmax(seg_skl_patch, dim=1).cpu().data.numpy()) / 2
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                # sliding window account
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    # merge from patches to image         
    score_map = score_map/np.expand_dims(cnt,axis=0)
    skl_off = skl_off/np.expand_dims(cnt,axis=0)
    cnt_off = cnt_off/np.expand_dims(cnt,axis=0)
    #label_map = np.argmax(score_map, axis = 0)
    score_map = (score_map[1, :, :, :] > 0.9).astype(np.float32)
    label_map = score_map

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        cnt_off = cnt_off[:, wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        skl_map = skl_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]

    # centroid clustering
    centroids = cen_cluster(label_map, cnt_off)
    # voxel map to the skeleton
    ins_skl_map = map_cntToskl(centroids, label_map, skl_off, cnt_off)

    
    return ins_skl_map
    
    
def cnt_skl_detection(net_cnt, net_skl, image, stride_xy, stride_z, patch_size):
    ins_skl_map = detect(net_cnt, net_skl, image, stride_xy, stride_z, patch_size)
    return centroids
    
