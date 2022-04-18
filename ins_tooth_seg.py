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
from skimage import measure
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize_3d
from networks.vnet_ins_seg import VNet_singleTooth




def tooth_seg(net_seg, image, centroids, patch_size):
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
    
    # crop the patch size from original image
    crop_size = patch_size
    image_list, skeleton_list, crop_coord_min_list = [], [], []
    teeth_ids = centroids.shape[1]
    for i in range(teeth_ids):
        tooth_id = i+1

        mean_coord = (centroids[0, i], centroids[1, i], centroids[2, i])
        # generate the crop coords
        crop_coord_min = mean_coord - crop_size/2
        np.clip(crop_coord_min, (0, 0, 0), image.shape - crop_size, out = crop_coord_min)
        crop_coord_min = crop_coord_min.astype(int)

        crop_cnt = np.zeros((crop_size[0], crop_size[1], crop_size[2]))
        crop_cnt[mean_coord[0] - crop_coord_min[0], mean_coord[1] - crop_coord_min[1], mean_coord[2] - crop_coord_min[2]] = 1
        crop_cnt = gaussian_filter(crop_cnt, sigma=5)
        crop_cnt = (crop_cnt - crop_cnt.min())/(crop_cnt.max() - crop_cnt.min())
        
        image_list.append(image[crop_coord_min[0]:(crop_coord_min[0]+crop_size[0]), crop_coord_min[1]:(crop_coord_min[1]+crop_size[1]), crop_coord_min[2]:(crop_coord_min[2]+crop_size[2])])
        skeleton_list.append(crop_cnt)
        crop_coord_min_list.append(crop_coord_min)
    patches_coord_min = np.asarray(crop_coord_min_list)
    image_patches = np.asarray(image_list)
    skeleton_patches = np.asarray(skeleton_list)

    image_patches = torch.from_numpy(image_patches[:, None, :, :, :]).float().cuda(3)
    skeleton_patches = torch.from_numpy(skeleton_patches[:, None, :, :, :]).float().cuda(3)
    with torch.no_grad():
        seg_patches_1, bd_patches_1 = net_seg(image_patches[:10, :, :, :, :], skeleton_patches[:10, :, :, :, :])
        seg_patches_2, bd_patches_2 = net_seg(image_patches[10:20, :, :, :, :], skeleton_patches[10:20, :, :, :, :])
        seg_patches_3, bd_patches_3 = net_seg(image_patches[20:, :, :, :, :], skeleton_patches[20:, :, :, :, :])
        seg_patches = torch.cat((seg_patches_1, seg_patches_2), 0)
        seg_patches = torch.cat((seg_patches, seg_patches_3), 0)
    seg_patches = F.softmax(seg_patches, dim=1)
    seg_patches = torch.argmax(seg_patches, dim = 1)
    seg_patches = seg_patches.cpu().data.numpy()
    w2, h2, d2 = image.shape
    count = 0
    image_label = np.zeros((w2, h2, d2), dtype=int)
    image_vote_flag = np.zeros((w2, h2, d2), dtype=int)
    for crop_i in range(patches_coord_min.shape[0]):
        # label patch
        labels, num = measure.label(seg_patches[crop_i, :, :, :], connectivity=2, background=0, return_num=True)
        if num > 1:
            max_num = -1e10
            for lab_id in range(1, num+1):
                if np.sum(labels == lab_id) > max_num:
                    max_num = np.sum(labels == lab_id)
                    true_id = lab_id
            seg_patches[crop_i, :, :, :] = (labels == true_id)
        coord = np.array(np.nonzero((seg_patches[crop_i, :, :, :] == 1)))
        coord[0] = coord[0] + patches_coord_min[crop_i, 0]
        coord[1] = coord[1] + patches_coord_min[crop_i, 1]
        coord[2] = coord[2] + patches_coord_min[crop_i, 2]             
        image_vote_flag[coord[0], coord[1], coord[2]] = 1
        if np.sum((image_vote_flag > 0.5) * (image_label > 0.5)) > 2000:
            image_vote_flag[coord[0], coord[1], coord[2]] = 0
            continue
        count = count + 1
        image_label[coord[0], coord[1], coord[2]] = count
        image_vote_flag[coord[0], coord[1], coord[2]] = 0

    if add_pad:
        image_label = image_label[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return image_label

def ins_tooth_seg(ins_net, image, centroids, patch_size):
    #net = load_model()
    label_map = tooth_seg(ins_net, image, centroids, patch_size)
    #del net
    return label_map