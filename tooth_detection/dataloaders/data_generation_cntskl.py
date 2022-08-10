import numpy as np
import h5py
import re
import os
import random
import numpy as np
import nibabel as nib
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter
from skimage import morphology

output_size =[112, 112, 80]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  
def rescale(image_label, w_ori, h_ori, z_ori, flag):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    # resize label map (int)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = torch.zeros((w_ori, h_ori, z_ori)).cuda(0)
        image_label = torch.from_numpy(image_label.astype(float)).cuda(0)
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id]).float()
            #image_label_bn = torch.from_numpy(image_label_bn.astype(float))
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori), mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori.cpu().data.numpy()
    
    if flag == 'nearest':
        image_label = torch.from_numpy(image_label.astype(float))
        image_label = image_label[None, None, :, :, :]
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].numpy()
    return image_label


    
def read_data(data_path_img, data_patch_lab):
    
    src_data_file = os.path.join(data_path_img)
    src_data_vol = nib.load(src_data_file)
    images = src_data_vol.get_data()
    spacing = src_data_vol.header['pixdim'][1:4]
    w, h, d = images.shape
    images = rescale(images, w*(spacing[0]/0.4), h*(spacing[0]/0.4), d*(spacing[0]/0.4), 'nearest')

    low_bound = np.percentile(images, 5)
    up_bound = np.percentile(images, 99.9)
    images[images < 500] = 500
    images[images > 2500] = 2500
    images = (images - 500)/(2500 - 500)
    
    #images = rescale(images, (int(ori_w*spacing[0]/1), int(ori_h*spacing[1]/1), int(ori_d*spacing[2]/1)))
    
    lab_data_file = os.path.join(data_patch_lab)
    lab_data_vol = nib.load(lab_data_file)
    labels = lab_data_vol.get_data()
    labels = rescale(labels, w*(spacing[0]/0.4), h*(spacing[0]/0.4), d*(spacing[0]/0.4), 'trilinear')

    #labels = rescale_mask(labels, (int(ori_w*spacing[0]/1), int(ori_h*spacing[1]/1), int(ori_d*spacing[2]/1)))
    
    return images, labels, labels


def data_pad(image, label):
    """
    pad the image and label
    """
    dim_max = np.max(image.shape)
    print(image.shape)
    if label.shape[0] <= dim_max or label.shape[1] <= dim_max or label.shape[2] <= dim_max:
        pw = max((dim_max - label.shape[0]) // 2 + 0, 0)
        ph = max((dim_max - label.shape[1]) // 2 + 0, 0)
        pd = max((dim_max - label.shape[2]) // 2 + 0, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
    return image, label



def data_rot(image, label, skeletons):
    """
    random rotation
    """
    image_list, label_list, skeletons_list = [], [], []
    #image_list.append(image)
    #label_list.append(label)
    #skeletons_list.append(skeletons)
    for k in range(3):
        image_list.append(np.rot90(image, k))
        label_list.append(np.rot90(label, k))
        skeletons_list.append(np.rot90(skeletons, k))
    return image_list, label_list, skeletons_list


def random_crop(image, label, offset_cnt, offset_skl):
    output_size = (128, 128, 128)
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        offset_cnt = np.pad(offset_cnt, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        offset_skl = np.pad(offset_skl, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


    (w, h, d) = image.shape
    image_list, label_list, offset_cnt_list, offset_skl_list = [], [], [], []
    for i in range(10):
        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])
        print('print the random coord:', w1, h1, d1)

        label_list.append(label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        image_list.append(image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        offset_cnt_list.append(offset_cnt[:, w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        offset_skl_list.append(offset_skl[:, w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        
    return image_list, label_list, offset_cnt_list, offset_skl_list


def covert_h5(images, labels, skeletons, data_id, data_path):

    path_pos_1 = [sub_data_path.start() for sub_data_path in re.finditer('/', data_path)][-2]
    path_pos_2 = [sub_data_path.start() for sub_data_path in re.finditer('/', data_path)][-1]
    path_pos_3 = [sub_data_path.start() for sub_data_path in re.finditer('.nii.gz', data_path)][-1]

    
    teeth_ids = np.unique(labels)
    
    # remove noise label
    for i in range(len(teeth_ids)):
         tooth_id = teeth_ids[i]
         if (labels == tooth_id).sum() < 500:
             print('-- find one:', (labels == tooth_id).sum())
             labels[labels == tooth_id] = 0
    teeth_ids = np.unique(labels)
    
    image_bbox = (labels > 0)
    if image_bbox.sum() > 10000:
            
        x_min = np.nonzero(image_bbox)[0].min()-16
        x_max = np.nonzero(image_bbox)[0].max()+16
    
        y_min = np.nonzero(image_bbox)[1].min()-16
        y_max = np.nonzero(image_bbox)[1].max()+16

        z_min = np.nonzero(image_bbox)[2].min()-16
        z_max = np.nonzero(image_bbox)[2].max()+16
            
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if z_min < 0:
            z_min = 0
        if x_max > image_bbox.shape[0]:
            x_max = image_bbox.shape[0]
        if y_max > image_bbox.shape[1]:
            y_max = image_bbox.shape[1]
        if z_max > image_bbox.shape[2]:
            z_max = image_bbox.shape[2]
    
    images = images[x_min:x_max, y_min:y_max, z_min:z_max]
    labels = labels[x_min:x_max, y_min:y_max, z_min:z_max]
    skeletons = skeletons[x_min:x_max, y_min:y_max, z_min:z_max]


    #f_txt = open('/userhome/35/zmcui/TMI_ToothSegmentation/data/h5/train.list', 'w')
    #images, labels = data_pad(images, labels)
    
    #images, labels, skeletons = data_rescale(images, labels, skeletons)
    '''
    # pre-processing (normalization)
    image_sort = np.sort(images.copy(), axis = None)
    max_value = image_sort[-500]
    min_value = image_sort[0]
    images[images > max_value] = max_value
    images = (images - min_value) / (max_value - min_value)
    '''
    # data argumentation
    #images_list, labels_list, skeleton_list = data_rot(images, labels, skeletons)
    
    for i_rot in range(1):
        offset = np.zeros((3, images.shape[0], images.shape[1], images.shape[2]))
        offset_skl = np.zeros((3, images.shape[0], images.shape[1], images.shape[2]))
        label_update = np.zeros(labels.shape)
        centroids = []
        for i in range(len(teeth_ids)):
            print('tooth id:', i)
            tooth_id = teeth_ids[i]
            if tooth_id == 0:
                continue
            annotation = np.zeros(labels.shape)
            bin_tooth_label = morphology.remove_small_objects((labels == tooth_id), 500)
            coord = np.nonzero(bin_tooth_label)
            if coord[0].shape[0] < 500:
                continue
            meanx = int(np.mean(coord[0]))
            meany = int(np.mean(coord[1]))
            meanz = int(np.mean(coord[2]))
            
            offset[0, coord[0], coord[1], coord[2]] = meanx - coord[0]
            offset[1, coord[0], coord[1], coord[2]] = meany - coord[1]
            offset[2, coord[0], coord[1], coord[2]] = meanz - coord[2]
            
            centroids.append([meanx, meany, meanz])
            label_update[labels == tooth_id] = len(centroids)
            
            '''
            # skeleton offset
            coord_skl = np.nonzero((skeletons == tooth_id))
            coord_skl = np.asarray(coord_skl)
            coord = np.asarray(coord)
            coord_skl_mat = np.repeat(coord_skl[:, :, None], coord.shape[1], axis = 2)
            coord_mat = np.repeat(coord[:, None, :], coord_skl.shape[1], axis = 1)
            dis_mat = np.sqrt(np.square(coord_skl_mat - coord_mat).sum(axis = 0))
            if dis_mat.shape[0] < 1:
                print('continue:', coord_skl.shape)
                continue
            dis_index = np.argsort(dis_mat, axis = 0)[0, :]
            print(coord.shape, dis_mat.shape, dis_index.shape)
            offset_skl[0, coord[0, :], coord[1, :], coord[2, :]] = coord_skl[0, dis_index] - coord[0, :]
            offset_skl[1, coord[0, :], coord[1, :], coord[2, :]] = coord_skl[1, dis_index] - coord[1, :]
            offset_skl[2, coord[0, :], coord[1, :], coord[2, :]] = coord_skl[2, dis_index] - coord[2, :]
            '''
            
        centroids = np.asarray(centroids)
        
        
        image_list, label_list, offset_cnt_list, offset_skl_list = random_crop(images, labels, offset, offset)

        for i_file in range(len(image_list)):
            print('---save file:', i_file)
            f = h5py.File('/public_bme/data/czm/NC_CBCT/h5_1st_stage_v1/' + data_path[(path_pos_2+1):path_pos_3] + '_' + str(i_file) + '.h5', 'w')
            f.create_dataset('image', data = image_list[i_file])
            f.create_dataset('label', data = label_list[i_file].astype(int))
            f.create_dataset('cnt_offset', data = offset_cnt_list[i_file])
            #f.create_dataset('skl_offset', data = offset_skl_list[i_file])
            #nib.save(nib.Nifti1Image(image_list[i_file].astype(np.float32), np.eye(4)), '/hpc/data/home/bme/v-cuizm/data/ToothSeg_CBCT/h5/h5_1st_stage/' + data_path[(path_pos_2+1):path_pos_3] + '_' + str(i_file) + '.nii.gz')
            #nib.save(nib.Nifti1Image(offset_cnt_list[i_file][0, :, :, :].astype(np.float32), np.eye(4)), '/hpc/data/home/bme/v-cuizm/data/ToothSeg_CBCT/h5/h5_1st_stage/' + data_path[(path_pos_2+1):path_pos_3] + '_' + str(i_file) + '_offset.nii.gz')
            #nib.save(nib.Nifti1Image(label_list[i_file].astype(np.float32), np.eye(4)), '/hpc/data/home/bme/v-cuizm/data/ToothSeg_CBCT/h5/h5_1st_stage/' + data_path[(path_pos_2+1):path_pos_3] + '_' + str(i_file) + 'label.nii.gz')
            f.close()

        
if __name__ == '__main__':
    with open('/public_bme/data/czm/NC_CBCT/nii/CQ/300patients_547_CBCT_scans/resting_447/file_polish.list', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n','') for item in image_list]


    with open('/public_bme/data/czm/NC_CBCT/nii/CQ/300patients_547_CBCT_scans_label/refine/file_polish.list', 'r') as f:
        label_list = f.readlines()
    label_list = [item.replace('\n','') for item in label_list]


    for data_id in range(210, len(image_list)):
        print('process the data:', data_id)
        images, labels, skeletons = read_data(image_list[data_id], label_list[data_id])
        covert_h5(images, labels, skeletons, data_id, image_list[data_id])