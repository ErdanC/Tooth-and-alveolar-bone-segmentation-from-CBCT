import numpy as np
import h5py
import os
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter

output_size =[112, 112, 80]

def rescale(img, size):
    img = torch.from_numpy(img.astype(float))
    img = img[None, None, :, :, :]
    img = torch.nn.functional.interpolate(img, size=size, mode='trilinear')
    img = img[0, 0, :, :, :].numpy()
    return img
    
def rescale(image_label, w_ori, h_ori, z_ori, flag):
    # resize label map (int)
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = np.zeros((w_ori, h_ori, z_ori))
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id])
            image_label_bn = torch.from_numpy(image_label_bn.astype(float))
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori), mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :].numpy()
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori
    
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
    labels[labels > 0.5] = 1
    labels[labels < 0.5] = 0
    labels = rescale(labels, w*(spacing[0]/0.4), h*(spacing[0]/0.4), d*(spacing[0]/0.4), 'trilinear')

    #labels = rescale_mask(labels, (int(ori_w*spacing[0]/1), int(ori_h*spacing[1]/1), int(ori_d*spacing[2]/1)))
    
    return images, labels


def random_crop(image, label):
    output_size = (256, 256, 256)
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


    (w, h, d) = image.shape
    image_list, label_list = [], []
    for i in range(5):
        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])
        print('print the random coord:', w1, h1, d1)

        label_list.append(label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        image_list.append(image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
        
    return image_list, label_list

def covert_h5(images, labels):
    image_list, label_list = random_crop(images, labels)
    for file_i in range(len(image_list)):
        f = h5py.File('/public_bme/data/czm/NC_CBCT/h5_roi/' + str(data_id+678) + '_' + str(file_i) + '_roi.h5', 'w')
        f.create_dataset('image', data = image_list[file_i])
        f.create_dataset('label', data = label_list[file_i])
        f.close()
    
    
    
    
if __name__ == '__main__':
    with open('/public_bme/data/czm/NC_CBCT/nii/CQ/300patients_547_CBCT_scans/resting_447/file_polish.list', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n','') for item in image_list]

    with open('/public_bme/data/czm/NC_CBCT/nii/CQ/300patients_547_CBCT_scans_label/refine/file_polish.list', 'r') as f:
        image_list1 = f.readlines()
    image_list1 = [item.replace('\n','') for item in image_list1]

    for data_id in range(210, len(image_list)):
        print('process the data:', data_id)
        domain_flag = 0
        images, labels = read_data(image_list[data_id], image_list1[data_id])
        covert_h5(images, labels)
        