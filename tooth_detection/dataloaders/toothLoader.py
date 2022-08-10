import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

class toothLoader(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = '/public_bme/data/czm/NC_CBCT/h5_1st_stage_v1/'
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/file.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(image_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        offset_cnt = h5f['cnt_offset'][:]
        #offset_skl = h5f['skl_offset'][:]
        sample = {'image': image, 'offset_cnt': offset_cnt, 'offset_skl': offset_cnt, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, offset, label = sample['image'], sample['offset'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        annot = np.rot90(annot, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        offset = np.flip(offset, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {'image': image, 'offset': offset, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float)
        offset_cnt = sample['offset_cnt']
        offset_cnt = offset_cnt.reshape(offset_cnt.shape[0], offset_cnt.shape[1], offset_cnt.shape[2], offset_cnt.shape[3]).astype(np.float)
        offset_skl = sample['offset_skl']
        offset_skl = offset_skl.reshape(offset_skl.shape[0], offset_skl.shape[1], offset_skl.shape[2], offset_skl.shape[3]).astype(np.float)        
        label = sample['label']   
        label = label.reshape(label.shape[0], label.shape[1], label.shape[2]).astype(np.float)
        return {'image': torch.from_numpy(image).float(), 'offset_cnt': torch.from_numpy(offset_cnt).float(), 'offset_skl': torch.from_numpy(offset_skl).float(), 'label': torch.from_numpy(label).long()}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class LabelCrop(object):
    def __call__(self, sample):
        image, label, centroids = sample['image'], sample['label'], sample['centroids']
        w, h, d = label.shape

        label_crop = label.copy()
        label_crop = (label_crop > 0)
        
        tempL = np.nonzero(label_crop)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        minx = max(minx - np.random.randint(10, 20), 0)
        maxx = min(maxx + np.random.randint(10, 20), w)
        miny = max(miny - np.random.randint(10, 20), 0)
        maxy = min(maxy + np.random.randint(10, 20), h)
        minz = max(minz - np.random.randint(10, 20), 0)
        maxz = min(maxz + np.random.randint(10, 20), d)
    
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]
        
        centroids[:, 0]  = centroids[:, 0] - minx
        centroids[:, 1]  = centroids[:, 1] - miny
        centroids[:, 2]  = centroids[:, 2] - minz
        
        return {'image': image, 'offset': sample['offset'], 'label': label, 'centroids': centroids}

class DataScale(object):
    """
    Scale data to a fix size
    Args:
    output_size (int): Desired output size
    """
    def __init__(self):
        self.output_size = (256, 256, 208)
        
    def __call__(self, sample):
        image_t, label_t, offset_t, centroids_t = sample['image'], sample['label'], sample['offset'], sample['centroids']
        #self.output_size[2] = image_t.shape[2]
        w, h, d = sample['image'].shape
        m = nn.Upsample(self.output_size, mode='nearest')
        
        image_t = Variable(torch.from_numpy(image_t.astype(np.float)).type(torch.FloatTensor))
        label_t = Variable(torch.from_numpy(label_t.astype(np.float)).type(torch.FloatTensor))
        offset_t = Variable(torch.from_numpy(offset_t.astype(np.float)).type(torch.FloatTensor))
        
        image_t = m(image_t[None, None, :, :, :])[0, 0, :, :, :]
        label_t = m(label_t[None, None, :, :, :])[0, 0, :, :, :]
        offset_t = m(offset_t[None, :, :, :, :])[0, :, :, :, :]
        
        image_t = image_t.data.cpu().numpy()
        label_t = label_t.data.cpu().numpy()
        offset_t = offset_t.data.cpu().numpy()
        offset_t[0, :, :, :] = offset_t[0, :, :, :] * 256 / w
        offset_t[1, :, :, :] = offset_t[1, :, :, :] * 256 / h
        offset_t[2, :, :, :] = offset_t[2, :, :, :] * 208 / d
        centroids_t[:, 0] = centroids_t[:, 0] * 256 / w
        centroids_t[:, 1] = centroids_t[:, 1] * 256 / h
        centroids_t[:, 2] = centroids_t[:, 2] * 208 / d
        return {'image': image_t, 'offset': offset_t, 'label': label_t, 'centroids': centroids_t}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, offset_cnt, offset_skl, label = sample['image'], sample['offset_cnt'], sample['offset_skl'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            offset_cnt = np.pad(offset_cnt, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            offset_skl = np.pad(offset_skl, [(0, 0), (pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]][0:256:2, 0:256:2, 0:256:2]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]][0:256:2, 0:256:2, 0:256:2]
        offset_cnt = offset_cnt[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]][:, 0:256:2, 0:256:2, 0:256:2]/2
        offset_skl = offset_skl[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]][:, 0:256:2, 0:256:2, 0:256:2]/2
        return {'image': image, 'offset_cnt': offset_cnt, 'offset_skl': offset_skl, 'label': label}



class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)