import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet_singleTooth
from utils.losses import dice_loss
from dataloaders.singeToothLoader import singeToothLoader, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler, LabelCrop, DataScale, singleToothCrop

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/../', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='single_tooth_seg', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "/../" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (96, 96, 128)
num_classes = 3

seg_criterion = torch.nn.BCELoss()

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    net = VNet_singleTooth(n_channels=2, n_classes=2, normalization='batchnorm', has_dropout=True)
    net.cuda()
    #net = torch.nn.DataParallel(net)

    db_train = singeToothLoader(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                           #LabelCrop(),
                           #RandomRotFlip(),
                           ToTensor()
                       ]))
    db_test = singeToothLoader(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           ToTensor()
                       ]))
                       
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path)
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch, skeleton_batch, boundary_patch, kp_patch = sampled_batch['image'], sampled_batch['label'], sampled_batch['skeleton'],  sampled_batch['boundary'], sampled_batch['keypoints']
            volume_batch, label_batch, skeleton_batch, boundary_patch, kp_patch = volume_batch.cuda(), label_batch.cuda(), skeleton_batch.cuda(), boundary_patch.cuda(), kp_patch.cuda()
            print('the shape:', volume_batch.shape)
            outputs_seg, output_bd, output_kp = net(volume_batch, skeleton_batch)

            # loss for seg and kp
            ## seg
            label_batch[label_batch > 0.5] = 1
            loss_seg = F.cross_entropy(outputs_seg, label_batch)
            outputs_soft = F.softmax(outputs_seg, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch[:, :, :, :])
            
            ## bd loss
            boundary_patch[boundary_patch > 0.5] = 1
            loss_bd = F.cross_entropy(output_bd, boundary_patch)
            outputs_soft = F.softmax(output_bd, dim=1)
            loss_bd_dice = dice_loss(outputs_soft[:, 1, :, :, :], boundary_patch[:, :, :, :])

            ## kp loss
            kp_patch[kp_patch > 0.5] = 1
            loss_kp = F.cross_entropy(output_kp, kp_patch)
            outputs_soft = F.softmax(output_kp, dim=1)
            loss_kp_dice = dice_loss(outputs_soft[:, 1, :, :, :], kp_patch[:, :, :, :])

            
            loss = 0.5 * (loss_seg_dice + loss_seg) + 0.1*(loss_bd + loss_bd_dice) + 0.1*(loss_kp + loss_kp_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f, %f' % (iter_num, loss.item(), loss_seg_dice.item()))
                
            
            
        if iter_num > max_iterations:
            break
        
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()