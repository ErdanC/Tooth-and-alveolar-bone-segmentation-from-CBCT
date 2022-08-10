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
import pynvml
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.toothLoader import toothLoader, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler, LabelCrop, DataScale

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/userhome/35/zmcui/TMI_ToothSegmentation/data/h5/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='NC_1st_stage_cntV2_HZ_02_(1000data_256size_intensityClip)', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/ours_transformer"

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

patch_size = (256, 256, 256)
num_classes = 3

reg_criterion = torch.nn.L1Loss()

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

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()
    #net = torch.nn.DataParallel(net)
    
    # load the model
    # save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    # net.load_state_dict(torch.load('/u2/home/czm/project_test/CBCT_v2/TMI/model/vnet_annotation/iter_6000.pth'))
    
    db_train = toothLoader(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                           #RandomCrop(patch_size),
                           #DataScale(),
                           ToTensor()
                       ]))
    db_test = toothLoader(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           #RandomCrop(patch_size),
                           #DataScale(),
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
            volume_batch, offset_batch, offset_skl_batch, label_batch = sampled_batch['image'], sampled_batch['offset_cnt'], sampled_batch['offset_skl'], sampled_batch['label']
            volume_batch, offset_batch, offset_skl_batch, label_batch = volume_batch.cuda(), offset_batch.cuda(), offset_skl_batch.cuda(), label_batch.cuda()
            outputs_off, outputs_seg = net(volume_batch)

            # loss for seg and off
            ## coarse seg and off
            label_batch[label_batch > 0.5] = 1
            loss_seg = F.cross_entropy(outputs_seg, label_batch)
            outputs_soft = F.softmax(outputs_seg, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            loss_off = reg_criterion(outputs_off[:, :, label_batch[0, :, :, :]==1], offset_batch[:, :, label_batch[0, :, :, :]==1])
            
            print('test for seg:', loss_seg_dice.item())
            loss = 0.5 * (loss_seg_dice + loss_seg) + loss_off
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            
            del volume_batch, offset_batch, offset_skl_batch, label_batch, loss_seg, outputs_soft, loss_seg_dice, loss_off

            ## change lr
            if iter_num % 4000 == 0 and iter_num < 8001:
                lr_ = base_lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 10000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
            
            if iter_num % 600 == 0:
                net.eval()
                test_loss = 0
                iter_test = 0
                for i_batch, sampled_batch in enumerate(testloader):
                    volume_batch, offset_batch, offset_skl_batch, label_batch = sampled_batch['image'], sampled_batch['offset_cnt'], sampled_batch['offset_skl'], sampled_batch['label']
                    volume_batch, offset_batch, offset_skl_batch, label_batch = volume_batch.cuda(), offset_batch.cuda(), offset_skl_batch.cuda(), label_batch.cuda()
                    with torch.no_grad():
                        outputs_off, outputs_seg = net(volume_batch)

                        label_batch[label_batch > 0.5] = 1
                        loss_seg = F.cross_entropy(outputs_seg, label_batch)
                        outputs_soft = F.softmax(outputs_seg, dim=1)
                        loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
                        loss_off = reg_criterion(outputs_off[:, :, label_batch[0, :, :, :]==1], offset_batch[:, :, label_batch[0, :, :, :]==1])

                        print('---test for seg:', 1 - loss_seg_dice.item())
                        test_loss = test_loss + loss
                        iter_test = iter_test + 1
                writer.add_scalar('loss_test/test_loss', test_loss/iter_test, iter_num)
                net.train()
                del volume_batch, offset_batch, offset_skl_batch, label_batch, loss_seg, outputs_soft, loss_seg_dice, loss_off
            
                                
        if iter_num > max_iterations:
            break
        
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()