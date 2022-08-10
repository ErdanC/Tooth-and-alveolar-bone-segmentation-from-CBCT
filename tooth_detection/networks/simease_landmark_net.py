import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F

NUM_LANDMARKS = 32

class ReconLoss(nn.Module):

    def __init__(self):
        super(ReconLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_recon, img_gt, pred_seg0, label_gt, mr_flag):
        
        if mr_flag == False:
            label_pred = F.softmax(pred_seg0, dim=1)
            label_pred = torch.argmax(label_pred, dim=1)
            if torch.sum(label_pred) < 1:
                label_pred[0, 0, 0, 0] = 1
                label_pred[0, 0, 0, 1] = 2
                label_pred[0, 0, 0, 2] = 3
                label_pred[0, 0, 0, 3] = 4
            label_pred = label_pred[:, None, :, :, :]
            img_recon = img_recon[label_pred > 0.5].view(-1)
            img_gt = img_gt[label_pred > 0.5].view(-1)
            
            
        if mr_flag == True:
            img_recon = img_recon[label_gt > 0.5].view(-1)
            img_gt = img_gt[label_gt > 0.5].view(-1)
        
        label_gt = label_gt.type(torch.LongTensor).cuda()
        loss_rec = self.mse(img_recon, img_gt)
        loss_seg = self.ce(pred_seg0, label_gt[:, 0, :, :, :])
        if mr_flag == True:
            loss = loss_rec + loss_seg
        if mr_flag == False:
            loss = loss_rec
        return loss_rec, loss_seg, loss
        

class SegLoss(nn.Module):

    def __init__(self):
        super(SegLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_seg0, label_gt):
        label_gt = label_gt.type(torch.LongTensor).cuda()
        loss_seg = self.ce(pred_seg0, label_gt[:, 0, :, :, :])
        return loss_seg

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(4, 4, 4), stride=1, padding=0),
        )
        self.linear = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.sigmoid(self.linear(x))
        return x
        
class seg_refine(nn.Module):
    def __init__(self):
        super(seg_refine, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32, track_running_stats=False),
            nn.ReLU()

        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )
        
        
        # decoder
        self.conv0_d = nn.Sequential(
            nn.ConvTranspose3d(256, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )

        self.conv1_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.conv2_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )

        self.conv3_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU()
        )
        self.final_block = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1)
        )
        
        
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv0_d(x)
        x = self.conv1_d(x)
        x = self.conv2_d(x)
        x = self.conv3_d(x)
        x = self.final_block(x)
        return x

        


class FeatExtNet(nn.Module):

    def __init__(self):
        super(FeatExtNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32, track_running_stats=False),
            nn.ReLU()

        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class LandmarkExtNet(nn.Module):

    def __init__(self):
        super(LandmarkExtNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32, track_running_stats=False),
            nn.ReLU()

        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU() 
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, track_running_stats=False),
            nn.ReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=NUM_LANDMARKS, kernel_size=1, stride=1, padding=0)
        )
        self.softmax = nn.Softmax(dim=2)


    def normalize(self, feat):
        exp_feat = torch.exp(feat)
        exp_feat_sum = torch.sum(exp_feat, (3, 2))[:, :, None, None]
        feat_normalized = exp_feat / exp_feat_sum
        del exp_feat, exp_feat_sum
        return feat_normalized

    def get_weighted_coordinate(self, heat_map):

        dim0 = heat_map.shape[2]
        dim1 = heat_map.shape[3]
        dim2 = heat_map.shape[4]
        grid_0, grid_1, grid_2 = torch.meshgrid(torch.arange(dim0), torch.arange(dim1), torch.arange(dim2))
        self.coordinates = torch.cat((grid_0[None, :, :, :], grid_1[None, :, :, :], grid_2[None, :, :, :]), 0).type(torch.FloatTensor).cuda()
        self.coordinates.requires_grad = False

        denominator = torch.sum(heat_map, (4, 3, 2))  # it should be one, delete when varified
        coord_0 = torch.div(torch.sum(self.coordinates[None, None, 0, :, :, :] * heat_map, (4, 3, 2)), denominator)
        coord_1 = torch.div(torch.sum(self.coordinates[None, None, 1, :, :, :] * heat_map, (4, 3, 2)), denominator)
        coord_2 = torch.div(torch.sum(self.coordinates[None, None, 2, :, :, :] * heat_map, (4, 3, 2)), denominator)

        res = torch.cat((coord_0[:, :, None], coord_1[:, :, None], coord_2[:, :, None]), 2)  # the order is x, y
        del grid_0, grid_1, grid_2, coord_0, coord_1, coord_2
        return res

    def coord2gaussian(self, pred_coords, im_size, sigma):  # pred_coords: bn, 17, 3
        pred_coords_ext = pred_coords[:, :, None, None, None, :]  # bn, 17, 1, 1, 2
        grid_0, grid_1, grid_2 = torch.meshgrid(torch.arange(im_size[0]), torch.arange(im_size[1]), torch.arange(im_size[2]))
        grid_0 = grid_0.type(torch.FloatTensor).cuda()
        grid_1 = grid_1.type(torch.FloatTensor).cuda()
        grid_2 = grid_2.type(torch.FloatTensor).cuda()
        grids = torch.cat((grid_0[:, :, :, None], grid_1[:, :, :, None], grid_2[:, :, :, None]), 3)[None, None, :, :, :, :]
        coords_diff = grids - pred_coords_ext  # bn, 17, h, w, 2, the pred_coords_ext stores the coord in x,y format, need to switch
        squared_dis = torch.sum(coords_diff * coords_diff, 5)  # bn, 17, h, w
        gaussian = torch.exp(- 1.0 / (2.0 * sigma*sigma) * squared_dis)  # bn, 17, h, w
        del squared_dis, coords_diff, grids, grid_0, grid_1, grid_2, pred_coords_ext
        return gaussian
        
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = self.conv4(x3)        
        heatmap = self.conv5(x3)
        heatmap_norm = heatmap.view(heatmap.shape[0], heatmap.shape[1], -1)
        heatmap_norm = self.softmax(heatmap_norm)
        heatmap_norm = heatmap_norm.view(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4])
        pred_coords_3d = self.get_weighted_coordinate(heatmap_norm)
        recon_gaussian = self.coord2gaussian(pred_coords_3d, im_size=[heatmap_norm.shape[2], heatmap_norm.shape[3], heatmap_norm.shape[4]], sigma=0.8)
        ratio = x.size()[2] / heatmap_norm.shape[2]
        pred_coords_3d_orig_scale = pred_coords_3d * ratio
        del x0, x1, x2, heatmap
        return recon_gaussian, x3, pred_coords_3d_orig_scale

class ImgDecoderNet(nn.Module):

    def __init__(self):
        super(ImgDecoderNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256 + NUM_LANDMARKS, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU()
        )
        self.final_block = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_block(x)
        return x


class ImgSegNet(nn.Module):

    def __init__(self):
        super(ImgSegNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32, track_running_stats=False),
            nn.ReLU()

        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )
        

        self.conv0_d = nn.Sequential(
            nn.ConvTranspose3d(256 + 32, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, track_running_stats=False),
            nn.ReLU()
        )

        self.conv1_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU()
        )

        self.conv2_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, track_running_stats=False),
            nn.ReLU()
        )

        self.conv3_d = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU()
        )
        self.final_block = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=5, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, h_mr):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        low_feature = self.conv3(x)
        hy_feature = torch.cat((low_feature, h_mr), 1)
        x = self.conv0_d(hy_feature)
        x = self.conv1_d(x)
        x = self.conv2_d(x)
        x = self.conv3_d(x)
        x = self.final_block(x)
        return x, low_feature







class SimeaseLandmarkNet(nn.Module):

    def __init__(self):
        super(SimeaseLandmarkNet, self).__init__()
        self.pose_extract_net = LandmarkExtNet()
        self.feat_extract_net = FeatExtNet()
        self.img_decoder_net = ImgDecoderNet()
        self.img_seg_net = ImgSegNet()

        self.coordinates = None

    def normalize(self, feat):
        exp_feat = torch.exp(feat)
        exp_feat_sum = torch.sum(exp_feat, (3, 2))[:, :, None, None]
        feat_normalized = exp_feat / exp_feat_sum
        del exp_feat, exp_feat_sum
        return feat_normalized

    def get_weighted_coordinate(self, heat_map):

        dim0 = heat_map.shape[2]
        dim1 = heat_map.shape[3]
        dim2 = heat_map.shape[4]
        grid_0, grid_1, grid_2 = torch.meshgrid(torch.arange(dim0), torch.arange(dim1), torch.arange(dim2))
        self.coordinates = torch.cat((grid_0[None, :, :, :], grid_1[None, :, :, :], grid_2[None, :, :, :]), 0).type(torch.FloatTensor).cuda()
        self.coordinates.requires_grad = False

        denominator = torch.sum(heat_map, (4, 3, 2))  # it should be one, delete when varified
        coord_0 = torch.div(torch.sum(self.coordinates[None, None, 0, :, :, :] * heat_map, (4, 3, 2)), denominator)
        coord_1 = torch.div(torch.sum(self.coordinates[None, None, 1, :, :, :] * heat_map, (4, 3, 2)), denominator)
        coord_2 = torch.div(torch.sum(self.coordinates[None, None, 2, :, :, :] * heat_map, (4, 3, 2)), denominator)


        res = torch.cat((coord_0[:, :, None], coord_1[:, :, None], coord_2[:, :, None]), 2)  # the order is x, y
        del grid_0, grid_1, grid_2, coord_0, coord_1, coord_2
        return res

    def coord2gaussian(self, pred_coords, im_size, sigma):  # pred_coords: bn, 17, 3
        pred_coords_ext = pred_coords[:, :, None, None, None, :]  # bn, 17, 1, 1, 2
        grid_0, grid_1, grid_2 = torch.meshgrid(torch.arange(im_size[0]), torch.arange(im_size[1]), torch.arange(im_size[2]))
        grid_0 = grid_0.type(torch.FloatTensor).cuda()
        grid_1 = grid_1.type(torch.FloatTensor).cuda()
        grid_2 = grid_2.type(torch.FloatTensor).cuda()
        grids = torch.cat((grid_0[:, :, :, None], grid_1[:, :, :, None], grid_2[:, :, :, None]), 3)[None, None, :, :, :, :]
        coords_diff = grids - pred_coords_ext  # bn, 17, h, w, 2, the pred_coords_ext stores the coord in x,y format, need to switch
        squared_dis = torch.sum(coords_diff * coords_diff, 5)  # bn, 17, h, w
        gaussian = torch.exp(- 1.0 / (2.0 * sigma*sigma) * squared_dis)  # bn, 17, h, w

        del squared_dis, coords_diff, grids, grid_0, grid_1, grid_2, pred_coords_ext
        return gaussian

    def forward(self, img0, img1):

        feat1 = self.feat_extract_net(img1)
        pred_heat_map, heatmap_feature = self.pose_extract_net(img0)

        pred_coords_3d = self.get_weighted_coordinate(pred_heat_map)

        recon_gaussian = self.coord2gaussian(pred_coords_3d, im_size=[pred_heat_map.shape[2], pred_heat_map.shape[3], pred_heat_map.shape[4]], sigma=0.8)
        
        cat_feat = torch.cat((feat1, recon_gaussian), 1)
        rec_img0 = self.img_decoder_net(cat_feat)
        seg_img0 = self.img_seg_net(recon_gaussian)
        ratio = img0.shape[2] / pred_heat_map.shape[2]
        pred_coords_3d_orig_scale = pred_coords_3d * ratio
        del cat_feat, pred_coords_3d, feat1,
        return pred_coords_3d_orig_scale, rec_img0, recon_gaussian, pred_heat_map, heatmap_feature, seg_img0
