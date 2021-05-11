import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.utils.postprocess import _transpose_and_gather_feat


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()


    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
          loss = loss - neg_loss
        else:
          loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def forward(self, pred, target):
        return self._neg_loss(pred, target)


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
   
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    def __init__(self, cfg):
        super(CenterNetLoss, self).__init__()

        if cfg.MODEL.DETECTOR_TYPE == 'hourglass':
            self.nstack = cfg.MODEL.HOURGLASS.NSTACK
        else:
            self.nstack = 1
        self.hm_loss_weight = cfg.SOLVER.HEATMAP_LOSS_WEIGHT
        self.wh_loss_weight = cfg.SOLVER.WH_LOSS_WEIGHT
        self.reg_loss_weight = cfg.SOLVER.REG_LOSS_WEIGHT

        self.hm_loss_fn = FocalLoss()
        self.wh_loss_fn = RegL1Loss()
        self.reg_loss_fn = RegL1Loss()


    def forward(self, preds, target):
        hm_loss, wh_loss, reg_loss = 0, 0, 0

        for pred in preds:
            device = pred['hm'].device
            pred['hm'] = pred['hm'].float()
            # pred['heatmap'] = pred['heatmap'].max(target['invalid_heatmap'].to(device))
            hm_loss += self.hm_loss_fn(pred['hm'], target['heatmap'].to(device)) / self.nstack
            wh_loss += self.reg_loss_fn(pred['wh'], target['reg_mask'].to(device), target['ind'].to(device), target['wh'].to(device)) / self.nstack
            reg_loss += self.reg_loss_fn(pred['reg'], target['reg_mask'].to(device), target['ind'].to(device), target['reg'].to(device)) / self.nstack

        loss = (self.hm_loss_weight * hm_loss) + (self.wh_loss_weight * wh_loss) + (self.reg_loss_weight * reg_loss)

        return loss


class DetectorLoss(nn.Module):
    def __init__(self, cfg):
        super(DetectorLoss, self).__init__()

        self.det_loss_fn = CenterNetLoss(cfg)

        if cfg.MODEL.D_LOSS_TYPE == 'cross_entropy':
            self.adv_loss_fn = BCELoss(d_train=False)
        elif cfg.MODEL.D_LOSS_TYPE == 'reconstruction':
            self.adv_loss_fn = ReconstructionLoss(d_train=False)
        elif cfg.MODEL.D_LOSS_TYPE == 'wasserstain':
            self.adv_loss_fn = WassersteinLoss(d_train=False)
        elif cfg.MODEL.D_LOSS_TYPE == 'hinge':
            self.adv_loss_fn = HingeLoss(d_train=False)

        self.contents_loss_fn = nn.L1Loss()
        self.style_loss_fn = StyleLoss()


    def forward(self, det_preds, targets, adv_preds, features):
        assert len(det_preds) == len(adv_preds)
        det_loss = []
        adv_loss = []
        contents_loss = []
        style_loss = []

        for i in range(len(det_preds)):
            det_loss.append(self.det_loss_fn(det_preds[i], targets[i]))
            adv_loss.append(self.adv_loss_fn(adv_preds[i], i))

        for i in range(len(features)):
            contents_loss.append(self.contents_loss_fn(features[i-1], features[i]))
            style_loss.append(self.style_loss_fn(features[i-1], features[i]))

        return det_loss, adv_loss, contents_loss, style_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, cfg):
        super(DiscriminatorLoss, self).__init__()

        if cfg.MODEL.D_LOSS_TYPE == 'cross_entropy':
            self.adv_loss_fn = BCELoss(d_train=True)
        elif cfg.MODEL.D_LOSS_TYPE == 'reconstruction':
            self.adv_loss_fn = ReconstructionLoss(d_train=True)
        elif cfg.MODEL.D_LOSS_TYPE == 'wasserstain':
            self.adv_loss_fn = WassersteinLoss(d_train=True)
        elif cfg.MODEL.D_LOSS_TYPE == 'hinge':
            self.adv_loss_fn = HingeLoss(d_train=True)

    def forward(self, adv_preds):
        adv_loss = []
        for i in range(len(adv_preds)):
            adv_loss.append(self.adv_loss_fn(adv_preds[i], i))

        return adv_loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

        self.loss_fn = nn.L1Loss()

    def forward(self, feat1, feat2):
        loss = self.loss_fn(self._gram_matrix(feat1), self._gram_matrix(feat2))

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class BCELoss(nn.Module):
    def __init__(self, d_train):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.d_train = d_train

    def forward(self, prediction, i):
        print(self.temp.device)
        loss = 0

        if type(prediction) != list:
            prediction = [prediction]

        for j in range(len(prediction)):
            if self.d_train:
                label = torch.zeros(prediction[j].shape)
                label[:, i, :, :] = 1
                label = label.to(prediction[j].device)
            else:
                label = torch.ones(prediction[j].shape)
                label[:, i, :, :] = 0
                label = label.to(prediction[j].device)

            loss += self.loss_fn(prediction[j], label)

        return loss / len(prediction)


class WassersteinLoss(nn.Module):
    def __init__(self, d_train):
        super(WassersteinLoss, self).__init__()
        self.d_train = d_train

    def forward(self, prediction, i):
        if type(prediction) != list:
            prediction = [prediction]

        loss = 0
        for j in range(len(prediction)):
            for k in range(prediction[j].shape[1]):
                if i == k:
                    if self.d_train:
                        loss += torch.mean(-prediction[j][:, k, :, :])
                    else:
                        loss += torch.mean(prediction[j][:, k, :, :])

                else:
                    if self.d_train:
                        loss += torch.mean(prediction[j][:, k, :, :])
                    else:
                        loss += torch.mean(-prediction[j][:, k, :, :])
            
        return loss / len(prediction) / prediction[0].shape[1]
                

class HingeLoss(nn.Module):
    def __init__(self, d_train):
        super(HingeLoss, self).__init__()
        self.d_train = d_train
        self.relu = nn.ReLU()

    def forward(self, prediction, i):
        if type(prediction) != list:
            prediction = [prediction]

        loss = 0
        for j in range(len(prediction)):
            for k in range(prediction[j].shape[1]):
                if i == k:
                    if self.d_train:
                        loss += torch.mean(self.relu(1.0 - prediction[j][:, k, :, :]))
                    else:
                        loss += torch.mean(prediction[j][:, k, :, :])

                else:
                    if self.d_train:
                        loss += torch.mean(self.relu(1.0 + prediction[j][:, k, :, :]))
                    else:
                        loss += torch.mean(-prediction[j][:, k, :, :])
            
        return loss / len(prediction) / prediction[0].shape[1]


class MSELoss(nn.Module):
    def __init__(self, d_train):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.d_train = d_train

    def forward(self, prediction, i):
        if self.d_train:
            label = torch.zeros(prediction.shape)
            label[:, i, :, :] = 1
            label = label.to(prediction.device)
        else:
            label = torch.ones(prediction.shape)
            label[:, i, :, :] = 0
            label = label.to(prediction.device)

        return self.loss_fn(prediction, label)


class ReconstructionLoss(nn.Module):
    def __init__(self, d_train):
        super(ReconstructionLoss, self).__init__()
        self.loss_fn = nn.L1Loss()
        self.d_train = d_train

    def forward(self, features, predictions):
        if self.d_train:
            return self.loss_fn(features, predictions)
        else:
            return -self.loss_fn(features, predictions)


# class HingeLoss(nn.Module):
#     def __init__(self, d_train):
#         super(HingeLoss, self).__init__()
#         self.relu = nn.ReLU()
#         self.d_train = d_train

#     def forward(self, prediction, i):
#         loss = 0
#         for j in range(prediction.shape[1]):
#             if i == j:
#                 if self.d_train:
#                     loss += torch.mean(self.relu(1.0 - prediction[:, i, :, :]))
#                 else:
#                     loss += torch.mean(prediction[:, i, :, :])

#             else:
#                 if self.d_train:
#                     loss += torch.mean(self.relu(1.0 + prediction[:, i, :, :]))
#                 else:
#                     loss += torch.mean(-prediction[:, i, :, :])

#         return loss


# class WassersteinLoss(nn.Module):
#     def __init__(self, d_train):
#         super(WassersteinLoss, self).__init__()
#         self.d_train = d_train

#     def forward(self, prediction, i):
#         loss = 0
#         for j in range(prediction.shape[1]):
#             if i == j:
#                 if self.d_train:
#                     loss += torch.mean(-prediction[:, i, :, :])
#                 else:
#                     loss += torch.mean(prediction[:, i, :, :])

#             else:
#                 if self.d_train:
#                     loss += torch.mean(prediction[:, i, :, :])
#                 else:
#                     loss += torch.mean(-prediction[:, i, :, :])

#         return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
        
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        
        return loss
