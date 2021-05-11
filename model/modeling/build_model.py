import torch
import torch.nn as nn
import torch.autograd as autograd

import itertools
from copy import deepcopy

from .base_networks import *
from .hourglass import MultiScaleHourglassNet
from .resnet import get_resnet18
from .discriminator import Discriminator, UNetDiscriminator, BeganDiscriminator
from model.engine.loss_functions import DetectorLoss, DiscriminatorLoss, CenterNetLoss, ReconstructionLoss
from model.utils.misc import fix_model_state_dict
from model.utils.overlay_heatmap import OverlayHeatmap

from model.modeling.extractor import *
from model.modeling.resnet import DetectionCore as CenterNet

from .resnet_original import get_pose_net
from .auto_encoder import AutoEncoder


class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        if cfg.MODEL.EXTRACTOR_TYPE == 'normal':
            self.extractors = nn.ModuleList([
                Extractor(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])
        elif cfg.MODEL.EXTRACTOR_TYPE == 'sft':
            self.extractors = nn.ModuleList([
                SFTExtractor(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])
        elif cfg.MODEL.EXTRACTOR_TYPE == 'normal_rfmatch':
            self.extractors = nn.ModuleList([
                Extractor_RFMatch(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])
        elif cfg.MODEL.EXTRACTOR_TYPE == 'normal_rfmatch_dilation':
            self.extractors = nn.ModuleList([
                Extractor_RFMatch_Dilation(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])


        if cfg.MODEL.DETECTOR_TYPE == 'centernet':
            self.detector = CenterNet(18)
        elif cfg.MODEL.DETECTOR_TYPE == 'ssd':
            self.detector = SSD()
        self.det_loss_fn = DetectorLoss(cfg)
        
        self.discriminator = Discriminator(cfg, 3)
        # self.discriminator = UNetDiscriminator(cfg)
        self.dis_loss_fn = DiscriminatorLoss(cfg)

        self.gradient_penalty = cfg.SOLVER.GRADIENT_PENALTY
        self.gradient_penalty_weight = cfg.SOLVER.GRADIENT_PENALTY_WEIGHT

        self.overley = OverlayHeatmap()
        self.explosion_check_thresh = 10.
        self.explosion_counter = 0


    def forward(self, x, target=None):
        if target is None:
            return self.dis_forward(x)

        else:
            return self.det_forward(x, target)

    def det_forward(self, x, targets):
        det_preds, adv_preds, features = [], [], []

        for i in range(len(x)):
            x[i]= x[i].to('cuda', non_blocking=True)
            for k in targets[i].keys():
                targets[i][k] = targets[i][k].to('cuda', non_blocking=True)

            feature = self.extractors[i](x[i])
            det_pred = self.detector(feature)
            adv_pred = self.discriminator(feature)

            det_preds.append(det_pred)
            adv_preds.append(adv_pred)
            features.append(feature)

        det_preds = tuple(det_preds)
        adv_preds = tuple(adv_preds)
        features = tuple(features)
        
        return self.det_loss_fn(det_preds, targets, adv_preds, features)


    # def det_forward(self, x, target, i):
    #     x = x.to('cuda')
    #     for k in target.keys():
    #         target[k] = target[k].to('cuda')

    #     feature = self.extractors[i](x)
    #     det_pred = self.detector(feature)
    #     adv_pred = self.discriminator(feature)

    #     det_loss, adv_loss = self.g_loss_fn(det_pred, target, adv_pred, feature, i)

    #     return det_loss, adv_loss


    def dis_forward(self, x):
        features, adv_preds = [], []
        for i in range(len(x)):
            x[i] = x[i].to('cuda', non_blocking=True)

            feature = self.extractors[i](x[i])
            adv_pred = self.discriminator(feature)

            features.append(feature)
            adv_preds.append(adv_pred)
        
        if self.gradient_penalty:
            gp = 0
            for i in range(len(features)):
                gp += self.gradient_penalty_weight * self._gradient_penalty(features[i-1], features[i]) / len(features)

            return self.dis_loss_fn(adv_preds), gp

        return self.dis_loss_fn(adv_preds)


    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(real_data.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad = True
        interpolated = interpolated.to(real_data.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(real_data.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()


    def _explosion_check(loss, x, pred, targets):
        if loss > self.explosion_check_thresh:
            self.explosion_counter += 1
            for i in range(len(x)):
                self.overlay(x[i], targets[i]['hm'])
                self.overlay(x[i], preds[i]['hm'])



# class ModelWithLoss(nn.Module):
#     def __init__(self, cfg):
#         super(ModelWithLoss, self).__init__()

#         self.valid_scale = cfg.MODEL.VALID_SCALE

#         if cfg.MODEL.DETECTOR_TYPE == 'hourglass':
#             self.model = MultiScaleHourglassNet(cfg)
#         elif cfg.MODEL.DETECTOR_TYPE == 'resnet18':
#             self.model = get_resnet18()
#             self.model.load_state_dict(fix_model_state_dict(torch.load('weights/ctdet_coco_resdcn18.pth')['state_dict']), strict=False)

#             # self.model = get_pose_net(18, {'heatmap': 80, 'reg':2, 'wh':2}, 64)

#         if cfg.MODEL.DISCRIMINATOR_TYPE == 'normal':
#             self.discriminator = Discriminator(cfg)
#         elif cfg.MODEL.DISCRIMINATOR_TYPE == 'unet':
#             self.discriminator = UNetDiscriminator(cfg)
#         elif cfg.MODEL.DISCRIMINATOR_TYPE == 'began':
#             self.discriminator = BeganDiscriminator(cfg)
#         else:
#             raise NotImplementedError()

#         self.loss_fn = DetectorLoss(cfg)
#         if cfg.MODEL.DISCRIMINATOR_TYPE == 'began':
#             self.d_loss_fn = ReconstructionLoss()
#         else:
#             self.d_loss_fn = DiscriminatorLoss(cfg)


#         if cfg.SOLVER.DIS_TRAIN_RATIO == 0:
#             self.not_use_discriminator = True
#         else:
#             self.not_use_discriminator = False

#         self.gp = cfg.SOLVER.GRADIENT_PENALTY
#         self.d_type = cfg.MODEL.DISCRIMINATOR_TYPE

#     def forward(self, x, targets=None, i=-1):
#         if targets == None and i == -1:
#             # assert len(x) == 4
#             inter_features = []
#             gan_preds = []
#             for i in range(len(x)):
#                 x[i] = x[i].to('cuda')
#                 inter_feature = self.model(x[i], i, d_train=True)
#                 inter_features.append(inter_feature)
#                 gan_preds.append(self.discriminator(inter_feature))

#             if self.d_type == 'began':
#                 loss = self.d_loss_fn(inter_features, gan_preds)
#             else:
#                 loss = self.d_loss_fn(gan_preds)

#             if self.gp:
#                 return loss + self._compute_gradient_penalty(inter_features)
#             else:
#                 return loss

#         else:
#             x = x.to('cuda')
#             for k in targets.keys():
#                 targets[k] = targets[k].to('cuda')
#             detect_pred, inter_feature = self.model(x, i)
#             if self.not_use_discriminator:
#                 gan_pred = None
#             else:
#                 gan_pred = self.discriminator(inter_feature)

#             return self.loss_fn(detect_pred, targets, gan_pred, i)


#     def _compute_gradient_penalty(self, gan_preds):
#         """Calculates the gradient penalty loss for WGAN GP"""
#         gradient_penalty = 0
#         for real_samples, fake_samples in itertools.combinations(gan_preds, 2):
#             # Random weight term for interpolation between real and fake samples
#             alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
#             # Get random interpolation between real and fake samples
#             interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#             d_interpolates = self.discriminator(interpolates)
#             fake = torch.ones(real_samples.shape[0], dtype=torch.float32, requires_grad=False)
#             # Get gradient w.r.t. interpolates
#             gradients = autograd.grad(
#                 outputs=d_interpolates,
#                 inputs=interpolates,
#                 grad_outputs=fake,
#                 create_graph=True,
#                 retain_graph=True,
#                 only_inputs=True,
#             )[0]
#             gradients = gradients.view(gradients.size(0), -1)
#             gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean()

#         return gradient_penalty

class Model(nn.Module):
    def __init__(self, cfg, mmd_calc=False):
        super(Model, self).__init__()

        if cfg.MODEL.EXTRACTOR_TYPE == 'normal':
            self.extractors = nn.ModuleList([
                Extractor(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])
        elif cfg.MODEL.EXTRACTOR_TYPE == 'sft':
            self.extractors = nn.ModuleList([
                SFTExtractor(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])
        elif cfg.MODEL.EXTRACTOR_TYPE == 'normal_rfmatch':
            self.extractors = nn.ModuleList([
                Extractor_RFMatch(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])
        elif cfg.MODEL.EXTRACTOR_TYPE == 'normal_rfmatch_dilation':
            self.extractors = nn.ModuleList([
                Extractor_RFMatch_Dilation(cfg, 3, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
            ])

        self.detector = CenterNet(18)

        self.mmd_calc = mmd_calc
        
    def forward(self, x, i):
        x = x.to('cuda')
        
        feature = self.extractors[i](x)
        if self.mmd_calc:
            return feature
        det_pred = self.detector(feature)

        return det_pred, feature


class AEWithLoss(nn.Module):
    def __init__(self, cfg):
        super(AEWithLoss, self).__init__()

        self.ae = AutoEncoder(input_size=cfg.TEST.AE_INPUT_PATCH, feat=cfg.MODEL.EXTRACTOR_FEAT, latent_size=cfg.TEST.AE_LATENT_SIZE)

        self.loss_fn = nn.L1Loss()


    def train_forward(self, x):
        x = x.to('cuda')
        target = x
        x = self.ae(x)
        loss = self.loss_fn(x, target)

        return loss

    def inference_forward(self, x):
        x = x.to('cuda')
        latent = self.ae(x, inference=True)

        return latent


class AE(nn.Module):
    def __init__(self, cfg):
        super(AE, self).__init__()

        self.ae = AutoEncoder(train=False, input_size=cfg.TEST.AE_INPUT_PATCH, feat=cfg.MODEL.EXTRACTOR_FEAT, latent_size=cfg.TEST.AE_LATENT_SIZE)
        self.loss_fn = nn.L1Loss()

        self.train = True

    def forward(self, x):
        x = x.to('cuda')
        if self.train:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


    def train_forward(self, x):
        target = x
        x = self.ae(x)
        loss = self.loss_fn(x, target)

        return loss


    def eval_forward(self, x):
        x = x.to('cuda')
        latent = self.ae(x, inference=True)

        return latent