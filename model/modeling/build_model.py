import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import combinations

from model.modeling.extractor import build_extractors
from model.modeling.detector import build_detector
from model.modeling.discriminator import build_discriminator
from model.engine.loss_functions import DetectorLoss, DiscriminatorLoss


class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        self.extractors = build_extractors(cfg)
        self.detector = build_detector(cfg)
        self.loss_fn = DetectorLoss(cfg)

        self.discriminator = build_discriminator(cfg)
        self.d_loss_fn = DiscriminatorLoss(cfg)

        if cfg.SOLVER.IMAGENET_PRETRAINED:
            self._load_pretrained_model(cfg.SOLVER.IMAGENET_PRETRAINED_MODEL)

        self.valid_scale = cfg.MODEL.VALID_SCALE
        self.mixed_precision = cfg.MIXED_PRECISION
        self.gp = cfg.SOLVER.DISCRIMINATOR.GP
        self.gp_weight = cfg.SOLVER.DISCRIMINATOR.GP_WEIGHT


    def forward(self, images, targets, pretrain=False):
        with torch.cuda.amp.autocast(enabled=(self.mixed_precision and not pretrain)):
            images = [x.to('cuda', non_blocking=True) for x in images]
            features = [None for _ in range(len(images))]
            predictions = [None for _ in range(len(images))]
            adv_predictions = [None for _ in range(len(images))]
            for i in self.valid_scale:
                feat = self.extractors[i](images[i])
                pred = self.detector(feat)
                adv_pred = self.discriminator(feat)

                features[i] = feat
                predictions[i] = pred
                adv_predictions[i] = adv_pred

        targets = [{k:v.to('cuda', non_blocking=True) for k, v in target.items()} for target in targets]

        det_loss, det_loss_dict = self.loss_fn(predictions, adv_predictions, targets)
        dis_loss, dis_loss_dict = self.d_loss_fn(adv_predictions)

        loss_dict = {**det_loss_dict, **dis_loss_dict}

        if self.gp:
            penalty = self._gradient_penalty(features)
            
            dis_loss += self.gp_weight * penalty
            loss_dict['gradient_penalty'] = penalty.detach()

        return det_loss, dis_loss, loss_dict

    def _load_pretrained_model(self, model_path):
        def remove_model(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                if name.startswith('model.'):
                    name = name[6:]  # remove 'model.' of keys
                new_state_dict[name] = v
            return new_state_dict

        self.load_state_dict(remove_model(torch.load(model_path)), strict=False)
        for i in range(1, len(self.extractors)):
            self.extractors[i].load_state_dict(self.extractors[0].state_dict())

    def _gradient_penalty(self, features):
        size = features[0].size()
        batch_size = size[0]
        device = features[0].device

        penalty = 0
        for i, j in combinations(self.valid_scale, 2):
            alpha = torch.rand(batch_size, 1, 1, 1)
            alpha = alpha.expand(size)
            alpha = alpha.to(device)

            interpolated = alpha * features[i] + (1 - alpha) * features[j]
            # interpolated.requires_grad = True
            # interpolated = interpolated.to(device)

            pred_interpolated = self.discriminator(interpolated)

            gradients = torch.autograd.grad(inputs=interpolated, outputs=pred_interpolated, grad_outputs=torch.ones(pred_interpolated.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(batch_size, -1)
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            penalty += ((gradients_norm - 1) ** 2).mean()

        return penalty / len(list(combinations(self.valid_scale, 2)))


    # def _gradient_penalty(self, real_data, generated_data):
    #     batch_size = real_data.size()[0]

    #     # Calculate interpolation
    #     alpha = torch.rand(batch_size, 1, 1, 1)
    #     alpha = alpha.expand_as(real_data)
    #     alpha = alpha.to(real_data.device)
    #     interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    #     interpolated.requires_grad = True
    #     interpolated = interpolated.to(real_data.device)

    #     # Calculate probability of interpolated examples
    #     prob_interpolated = self.discriminator(interpolated)

    #     # Calculate gradients of probabilities with respect to examples
    #     gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
    #                            grad_outputs=torch.ones(prob_interpolated.size()).to(real_data.device),
    #                            create_graph=True, retain_graph=True)[0]

    #     # Gradients have shape (batch_size, num_channels, img_width, img_height),
    #     # so flatten to easily take norm per example in batch
    #     gradients = gradients.view(batch_size, -1)
    #     # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    #     # Derivatives of the gradient close to 0 can cause problems because of
    #     # the square root, so manually calculate norm and add epsilon
    #     gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    #     # Return gradient penalty
    #     return ((gradients_norm - 1) ** 2).mean()