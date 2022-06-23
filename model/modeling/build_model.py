import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.model =

    def forward(self, x):
        x = x.to('cuda', non_blocking=True)
        pred = self.model(x)

        return pred


class ModelWithLoss(Model):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.discriminator = 
        self.loss_fn = 

        self.mixed_precision = cfg.MIXED_PRECISION

    def forward(self, x, target):
        with torch.cuda.amp.autocast(enabled=(self.mixed_precision)):
            x = x.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

            det_preds, features = self.model(x)
            adv_preds = self.discriminator(features)

            loss = self.loss_fn(det_preds, adv_preds, target)

            return loss