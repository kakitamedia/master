import torch.nn as nn

class ModelWithLoss(nn.Module):
    def __init__(self, cfg):
        super(ModelWithLoss, self).__init__()

        self.model = 
        self.loss_fn = 

    def forward(self, x, target):
        x = x.to('cuda')
        target = target.to('cuda')
        pred = self.model(x)
        loss = self.loss_fn(pred, target)

        return loss


