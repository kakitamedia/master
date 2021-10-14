import torch
import torch.nn as nn

from torch import linalg as LA

from model.config import cfg
from model.modeling.base_network import *
from model.modeling.extractor import build_extractors
from model.modeling.build_model import Model


# model = nn.Sequential(
#     ConvBlock(3, 64, kernel_size=3, stride=1, padding=1, normalization='batch', activation='lrelu'),
#     ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, normalization='batch', activation='lrelu'),
#     ConvBlock(64, 3, kernel_size=3, stride=1, padding=1, normalization='batch', activation='lrelu'),
# )

model = Model(cfg)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

loss_fn = nn.L1Loss()

model = model.to('cuda')
loss_fn = loss_fn.to('cuda')

for _ in range(1000):
    # image = [torch.tensor([i for i in range(1*3*128*128)]).view(1, 3, 128, 128).float().to('cuda') for i in range()
    images = [torch.ones((1, 3, 128*scale, 128*scale)) for scale in cfg.MODEL.DOWN_RATIOS]
    # target = torch.tensor([i for i in range(5*3*128*128)]).view(5, 3, 128, 128).float().to('cuda')

    predictions = model(images)

    loss = 0
    for i in range(len(predictions)):
        for key in predictions[i]:
            print(predictions[i][key].shape)
            loss += predictions[i][key].mean()

    loss.backward()
    optimizer.step()

    norm = 0
    for param in model.parameters():
        if param.grad is None:
            print('before', param.grad, param.shape)
        else:
            norm += LA.norm(param.grad)
            print('before', LA.norm(param.grad), param.shape)
        # break
    print(norm)

    optimizer.zero_grad()

    # norm = 0
    # for param in model.detector.parameters():
    #     norm += LA.norm(param.grad)
    #     print('after', LA.norm(param.grad))
    #     # break
    # print(norm)