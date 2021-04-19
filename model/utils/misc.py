import torch
from collections import OrderedDict
import cv2

def _sigmoid(x):
	y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
	return y


def str2bool(s):
	return s.lower() in ('true', '1')


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def overlay_heatmap(image, heatmap):
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(jet_heatmap, 0.4, image, 0.6, 0)
    
    return overlayed_image


def init_layer(layer):
    for m in layer:
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('ConvTranspose2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()