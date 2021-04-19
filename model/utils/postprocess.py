import numpy as np

import torch
import torch.nn as nn


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


# def decode_predictions(predictions, K):
#     heatmap = predictions['heatmap']
#     wh = predictions['wh']
#     reg = predictions['reg']

#     # print(heatmap.shape)
#     # print(wh.shape)
#     # print(reg.shape)

#     batch_size, num_channels, height, width = heatmap.size()

#     heatmap = _nms(heatmap)

#     scores, inds, classes, ys, xs = _topk(heatmap, K=K)
    
#     reg = _transpose_and_gather_feat(reg, inds)
#     reg = reg.view(batch_size, K, 2)
#     xs = xs.view(batch_size, K, 1) + reg[:, :, 0:1]
#     ys = ys.view(batch_size, K, 1) + reg[:, :, 1:2]

#     wh = _transpose_and_gather_feat(wh, inds)
#     wh = wh.view(batch_size, K, 2)

#     classes = classes.view(batch_size, K, 1).float()
#     scores = scores.view(batch_size, K, 1)
#     boxes = torch.cat([xs - wh[..., 0:1] / 2,
#                        ys - wh[..., 1:2] / 2,
#                        xs + wh[..., 0:1] / 2,
#                        ys + wh[..., 1:2] / 2], dim=2)
    
#     # detections = torch.cat([boxes, scores, classes], dim=2)
#     detections = []
#     for i in range(batch_size):
#         detections.append([boxes[i], scores[i], classes[i]])

#     return detections


scale_dict = {0:4, 1:2, 2:1, 3:0.5}

def decode_predictions(predictions, K, i):
    heatmap = predictions['hm']
    wh = predictions['wh']
    reg = predictions['reg']

    batch_size, num_channels, height, width = heatmap.size()

    heatmap = _nms(heatmap)

    scores, inds, classes, ys, xs = _topk(heatmap, K=K)
    
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch_size, K, 2)
    xs = xs.view(batch_size, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch_size, K, 1) + reg[:, :, 1:2]

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch_size, K, 2)

    xs *= scale_dict[i]
    ys *= scale_dict[i]
    wh *= scale_dict[i]

    classes = classes.view(batch_size, K).float()
    scores = scores.view(batch_size, K)
    boxes = torch.cat([xs - wh[..., 0:1] / 2,
                       ys - wh[..., 1:2] / 2,
                       xs + wh[..., 0:1] / 2,
                       ys + wh[..., 1:2] / 2], dim=2)

    return boxes, classes, scores


def post_process(detections, scale=1, num_classes=80):
    detections = detections.detach().cpu().numpy()
    detections = detections.reshape(1, -1, detections.shape[2])
    detections = temp(detections, num_classes)

    for i in range(1, num_classes+1):
        detections[0][i] = np.array(detections[0][i], dtype=np.float32).reshape(-1, 5)
        detections[0][i][:, :4] /= scale

    print(detections)
    return detections[0]



def temp(detections, num_classes):
    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j+1] = np.concatenate([
                detections[i, inds, :4].astype(np.float32),
                detections[i, inds, 4:5].astype(np.float32)
            ], axis=1).tolist()
        ret.append(top_preds)
    return ret


def merge_results(cfg, detections):
    num_classes = cfg.MODEL.NUM_CLASSES
    max_per_image = cfg.TEST.MAX_OBJECTS
    results = {}
    for i in range(1, num_classes+1):
        results[i] = np.concatenate([detection[i].cpu() for detection in detections], axis=0).astype(np.float32)
        
    scores = np.hstack([results[i][:, 4] for i in range(1, num_classes+1)])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for i in range(1, num_classes + 1):
            mask = results[i][:, 4] >= thresh
            results[j] = results[j][mask]

    return results
