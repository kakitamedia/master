import json
import os

import torch
from tqdm import tqdm
from copy import copy

from model.utils.postprocess import decode_predictions, post_process
from model.utils.nms import boxes_nms
from overlay_heatmap import OverlayHeatmap
from model.engine.loss_functions import MMD_loss

def multiscale_inference(args, cfg, model, eval_dataset, eval_loader, device, summary_wirter=None):
    overlay = OverlayHeatmap()
    features = [torch.empty((0, 6400)) for _ in range(3)]
    # mmd_loss = MMD_loss()
    mmds = [0, 0, 0] #[41, 12, 24]
    results = []
    j = 0
    for images in tqdm(eval_loader):
        batch_size = images[0].shape[0]
        boxes = torch.empty((batch_size, 0, 4))
        labels = torch.empty((batch_size, 0))
        scores = torch.empty((batch_size, 0))

        for i, image in enumerate(images):
            if i not in cfg.MODEL.VALID_SCALE:
                continue
            with torch.no_grad():
                image = image.to(device)
                predictions, feature = model(image, i)
                # feature = feature.to('cpu')
                # features = model(image, i)
                # batch, channel, height, width = feature.shape
                # center_point = [int(height/2), int(width/2)]
                # # print(feature[:, :, center_point[0]-80:center_point[0]+80, center_point[1]-80:center_point[1]+80].contiguous().view(1, -1).shape)
                # features[i] = torch.cat([features[i], feature[:, :, center_point[0]-5:center_point[0]+5, center_point[1]-5:center_point[1]+5].contiguous().view(1, -1)], dim=0)

                # print(feature[:, :, center_point[0]-80:center_point[0]+80, center_point[1]-80:center_point[1]+80].shape)
                # features[i] = torch.cat([features[i], feature[:, :, center_point[0]-80:center_point[0]+80, center_point[1]-80:center_point[1]+80]], dim=0)
                # features[i].append(feature[:, :, center_point[0]-80:center_point[0]+80, center_point[1]-80:center_point[1]+80])

                # overlay(copy(image), copy(predictions[-1]), 'temp/{}_x{}.png'.format(j, i))

                predictions = decode_predictions(predictions[-1], cfg.TEST.MAX_OBJECTS, i)
            
            boxes = torch.cat((boxes, predictions[0].cpu()), dim=1)
            labels = torch.cat((labels, predictions[1].cpu()), dim=1)
            scores = torch.cat((scores, predictions[2].cpu()), dim=1)

        for i in range(batch_size):
            results.append(multiscale_nms(boxes[i], labels[i], scores[i], cfg.TEST.NMS_THRESHOLD))
            

    return coco_evaluation(eval_dataset, results, cfg.OUTPUT_DIR)


def calculate_mmd_with_ae(args, cfg, model, auto_encoder, optimizer, train_loader, eval_loader):
    logging_loss = [0, 0, 0]
    ### AE Training
    model.mmd_calc = True
    model.eval()
    auto_encoder.train()
    for iteration, images in enumerate(train_loader, 1):
        for i, image in enumerate(images):
            with torch.no_grad():
                feature = model(image, i)
            loss = auto_encoder.train_forward(feature)
            loss = loss.mean()

            logging_loss[i] += loss.item()            
            
            loss.backward()
            optimizer.step()
    
        if iteration % args.log_step == 0:
            for i in range(len(logging_loss)):
                logging_loss[i] /= args.log_step
            print('===> Iter: {:07d}, Loss: {}'.format(iteration, logging_loss))
            logging_loss = [0, 0, 0]

    ### Calcurate MMD
    all_features = [[], [], []]
    crop_size = cfg.TEST.AE_INPUT_PATCH // 2
    auto_encoder.eval()
    for images in tqdm(eval_loader):
        for i, image in enumerate(images):
            with torch.no_grad():
                feature = model(image, i)

                batch, channel, height, width = feature.shape
                center_point = [int(height/2), int(width/2)]
                feature = feature[:, :, center_point[0]-crop_size:center_point[0]+crop_size, center_point[1]-crop_size:center_point[1]+crop_size]
                
                feature = auto_encoder.inference_forward(feature)
                all_features.append(feature.to('cpu'))


    for i in range(len(all_features)):
        all_features[i] = torch.cat(all_features[i], dim=0)

    print('MMD between x1 and x2: {}'.format(MMD(all_features[0], all_features[1])))
    print('MMD between x2 and x4: {}'.format(MMD(all_features[1], all_features[2])))
    print('MMD between x4 and x1: {}'.format(MMD(all_features[2], all_features[0])))



def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    print(numerator)
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


def multiscale_nms(boxes, labels, scores, nms_thresh):
    keep = boxes_nms(boxes, scores, nms_thresh)

    boxes = boxes[keep, :]
    labels = labels[keep]
    scores = scores[keep]

    # scores, ind = torch.topk(scores, k=100)
    # labels = labels[ind]
    # boxes = boxes[ind, :]

    boxes = boxes.tolist()
    labels = labels.tolist()
    scores = scores.tolist()

    return [boxes, labels, scores]

# def do_evaluate(args, cfg, model, eval_dataset, eval_loader, device, summary_writer=None):
#     results = []
#     total_loss = 0
#     for i in tqdm(range(len(eval_dataset))):
#         images, targets = eval_dataset.__getitem__(i)
#         images = images[0]
#         targets = targets[0]
#         images = images.unsqueeze(0).to(device)
#         # print(images.min(), images.mean(), images.max())
#         for k in targets.keys():  
#             targets[k] = torch.from_numpy(targets[k]).unsqueeze(0).to(device)

#         with torch.no_grad():
#             predictions, loss = model(images, targets)

#         # print(predictions[cfg.MODEL.NSTACK-1], targets)
#         total_loss += loss.item()

#         detection = decode_predictions(predictions[cfg.MODEL.NSTACK-1], K=cfg.TEST.MAX_OBJECTS)
#         # detection = decode_predictions(targets, K=cfg.TEST.MAX_OBJECTS)
#         # results[i] = post_process(detection, num_classes=cfg.MODEL.NUM_CLASSES)
#         # results.append(post_process(detection, num_classes=cfg.MODEL.NUM_CLASSES))
#         results.append(detection)

#         # results = merge_results(cfg, results)

#     # print(results)
#     return total_loss, coco_evaluation(eval_dataset, results, cfg.OUTPUT_DIR)


def do_evaluate_multiscale(args, cfg, model, eval_dataset, eval_loader, device, summary_writer=None):
    results = []
    for images, targets in tqdm(eval_loader):
        for i in range(len(images)):
            images[i] = images[i].to(device)
            for k in targets[i].keys():
                targets[i][k] = targets[i][k].to(device)

        with torch.no_grad():
            result = []
            for i in range(4):
                predictions = model(images[0], targets=targets, i=i)
                boxes, scores, labels = decode_predictions(predictions[cfg.MODEL.NSTACK-1], K=cfg.TEST.MAX_OBJECTS)
                if i == 0:
                    all_boxes = boxes
                    all_scores = scores
                    all_labels = labels
                else:
                    all_boxes = torch.cat((all_boxes, boxes), dim=1)
                    all_scores = torch.cat((all_scores, scores), dim=1)
                    all_labels = torch.cat((all_labels, labels), dim=1)

            all_boxes = all_boxes.squeeze()
            all_scores = all_scores.squeeze()
            all_labels = all_labels.squeeze()

            # print(all_boxes.shape, all_scores.shape, all_labels.shape)

            keep = boxes_nms(all_boxes, all_scores, cfg.TEST.NMS_THRESHOLD, cfg.TEST.MAX_OBJECTS)
            all_boxes = all_boxes[keep, :]
            all_scores = all_scores[keep]
            all_labels = all_labels[keep]

            # result.extend([all_boxes, all_scores, all_labels])

            results.append([all_boxes, all_scores, all_labels])

    
        # heatmap, wh, reg = merge_predictions(heatmap, wh, reg)

        # predictions = {'heatmap': heatmap, 'wh': wh, 'reg': reg}

        # detection = decode_predictions(predictions, K=cfg.TEST.MAX_OBJECTS)
        # results.append(detection)

    return coco_evaluation(eval_dataset, results, cfg.OUTPUT_DIR)


def merge_predictions(heatmap, wh, reg):
    batch, channel, height, width = wh[0].shape
    heatmap = torch.stack(heatmap)
    wh = torch.stack(wh)
    reg = torch.stack(reg)

    index = heatmap.argmax(dim=0)
    _, batch, channel, height, width = wh.shape
    # print(wh.shape)
    # print(reg.shape)
    # print(index.shape)
    # print(wh[index].shape)
    wh = wh[index, [i for i in range(batch)], [i for i in range(channel)], [i for i in range(height)], [i for i in range(width)]]
    _, batch, channel, height, width = reg.shape
    # reg = reg[index, [i for i in range(batch)], [i for i in range(channel)], [i for i in range(height)], [i for i in range(width)]]

    heatmap = heatmap.max(dim=0)

    return hetamap, wh, reg


def do_evaluate(args, cfg, model, eval_dataset, eval_loader, device, summary_writer=None):
    results = []
    total_loss = 0
    for images, targets in tqdm(eval_loader):
        images = images[0].to(device)
        targets = targets[0]
        for k in targets.keys():
            targets[k] = targets[k].to(device)

        with torch.no_grad():
            predictions = model(images)

        total_loss += loss.mean().item()

        # for i in range(batch):
        detection = decode_predictions(predictions[cfg.MODEL.NSTACK-1], K=cfg.TEST.MAX_OBJECTS)
        # results.append(detection)
        results.extend(detection)

    total_loss /= len(eval_loader)

    return total_loss, coco_evaluation(eval_dataset, results, cfg.OUTPUT_DIR)


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def coco_evaluation(dataset, predictions, output_dir):
    coco_results = []
    for i, pred in enumerate(predictions):
        boxes, labels, scores = pred
        image_id, annotation = dataset.get_annotation(i)
        class_mapper = dataset.contiguous_id_to_coco_id

        if len(labels) == 0:
            # print('continue')
            continue

        coco_results.extend(
            [
                {
                    'image_id': image_id,
                    'category_id': class_mapper[labels[k]+1],
                    'bbox': [box[0], box[1], (box[2] - box[0]), (box[3] - box[1])],
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    iou_type = 'bbox'
    json_result_file = os.path.join(output_dir, iou_type + ".json")
    os.makedirs(os.path.dirname(json_result_file), exist_ok=True)
    print('Writing resuls to {}...'.format(json_result_file))
    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(json_result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


