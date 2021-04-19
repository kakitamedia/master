import os
import numpy as np
import cv2
from PIL import Image
import time
from joblib import Parallel, delayed

from torch.utils.data import Dataset

from copy import copy
from overlay_heatmap import OverlayHeatmap


class COCODataset(Dataset):
    class_names = ('__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, data_dir, ann_file, transform=None, target_transform=None, remove_empty=False, pretrain=False):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        if self.remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

        self.overlay_heatmap = OverlayHeatmap()

        num_workers = 24
        # self.images = self.load_all_images(num_workers)
        # print('WWWWWWWWWWWWWWWWWWWWww', len(self.ids))

    def load_all_images(self, num_workers):
        print('loading all images into memory...')
        def _load_image(i):
            image_id = self.ids[i]
            images[image_id] = self._read_image(image_id)

        images = {}
        Parallel(n_jobs=num_workers, backend='threading')([delayed(_load_image)(index) for index in range(len(self.ids))])

        return images

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        # image = self.images[image_id]

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            image, target = self.target_transform(image, boxes, labels)
            # self.overlay_heatmap(copy(image[1]), copy(target[1]), 'temp/{}.png'.format(image_id))
            
            return image, target

        return image


    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def _read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)[:, :, ::-1]
        return image
