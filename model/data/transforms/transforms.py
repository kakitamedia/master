import numpy as np
import cv2
import random
import math
import time

import torch


def multiple_list(list1, list2):
    assert len(list1) == len(list2)
    output_list = [None for _ in range(len(list1))]
    for i in range(len(output_list)):
        output_list[i] = list1[i] * list2[i]

    return output_list

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def extract_valid_boxes(boxes, scale_range):
    box_sizes = boxes[:, 2:] - boxes[:, :2] # [[w, h]]
    box_sizes = box_sizes.max(1)

    m1 = box_sizes > scale_range[0]
    m2 = box_sizes < scale_range[1]
    
    # print(box_sizes, m1*m2)

    return m1 * m2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            start = time.time()
            img, boxes, labels = t(img, boxes, labels)
            end = time.time()
            # print(end - start)

        return img, boxes, labels


class RandomResize(object):
    def __init__(self):
        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.resize_range = (0.6, 1.4)

    def __call__(self, image, boxes=None, labels=None):
        interp_method = self.interp_methods[random.randint(0, 4)]
        
        fx, fy = random.uniform(self.resize_range[0], self.resize_range[1]), random.uniform(self.resize_range[0], self.resize_range[1])
        # fx, fy = 512/640, 512/640
        height, width, _ = image.shape
        size = (int(width*fx), int(height*fy))
        image = cv2.resize(image, size, interpolation=interp_method)

        if boxes is not None:
            boxes[:, 1::2] = boxes[:, 1::2] * fy
            boxes[:, 0::2] = boxes[:, 0::2] * fx

        height, width, _ = image.shape
        height = (height // 4) * 4
        width = (width // 4) * 4

        image = image[:height, :width, :]
        if boxes is not None:
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        
        return image, boxes, labels


class Resize(object):
    def __init__(self, size=512):
        self.size = size
        self.interp_method = cv2.INTER_CUBIC

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size), interpolation=self.interp_method)

        return image, boxes, labels


class SampleCrop(object):
    def __init__(self, cfg, scale_range=(0, 640), max_shift=200, crop_size=512):
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.crop_size = crop_size


    def __call__(self, image, boxes=None, labels=None):
        if boxes is None and labels is None:
            return image, boxes, labels

        height, width, _ = image.shape
        box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        valid_boxes = extract_valid_boxes(boxes, self.scale_range)
        if (~valid_boxes).all():
            image_center = np.zeros(2)
            image_center[0] = np.random.randint(width)
            image_center[1] = np.random.randint(height)

        else:
            valid_centers = box_centers[valid_boxes]
            
            image_center = valid_centers[np.random.randint(len(valid_centers))]
            image_center += np.random.uniform(-self.max_shift, self.max_shift, (image_center.shape))
            image_center = image_center.astype(np.int16)

        left_pad = int(np.clip(self.crop_size/2 - image_center[0], 0, None))
        right_pad = int(np.clip(self.crop_size/2 - (width - image_center[0]), 0, None))
        upper_pad = int(np.clip(self.crop_size/2 - image_center[1], 0, None))
        under_pad = int(np.clip(self.crop_size/2 - (height - image_center[1]), 0, None))

        image = np.pad(image, [(upper_pad, under_pad), (left_pad, right_pad), (0, 0)], 'constant')
        boxes[:, :2] = boxes[:, :2] + np.array([left_pad, upper_pad])
        boxes[:, 2:] = boxes[:, 2:] + np.array([left_pad, upper_pad])

        image_center += np.array([left_pad, upper_pad]).astype(np.uint16)

        crop_area = np.array([image_center[0] - self.crop_size/2, image_center[1] - self.crop_size/2,
                              image_center[0] + self.crop_size/2, image_center[1] + self.crop_size/2]).astype(np.uint16)
        boxes[:, :2] = boxes[:, :2] - crop_area[:2]
        boxes[:, 2:] = boxes[:, 2:] - crop_area[:2]
        
        boxes = np.clip(boxes, 0, self.crop_size)
        box_sizes = boxes[:, 2:] - boxes[:, :2]

        mask = box_sizes > 0
        mask = mask.all(1)

        mask = multiple_list(mask, valid_boxes)

        image = image[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]
        boxes = boxes[mask]
        labels = labels[mask]

        # cv2.imwrite('/home/akita_15002/workspace/object_detection/SR_CenterNet/images/www.jpg', image)

        return image, boxes, labels


class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), boxes, labels

class ToNumpy(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.numpy().transpose((1, 2, 0)), boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ConvertToInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.uint8), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean=0):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, cfg):
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)/255
        image -= self.mean
        image /= self.std

        return image.astype(np.float32), boxes, labels


class Denormalize(object):
    def __init__(self, cfg):
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD

    def __call__(self, image, boxes=None, labels=None):
        image *= self.std
        image += self.mean
        image = image*255

        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes=None, classes=None):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class Clamp(object):
    def __init__(self, min=0., max=255.):
        self.min = min
        self.max = max

    def __call__(self, image):
        return torch.clamp(image, min=self.min, max=self.max)


class CenterCrop(object):
    def __init__(self, crop_size=64):
        self.crop_size = crop_size

    def __call__(self, hr_img, lr_img=None):
        height, width, _ = hr_img.shape
        crop_area = [(height - self.crop_size)/2, (height + self.crop_size)/2, (width - self.crop_size)/2, (width + self.crop_size)/2]
        crop_area = [int(i) for i in crop_area]
        
        return hr_img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]], lr_img


class ConstantPadding(object):
    def __init__(self, image_size=640):
        self.image_size = image_size

    def __call__(self, image, boxes=None, labels=None):
        height, width, channel = image.shape
        padded_image = np.zeros((self.image_size, self.image_size, channel))

        padded_image[:height, :width, :] = image
    
        return padded_image, boxes, labels


# class SampleCrop_MakeHeatmap(object):
#     def __init__(self, cfg, down_ratio=4):
#         # self.input_size = cfg.INPUT.IMAGE_SIZE
#         self.num_classes = cfg.MODEL.NUM_CLASSES
#         self.max_objects = cfg.INPUT.MAX_OBJECTS

#         self.down_ratio = down_ratio
#         # self.scale_range = (0, 128) if down_ratio == 1/2 else (32, 128)
#         self.scale_range = (0, 256)
#         self.max_shift = 24
#         self.crop_size = 128

#         # self.down_ratio = [4, 2, 1, 0.5]
#         # self.scale_range = [(128, 512), (64, 256), (32, 128), (0, 64)]
#         # self.max_shift = [200, 100, 50, 25]
#         # self.crop_size = [512, 256, 128, 64]

#         # self.down_ratio = 4
#         # self.crop_size = 128
#         # self.max_shift = 32
#         # self.scale_range = (0, 128)

#         # self.down_ratio = 1/2
#         # self.crop_size = 128
#         # self.max_shift = 32
#         # self.scale_range = (0, 128)

#     def __call__(self, image, boxes, labels):
#         height, width, _ = image.shape
#         hm_height, hm_width = height // self.down_ratio, width // self.down_ratio
#         hm_height, hm_width = int(hm_height), int(hm_width)
#         for i in range(len(boxes)):
#             for j in range(len(boxes[i])):
#                 boxes[i][j] /= self.down_ratio

#         box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
#         valid_masks = extract_valid_boxes(boxes, self.scale_range)

#         heatmap = np.zeros((self.num_classes, self.crop_size, self.crop_size), dtype=np.float32)
#         invalid_heatmap = np.zeros((self.num_classes, self.crop_size, self.crop_size), dtype=np.float32)

#         if (~valid_masks).all():
#             image_center = np.zeros(2)
#             image_center[0] = np.random.randint(hm_width)
#             image_center[1] = np.random.randint(hm_height)
#         else:
#             valid_centers = box_centers[valid_masks]
#             image_center = valid_centers[np.random.randint(len(valid_centers))]
#             image_center += np.random.uniform(-self.max_shift, self.max_shift, (image_center.shape))
#             image_center = image_center.astype(np.int16)

#         left_pad = int(np.clip(self.crop_size/2 - image_center[0], 0, None))
#         right_pad = int(np.clip(self.crop_size/2 - (hm_width - image_center[0]), 0, None))
#         upper_pad = int(np.clip(self.crop_size/2 - image_center[1], 0, None))
#         under_pad = int(np.clip(self.crop_size/2 - (hm_height - image_center[1]), 0, None))

#         image_center += np.array([left_pad, upper_pad]).astype(np.uint16)

#         crop_area = np.array([image_center[0] - self.crop_size/2, image_center[1] - self.crop_size/2,
#                               image_center[0] + self.crop_size/2, image_center[1] + self.crop_size/2]).astype(np.uint16)

#         image = np.pad(image, [(int(round(upper_pad*self.down_ratio)), int(round(under_pad*self.down_ratio))), (int(round(left_pad*self.down_ratio)), int(round(right_pad*self.down_ratio))), (0, 0)], 'constant')

        

#         # MakeHeatmap
#         tic = time.time()
#         for i in range(len(boxes)):
#             box = boxes[i]
#             label = labels[i]
#             class_id = label - 1
#             valid_label = valid_masks[i]
#             hm = np.zeros((hm_height, hm_width), dtype=np.float32)
#             invalid_hm = np.zeros((hm_height, hm_width), dtype=np.float32)

#             box_width = (box[2] - box[0])
#             box_height = (box[3] - box[1])
#             center = ((box[:2] + box[2:]) / 2)
#             center_int = center.astype(np.uint16)
#             radius = max(0, int(gaussian_radius((math.ceil(box_height), math.ceil(box_width)))))

#             hm = draw_umich_gaussian(hm, center.astype(np.uint16), radius)
#             hm = np.pad(hm, [(upper_pad, under_pad), (left_pad, right_pad)], 'constant')
#             heatmap[class_id] = np.maximum(heatmap[class_id], hm[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]])
#             if not valid_label:
#                 invalid_hm = draw_umich_gaussian(invalid_hm, center.astype(np.uint16), radius)
#                 invalid_hm = np.pad(invalid_hm, [(upper_pad, under_pad), (left_pad, right_pad)], 'constant')
#                 invalid_heatmap[class_id] = np.maximum(invalid_heatmap[class_id], invalid_hm[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]])


#         boxes[:, :2] = boxes[:, :2] + np.array([left_pad, upper_pad])
#         boxes[:, 2:] = boxes[:, 2:] + np.array([left_pad, upper_pad])

#         boxes[:, :2] = boxes[:, :2] - crop_area[:2]
#         boxes[:, 2:] = boxes[:, 2:] - crop_area[:2]

#         box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        
#         m1 = box_centers > 0
#         m1 = m1.all(1)
#         m2 = box_centers < self.crop_size
#         m2 = m2.all(1)

#         mask = m1 * m2

#         image = image[int(round(crop_area[1]*self.down_ratio)):int(round(crop_area[3]*self.down_ratio)), int(round(crop_area[0]*self.down_ratio)):int(round(crop_area[2]*self.down_ratio)), :]
#         # heatmap = heatmap[:, crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]
#         # invalid_heatmap = invalid_heatmap[:, crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]

#         boxes = boxes[mask]
#         labels = labels[mask]

        
#         wh = np.zeros((self.max_objects, 2), dtype=np.float32)
#         reg = np.zeros((self.max_objects, 2), dtype=np.float32)
#         ind = np.zeros((self.max_objects), dtype=np.int64)
#         reg_mask = np.zeros((self.max_objects), dtype=np.uint8)
        
#         for i in range(len(boxes)):
#             box = boxes[i]
#             label = labels[i]
#             class_id = label - 1

#             box_width = box[2] - box[0]
#             box_height = box[3] - box[1]
#             center = (box[:2] + box[2:]) / 2
#             center_int = center.astype(np.uint16)

#             wh[i] = 1. * box_width, 1. * box_height
#             ind[i] = int(center_int[1] * self.crop_size + center_int[0])
#             reg[i] = center - center_int
#             reg_mask[i] = 1

#         ret = {'heatmap': heatmap, 'invalid_heatmap': invalid_heatmap.clip(1e-4, 1-1e-4), 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

#         height, width, _ = image.shape
#         # print(self.crop_size*self.down_ratio, height, width)
#         error_flag = False
#         if height != int(self.crop_size*self.down_ratio):
#             error_flag = True
#         if width != int(self.crop_size*self.down_ratio):
#             error_flag = True

#         if error_flag:
#             print(left_pad, right_pad, upper_pad, under_pad, crop_area)

#         return image, ret

# class SampleCrop_MakeHeatmap(object):
#     def __init__(self, cfg, down_ratio=4):
#         self.num_classes = cfg.MODEL.NUM_CLASSES
#         self.max_objects = cfg.INPUT.MAX_OBJECTS

#         self.down_ratio = down_ratio
#         self.scale_range = (0, 256)
#         self.max_shift = 24
#         self.crop_size = 128

    
#     def __call__(self, image, boxes, labels):
#         height, width, _ = image.shape
#         hm_height, hm_width = self.crop_size // self.down_ratio, self.crop_size // self.down_ratio
#         hm_height, hm_width = int(hm_height), int(hm_width)
#         for i in range(len(boxes)):
#             for j in range(len(boxes[i])):
#                 boxes[i][j] /= self.down_ratio

#         box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
#         valid_masks = extract_valid_boxes(boxes, self.scale_range)

#         if (~valid_masks).all():
#             image_center = np.zeros(2)
#             image_center[0] = np.random.randint(width)
#             image_center[1] = np.random.randint(height)
#         else:
#             valid_centers = box_centers[valid_masks]
#             image_center = valid_centers[np.random.randint(len(valid_centers))]
#             image_center += np.random.uniform(-self.max_shift, self.max_shift, (image_center.shape))
#             image_center = image_center.astype(np.int16)

#         left_pad = int(np.clip(self.crop_size/2 - image_center[0], 0, None))
#         right_pad = int(np.clip(self.crop_size/2 - (width - image_center[0]), 0, None))
#         upper_pad = int(np.clip(self.crop_size/2 - image_center[1], 0, None))
#         under_pad = int(np.clip(self.crop_size/2 - (height - image_center[1]), 0, None))

#         image = np.pad(image, [(int(round(upper_pad*self.down_ratio)), int(round(under_pad*self.down_ratio))), (int(round(left_pad*self.down_ratio)), int(round(right_pad*self.down_ratio))), (0, 0)], 'constant')

#         boxes[:, :2] = boxes[:, :2] + np.array([left_pad, upper_pad])
#         boxes[:, 2:] = boxes[:, 2:] + np.array([left_pad, upper_pad])

#         image_center += np.array([left_pad, upper_pad]).astype(np.uint16)

#         crop_area = np.array([image_center[0] - self.crop_size/2, image_center[1] - self.crop_size/2,
#                               image_center[0] + self.crop_size/2, image_center[1] + self.crop_size/2]).astype(np.uint16)

#         print(left_pad, right_pad, upper_pad, under_pad, image.shape, crop_area*4)

#         boxes[:, :2] = boxes[:, :2] - crop_area[:2]
#         boxes[:, 2:] = boxes[:, 2:] - crop_area[:2]
        
#         boxes = np.clip(boxes, 0, self.crop_size)
#         box_sizes = boxes[:, 2:] - boxes[:, :2]

#         mask = box_sizes > 0
#         mask = mask.all(1)

#         image = image[int(round(crop_area[1]*self.down_ratio)):int(round(crop_area[3]*self.down_ratio)), int(round(crop_area[0]*self.down_ratio)):int(round(crop_area[2]*self.down_ratio)), :]
#         boxes = boxes[mask]
#         labels = labels[mask]

#         heatmap = np.zeros((self.num_classes, self.crop_size, self.crop_size), dtype=np.float32)
#         wh = np.zeros((self.max_objects, 2), dtype=np.float32)
#         reg = np.zeros((self.max_objects, 2), dtype=np.float32)
#         ind = np.zeros((self.max_objects), dtype=np.int64)
#         reg_mask = np.zeros((self.max_objects), dtype=np.uint8)

#         for i in range(len(boxes)):
#             box = boxes[i]
#             label = labels[i]
#             class_id = label - 1

#             box_width = box[2] - box[0]
#             box_height = box[3] - box[1]
#             center = (box[:2] + box[2:]) / 2
#             center_int = center.astype(np.uint8)
#             radius = max(0, int(gaussian_radius((math.ceil(box_height), math.ceil(box_width)))))

#             heatmap[class_id] = draw_umich_gaussian(heatmap[class_id], center.astype(np.uint8), radius)
#             wh[i] = 1. * box_width, 1. * box_height
#             ind[i] = int(center_int[1] * hm_width + center_int[0])
#             reg[i] = center - center_int
#             reg_mask[i] = 1
            
#         ret = {'heatmap': heatmap, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

#         return image, ret

class SampleCrop_MakeHeatmap(object):
    def __init__(self, cfg, scale_range=(0, 640), max_shift=200, crop_size=512, down_ratio=4):
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.max_objects = cfg.INPUT.MAX_OBJECTS

        self.down_ratio = down_ratio
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.crop_size = crop_size


    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
        hm_height, hm_width = height // self.down_ratio, width // self.down_ratio
        hm_height, hm_width = int(hm_height), int(hm_width)
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j] /= self.down_ratio

        box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        valid_masks = extract_valid_boxes(boxes, self.scale_range)

        heatmap = np.zeros((self.num_classes, self.crop_size, self.crop_size), dtype=np.float32)
        invalid_heatmap = np.zeros((self.num_classes, self.crop_size, self.crop_size), dtype=np.float32)

        if (~valid_masks).all():
            image_center = np.zeros(2)
            image_center[0] = np.random.randint(hm_width)
            image_center[1] = np.random.randint(hm_height)
        else:
            valid_centers = box_centers[valid_masks]
            image_center = valid_centers[np.random.randint(len(valid_centers))]
            image_center += np.random.uniform(-self.max_shift, self.max_shift, (image_center.shape))
            image_center = image_center.astype(np.int16)

        left_pad = int(np.clip(self.crop_size/2 - image_center[0], 0, None))
        right_pad = int(np.clip(self.crop_size/2 - (hm_width - image_center[0]), 0, None))
        upper_pad = int(np.clip(self.crop_size/2 - image_center[1], 0, None))
        under_pad = int(np.clip(self.crop_size/2 - (hm_height - image_center[1]), 0, None))

        image_center += np.array([left_pad, upper_pad]).astype(np.uint16)

        crop_area = np.array([image_center[0] - self.crop_size/2, image_center[1] - self.crop_size/2,
                              image_center[0] + self.crop_size/2, image_center[1] + self.crop_size/2]).astype(np.uint16)

        image = np.pad(image, [(int(round(upper_pad*self.down_ratio)), int(round(under_pad*self.down_ratio))), (int(round(left_pad*self.down_ratio)), int(round(right_pad*self.down_ratio))), (0, 0)], 'constant')
        

        # MakeHeatmap
        tic = time.time()
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            class_id = label - 1
            valid_label = valid_masks[i]
            hm = np.zeros((hm_height, hm_width), dtype=np.float32)
            invalid_hm = np.zeros((hm_height, hm_width), dtype=np.float32)

            box_width = (box[2] - box[0])
            box_height = (box[3] - box[1])
            center = ((box[:2] + box[2:]) / 2)
            center_int = center.astype(np.uint16)
            radius = max(0, int(gaussian_radius((math.ceil(box_height), math.ceil(box_width)))))

            hm = draw_umich_gaussian(hm, center.astype(np.uint16), radius)
            hm = np.pad(hm, [(upper_pad, under_pad), (left_pad, right_pad)], 'constant')
            heatmap[class_id] = np.maximum(heatmap[class_id], hm[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]])
            if not valid_label:
                invalid_hm = draw_umich_gaussian(invalid_hm, center.astype(np.uint16), radius)
                invalid_hm = np.pad(invalid_hm, [(upper_pad, under_pad), (left_pad, right_pad)], 'constant')
                invalid_heatmap[class_id] = np.maximum(invalid_heatmap[class_id], invalid_hm[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]])


        boxes[:, :2] = boxes[:, :2] + np.array([left_pad, upper_pad])
        boxes[:, 2:] = boxes[:, 2:] + np.array([left_pad, upper_pad])

        boxes[:, :2] = boxes[:, :2] - crop_area[:2]
        boxes[:, 2:] = boxes[:, 2:] - crop_area[:2]

        box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        
        m1 = box_centers > 0
        m1 = m1.all(1)
        m2 = box_centers < self.crop_size
        m2 = m2.all(1)

        mask = m1 * m2

        image = image[int(round(crop_area[1]*self.down_ratio)):int(round(crop_area[3]*self.down_ratio)), int(round(crop_area[0]*self.down_ratio)):int(round(crop_area[2]*self.down_ratio)), :]
        # heatmap = heatmap[:, crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]
        # invalid_heatmap = invalid_heatmap[:, crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]

        boxes = boxes[mask]
        labels = labels[mask]

        
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)
        
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            class_id = label - 1

            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            center = (box[:2] + box[2:]) / 2
            center_int = center.astype(np.uint16)

            wh[i] = 1. * box_width, 1. * box_height
            ind[i] = int(center_int[1] * self.crop_size + center_int[0])
            reg[i] = center - center_int
            reg_mask[i] = 1

        ret = {'heatmap': heatmap, 'invalid_heatmap': invalid_heatmap.clip(1e-4, 1-1e-4), 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        height, width, _ = image.shape
        # print(self.crop_size*self.down_ratio, height, width)
        error_flag = False
        if height != int(self.crop_size*self.down_ratio):
            error_flag = True
        if width != int(self.crop_size*self.down_ratio):
            error_flag = True

        if error_flag:
            print(left_pad, right_pad, upper_pad, under_pad, crop_area)

        return image, ret

        
class MakeHeatmap(object):
    def __init__(self, cfg, down_ratio=4 ,eval=False):
        # self.input_size = cfg.INPUT.IMAGE_SIZE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.max_objects = cfg.INPUT.MAX_OBJECTS

        self.down_ratio = down_ratio
        self.eval = eval

    def __call__(self, image, boxes, labels):
        output_height, output_width, _ = image.shape
        # _, output_height, output_width = image.shape
        num_objects = min(len(boxes), self.max_objects)
        output_height = int(output_height / self.down_ratio)
        output_width = int(output_width / self.down_ratio)
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j] /= self.down_ratio


        heatmap = np.zeros((self.num_classes, output_height, output_width), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, output_width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, output_height - 1)

        for i in range(num_objects):
            box = boxes[i]
            label = labels[i]
            class_id = label - 1 # remove __background__ label

            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            center = (box[:2] + box[2:]) / 2
            center_int = center.astype(np.uint8)
            radius = max(0, int(gaussian_radius((math.ceil(box_height), math.ceil(box_width)))))

            heatmap[class_id] = draw_umich_gaussian(heatmap[class_id], center.astype(np.uint8), radius)
            wh[i] = 1. * box_width, 1. * box_height
            ind[i] = int(center_int[1] * output_width + center_int[0])
            reg[i] = center - center_int
            reg_mask[i] = 1

        

        ret = {'heatmap': heatmap, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        return image, ret


class FactorResize(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width/self.factor), int(height/self.factor)), cv2.INTER_AREA)

        return image


class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape

        image_center = np.zeros(2)
        image_center[0] = np.random.randint(width)
        image_center[1] = np.random.randint(height)

        left_pad = int(np.clip(self.crop_size/2 - image_center[0], 0, None))
        right_pad = int(np.clip(self.crop_size/2 - (width - image_center[0]), 0, None))
        upper_pad = int(np.clip(self.crop_size/2 - image_center[1], 0, None))
        under_pad = int(np.clip(self.crop_size/2 - (height - image_center[1]), 0, None))

        image_center += np.array([left_pad, upper_pad]).astype(np.uint16)

        crop_area = np.array([image_center[0] - self.crop_size/2, image_center[1] - self.crop_size/2,
                              image_center[0] + self.crop_size/2, image_center[1] + self.crop_size/2]).astype(np.uint16)

        image = np.pad(image, [(upper_pad, under_pad), (left_pad, right_pad), (0, 0)], 'constant')

        image = image[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]

        if boxes != None:
            boxes[:, :2] = boxes[:, :2] - crop_area[:2]
            boxes[:, 2:] = boxes[:, 2:] - crop_area[:2]

            boxes = np.clip(boxes, 0, self.crop_size)
            box_sizes = boxes[:, 2:] - boxes[:, :2]

            mask = box_sizes > 0
            mask = mask.all(1)
            mask = multiple_list(mask, boxes)

            boxes = boxes[mask]
            labels = labels[mask]

        return image, boxes, labels