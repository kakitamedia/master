import numpy as np
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

class ToNumpy(object):
    def __call__(self, image):
        return image.numpy().transpose((1, 2, 0))

class ConvertToInts(object):
    def __call__(self, image):
        return np.clip(image, 0, 255).astype(np.uint8)

class Denormalize(object):
    def __init__(self):
        self.mean = [0.408, 0.447, 0.470]
        self.std =  [0.289, 0.274, 0.278]

    def __call__(self, image):
        image *= self.std
        image += self.mean
        image = image*255

        return image

class Resize(object):
    def __init__(self, size=512):  
        self.size = size
        self.interp_method = cv2.INTER_CUBIC

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size), interpolation=self.interp_method)

        return image


class OverlayHeatmap(object):
    def __init__(self):
        self.image_transforms = Compose([
            ToNumpy(),
            Denormalize(),
            ConvertToInts(),
        ])
        self.heatmap_transforms = Compose([
            ToNumpy(),
            Resize(size=640),
        ])

    def __call__(self, image, target, save_path):
        image, heatmap = image.cpu(), target['hm'].cpu()
        if len(image.shape) == 4:
            image, heatmap = image.squeeze(0), heatmap.squeeze(0)
        # print(image.shape, heatmap.shape)
        image = self.image_transforms(image)
        heatmap = self.heatmap_transforms(heatmap)

        heatmap = (np.max(heatmap, 2)*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlayed_image = cv2.addWeighted(heatmap, 0.6, image, 0.4, 0)
        cv2.imwrite(save_path, overlayed_image)
