from yacs.config import CfgNode as CN

_C = CN()


_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 80
_C.MODEL.DOWN_RATIOS = [4, 2, 1]
_C.MODEL.VALID_SCALE = [0, 1, 2]

_C.MODEL.EXTRACTOR = CN()
_C.MODEL.EXTRACTOR.TYPE = 'unet'

_C.MODEL.DETECTOR = CN()
_C.MODEL.DETECTOR.IMAGE_INPUT = False
_C.MODEL.DETECTOR.TYPE = 'resnet18'
_C.MODEL.DETECTOR.INPUT_CHANNEL = 64
_C.MODEL.DETECTOR.ACTIVATION = 'lrelu'
_C.MODEL.DETECTOR.NORMALIZATION = 'batch'

_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.TYPE = 'residual'
_C.MODEL.DISCRIMINATOR.INPUT_CHANNEL = 64
_C.MODEL.DISCRIMINATOR.HIDDEN_CHANNEL = 64
_C.MODEL.DISCRIMINATOR.ACTIVATION = 'lrelu'
_C.MODEL.DISCRIMINATOR.NORMALIZATION = 'spectral'

# ---------------------
_C.MODEL.DISCRIMINATOR.MASKING = False  # using masking (True) or not (False) in discriminator
_C.MODEL.DISCRIMINATOR.MASKING_CLASS_AGNOSTIC = False  # instance_level (True) or class_level (False)
# Normalize the loss only if (MODEL.DISCRIMINATOR.MASKING) = True AND
#   (MODEL.DISCRIMINATOR.NORMALIZE_LOSS_WITH_MASK = True)
_C.MODEL.DISCRIMINATOR.NORMALIZE_LOSS_WITH_MASK = True
# ---------------------

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.MAX_ITER = 900000
_C.SOLVER.LR_DECAY = [400000, 800000]
_C.SOLVER.SYNC_BATCHNORM = True
_C.SOLVER.IMAGENET_PRETRAINED = True
_C.SOLVER.IMAGENET_PRETRAINED = True
_C.SOLVER.IMAGENET_PRETRAINED_MODEL = 'weights/pretrained_unet_extractor.pth'
_C.SOLVER.PRETRAINED = False
_C.SOLVER.PRETRAINED_MODEL = ''
_C.SOLVER.CONTENTS_MATCH = False
_C.SOLVER.ADV_LOSS_FN = 'wasserstain'

_C.SOLVER.EXTRACTOR = CN()
_C.SOLVER.EXTRACTOR.WEIGHT_FIX = False

_C.SOLVER.DETECTOR = CN()
_C.SOLVER.DETECTOR.LR = 5e-4
_C.SOLVER.DETECTOR.HM_LOSS_WEIGHT = 1.
_C.SOLVER.DETECTOR.HM_LOSS_ALPHA = 2
_C.SOLVER.DETECTOR.HM_LOSS_BETA = 4
_C.SOLVER.DETECTOR.WH_LOSS_WEIGHT = 0.1
_C.SOLVER.DETECTOR.REG_LOSS_WEIGHT = 1.
_C.SOLVER.DETECTOR.ADV_LOSS_WEIGHT = 0.1
_C.SOLVER.DETECTOR.RECON_LOSS_WEIGHT = 1.
_C.SOLVER.DETECTOR.GRADIENT_CLIP = 0.
_C.SOLVER.DETECTOR.INIT_TRAIN_ITER = 5000
_C.SOLVER.DETECTOR.WEIGHT_FIX = False

_C.SOLVER.DISCRIMINATOR = CN()
_C.SOLVER.DISCRIMINATOR.LR = 2.5e-4
_C.SOLVER.DISCRIMINATOR.ADV_LOSS_WEIGHT = 0.1
_C.SOLVER.DISCRIMINATOR.GRADIENT_CLIP = 0.
_C.SOLVER.DISCRIMINATOR.INIT_TRAIN_ITER = 5000
_C.SOLVER.DISCRIMINATOR.GP = False
_C.SOLVER.DISCRIMINATOR.GP_WEIGHT = 1.


_C.SOLVER.DATA = CN()
_C.SOLVER.DATA.RESIZE_RANGE = (0.6, 1.4)
_C.SOLVER.DATA.HEATMAP_SIZE = (128, 128)
_C.SOLVER.DATA.BOX_THRESHOLD = [0.3, 0.95, 0.95]
_C.SOLVER.DATA.MAX_OBJECTS = 128


_C.INPUT = CN()
_C.INPUT.MEAN = [0.408, 0.447, 0.470]
_C.INPUT.STD = [0.289, 0.274, 0.278]


_C.DATASET = CN()
_C.DATASET.TRAIN = ('coco_2017_train',)
_C.DATASET.VAL = ('coco_2014_minival',)


_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.5
_C.TEST.MAX_OBJECTS = 100
_C.TEST.NMS = True
_C.TEST.NMS_THREDSHOLD = 0.5


_C.MIXED_PRECISION = False
_C.OUTPUT_DIR = ''
_C.SEED = 123
