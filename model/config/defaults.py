from yacs.config import CfgNode as CN
import os

# hostname = os.uname()[1]
# if not hostname in ['dl10', 'dl30', 'dl31']:

hostname = 'dl001'

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda'

_C.MODEL.NUM_CLASSES = 80
_C.MODEL.EXTRACTOR_TYPE = 'normal'
_C.MODEL.EXTRACTOR_FEAT = 64
_C.MODEL.EXTRACTOR_OUTPUT_CHANNEL = 64
_C.MODEL.DETECTOR_TYPE = 'centernet'
_C.MODEL.DETECTOR_ACTIVATION = 'lrelu'
_C.MODEL.DETECTOR_NORM = 'batch'
_C.MODEL.DISCRIMINATOR_ACTIVATION = 'lrelu'
_C.MODEL.DISCRIMINATOR_TYPE = 'normal'
_C.MODEL.DISCRIMINATOR_INPUT_CHANNEL = 64
_C.MODEL.D_LOSS_TYPE = 'cross_entropy'
_C.MODEL.DISCRIMINATOR_NORM = 'spectral'
_C.MODEL.VALID_SCALE = [0, 1, 2]
# _C.MODEL.VALID_SCALE = [0]
_C.MODEL.DOWN_RATIOS = [4, 2, 1]

_C.MODEL.HOURGLASS = CN()
_C.MODEL.HOURGLASS.NSTACK = 2
_C.MODEL.HOURGLASS.DIMS = [256, 256, 384, 384, 384, 512]
_C.MODEL.HOURGLASS.MODULES = [2, 2, 2, 2, 2, 4]

_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = 512
_C.INPUT.MAX_OBJECTS = 128
_C.INPUT.MEAN = [0.408, 0.447, 0.470]
_C.INPUT.STD = [0.289, 0.274, 0.278]

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 900000
_C.SOLVER.LR_DACAY = 400000
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.DETECTOR_LR = 2.5e-4
_C.SOLVER.DISCRIMINATOR_LR = 5e-5
_C.SOLVER.HEATMAP_LOSS_WEIGHT = 1
_C.SOLVER.WH_LOSS_WEIGHT = 0.1
_C.SOLVER.REG_LOSS_WEIGHT = 1.
_C.SOLVER.ADV_LOSS_INIT_WEIGHT = 0.1
_C.SOLVER.ADV_LOSS_GAIN = 0.

_C.SOLVER.GEN_TRAIN_RATIO = 1
_C.SOLVER.DIS_TRAIN_RATIO = 1
_C.SOLVER.INIT_DIS_TRAIN = 500
_C.SOLVER.INIT_GEN_TRAIN = 0
_C.SOLVER.SYNC_BATCHNORM = True
_C.SOLVER.WEIGHT_FIX = False
_C.SOLVER.SCRATCH = False
_C.SOLVER.GRADIENT_PENALTY = False
_C.SOLVER.GRADIENT_PENALTY_WEIGHT = 10

_C.SOLVER.CONTENTS_MATCH = False

_C.DATASETS = CN()
_C.DATASETS.TRAIN = ('coco_2017_train',)
_C.DATASETS.VAL = ('coco_2014_minival',)
# _C.DATASETS.VAL = ('coco_2014_tinyval',)
_C.DATASETS.SCALE_RANGES = [[0, 640], [0, 320], [0, 160]]
_C.DATASETS.MAX_SHIFTS = [200, 100, 50]
_C.DATASETS.CROP_SIZES = [512, 256, 128]

_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.5
_C.TEST.MAX_OBJECTS = 100

_C.TEST.AE_INPUT_PATCH = 64
_C.TEST.AE_LATENT_SIZE = 128
_C.TEST.AE_TRAIN_ITERATION = 50000
_C.TEST.AE_TRAIN_BATCH_SIZE = 32
_C.TEST.AE_TRAIN_LR = 1e-4

_C.OUTPUT_DIR = os.path.join('output', hostname)
# _C.PRETRAINED_MODEL = 'weights/multiscale_centernet_pretrain.pth'
_C.PRETRAINED_MODEL = 'weights/temp.pth'

_C.SEED = 123
_C.DEBUG = False