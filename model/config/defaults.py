from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.SCALE_FACTORS = [1, 2, 4]

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-4
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.MAX_ITER = 500000
_C.SOLVER.LR_DECAY = [300000, 450000]
_C.SOLVER.SYNC_BATCHNORM = True


_C.DATASET = CN()

_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.DATA_DIR = 'datasets/coco/train2017'
_C.DATASET.TRAIN.ANN_FILE = 'datasets/coco/annotations/instances_train2017.json'

_C.DATASET.VAL = CN()
_C.DATASET.VAL.DATA_DIR = 'datasets/coco/val2014'
_C.DATASET.VAL.ANN_FILE = 'datasets/coco/annotations/instances_val2014.json'

_C.MIXED_PRECISION = False
_C.OUTPUT_DIR = ''
_C.SEED = 123
