import os

hostname = os.uname()[1]
if not hostname in ['dl10', 'dl20', 'dl30', 'dl31']:
    hostname = 'dl001'
    # hostname = 'local'

class DatasetCatalog:
    DATA_DIR = 'datasets/{}'.format(hostname)
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "voc/VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "voc/VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "voc/VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        'coco_2014_tinyval': {
            'data_dir': 'coco/val2014',
            'ann_file': 'coco/annotations/instances_tinyval2014.json'
        },
        'coco_2017_train': {
            'data_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'data_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        }
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))