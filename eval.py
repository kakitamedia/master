import argparse
import datetime
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler


from model.config import cfg
from model.modeling.hourglass import MultiScaleHourglassNet
from model.modeling.resnet import get_resnet18
from model.engine.inference import multiscale_inference
from model.data.transforms.data_preprocess import EvalTransform
from model.data.transforms.transforms import MakeHeatmap
from model.data.datasets import build_dataset
from model.utils.misc import fix_model_state_dict
from model.modeling.build_model import Model


def eval(args, cfg):
    device = torch.device(cfg.MODEL.DEVICE)

    # if cfg.MODEL.DETECTOR_TYPE == 'hourglass':
    #     model = MultiScaleHourglassNet(cfg, inference=True).to(device)
    # elif cfg.MODEL.DETECTOR_TYPE == 'resnet18':
    #     model = get_resnet18(inference=True).to(device)
    model = Model(cfg).to(device)
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets')
    eval_transform = EvalTransform(cfg)
    eval_dataset = build_dataset(dataset_list=cfg.DATASETS.VAL, transform=eval_transform)
    sampler = SequentialSampler(eval_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    eval_loader = DataLoader(eval_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    
    extractor_path = os.path.join(cfg.OUTPUT_DIR, 'extractor', 'iteration_{}.pth'.format(args.iteration))
    detector_path = os.path.join(cfg.OUTPUT_DIR, 'detector', 'iteration_{}.pth'.format(args.iteration))

    model.extractors.load_state_dict(fix_model_state_dict(torch.load(extractor_path, map_location=lambda storage, loc:storage)))
    model.detector.load_state_dict(fix_model_state_dict(torch.load(detector_path, map_location=lambda storage, loc:storage)))
    print('Trained model was loaded from {}'.format(extractor_path))

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    model.eval()
    results = multiscale_inference(args, cfg, model, eval_dataset, eval_loader, device)
    print(results.stats[0])
    

def main():
    parser = argparse.ArgumentParser(description='Scale and Degradation Specific Object Detection')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE')
    parser.add_argument('--output_dirname', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--trained_model', type=str, default='weights/multiscale_centernet_pretrain.pth')
    parser.add_argument('--iteration', type=int, default=0)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = str(dt_now.date()) + '_' + str(dt_now.time())
    else:
        output_dirname = args.output_dirname
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_dirname)
    cfg.freeze()

    print('Running with config:\n{}'.format(cfg))

    eval(args, cfg)


if __name__ == '__main__':
    main()