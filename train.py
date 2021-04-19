import argparse
import datetime
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.utils.sync_batchnorm import convert_model

from model.config import cfg
from model.modeling.build_model import ModelWithLoss
from model.data.transforms.data_preprocess import TrainTransform, TrainTargetTransform, TrainTragetTransform_ContentsMatch
from model.data.datasets import build_dataset
from model.engine.trainer import do_train
from model.data import samplers
from model.utils.misc import str2bool, fix_model_state_dict
from model.engine.loss_functions import CenterNetLoss

import shutil


def train(args, cfg):
    device = torch.device(cfg.MODEL.DEVICE)
    model = ModelWithLoss(cfg).to(device)
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    train_transform = TrainTransform(cfg)
    if cfg.SOLVER.CONTENTS_MATCH:
        target_transform = TrainTragetTransform_ContentsMatch(cfg)
    else:
        target_transform = TrainTargetTransform(cfg)
    train_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform, target_transform=target_transform)
    sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    eval_transform = None
    eval_target_transform = None
    eval_dataset = build_dataset(dataset_list=cfg.DATASETS.VAL, transform=eval_transform, target_transform=target_transform)

    if cfg.SOLVER.WEIGHT_FIX:
        for param in model.model.parameters():
            param.requires_grad = False
        
        for i in range(0, len(cfg.MODEL.DOWN_RATIOS)):
            for param in model.model.init_convs[i].parameters():
                param.requires_grad = True
        

    optimizer = torch.optim.Adam(list(model.extractors.parameters()) + list(model.detector.parameters()), lr=cfg.SOLVER.DETECTOR_LR)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.SOLVER.DISCRIMINATOR_LR)

    # if not args.scratch:
    #     model.model.load_state_dict(fix_model_state_dict(torch.load(cfg.PRETRAINED_MODEL, map_location=lambda storage, loc:storage)))
    #     print('Pretrained model was loaded from {}'.format(cfg.PRETRAINED_MODEL))

    if len(args.pretrained_extractor) != 0:
        print('Load Pretrained Extractor from {}'.format(args.pretrained_extractor))
        model.extractors.load_state_dict(fix_model_state_dict(torch.load(args.pretrained_extractor)))
        # model.model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(args.pretrained_model)))

    if len(args.pretrained_detector) != 0:
        print('Load Pretrained Detector from {}'.format(args.pretrained_detector))
        model.detector.load_state_dict(fix_model_state_dict(torch.load(args.pretrained_detector)))

    if args.resume_iter != 0:
        print('Resume from {}'.format(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        model.extractors.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'extractor', 'iteration_{}.pth'.format(args.resume_iter)))))
        model.detector.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'detector', 'iteration_{}.pth'.format(args.resume_iter)))))
        optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(args.resume_iter))))
        if cfg.SOLVER.DIS_TRAIN_RATIO != 0:
            model.discriminator.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'discriminator', 'iteration_{}.pth'.format(args.resume_iter)))))
            d_optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'd_optimizer', 'iteration_{}.pth'.format(args.resume_iter))))


    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to(device)

    if args.mixed_precision:
        [model.model, model.discriminator], [optimizer, d_optimizer] = apex.amp.initialize(
            [model.model, model.discriminator], [optimizer, d_optimizer], opt_level='O1')

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    # if args.tensorboard and not cfg.DEBUG:
    if not args.debug:
        summary_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None


    do_train(args, cfg, model, optimizer, d_optimizer, train_loader, device, summary_writer)


def main():
    parser = argparse.ArgumentParser(description='Scale and Degradetion Specific Object Detection')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('--output_dirname', type=str, default='', help='')
    parser.add_argument('--num_workers', type=int, default=12, help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--save_step', type=int, default=10000)
    parser.add_argument('--eval_step', type=int, default=9999999)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--mixed_precision', type=str2bool, default=False)
    # parser.add_argument('--tensorboard', type=str2bool, default=False)
    parser.add_argument('--scratch', type=str2bool, default=False)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--pretrained_extractor', type=str, default='')
    parser.add_argument('--pretrained_detector', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)

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
    if not args.debug and args.resume_iter == 0:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    train(args, cfg)
    

if __name__ == '__main__':
    main()
