import argparse
import os
import shutil
import numpy as np
import datetime
import wandb

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.optim.lr_scheduler import MultiStepLR

from model.config import cfg
from model.engine.trainer import do_train
from model.modeling.build_model import ModelWithLoss
from model.data.samplers import IterationBasedBatchSampler
from model.utils.sync_batchnorm import convert_model
from model.utils.misc import str2bool, fix_model_state_dict
from model.data.datasets.coco_dataset import COCODataset

def parse_args():
    parser = argparse.ArgumentParser(description='pytorch training code')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--run_name', type=str, default='', help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--eval_step', type=int, default=50000, help='')
    parser.add_argument('--save_step', type=int, default=50000, help='')
    parser.add_argument('--num_gpus', type=int, default=1, help='')
    parser.add_argument('--num_workers', type=int, default=16, help='')
    parser.add_argument('--resume_iter', type=int, default=0, help='')

    return parser.parse_args()

def train(args, cfg):
    model = ModelWithLoss(cfg).to('cuda')
    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to('cuda')
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    data_loader = {}

    train_dataset = [
        COCODataset(cfg.DATASET.TRAIN.DATA_DIR, cfg.DATASET.TRAIN.ANN_FILE, transform=TrainTransform(factor)),
        for factor in cfg.MODEL.SCALE_FACTORS        
        ]
    sampler = RandomSampler(train_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    data_loader['train'] = train_loader

    if args.eval_step != 0:
        val_transforms = 
        val_dataset = 
        sampler = SequentialSampler(val_dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=args._eval_batchsize, drop_last=False)
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

        data_loader['val'] = val_loader

    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR)
    scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.MIXED_PRECISION)

    if args.resume_iter != 0:
        print('Resume from {}'.format(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter)))))
        optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(args.resume_iter))))
        scheduler.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'scheduler', 'iteration_{}.pth'.format(args.resume_iter))))
        scaler.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'scaler', 'iteration_{}.pth'.format(args.resume_iter))))

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    do_train(args, cfg, model, optimizer, scheduler, scaler, data_loader)


def main():
    args = parse_args()

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if len(args.run_name) == 0:
        dt_now = datetime.datetime.now()
        args.run_name = str(dt_now.date()) + '_' + str(dt_now.time())

    output_dirname = os.path.join('output', args.run_name)
    cfg.OUTPUT_DIR = output_dirname
    cfg.freeze()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)
    else:
        raise Exception('GPU not found.')

    if not args.debug:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        if not len(args.config_file) == 0:
            shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))
        wandb.init(project='project_name', entity='kakita', config=dict(yaml=cfg))
        wandb.run.name = args.run_name
        wandb.run.save()

    train(args, cfg)


if __name__ == '__main__':
    main()
