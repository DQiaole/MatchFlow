from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse, configparser
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim

from network import RAFTGMA_QuadtreeFnet
import datasets
import evaluate
import evaluate_tile
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from common import torch_init_model
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(train_outputs, image1, image2, flow_gt, valid, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    flow_preds = train_outputs

    # original RAFT loss
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < MAX_FLOW)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None].float() * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW([{
        "params": [p for n, p in model.named_parameters() if
                   ('pos_block' not in n) and p.requires_grad],
    },
    {
        "params": [p for n, p in model.named_parameters() if
                   ('pos_block' in n) and p.requires_grad],
        "weight_decay": 0.0,
    }, ], lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def main(args):
    model = nn.DataParallel(RAFTGMA_QuadtreeFnet(args), device_ids=args.gpus)

    if args.restore_ckpt is not None:
        print('restore from ', args.restore_ckpt)
        torch_init_model(model, torch.load(args.restore_ckpt, map_location='cpu'), key='model')

    model.cuda()
    model.train()

    if args.stage != 'chairs' and args.raft:
        model.module.freeze_bn()

    print(f"Parameter Count: {count_parameters(model)}")

    if args.eval_only:
        for val_dataset in args.validation:
            if val_dataset == 'things':
                evaluate.validate_things(model.module)
            elif val_dataset == 'sintel':
                evaluate.validate_sintel(model.module)
            elif val_dataset == 'sintel_submission':
                evaluate.create_sintel_submission_vis(model.module, warm_start=True)
            elif val_dataset == 'kitti':
                evaluate_tile.validate_kitti(model.module)
            elif val_dataset == 'kitti_submission':
                evaluate_tile.create_kitti_submission(model)
        return

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000

    while total_steps <= args.num_steps:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]
            flow_predictions = model(image1, image2, iters=args.iters)
            loss, metrics = sequence_loss(flow_predictions, image1, image2, flow, valid, gamma=args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = args.output + f'/{logger.total_steps + 1}_{args.name}.pth'
                torch.save({
                    'iteration': total_steps,
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict()
                }, PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))

                logger.write_dict(results)

                model.train()

            total_steps += 1
            if total_steps > args.num_steps:
                break

    logger.close()
    PATH = args.output + f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    parser.add_argument('--matching_model_path', default='./ckpts/outdoor.ckpt', type=str,
                        help='path to pretrained matching model')
    parser.add_argument('--eval_only', action='store_true', default=False, help='eval only')
    parser.add_argument('--raft', action='store_true', default=False, help='do not use GMA module')

    args = parser.parse_args()

    from configs.default import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.backends.cudnn.benchmark = True

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    main(cfg)
