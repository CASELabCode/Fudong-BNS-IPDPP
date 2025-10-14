import argparse
import datetime
import json
import math
import sys

import numpy as np
import os
import time
from pathlib import Path

from torchvision.datasets import CIFAR100
from torchvision import transforms

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory
from timm.utils import accuracy

from datasets.imbalance_cifar import ImbalanceCIFAR100

from utils import misc
from sampling import sampler

from models import models_resnet, models_simclr, models_linprob
from utils.lt_metrics import compute_class_counts, categorize_classes, evaluate_model

from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from typing import Iterable
import utils.lr_sched as lr_sched

from datasets import data_augmentation

import warnings
warnings.filterwarnings('ignore')



def get_args_parser():
    parser = argparse.ArgumentParser('BNS Linear Probing on CIFAR-100', add_help=False)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--imbalanced_factor', default=100, type=int, help='imbalanced factor')
    parser.add_argument('--input_size', default=32, type=int, help='images input size')
    parser.add_argument('--embed_dim', default=128, type=int, help='embed dimensions')
    parser.add_argument('--num_views', default=1, type=int, help='pairs of data augmentation')

    parser.add_argument('--num_classes', default=100, type=int, help='number of classes')
    parser.add_argument('--num_samples', default=50, type=int, help='number of samples per class')

    parser.add_argument('--resume', default='',help='resume from checkpoint')
    parser.add_argument('--eval', default='', help='checkpoint to evaluate')

    # dataset
    parser.add_argument('-dr', '--data_root', type=str, default='/mnt/data/cfiar100')
    parser.add_argument('-sf', '--save_freq', type=int, default=5, help='saving frequency')

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer decay (default: 0.75)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir/cifar100/lineprob_bns_cifar100',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/cifar100/lineprob_bns_cifar100',
                        help='path where to tensorboard log')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_transform = data_augmentation.get_simclr_pipeline_transform(args.input_size, args.num_views)
    test_transform = transforms.ToTensor()

    original_train_set = ImbalanceCIFAR100(root=args.data_root, imb_factor=1/args.imbalanced_factor, train=True, transform=train_transform, download=True)
    test_set = CIFAR100(root=args.data_root, train=False, transform=test_transform, download=False)

    # Compute class counts and categorize classes
    class_counts = compute_class_counts(original_train_set)
    many_shot_classes, medium_shot_classes, few_shot_classes = categorize_classes(class_counts, many_threshold=100, few_threshold=20)

    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    base_encoder = models_resnet.ResNet34(num_classes=args.num_classes)
    pre_model = models_simclr.SimCLR(base_encoder, projection_dim=args.embed_dim)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pre_model.load_state_dict(checkpoint['model'])

    model = models_linprob.LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=args.num_classes)
    model.to(device)

    # freeze all but the linear classifier
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.lin.named_parameters():
        p.requires_grad = True

    train_set = sampler.sample_balanced_subset(original_train_set, samples_per_class=args.num_samples)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()


    # evaluate the model only
    if args.eval:
        checkpoint = torch.load(args.eval, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        test_stats = evaluate(test_loader, model, device)
        print(f"Accuracy of the network on the {len(test_set)} test images: {test_stats['acc1']:.1f}%")

        many_shot_acc, medium_shot_acc, few_shot_acc = evaluate_model(model, test_loader, many_shot_classes,
                                                                      medium_shot_classes, few_shot_classes)
        print(
            '* Many-Shot Acc@1 {many_shot_acc:.3f} Medium-Shot Acc@1 {medium_shot_acc:.3f} Few-Shot Acc@1 {few_shot_acc:.3f}'
            .format(many_shot_acc=many_shot_acc, medium_shot_acc=medium_shot_acc, few_shot_acc=few_shot_acc))

        exit(0)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(test_loader, model, device)
        print(f"Accuracy of the network on the {len(test_set)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        many_shot_acc, medium_shot_acc, few_shot_acc = evaluate_model(model, test_loader, many_shot_classes,
                                                                      medium_shot_classes, few_shot_classes)
        print(
            '* Many-Shot Acc@1 {many_shot_acc:.3f} Medium-Shot Acc@1 {medium_shot_acc:.3f} Few-Shot Acc@1 {few_shot_acc:.3f}'
            .format(many_shot_acc=many_shot_acc, medium_shot_acc=medium_shot_acc, few_shot_acc=few_shot_acc))

        if args.log_dir is not None:
            log_stats = {
                'many-shot': f"{many_shot_acc * 100:.2f}%",
                'medium-shot': f"{medium_shot_acc * 100:.2f}%",
                'few-shot': f"{few_shot_acc * 100:.2f}%",
                'test_acc': f"{test_stats['acc1']:.2f}%",
                'max_test_acc': f"{max_accuracy:.2f}%",
                'epoch': epoch
            }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device)
        target = target.to(device)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
