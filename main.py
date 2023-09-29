import os

os.environ['OMP_NUM_THREADS'] = '1'
import json
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from src.datasets import *
from src.models.mvdet import MVDet
from src.utils.logger import Logger
from src.utils.draw_curve import draw_curve
from src.utils.str2bool import str2bool
from src.trainer import PerspectiveTrainer


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    if args.dataset == 'carlax':
        with open(args.cfg_path, "r") as fp:
            dataset_config = json.load(fp)
        base = CarlaX(dataset_config, args.host, args.port, args.tm_port, args.spawn_strategy, args.carla_seed)

        args.task = 'mvdet'
        args.num_workers = 0

        train_set = frameDataset(base, split='trainval', world_reduce=args.world_reduce,
                                 img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                 img_kernel_size=args.img_kernel_size,
                                 augmentation=args.augmentation, interactive=args.interactive)
        test_set = frameDataset(base, split='test', world_reduce=args.world_reduce,
                                img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                img_kernel_size=args.img_kernel_size, interactive=args.interactive)
    else:
        if args.dataset == 'wildtrack':
            base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
        elif args.dataset == 'multiviewx':
            base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
        else:
            raise Exception('must choose from [wildtrack, multiviewx]')

        args.task = 'mvdet'

        train_set = frameDataset(base, split='trainval', world_reduce=args.world_reduce,
                                 img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                 img_kernel_size=args.img_kernel_size,
                                 augmentation=args.augmentation)
        test_set = frameDataset(base, split='test', world_reduce=args.world_reduce,
                                img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                img_kernel_size=args.img_kernel_size)

    if args.interactive:
        args.lr /= 5
        # args.epochs *= 2

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    lr_settings = f'base{args.base_lr_ratio}other{args.other_lr_ratio}' + \
                  f'control{args.control_lr}std_lr_factor{args.std_lr_factor}' + \
                  f'vfratio{args.vf_ratio}' if args.interactive else ''
    logdir = f'logs/{args.dataset}/{"DEBUG_" if is_debug else ""}{"FINE_TUNE_" if args.fine_tune else ""}' \
             f'{args.arch}_{args.aggregation}_down{args.down}_seed{args.seed}_carlaseed{args.carla_seed}_' \
             f'{f"RL_reward{args.reward}_spawn{args.spawn_strategy}_arch{args.rl_variant}_logstd{args.log_std}_" if args.interactive else ""}' \
             f'lr{args.lr}{lr_settings}_b{args.batch_size}_e{args.epochs}_' \
             f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}' if not args.eval \
        else f'logs/{args.dataset}/EVAL_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
    os.makedirs(logdir, exist_ok=True)
    copy_tree('src', logdir + '/scripts/src')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVDet(train_set, args.arch, args.aggregation,
                  args.use_bottleneck, args.hidden_dim, args.outfeat_dim,
                  args.rl_variant if args.interactive else '', seed=args.seed, log_std=args.log_std).cuda()
    
    # different modes of training
    if args.eval:
        # eval must be with resume
        assert args.resume is not None, 'must provide a checkpoint for evaluation'
        print(f'loading checkpoint: logs/{args.dataset}/{args.resume}')
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    elif args.fine_tune:
        assert args.resume is not None, 'must provide a checkpoint for fine-tuning'
        # resume training is used for fine-tuning the output head
        print(f'loading checkpoint: logs/{args.dataset}/{args.resume}')
        pretrained_dict = torch.load(f'logs/{args.dataset}/{args.resume}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # NOTE: only fine-tune the output head
        for n, p in model.named_parameters():
            if 'world' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # set a lower starting learning rate for fine-tuning
        args.lr /= 4
        # set a lower number of training epochs for fine-tuning
        args.epochs //=  2
    
    # load base checkpoint
    elif args.interactive:
        with open(f'logs/{args.dataset}/{args.arch}_{args.spawn_strategy}_expert.txt', 'r') as fp:
            load_dir = fp.read()
        print(load_dir)
        pretrained_dict = torch.load(f'{load_dir}/model.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'control' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    param_dicts = [{"params": [p for n, p in model.named_parameters()
                               if 'base' not in n and 'control' not in n and p.requires_grad],
                    "lr": args.lr * args.other_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, },
                   {"params": [p for n, p in model.named_parameters() if 'control' in n and 'log_std' not in n and p.requires_grad],
                    "lr": args.control_lr, }, 
                   {"params": [p for n, p in model.named_parameters() if 'control' in n and 'log_std' in n and p.requires_grad],
                    "lr": args.control_lr * args.std_lr_factor, }]
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def warmup_lr_scheduler(epoch, warmup_epochs=0.1 * args.epochs):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, args, args.fine_tune)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []

    # learn if 1. not in evaluation mode for all other rl_variants; 2. only when fine-tuning the random_action model
    if not args.eval and (args.rl_variant != "random_action" or args.fine_tune):
        # test for initial model performance
        if args.fine_tune:
            trainer.test(test_loader)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, scheduler, log_interval=args.log_interval)
            if epoch % max(args.epochs // 10, 1) == 0:
                print('Testing...')
                test_loss, test_prec = trainer.test(test_loader)

                # draw & save
                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_prec_s.append(train_prec)
                test_loss_s.append(test_loss)
                test_prec_s.append(test_prec[0])
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s,
                           train_prec_s, test_prec_s)
                torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))

    print('Test loaded model...')
    print(logdir)
    trainer.test(test_loader)

    # close carla env before quitting
    if args.dataset == 'carlax':
        base.env.close()

    # copy base model file if 
    if args.dataset == 'carlax' and not args.interactive:
        with open(f'logs/{args.dataset}/{args.arch}_{args.spawn_strategy}_expert.txt', 'w') as fp:
            fp.write(logdir)


if __name__ == '__main__':
    # Common settings
    parser = argparse.ArgumentParser(description='camera position control for multiview classification & detection')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'shufflenet0.5'])
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--fine_tune', action='store_true', help='fine tune the output head')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-d', '--dataset', type=str, default='carlax',
                        choices=['wildtrack', 'multiviewx', 'carlax'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for task network')
    parser.add_argument('--control_lr', type=float, default=1e-4, help='learning rate for MVcontrol')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0)
    parser.add_argument('--other_lr_ratio', type=float, default=1.0)
    parser.add_argument('--std_lr_factor', type=float, default=100., help="factor of log_std learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options')
    parser.add_argument('--log_interval', type=int, default=100)
    # CarlaX settings
    parser.add_argument('--cfg_path', type=str, help='path to the config file')
    parser.add_argument('--spawn_strategy', type=str, choices=['uniform', 'gmm'])
    parser.add_argument('--carla_seed', type=int, default=2023, help='random seed for CarlaX')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='CarlaX host; defaults to "127.0.0.1"')
    parser.add_argument('--port', type=int, default=2000, help='CarlaX port; defaults to 2000')
    parser.add_argument('--tm_port', type=int, default=8000, help='TrafficManager port; defaults to 8000')
    # MVcontrol settings
    parser.add_argument('--interactive', action='store_true', help='interactive mode')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor (default: 0.99)')
    parser.add_argument('--log_std', type=float, default=-1., help='initial log std of action distribution')
    parser.add_argument('--vf_ratio', type=float, default=0.5, help='value loss ratio')
    parser.add_argument('--reward', type=str, help='type of reward used',
                        choices=['epi_loss', 'delta_loss', 'epi_cover_mean', 'epi_cover_max',
                                 'step_cover', 'epi_moda', 'delta_moda', "cover+delta_moda"])
    parser.add_argument('--rl_variant', type=str, help='architecture variants of the RL module',
                        choices=["random_action", "conv_base", "conv_deep_leaky"])
    # Multiview detection specific settings
    parser.add_argument('--reID', action='store_true', help='reID task')
    parser.add_argument('--aggregation', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--down', type=int, default=1, help='down sample the image to 1/N size')
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', action='store_true', help='use mse loss for regression')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    args = parser.parse_args()

    main(args)
