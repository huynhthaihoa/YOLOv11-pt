import copy
import csv
import os
import sys
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

# data_dir = '../Dataset/COCO'


def train(args):#, params):
    # Model
    # variant = params["variant"]
    if args.variant == "n":
        model = nn.yolo_v11_n(len(args.names))
    elif args.variant == "s":
        model = nn.yolo_v11_s(len(args.names))
    elif args.variant == "m":
        model = nn.yolo_v11_m(len(args.names))
    elif args.variant == "l":
        model = nn.yolo_v11_l(len(args.names))
    elif args.variant == "t":
        model = nn.yolo_v11_t(len(args.names))
    else:
        model = nn.yolo_v11_x(len(args.names))
    
    model.cuda()
    
    #Load pretrained models
    if args.pretrained:
        ckpt = f"v11_{args.variant}.pt"
        if not os.path.exists(ckpt):
            os.system(f"wget https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_{args.variant}.pt")
            dst = model.state_dict()
            src = torch.load(ckpt, 'cpu', weights_only=False)['model'].float().state_dict()
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            model.load_state_dict(state_dict=ckpt, strict=False)

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    args.weight_decay *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, args.weight_decay),
                                args.min_lr, args.momentum, nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    # Dataset
    filenames = []
    # with open(f'{data_dir}/train2017.txt') as f:
    #     for filename in f.readlines():
    #         filename = os.path.basename(filename.rstrip())
    #         filenames.append(f'{data_dir}/images/train2017/' + filename)

    sampler = None
    dataset = Dataset(args)#filenames, args.input_size, args, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    scheduler = util.LinearLR(args, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = [-1, -1, -1, -1]
    cache_last = [-1, -1, -1, -1]
    metrics = ["mAP", "mAP50", "recall", "precision"]
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, args)

    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)  # unscale gradients
                    # util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                last = test(args, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                # Update best mAP
                for i, metric in enumerate(metrics):
                    
                    # Save model
                    save = {'epoch': epoch + 1,
                            'model': copy.deepcopy(ema.ema)}

                    if last[i] > best[i]:
                        # Delete old best (if exists) and save new best
                        if best[i] != -1:
                            os.system(f'rm ./weights/best_{metric}_{best[i]}.pt')
                        best[i] = last[i]
                        torch.save(save, f=f'./weights/best_{metric}_{best[i]}.pt')

                    # Delete old last (if exists) and save new last
                    if cache_last[i] != -1:
                        os.system(f'rm ./weights/last_{metric}_{cache_last[i]}.pt')
                    cache_last[i] = last[i]
                    torch.save(save, f=f'./weights/last_{metric}_{cache_last[i]}.pt')

                    del save

    if args.local_rank == 0:
        for i, metric in enumerate(metrics):
            util.strip_optimizer(f'./weights/best_{metric}_{best[i]}.pt')  # strip optimizers
            util.strip_optimizer(f'./weights/last_{metric}_{cache_last[i]}.pt')  # strip optimizers

@torch.no_grad()
def test(args, model=None):
    # filenames = []
    # with open(f'{data_dir}/val2017.txt') as f:
    #     for filename in f.readlines():
    #         filename = os.path.basename(filename.rstrip())
    #         filenames.append(f'{data_dir}/images/val2017/' + filename)

    dataset = Dataset(args)#filenames, args.input_size, augment=args.use_augment)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        _, _, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=args.names)
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args):#, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    
    if args.variant == "n":
        model = nn.yolo_v11_n(len(args.names)).fuse()
    elif args.variant == "s":
        model = nn.yolo_v11_s(len(args.names)).fuse()
    elif args.variant == "m":
        model = nn.yolo_v11_m(len(args.names)).fuse()
    elif args.variant == "l":
        model = nn.yolo_v11_l(len(args.names)).fuse()
    elif args.variant == "t":
        model = nn.yolo_v11_t(len(args.names)).fuse()
    else:
        model = nn.yolo_v11_x(len(args.names)).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    parser.add_argument('--variant', default='n', type=str)
    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--names',                            help='Class list (default is [])',                              default = [], type=lambda s: [str(item) for item in s.split(',')])

    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
        
    # training hyperparams
    parser.add_argument('--min_lr', default=1e-4, type=float)
    parser.add_argument('--max_lr', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.937, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--warmup-epochs', default=3, type=int)
    parser.add_argument('--box', default=0.05, type=float)
    parser.add_argument('--cls', default=0.5, type=float)
    parser.add_argument('--dfl', default=0.5, type=float)
    
    # augmentation hyperparams
    parser.add_argument('--use_augment', action='store_true')

    parser.add_argument('--hsv_h', default=0.015, type=float)
    parser.add_argument('--hsv_s', default=0.7, type=float)
    parser.add_argument('--hsv_v', default=0.4, type=float)
    parser.add_argument('--degrees', default=0.0, type=float)
    parser.add_argument('--translate', default=0.1, type=float)
    parser.add_argument('--scale', default=0.5, type=float)
    parser.add_argument('--shear', default=0.0, type=float)
    parser.add_argument('--flip_ud', default=0.0, type=float)
    parser.add_argument('--flip_lr', default=0.0, type=float)
    parser.add_argument('--mosaic', default=1.0, type=float)
    parser.add_argument('--mix_up', default=0.0, type=float)
    
    if sys.argv.__len__() == 2:
        parser.convert_arg_line_to_args = util.convert_arg_line_to_args
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    if args.config != '':
        args = parser.parse_args()
        yaml_data = yaml.safe_load(open(args.config))#, Loader=yaml.FullLoader)
        args_dict = args.__dict__
        for key, value in yaml_data.items():
            if isinstance(value, list):
                for v in value:
                    args_dict[key].append(v)
            else:
                args_dict[key] = value 
                
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    profile(args)

    if args.train:
        train(args)
    if args.test:
        test(args)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
