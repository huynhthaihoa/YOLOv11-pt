import copy
import csv
import os
import random
import sys
import warnings
import time

import cv2
import torch
import tqdm
import yaml
import numpy

from argparse import ArgumentParser
from loguru import logger

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
    
    #Load MS COCO pretrained model
    if args.pretrained:
        ckpt = f"v11_{args.variant}.pt"
        if not os.path.exists(ckpt):
            os.system(f"wget https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_{args.variant}.pt")
        # dst = model.state_dict()
        # src = torch.load(ckpt, 'cpu', weights_only=False)['model'].float().state_dict()
        model = util.load_weight(model, ckpt)
        # ckpt = {}
        # for k, v in src.items():
        #     if k in dst and v.shape == dst[k].shape:
        #         print(k)
        #         ckpt[k] = v
        # model.load_state_dict(state_dict=ckpt, strict=False)

    # Load custom pretrained model
    if args.weight_path:
        model = util.load_weight(model, args.weight_path)
    
    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    args.weight_decay *= args.batch_size * args.world_size * accumulate / 64
    # end_learning_rate = args.lr0 * args.lrf

    optimizer = torch.optim.SGD(util.set_params(model, args.weight_decay),
                                args.lr0, args.momentum, nesterov=True)

    # EMA
    if args.use_ema and args.local_rank == 0:
        ema = util.EMA(model) 
    else:
        ema = None
    # if args.local_rank == 0 else None

    # Dataset
    # filenames = []
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

    # eval_dataset = Dataset(args, False, False)
    # eval_loader = data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4,
    #                          pin_memory=True, collate_fn=Dataset.collate_fn)
    
    # Scheduler
    steps_per_epoch = len(loader)
    # num_total_steps = args.epochs * steps_per_epoch
    
    scheduler = util.LinearLR(args, steps_per_epoch)

    # if args.use_early_stopping:
    #     early_stopping = util.EarlyStopping(patience=args.lr_patience, verbose=True)
    #     eval_loss_meter = util.AverageMeter()
    #     eval_avg_box_loss = util.AverageMeter()
    #     eval_avg_cls_loss = util.AverageMeter()
    #     eval_avg_dfl_loss = util.AverageMeter()

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = [-1, -1, -1, -1]
    cache_last = [-1, -1, -1, -1]
    metrics = ["mAP", "mAP50", "recall", "precision"]
    
    # Auto mixed precision training
    if args.use_amp:
        amp_scaler = torch.amp.GradScaler()
    else:
        amp_scaler = None
        
    criterion = util.ComputeLoss(model, args)

    with open(f'{args.log_dir}/step.csv', 'w') as log:
        if args.local_rank == 0:
            dict_logger = csv.DictWriter(log, fieldnames=['epoch',
                                                    'box', 'cls', 'dfl',
                                                    'Recall', 'Precision', 'mAP@50', 'mAP'])
        dict_logger.writeheader()

        for epoch in range(args.epochs):

            # if args.use_early_stopping and early_stopping.early_stop:
            #     break             
            
            # training part
            model.train()
            
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=steps_per_epoch)

            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets, _) in p_bar:
                
                step = i + steps_per_epoch * epoch
                
                # if args.use_ema:
                scheduler.step(step, optimizer)

                samples = samples.cuda()#.float() #/ 255

                assert not torch.any(torch.isnan(samples)), "Input is not NaN!"
                
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Forward: Make predictions for this batch
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                assert not torch.any(torch.isnan(loss_box)), "Box Loss is not NaN!"
                assert not torch.any(torch.isnan(loss_cls)), "Cls Loss is not NaN!"
                assert not torch.any(torch.isnan(loss_dfl)), "Dfl Loss is not NaN!"

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode
                
                total_loss = loss_box + loss_cls + loss_dfl

                assert not torch.any(torch.isnan(total_loss)), "Loss is not NaN!"
                
                if args.use_amp:
                    # Backward: Compute the loss and its gradients
                    amp_scaler.scale(total_loss).backward()

                    # Optimize: adjust learning weights
                    # for param_group in optimizer.param_groups:
                    #     current_lr = (args.lr0 - end_learning_rate) * (1 - step / num_total_steps) ** 0.9 + end_learning_rate
                    #     param_group['lr'] = current_lr

                    if step % accumulate == 0:
                        # amp_scaler.unscale_(optimizer)  # unscale gradients
                        # util.clip_gradients(model)  # clip gradients
                        amp_scaler.step(optimizer)  # optimizer.step
                        amp_scaler.update()
                        # optimizer.zero_grad()
                        if ema:
                            ema.update(model)
                else:
                    # Backward: Compute the loss and its gradients
                    total_loss.backward()

                    # Optimize: adjust learning weights
                    # for param_group in optimizer.param_groups:
                    #     current_lr = (args.lr0 - end_learning_rate) * (1 - step / num_total_steps) ** 0.9 + end_learning_rate
                    #     param_group['lr'] = current_lr

                    if step % accumulate == 0:
                        # util.clip_gradients(model)  # clip gradients
                        optimizer.step()
                        # optimizer.zero_grad()
                        if ema:
                            ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)
                    
                # for param in model.parameters():
                #     print(param.data)
            #
            # ====================================
            # validation part
            
            if ema:
                # mAP
                last = test(args, ema.ema, epoch)
            else:
                last = test(args, model, epoch)
            
            # if args.use_early_stopping:
            #     scheduler.step(eval_loss_meter.avg)
            #     early_stopping(eval_loss_meter.avg)
            #     if early_stopping.early_stop:
            #         logger.info("Early stopping!")
            #         break 
            #     else:
            #         logger.info(f"Early stopping counter percentage: {early_stopping.percentage()} %")

            dict_logger.writerow({'epoch': str(epoch + 1).zfill(3),
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
                if ema:
                    save = {'epoch': epoch + 1,
                            'model': copy.deepcopy(ema.ema)}
                else:
                    save = {'epoch': epoch + 1,
                            'model': copy.deepcopy(model)}

                if last[i] > best[i]:
                    # Delete old best (if exists) and save new best
                    if best[i] != -1:
                        os.system(f'rm {args.log_dir}/best_{metric}_{best[i]}.pt')
                    best[i] = last[i]
                    torch.save(save, f=f'{args.log_dir}/best_{metric}_{best[i]}.pt')

                # Delete old last (if exists) and save new last
                if cache_last[i] != -1:
                    os.system(f'rm {args.log_dir}/last_{metric}_{cache_last[i]}.pt')
                cache_last[i] = last[i]
                torch.save(save, f=f'{args.log_dir}/last_{metric}_{cache_last[i]}.pt')

                del save

    if args.local_rank == 0:
        for i, metric in enumerate(metrics):
            util.strip_optimizer(f'{args.log_dir}/best_{metric}_{best[i]}.pt')  # strip optimizers
            util.strip_optimizer(f'{args.log_dir}/last_{metric}_{cache_last[i]}.pt')  # strip optimizers

@torch.no_grad()
def test(args, model=None, epoch=None):
    # filenames = []
    # with open(f'{data_dir}/val2017.txt') as f:
    #     for filename in f.readlines():
    #         filename = os.path.basename(filename.rstrip())
    #         filenames.append(f'{data_dir}/images/val2017/' + filename)

    dataset = Dataset(args, for_training=False, use_augment=False)#filenames, args.input_size, augment=args.use_augment)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        
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

        model = util.load_weight(model, args.weight_path)
        model = model.float().fuse().cuda()

    model.half()
    model.eval()
    
    # class color pallete
    color_dict = dict()
    color_set = set()
    n_classes = len(args.names)
    board = numpy.ones((n_classes * 50 + 10, 100, 3), dtype=numpy.uint8) * 255
    for i in range(len(args.names)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        if (r, g, b) in color_set or (r, g, b) == (255, 255, 255):
            continue
        color_dict[i] = (r, g, b)
        color_set.add((r, g, b))
        cv2.putText(board, f"{i}. {args.names[i]}", (10, i * 50 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[i], 1)
    if epoch:
        log_dir = f'{args.log_dir}/{epoch}'
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    cv2.imwrite(f'{log_dir}.jpg', board)
    
    # if eval_loss_meter:
    #     eval_criterion = util.ComputeLoss(model, args)
    
    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets, names in p_bar:
        with torch.no_grad():    
            samples = torch.autograd.Variable(samples.cuda())#samples.cuda()
            samples = samples.half()  # uint8 to fp16/32
            # samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
            _, _, h, w = samples.shape  # batch-size, channels, height, width
            scale = torch.tensor((w, h, w, h)).cuda()
            
            # Inference
            outputs = model(samples)
        
        outputs = util.non_max_suppression(outputs, args.conf_thres, args.iou_thres)
        
        # Metrics
        for i, output in enumerate(outputs):

            # ground truth
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            #should visualize from here
            util.plot_bboxes(color_dict, names[i], h, w, output, cls, box, log_dir)
            
            cls = cls.cuda() #(N of boxes, 1)
            box = box.cuda() #(N of boxes, 4)

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    # logger.warning("Cannot detect anything!")
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # else:
            #     logger.info("Detect something!")
                
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        _, _, m_pre, m_rec, map50, mean_ap = util.compute_ap(args.log_dir, *metrics, plot=plot, names=args.names)
    else:
        logger.warning(f"Something must be wrong here!: {len(metrics)}, {metrics[0].any()}")
        exit(0)
        
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
        logger.info(f'Number of parameters: {num_params}')
        logger.info(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    # Seed value
    parser.add_argument('--seed', type=int, help='Seed value', default=42)

    parser.add_argument('--variant', default='n', type=str)
    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--names',                            help='Class list (default is [])',                              default = [], type=lambda s: [str(item) for item in s.split(',')])

    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    # Load training data
    parser.add_argument('--train_dir',                 type=str,   help='path to the training data', required=False)
    parser.add_argument('--val_dir',                   type=str,   help='path to the validation data', required=False)

    # Log and save
    parser.add_argument('--log_dir',                   type=str,   help='directory to save checkpoints and summaries', default='')
    
    # Load weights
    parser.add_argument('--weight_path',                     type=str,   help='path to the weights file', required=False)
    
    # threshold hyperparams
    parser.add_argument('--conf_thres', default=0.001, type=float)
    parser.add_argument('--iou_thres', default=0.6, type=float)
    
    # training hyperparams
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--use_amp',                               help='if set, use automatic mixed precision training',           action='store_true')
    parser.add_argument('--reduction', default='none', type=str)

    parser.add_argument('--lr0', default=1e-4, type=float)
    parser.add_argument('--lrf', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.937, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--warmup_epochs', default=3, type=int)

    parser.add_argument('--use_early_stopping',                    help='if set, use early stopping', action='store_true')
    parser.add_argument('--lr_patience',               type=int,   help='learning rate patience threshold', default=5)
    
    parser.add_argument('--box', default=0.05, type=float, help='box loss weight')
    parser.add_argument('--cls', default=0.5, type=float, help='cls loss weight')
    parser.add_argument('--dfl', default=0.5, type=float, help='dfl loss weight')
   
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
    parser.add_argument('--use_albumentations', action='store_true')
    
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

    assert args.train or args.test, "You must specify either --train or --test"
                
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # if args.local_rank == 0:
    #     if not os.path.exists('weights'):
    #         os.makedirs('weights')

    util.setup_seed(args.seed)
    util.setup_multi_processes()

    profile(args)

    if args.train:

        # record config file for reproduction
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        args.log_dir = os.path.join(args.log_dir, time_stamp)        
        os.makedirs(args.log_dir, exist_ok=True)
        
        train_config_path = os.path.join(args.log_dir, f"train_config.yml")
        with open(train_config_path, 'w+') as f:
            yaml.dump(args.__dict__, f)
            
        train(args)
        
    if args.test:
        test(args)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    

    
    main()
