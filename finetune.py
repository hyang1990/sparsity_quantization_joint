import argparse
import copy
import math
import os
import sys
import time

import torch.utils.data
import torchvision
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torchvision.models import mnasnet1_0

import misc
from datasets import cifar, mnist, imagenet
from misc import model_snapshot, AverageMeter, validate, apply_weight_decay, load_pretrained_model, array1d_repr
from model.caffelenet import CaffeLeNet
from model.cifar_resnet import ResNet50
from model.imagenet_alexnet import alexnet
from model.imagenet_mobilenetv1 import mobilenetv1
from proxyless_nas import proxyless_mobile, proxyless_gpu
from util import tensor_round, get_optim_val_dict, km_quantize_tensor, quantize_with_bits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prune-Quant finetune in pytorch')
    parser.add_argument('--dataset', default='mnist', help='dataset used in the experiment')
    parser.add_argument('--data_dir', default='./ILSVRC_CLS', help='dataset dir in this machine')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='caffelenet')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--warmupT', default=0, type=float, help='number of total iterations for warmup')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.025, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--lr_sched', default=None, type=str, help='lr scheduler')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--gclip', default=-1, type=float, help='gradient clip')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrain', default=None, required=True, help='file to load pretrained model')
    parser.add_argument('--logdir',
                        help='The directory used to save the trained models',
                        default='log/default', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=-1)
    parser.add_argument('--mgpu', action='store_true', help='enable using multiple gpus')
    parser.add_argument('--log_interval', type=int, default=-1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--eval_tr', action='store_true', help='evaluate training set')
    parser.add_argument('--prox', action='store_true', help='use proximal op for primal update')
    parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--quant', action='store_true', help='only perform quantization')
    parser.add_argument('--weightbits', default=None, type=str, help='mannual weightbits')
    parser.add_argument('--optim', default='sgd', help='optimizer to use')
    parser.add_argument('--kdtemp', default=0.0, type=float, help='knowledge distillation temperature')

    best_acc = 0
    old_file = None

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    torch.backends.cudnn.benchmark = True

    # set up random seeds
    misc.seed_torch(args.seed)

    # create log file
    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)

    if os.path.exists(args.logdir):
        ans = misc.query_yes_no('Are you sure to overwrite the original directory: {}?'.format(args.logdir))
        if ans:
            # rm old contents in dir
            print('remove old contents in {}'.format(args.logdir))
            os.system('rm -rf ' + args.logdir)
        else:
            exit()
    misc.logger.init(args.logdir, 'train_log')
    print = misc.logger.info

    print('command:\npython {}'.format(' '.join(sys.argv)))

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    # create model
    teacher_model = None
    if args.dataset == 'cifar10':
        if args.arch == 'resnet50':
            model = ResNet50()
        else:
            raise NotImplementedError
    elif args.dataset == 'mnist':
        if args.arch == 'caffelenet':
            model = CaffeLeNet()
        else:
            raise NotImplementedError
    elif args.dataset == 'imagenet':
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=args.pretrain == 'pytorch')
        elif args.arch == 'alexnet':
            model = alexnet(pretrained=args.pretrain == 'pytorch', dropout=args.dp)
            if args.kdtemp > 0.0:
                # + knowledge distillation loss
                teacher_model = alexnet(pretrained=True)
                for param in teacher_model.parameters():
                    param.requires_grad = False
                teacher_model.eval()
        elif args.arch == 'mobilenetv1':
            model = mobilenetv1(pretrained=args.pretrain == 'pytorch')
        elif args.arch == 'mnasnet1_0':
            model = mnasnet1_0(pretrained=args.pretrain == 'pytorch')
        elif args.arch == 'proxyless_mobile':
            model = proxyless_mobile(pretrained=args.pretrain == 'pytorch')
        elif args.arch == 'proxyless_gpu':
            model = proxyless_gpu(pretrained=args.pretrain == 'pytorch')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    model.register_buffer('weight_bits', torch.tensor([0] * len([m for m in model.modules()
                                                                 if isinstance(m, nn.Conv2d)
                                                                 or isinstance(m, nn.Linear)])))

    # pretrained model
    assert args.pretrain != 'pytorch'
    load_pretrained_model(args.pretrain, model, strict=False)

    net_model = model

    if args.mgpu:
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if teacher_model is not None:
                teacher_model.features = torch.nn.DataParallel(teacher_model.features)
        else:
            model = torch.nn.DataParallel(model)
            if teacher_model is not None:
                teacher_model = torch.nn.DataParallel(teacher_model)

    if args.cuda:
        model.cuda()
        if teacher_model is not None:
            teacher_model.cuda()
            
    if args.dataset == 'cifar10':
        train_loader, val_loader = cifar.get10(batch_size=args.batch_size, data_root='./.data', train=True, val=True,
                                               num_workers=args.workers)
        train_loader4eval = train_loader
    elif args.dataset == 'mnist':
        train_loader, val_loader = mnist.get(batch_size=args.batch_size, data_root='./.data', train=True, val=True,
                                             num_workers=args.workers)
        train_loader4eval = train_loader
    elif args.dataset == 'imagenet':
        train_loader, val_loader, train_loader4eval = imagenet.get_data_loaders(args.data_dir,
                                                                                batch_size=args.batch_size,
                                                                                val_batch_size=args.batch_size,
                                                                                num_workers=args.workers,
                                                                                nsubset=-1,
                                                                                normalize=None)
    else:
        raise NotImplementedError

    loss_func = lambda m, x, y: misc.classify_loss(m, x, y, teacher_model, args.kdtemp)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=0.0,
                                    nesterov=False)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0.0)

    warmupT = int(args.warmupT * len(train_loader))

    # lr scheduler setup
    if args.lr_sched is not None:
        train_loader_len = len(train_loader)
        lr_sched = args.lr_sched.split(',')
        if lr_sched[0] == 'cos':
            if len(lr_sched) > 1:
                min_lr = float(lr_sched[1])
            else:
                min_lr = 0.0
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      train_loader_len * (args.epochs - args.warmupT),
                                                                      eta_min=min_lr,
                                                                      last_epoch=len(
                                                                          train_loader) * args.start_epoch - 1)
        elif lr_sched[0] == 'plat':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(lr_sched[1]),
                                                                      threshold=float(lr_sched[2]),
                                                                      patience=2)
        elif lr_sched[0] == 'step':
            lr_milestones = [int(i) for i in lr_sched[1:]]
            print('lr multi-step decay, milestones={}'.format(lr_milestones))
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                gamma=0.5,
                                                                milestones=lr_milestones,
                                                                last_epoch=args.start_epoch - 1)
        elif lr_sched[0] == 'exp':
            factor = float(lr_sched[1])
            print('lr exp decay, factor={}'.format(factor))
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor)
        else:
            raise NotImplementedError
    else:
        lr_scheduler = None

    model_weights = [p for name, p in model.named_parameters() if name.endswith('weight')]
    conv2d_weights = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)]
    linear_weights = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]

    conv_weights = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
    n_conv_layers = len(conv_weights)
    if args.weightbits is None:
        weight_bits = net_model.weight_bits.tolist()
    else:
        weight_bits = [int(b) for b in args.weightbits.split(',')]
        assert len(weight_bits) == len(conv_weights)
    print('weight_bits: {}'.format(weight_bits))
    num_weights = [w.numel() for w in conv_weights]
    print('number of weights: {}'.format(num_weights))
    if not args.quant:
        num_nnz = [w.data.nonzero().shape[0] for w in conv_weights]
        conv_weights_mask = [(w.data != 0.0).float() for w in conv_weights]
        for i, w in enumerate(conv_weights):
            assert conv_weights_mask[i].sum().item() == num_nnz[i]
    else:
        num_nnz = [num for num in num_weights]
        conv_weights_mask = None

    reserved_cluster = 0 if args.quant else 1
    print('num_nnz: {}'.format(num_nnz))
    model_size = sum([weight_bits[i] * num_nnz[i] for i in range(n_conv_layers)])
    full_model_size = sum([w.numel() * 32 for w in conv_weights])
    print('target model size={:.4e} bits / {:.4e} bits (compression rate:{:.4e})'.format(model_size,
                                                                                         full_model_size,
                                                                                         float(
                                                                                             model_size) / full_model_size))
    assert min(weight_bits) > 0

    if args.evaluate:
        quantize_with_bits(conv_weights, weight_bits, dictnz=args.quant)
        num_nnz_eval = [float(m.weight.nonzero().shape[0]) for m in model.modules()
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
        weight_bits_eval = [math.log2(max(1.0, tensor_round(m.weight.data, n=6).unique().shape[0] - reserved_cluster))
                            for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
        print('num_nnz_eval=\t{}'.format(array1d_repr([num_nnz_eval[i] / conv_weights[i].numel()
                                                       for i in range(n_conv_layers)], format='{:.3f}')))
        print('n_nnz_eval=\t{}'.format(num_nnz_eval))
        print('nnz_eval=\t{:.4e}'.format(sum(num_nnz_eval) / sum([conv_weights[i].numel()
                                                                  for i in range(n_conv_layers)])))
        print('weight_bits_eval=\t{}'.format(array1d_repr(weight_bits_eval, format='{:.0f}')))
        print('ave_weight_bits_eval=\t{:.4e}'.format(sum([num_nnz_eval[i] * weight_bits_eval[i]
                                                          for i in range(len(num_nnz_eval))]) / sum(num_nnz_eval)))
        validate(val_loader, model, loss_func=loss_func)
        exit()
    else:
        num_nnz_eval = [float(m.weight.nonzero().shape[0]) for m in model.modules()
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
        weight_bits_eval = [math.log2(max(1.0, tensor_round(m.weight.data, n=6).unique().shape[0] - reserved_cluster))
                            for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
        print('num_nnz_eval=\t{}'.format(array1d_repr([num_nnz_eval[i] / conv_weights[i].numel()
                                                       for i in range(n_conv_layers)], format='{:.3f}')))
        print('n_nnz_eval=\t{}'.format(num_nnz_eval))
        print('nnz_eval=\t{:.4e}'.format(sum(num_nnz_eval) / sum([conv_weights[i].numel()
                                                                  for i in range(n_conv_layers)])))
        print('weight_bits_eval=\t{}'.format(array1d_repr(weight_bits_eval, format='{:.0f}')))
        print('ave_weight_bits_eval=\t{:.4e}'.format(sum([num_nnz_eval[i] * weight_bits_eval[i]
                                                          for i in range(len(num_nnz_eval))]) / sum(num_nnz_eval)))

    log_tic = time.time()
    losses = AverageMeter()
    if args.log_interval <= 0:
        args.log_interval = len(train_loader)
    conv_weights_copy = [w.data.clone() for w in conv_weights]
    conv_weights_dict = []
    for i, w in enumerate(conv_weights):
        if conv_weights_mask is not None:
            quant_data = w.data[conv_weights_mask[i].bool()]
        else:
            quant_data = w.data
        # val_dict = get_optim_val_dict(quant_data, int(weight_bits[i]), niter=100, dictnz=True)
        val_dict = get_optim_val_dict(quant_data, int(weight_bits[i]), niter=100, dictnz=args.quant)
        conv_weights_dict.append(val_dict)

    dict_update_int = int(round(len(train_loader) / 20))
    for epoch in range(args.start_epoch, args.epochs):
        losses.reset()
        gclip_time = 0
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        for batch_idx, (data, target) in enumerate(train_loader):
            # lr schedule
            t = float(batch_idx + epoch * len(train_loader))
            if t < warmupT:
                lr = min(1.0, (t + 1) / float(warmupT)) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            model.train()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # pre-forward: quantization
            for i, w in enumerate(conv_weights):
                conv_weights_copy[i].data.copy_(w.data)
                w.data.copy_(km_quantize_tensor(w.data, int(weight_bits[i]), conv_weights_dict[i], dictnz=args.quant)[0])
            w_loss = loss_func(model, data, target)
            # after-forward: restore to backup weights
            for i, w in enumerate(conv_weights):
                w.data.copy_(conv_weights_copy[i].data)
            primal_loss = w_loss
            # losses stats
            losses.update(primal_loss.item(), data.size(0))
            # update network weights
            optimizer.zero_grad()
            primal_loss.backward()
            # apply weight_decay
            apply_weight_decay(model_weights, args.weight_decay)
            # gradient norm clip
            if args.gclip > 0:
                total_norm = clip_grad_norm_(model.parameters(), args.gclip, norm_type=float('inf'))
                if total_norm > args.gclip:
                    gclip_time += 1
            optimizer.step()

            if conv_weights_mask is not None:
                for i, w in enumerate(conv_weights):
                    w.data *= conv_weights_mask[i]

            if epoch >= args.warmupT and lr_scheduler is not None:
                # increment lr_scheduler each epoch or cosine decay each iteration
                if batch_idx == len(train_loader) - 1 or lr_sched[0] == 'cos':
                    lr_scheduler.step()

            if (batch_idx + 1) % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('+-------------- epoch {}, batch {}/{} ----------------+'.format(epoch, batch_idx + 1,
                                                                                       len(train_loader)))
                log_toc = time.time()
                print('Primal update: Loss={:.4f} (losses_avg={:.4f})'
                      ', lr={:.4e}, time_elapsed={:.3f}s'.format(
                    losses.val, losses.avg, optimizer.param_groups[0]['lr'], log_toc - log_tic))
                if args.gclip > 0:
                    print('gclip times={}'.format(gclip_time))
                # print(layers_stat(model, param_names='weight', param_filter=lambda p: p.dim() > 1))

                print('num_nnz=\t{}'.format(array1d_repr([num_nnz[i] / conv_weights[i].numel()
                                                          for i in range(n_conv_layers)], format='{:.3f}')))
                print('weight_bits=\t{}'.format(array1d_repr(weight_bits, format='{:.0f}')))

                log_tic = time.time()
                print('+-----------------------------------------------------+')

            if (batch_idx + 1) % dict_update_int == 0 or batch_idx == len(train_loader) - 1:
                for i, w in enumerate(conv_weights):
                    conv_weights_dict[i] = get_optim_val_dict(w.data, int(weight_bits[i]), niter=5,
                                                              val_dict=conv_weights_dict[i], dictnz=args.quant)

        # evaluate and print info
        print(misc.layers_stat(model, param_names='weight', param_filter=lambda p: p.dim() > 1))
        # pre-forward: quantization
        for i, w in enumerate(conv_weights):
            conv_weights_copy[i].data.copy_(w.data)
            w.data.copy_(km_quantize_tensor(w.data, int(weight_bits[i]), conv_weights_dict[i], dictnz=args.quant)[0])

        num_nnz_eval = [float(m.weight.nonzero().shape[0]) for m in model.modules()
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
        weight_bits_eval = [math.log2(max(1.0, tensor_round(m.weight.data, n=6).unique().shape[0] - reserved_cluster))
                            for m in model.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]
        print('n_nnz_eval=\t{}'.format(num_nnz_eval))
        print('num_nnz_eval=\t{}'.format(array1d_repr([num_nnz_eval[i] / conv_weights[i].numel()
                                                       for i in range(n_conv_layers)], format='{:.3f}')))
        print('nnz_eval=\t{:.4e}'.format(sum(num_nnz_eval) / sum([conv_weights[i].numel()
                                                                  for i in range(n_conv_layers)])))
        print('weight_bits_eval=\t{}'.format(array1d_repr(weight_bits_eval, format='{:.0f}')))
        print('ave_weight_bits_eval=\t{:.4e}'.format(sum([num_nnz_eval[i] * weight_bits_eval[i]
                                                          for i in range(len(num_nnz_eval))]) / sum(num_nnz_eval)))
        if args.eval_tr:
            print('training set:')
            validate(train_loader4eval, model, loss_func=loss_func)

        print('test set:')
        prec1 = validate(val_loader, model, loss_func=loss_func)[0]

        print('current model size={:.4e} bits'.format(
            sum([weight_bits_eval[i] * num_nnz_eval[i] for i in range(n_conv_layers)])))
        print(
            'compression rate={:.4e}'.format(sum([weight_bits_eval[i] * num_nnz_eval[i] for i in range(n_conv_layers)])
                                             / full_model_size))
        print('compression rate={:.4e} (train)'
              .format(sum([weight_bits[i] * num_nnz[i] for i in range(n_conv_layers)]) / full_model_size))
        print('======================================================')

        # remember best prec@1 and save checkpoint
        if prec1 > best_acc:
            print('find accuracy {:4f} > {:.4f}'.format(prec1, best_acc))
            new_file = os.path.join(args.logdir, 'model_best-{}.pkl'.format(epoch))
            misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
            best_acc = prec1
            old_file = new_file

        if epoch > 0 and args.save_every > 0 and epoch % args.save_every == 0:
            model_snapshot(model, os.path.join(args.logdir, 'model_epoch{}.pt'.format(epoch)))

        # save the lastest model
        model_snapshot(model, os.path.join(args.logdir, 'model_latest.pt'))

        # after-forward: restore to backup weights
        for i, w in enumerate(conv_weights):
            w.data.copy_(conv_weights_copy[i].data)

        # update val_dict
        if epoch != args.epochs - 1:
            for i, w in enumerate(conv_weights):
                conv_weights_dict[i] = get_optim_val_dict(w.data, int(weight_bits[i]), niter=100,
                                                          val_dict=conv_weights_dict[i], dictnz=args.quant)
