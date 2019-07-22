# ------------------------------------------------------------------------------
# Pytorch implementation of Hope
# Licensed under The MIT License [see LICENSE for details]
# Written by Zongwei Zhou; zongwei.zhou@nlpr.ia.ac.cn
# ------------------------------------------------------------------------------
# NOTE: At current time-step, the repository only supports Single GPU
# ------------------------------------------------------------------------------

import os
import numpy as np
import argparse
import time

import torch
from dataset.graph_dataset import graph_dataset
from model.net import ResGraphNet
from model.loss import global_loss, TripletLoss
from utils.average_meter_helper import AverageMeter
from utils.vis_tool import Visualizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.utils.net_utils import weights_normal_init, save_checkpoint, save_net, load_net
from model.utils.net_utils import clip_gradient, adjust_learning_rate
from model.criterion import evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Hope network.")
    parser.add_argument('--gpu', dest='gpu', help='index of gpu', choices=['0', '1'], default='0', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=80, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations t display',
                        default=5, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to save checkpoint')
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default='output/models',
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=20, type=int)
    parser.add_argument('--trainval', action='store_true', default=False)
    parser.add_argument('--bs', dest='bs', help='batch size', default=32, type=int)
    # optimizer
    parser.add_argument('-o', dest='optimizer', help='training optimizer', default='sgd', type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=0.01, type=float)
    parser.add_argument('--lr_decay_epoch', dest='lr_decay_epoch',
                        help='step to do learning rate decay, unit is epoch', default=50, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay(default: 1e-4)')

    # resume
    parser.add_argument('-r', dest='resume', help='resume checkpoint or not', default=False, type=bool)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model',
                        default=1, type=int)
    # log
    parser.add_argument('--env_name', dest='env_name', help='name of visdom environment',
                        default='HopeNet', type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] =  args.gpu
    device = torch.device('cuda:{}'.format(args.gpu))
    # logger
    vis = Visualizer(env=args.env_name)
    # dataset
    trainset = graph_dataset(subsets=('zara01', 'eth', 'hotel', 'univ'))
    validset = graph_dataset(subsets=('zara02',))
    train_dataloader = DataLoader(trainset, batch_size=args.bs, shuffle=True,
                                  collate_fn=trainset.collate_fn, num_workers=args.num_workers, pin_memory=True)
    valid_dataloader = DataLoader(validset, batch_size=args.bs, shuffle=False,
                                  collate_fn=validset.collate_fn, num_workers=args.num_workers, pin_memory=True)
    # net and optimizer
    model = ResGraphNet()
    lr = args.lr
    params = []
    for k, v in dict(model.named_parameters()).items():
        if v.requires_grad:
            if 'bias' in k:
                params += [{'params': [v], 'lr': 2 * lr, 'weight_decay': 1e-4}]
            else:
                params += [{'params': [v], 'lr': lr, 'weight_decay': args.weight_decay}]

    if args.optimizer == 'adam':
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(params, momentum=args.momentum)
    # resume
    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.resume:
        load_name = os.path.join(output_dir, '{}.pth'.format(args.checkpoint))
        print("loading checkpoint {}".format(load_name))
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint {}".format(load_name))
    model.to(device)

    # loss
    trip_loss = TripletLoss(margin=1)
    for epoch in range(args.start_epoch, args.max_epochs):
        # train
        model.train()
        start = time.time()
        if epoch % (args.lr_decay_epoch + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        avg = AverageMeter()
        print('Training ...')
        for step, value in tqdm(enumerate(train_dataloader)):
            end_time = time.time()
            batch_loss = 0
            for i in range(len(value[0])):
                visual_graph = value[0][i].to(device)
                points_graph = value[1][i].to(device)
                visual_feats = model(visual_graph)
                points_feats = model(points_graph)
                global_feats = torch.cat((visual_feats, points_feats), dim=0).contiguous()
                labels = torch.cat((visual_graph.y, points_graph.y), dim=0).contiguous()
                iloss, _, _, _, _, _ = global_loss(trip_loss, global_feats, labels)
                batch_loss += iloss
            batch_loss = batch_loss/len(value[0])
            optimizer.zero_grad()
            batch_loss.backward()
            # clip_gradient(model, 10)
            optimizer.step()

            batch_time = time.time() - end_time
            avg.update(batch_time=batch_time, loss=batch_loss.item())
            if (step+1) % args.disp_interval == 0:
                vis.plot('loss', avg.avg('loss'))
                log_str = '(Train) Epoch: [{0}][{1}/{2}]\t lr: {lr:.6f} \t {batch_time:s} \t {loss:s} \n'.format(
                    epoch, step+1, len(train_dataloader), lr=lr, batch_time=avg.batch_time, loss=avg.loss
                )
                vis.log(log_str)
        if args.trainval:
            # validation
            model.eval()
            valid_avg = AverageMeter()
            valid_disp_interval = int(args.disp_interval/len(train_dataloader)*len(valid_dataloader))
            print('Valiating ...')
            with torch.no_grad():
                for step, value in tqdm(enumerate(valid_dataloader)):
                    val_end_time = time.time()
                    batch_loss, Fm, Gm = 0, 0, 0
                    for i in range(len(value[0])):
                        visual_graph = value[0][i].to(device)
                        points_graph = value[1][i].to(device)
                        visual_feats = model(visual_graph)
                        points_feats = model(points_graph)
                        global_feats = torch.cat((visual_feats, points_feats), dim=0).contiguous()
                        labels = torch.cat((visual_graph.y, points_graph.y), dim=0).contiguous()
                        iloss, _, _, _, _, _ = global_loss(trip_loss, global_feats, labels)
                        batch_loss += iloss
                        # evaluation
                        f_m, g_m = evaluation(visual_feats, points_feats,
                                   visual_graph.y, points_graph.y)
                        Fm += f_m
                        Gm += g_m
                    batch_loss = batch_loss / len(value[0])
                    Fm = Fm/len(value[0])
                    Gm = Gm/len(value[0])

                    batch_time = time.time() - val_end_time
                    valid_avg.update(batch_time=batch_time, loss=batch_loss.item(), Fm=Fm, Gm=Gm)
                    if (step + 1) % valid_disp_interval == 0:
                        vis.plot_many({'valid_loss': valid_avg.avg('loss'),
                                       'evaluation_Fm': valid_avg.avg('Fm'),
                                       'evaluation_Gm': valid_avg.avg('Gm')})

                        log_str = '(Valid) Epoch: [{0}][{1}/{2}]\t lr: {lr:.6f} \t {batch_time:s} \t {loss:s} \n' \
                                  '{Fm:s}\t {Gm:s}\n'.format(
                            epoch, step + 1, len(valid_dataloader), lr=lr, batch_time=valid_avg.batch_time,
                            loss=valid_avg.loss, Fm=valid_avg.Fm, Gm=valid_avg.Gm
                        )
                        vis.log(log_str)

        save_name = os.path.join(output_dir, '{}.pth'.format(epoch%3))
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, save_name)
        print("save epoch {} model: {}".format(epoch, save_name))
        end = time.time()
        print('time spent in epoch: {} is {}'.format(epoch, end - start))


