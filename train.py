import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import cv2
from dataloader import prepare_dataloder, get_aug_image, avg_pseudo_label, get_consistencyloss
import os
import higher
from utils import *
import numpy as np
from Metrics import dice_coeff
from tensorboardX import SummaryWriter
from Models.Unetplpl_2D import NestedUNet
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--dataset', default='Prostate', type=str)
parser.add_argument('--checkpoint', default='', type=str, help='pretrained checkpoint')
parser.add_argument('--eval_batch_size', default=1, type=int,
                    help="batch size for eval.")
parser.add_argument('--size', default=[144, 144], type=int,
                    help="Resize shape.")
parser.add_argument('--train_root', default='./data/PROMISE12/train/', type=str,
                    help="image path for training (imperfect data).")
parser.add_argument('--meta_root', default='./data/PROMISE12/meta_train/', type=str,
                    help="image path for meta set (clean data).")
parser.add_argument('--vali_root', default='./data/PROMISE12/original_data/', type=str,
                    help="image path for validation.")
parser.add_argument('--baseline', default=False, type=bool,
                    help="whether to train under baseline setting.")
parser.add_argument('--reweight_label', default=True, type=bool,
                    help="whether to use MLB.")
parser.add_argument('--arch', default='Unet++_2d_ema', type=str,
                    help="name for the baseline.")
parser.add_argument('--gpuid', default=0, type=int,
                    help="GPU ID.")
parser.add_argument('--train_batch_size', default=1, type=int,
                    help="Batch size used during training.")
parser.add_argument('--meta_batch_size', default=2, type=int,
                    help="Batch size used for meta set.")
parser.add_argument('--print_freq', default=50, type=int,
                    help="print frequency.")
parser.add_argument('--warm_up', default=0, type=int,
                    help="# epoch of warming up.")

parser.add_argument('--wd', default=5e-4, type=float, help='weight decay for SGD')
parser.add_argument('--clipping_norm', default=0.2, type=float, help='')
parser.add_argument('--norm_type', default='org', type=str, help='weight map normalization type')
parser.add_argument('--temperature', default=10.0, type=float, help='')
parser.add_argument('--ema_decay', default=0.99, type=float, help='EMA decay rate')
parser.add_argument('--meta_opt', type=str, default='sgd',
                    help='type of meta optimizer')
parser.add_argument('--datasplitpath', default='', type=str, help='split information')

parser.add_argument('--meta_step', type=int, default=1,
                    help='training step of meta optimizer')
parser.add_argument('--hard_weight', action='store_true')

ROOT = os.getcwd()
SAVE_CHECKPOINT = './best_model/'


def create_ema_model(model):
    # Network definition
    for param in model.parameters():
        param.detach_()
    return model

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def main(args):
    global best_dice
    best_dice = 0

    net = NestedUNet(1, 1).cuda()
    ema_net = NestedUNet(1, 1).cuda()
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint))
        ema_net.load_state_dict(torch.load(args.checkpoint))

    ema_net = create_ema_model(ema_net)

    writer = SummaryWriter(log_dir=SAVE_CHECKPOINT + '/logs/ema_UNet++_mlb_PROMISE12/')
    print('Experiments is conducted on: baseline_' + args.arch)
    print('==> Preparing data..')
    if not os.path.exists(SAVE_CHECKPOINT + '/logs/'):
        os.mkdir(SAVE_CHECKPOINT + '/logs/')

    if args.baseline:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.BCELoss(reduction="none")
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler = prepare_dataloder(args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 20 / (epoch + 20))

    for epoch in range(args.num_epochs):
        net.train()
        train_loss = AverageMeter()
        meta_loss = AverageMeter()
        meta_dice = AverageMeter()
        train_dice = AverageMeter()
        train_dice_gt = AverageMeter()
        iter_num = 0
        for i, (image, labels, gtmasks) in enumerate(trainloader):
            image = image.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            org_label = labels
            if args.baseline:
                outputs = torch.sigmoid(net(image))
                l_f = criterion(outputs, labels)
                dice = 0
                for k in range(outputs.shape[0]):
                    dice += dice_coeff(org_label[k, 0, :, :].cpu().numpy(),
                                       outputs[k, 0, :, :].cpu().detach().numpy())

                dice = dice / outputs.shape[0]


            else:
                if epoch <= (args.warm_up - 1):
                    outputs = torch.sigmoid(net(image))

                    criterion_warm = nn.BCELoss().cuda()
                    l_f = criterion_warm(outputs, labels)

                    dice = 0
                    for k in range(outputs.shape[0]):
                        dice += dice_coeff(org_label[k, 0, :, :].cpu().numpy(),
                                           outputs[k, 0, :, :].cpu().detach().numpy())

                    dice = dice / outputs.shape[0]
                    dicegt = 0
                    for k in range(outputs.shape[0]):
                        dicegt += dice_coeff(gtmasks[k, 0, :, :].cpu().numpy(),
                                             outputs[k, 0, :, :].cpu().detach().numpy())

                    dicegt = dicegt / outputs.shape[0]

                else:
                    image2, labels_aug, gtmasks_aug, insize, outsize, flipindex = get_aug_image(image, labels, gtmasks, picksize=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
                    ema_input = image2 + (0.02) * torch.randn(image2.shape).cuda()

                    with higher.innerloop_ctx(net,  optimizer) as (meta_net,  meta_optimizer):
                        for s in range(args.meta_step):
                            y_f_hat_seg = meta_net(image2)
                            y_f_hat_seg = torch.sigmoid(y_f_hat_seg)

                            with torch.no_grad():
                                ema_output_seg = torch.sigmoid(ema_net(ema_input))

                            pseudo_labels_avg = avg_pseudo_label(y_f_hat_seg.detach(), insize, outsize, flipindex)
                            loss_consistency = get_consistencyloss(y_f_hat_seg, insize, outsize, flipindex)
                            loss_consistency_ema = torch.mean((y_f_hat_seg - ema_output_seg) ** 2)
                            pesudo_labels = (pseudo_labels_avg > 0.5).float()

                            if args.reweight_label:
                                cost1 = criterion(y_f_hat_seg, labels_aug)
                                cost2 = criterion(y_f_hat_seg, pesudo_labels)

                                cost = torch.cat((cost1, cost2), dim=1)

                            else:
                                cost = criterion(y_f_hat_seg, labels)

                            eps = torch.zeros(cost.size()).cuda()
                            eps = eps.requires_grad_()

                            l_f_meta = torch.sum(cost * eps) + loss_consistency + loss_consistency_ema

                            meta_optimizer.step(l_f_meta)

                        val_data, val_labels = next(iter(metaloader))
                        val_data = val_data.cuda()
                        val_labels = val_labels.cuda()
                        vali_data2, vali_labels2, _, vali_insize, vali_outsize, vali_flipindex = get_aug_image(val_data, val_labels, val_labels, picksize=[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

                        y_g_hat_seg = meta_net(vali_data2)
                        y_g_hat_seg = torch.sigmoid(y_g_hat_seg)


                        l_g_meta = torch.mean(criterion(y_g_hat_seg, vali_labels2))

                        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()

                    if args.reweight_label:
                        w_tilde = -grad_eps
                        b, c, w, h = w_tilde.shape


                        if args.hard_weight:
                            w_tilde = F.softmax(w_tilde, dim=-1)
                            zeros = torch.zeros_like(w_tilde).cuda()
                            weight = torch.where(w_tilde <= args.temperature, zeros, w_tilde)
                        else:

                            if args.norm_type == 'org':
                                w_tilde = torch.clamp(w_tilde, min=0)
                                norm_c = torch.sum(w_tilde) + 1e-10
                                if norm_c != 0:
                                    weight = w_tilde / norm_c
                                else:
                                    weight = w_tilde
                            elif args.norm_type == 'softmax':
                                weight = F.softmax(w_tilde.view(b, -1), dim=-1)
                            elif args.norm_type == 'sigmoid':
                                sigmoid = nn.Sigmoid()
                                weight = sigmoid(w_tilde)
                            else:
                                weight = w_tilde * 5
                        weight = weight.view(cost.shape)
                    else:
                        pass


                    outputs_seg = net(image2)
                    outputs_seg = torch.sigmoid(outputs_seg)
                    pseudo_labels_avg = avg_pseudo_label(outputs_seg.detach(), insize, outsize, flipindex)
                    pesudo_labels = (pseudo_labels_avg > 0.5).float()

                    with torch.no_grad():
                        ema_output_seg = torch.sigmoid(ema_net(ema_input))

                    loss_consistency = get_consistencyloss(outputs_seg, insize, outsize, flipindex)
                    loss_consistency_ema = torch.mean((outputs_seg - ema_output_seg) ** 2)

                    dice = 0
                    for k in range(outputs_seg.shape[0]):
                        dice += dice_coeff(labels_aug[k, 0, :, :].cpu().numpy(),
                                           outputs_seg[k, 0, :, :].cpu().detach().numpy())

                    dice = dice / outputs_seg.shape[0]

                    dicegt = 0
                    for k in range(outputs_seg.shape[0]):
                        dicegt += dice_coeff(gtmasks_aug[k, 0, :, :].cpu().numpy(),
                                             outputs_seg[k, 0, :, :].cpu().detach().numpy())

                    dicegt = dicegt / outputs_seg.shape[0]

                    if args.reweight_label:
                        cost1 = criterion(outputs_seg, labels_aug)
                        cost2 = criterion(outputs_seg, pesudo_labels)
                        cost = torch.cat((cost1, cost2), dim=1)

                    l_f = torch.sum(cost * weight) + loss_consistency + loss_consistency_ema

            optimizer.zero_grad()
            l_f.backward()
            if args.clipping_norm > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clipping_norm, norm_type=2)
            optimizer.step()
            if args.baseline:
                train_loss.update(l_f.item(), image.size(0))
            else:
                if epoch <= (args.warm_up - 1):
                    train_loss.update(l_f.item(), image.size(0))
                else:
                    train_loss.update(l_f.item(), image.size(0))
            train_dice.update(dice, image.size(0))
            train_dice_gt.update(dicegt, image.size(0))

            update_ema_variables(net, ema_net, args.ema_decay, i)


            vali_ema_input = vali_data2 + (0.02) * torch.randn(vali_data2.shape).cuda()
            vali_seg = net(vali_data2)

            with torch.no_grad():
                vali_ema_output_seg = torch.sigmoid(ema_net(vali_ema_input))

            vali_seg = torch.sigmoid(vali_seg)

            vali_loss_consistency = get_consistencyloss(vali_seg, vali_insize, vali_outsize, vali_flipindex)
            vali_loss_consistency_ema = torch.mean((vali_seg - vali_ema_output_seg) ** 2)

            vali_loss = torch.mean(criterion(vali_seg, vali_labels2)) + vali_loss_consistency + vali_loss_consistency_ema

            optimizer.zero_grad()
            vali_loss.backward()
            if args.clipping_norm > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clipping_norm, norm_type=2)
            optimizer.step()



            validice = 0
            for k in range(vali_seg.shape[0]):
                validice += dice_coeff(vali_labels2[k, 0, :, :].cpu().numpy(),
                                   vali_seg[k, 0, :, :].cpu().detach().numpy())

            validice = validice / vali_seg.shape[0]

            meta_loss.update(vali_loss.item(), vali_seg.size(0))
            meta_dice.update(validice, vali_seg.size(0))

            update_ema_variables(net, ema_net, args.ema_decay, i)

            if i % args.print_freq == 0:
                print(
                    'At epoch: {:03d} Step: {:03d}/{:03d} AVERAGE dice: {:.4f} AVERAGE gt dice: {:.4f} meta dice: {:.4f} AVERAGE TRAIN loss : {:.4f}'.
                    format(epoch, i, len(trainloader), train_dice.avg, train_dice_gt.avg,  meta_dice.avg, train_loss.val))


        writer.add_scalar('Train/loss', train_loss.avg, epoch)
        writer.add_scalar('Train/lr', optimizer.param_groups[-1]['lr'], epoch)
        print('At epoch: {:03d} the learning rate is : {:.4f}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('At epoch: {:03d} AVERAGE TRAIN loss : {:.4f}'.format(epoch, train_loss.avg))




        dice, test_loss = val(epoch, testloader, net, writer, args)

        scheduler.step()
        state = {
                    'net': net.state_dict(),
                    'dice': dice,
                    'epoch': epoch,
                }
        torch.save(state,
                   SAVE_CHECKPOINT + 'Newest_model.pth', _use_new_zipfile_serialization=False)
        if dice > best_dice:
            best_dice = dice
        
        print('-----------------------------------------------------')
        print('At epoch: {:03d} CURRENT loss: {:.4f}'.format(epoch, test_loss.avg))
        print('At epoch: {:03d} BEST dice: {:.4f}'.format(epoch, best_dice))
        print('At epoch: {:03d} CURRENT dice: {:.4f}'.format(epoch, dice))
        print('-----------------------------------------------------')





def val(epoch, testloader, net, writer, args):
    net.eval()
    test_loss = AverageMeter()
    testdice = AverageMeter()
    with torch.no_grad():
        for i, (inputs, targets, originalmask) in enumerate(testloader):
            inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3]).unsqueeze(1).cuda()
            targets = targets.view(-1, targets.shape[2], targets.shape[3]).unsqueeze(1).cuda()

            loss_sum = 0
            test_loss_criterion = nn.BCELoss().cuda()
            segl = 24
            seg = math.ceil(inputs.shape[0]//segl)
            for k in range(seg+1):
                inputs1 = inputs[k*segl:(k+1)*segl, :, :, :]
                targets1 = targets[k*segl:(k+1)*segl, :, :, :]
                if inputs1.shape[0] == 0:
                    break
                if len(inputs1.shape) == 3:
                    inputs1 = inputs1.unsqueeze(0)
                    targets1 = targets1.unsqueeze(0)
                outputs1_seg = net(inputs1)
                outputs1 = torch.sigmoid(outputs1_seg)
                loss1 = test_loss_criterion(outputs1, targets1)
                loss_sum = loss_sum + loss1.item()*targets1.shape[0]
                if k == 0:
                    outputs = outputs1.cpu().numpy()
                else:
                    outputs = np.concatenate([outputs, outputs1.cpu().numpy()], axis=0)

            dice = dice_coeff(originalmask.squeeze(0).numpy(), cv2.resize(outputs[:, 0, :, :].transpose(1, 2, 0),
                                                                          [originalmask.shape[-1],
                                                                           originalmask.shape[-2]]).transpose(2, 0, 1))


            test_loss.update(loss_sum/inputs.size(0), inputs.size(1))
            testdice.update(dice, 1)


    writer.add_scalar('Test/loss', test_loss.avg, epoch)
    writer.add_scalar('Test/acc', testdice.avg, epoch)
    writer.add_scalar('Test/best_acc', best_dice, epoch)

    return testdice.avg, test_loss

if __name__ == '__main__':
    args = parser.parse_args()
    torch.cuda.set_device(args.gpuid)

    print('baseline:', args.baseline, 'dataset:', args.dataset, 'Norm_type', args.norm_type)
    print('Img size:', args.size)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    main(args)


