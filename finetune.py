import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import utils.logger as logger
import torch.backends.cudnn as cudnn

from dataloader.preprocess import disp2depth
import models.anynet

parser = argparse.ArgumentParser(description='Anynet finetune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='carla',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=3,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='results/finetune_anynet/checkpoint.tar',
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')


args = parser.parse_args()

if args.datatype == 'kitti':
    from dataloader import KITTILoader as DA
    from dataloader import KITTIloader2015_ft as ls
    args.datapath = 'kitti/training/'
elif args.datatype == 'carla':
    from dataloader import CarlaLoader as DA
    from dataloader import CarlaSplit as ls
    args.datapath = '/data/cli/carla_0.9.6_data/'

loss_arr = []

def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 ==0:
            test(TestImgLoader, model, log)

    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        loss_arr.append(loss[2].item)

        if batch_idx % args.print_freq == 0:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    error_metrics = ['3-pixel error', 'threshold(1.25)', 'threshold(1.25^2)',
                     'threshold(1.25^3)', 'ard', 'srd', 'rmse', 'rmse_log']
    D1s = [[AverageMeter() for _ in range(len(error_metrics))] for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L, depth) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        if batch_idx in [0,2,4,6,8]:
            plt.imshow(np.array(disp_L.cpu())[0], cmap="viridis")
            plt.savefig("images/result-id{}-gt.png".format(batch_idx))
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x][0].update(3PixelError(output, disp_L).item())
                diff = error_eval(output, depth)
                for i in range(len(error_metrics)-1):
                    D1s[x][i+1].update(diff[i].item())
                if batch_idx in[0,2,4,6,8]:
                    mask = disp_L > 0
                    mask = mask * (disp_L < 192)
                    errmap = torch.abs(output[0] - disp_L)
                    err3 = ((errmap[mask] > 3.) & (errmap[mask] / disp_L[mask] > 0.05))
                    plt.imshow(np.array(err3.cpu()), cmap="viridis")
                    plt.savefig("images/result-id{}-stage{}.png".format(batch_idx, x))
        info_str = []
        for x in range(stages):
            stage_str = ', '.join(['{} = {:.4f}({:.4f})'.format(error_metrics[i],
                                                                D1s[x][i].val, D1s[x][i].avg) for i in range(len(error_metrics))])
            info_str.append('Stage {} {}'.format(x, stage_str))
        
        info_str='\n'.join(info_str)
        
        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)

def error_eval(disp, gt):
    mask = gt > 0.003
    mask = mask * (gt < 0.3)
    est = disp2depth(disp) * 1000
    gt = gt * 1000
    errmap = torch.max(est/gt, gt/est)[mask]
    err_thres1 = (errmap > 1.25).sum() / mask.sum()
    err_thres2 = (errmap > 1.25**2).sum() / mask.sum()
    err_thres3 = (errmap > 1.25**3).sum() / mask.sum()
    ard = (torch.abs(est-gt)/gt)[mask].sum() / mask.sum()
    srd = ((est-gt)**2/gt)[mask].sum() / mask.sum()
    rmse = torch.sqrt(((est-gt)**2)[mask].sum() / mask.sum())
    rmse_log = torch.sqrt(((torch.log(est)-torch.log(gt))**2)[mask].sum() / mask.sum())
    return (err_thres1, err_thres2, err_thres3, ard, srd, rmse, rmse_log)
    
def 3PixelError(disp, gt, maxdisp=192):
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
