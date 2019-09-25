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
import pdb
import cv2
# import imageio

import utils.logger as logger

import models.anynet

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default=None,
                    help='datapath')
parser.add_argument('--datatype', default='carla',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 4)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='results/pretrained_anynet',
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')


args = parser.parse_args()

if args.datatype == 'kitti':
    from dataloader import KITTILoader as DA
    from dataloader import KITTIloader2015 as lt
    args.datapath = 'kitti/training/'
elif args.datatype == 'carla':
    from dataloader import CarlaLoader as DA
    from dataloader import CarlaSplit as lt
    args.datapath = '/data/cli/carla_0.9.6_data/'

def main():
    global args

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    # pdb.set_trace()
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    start_full_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))

def add_afm_hook(module):
    """Add hook to module to save average feature map to `module._afm`"""
    def hook(module, input_, output):
        if not hasattr(module, '_afm_k'):
            module._afm_k = 0
            module._afm = output
            module._bytetype = 0
        else:
            module._afm_k += 1
            k = module._afm_k
            n = k + 1.
            
            # if output.type() != 'torch.FloatTensor':
            #     module._bytetype += 1
            #     output = output.float()
            module._afm = (k / n * module._afm) + (output / n)
    module.register_forward_hook(hook)

def save_afm(module, dir='./'):
    """Saves `module._afm` to disk, in dir"""
    path = "{}{}".format(dir, ".pth")
    afm = module._afm.cpu()
    torch.save(afm, path)
    vis_afm(afm, dir)

def vis_afm_signed(fm, dir='./', colormap=cv2.COLORMAP_HOT, fname='out'):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, "{}.jpg".format(fname))
    fm = normalize_fm(fm)
    cv2.imwrite(path, fm)

def vis_afm_negative(fm, dir='./', fname='out'):
    channel_neg = np.maximum(0, -fm)
    dir_neg = "{}{}".format(dir, "_neg")
    vis_afm_signed(channel_neg, dir_neg, cv2.COLORMAP_OCEAN, fname)

def vis_afm_positive(fm, dir='./', fname='out'):
    channel_pos = np.maximum(0, fm)
    dir_pos = "{}{}".format(dir, "_pos")
    vis_afm_signed(channel_pos, dir_pos, cv2.COLORMAP_HOT, fname)

def normalize_fm(fm):
    fm /= fm.max()
    fm *= 255
    fm = fm.astype(np.uint8)
    fm = cv2.applyColorMap(fm, cv2.COLORMAP_HOT)
    return fm

def vis_afm(afm, dir='./', vis_functions=(vis_afm_positive, vis_afm_negative)):
    """Saves `module._afm` to disk, in dir"""
    for i, channel in enumerate(afm[0]):
        channel = channel.data.numpy()
        for function in vis_functions:
            function(channel, dir, i)

def add_all_afm_hooks(net):
    for layer in get_all_afm_layers(list(net.children())[0]):
        add_afm_hook(layer)
    # for m in net:
    #     if isinstance(m, nn.Conv2d):
    #         add_afm_hook(m)

def save_all_afm(net, dir='./'):
    os.makedirs('output/vis', exist_ok=True)
    for name, layer in enumerate(get_all_afm_layers(list(net.children())[0])):
        try:
            save_afm(layer, dir=os.path.join('output/vis', str(name)))
            print(layer)
        except cv2.error:
            print('error')
            print(layer)

def get_all_afm_layers(net):
    """Return generator for layers, pulling out all Conv2d and GroupNorm layers."""
    if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.BatchNorm2d):
        yield net
    elif len(list(net.children())) == 0:
        # Some layer we don't care about for afm, e.g. Linear
        return
    else:
        for c in net.children():
            yield from get_all_afm_layers(c)
    
    #for nbranches, s, stage in (
    #                        (2, 2, net.stage2),
    #                        (3, 3, net.stage3),
    #                        (4, 4, net.stage4)):
    #    for b in range(nbranches):
    #        yield 'stg{}_br{}_convbn1'.format(s, b), \
    #            stage[0].branches[b][0].bn1
    #        yield 'stg{}_br{}_convbn2'.format(s, b), \
    #            stage[0].branches[b][0].bn2
    #        yield 'stg{}_br{}_conv1'.format(s, b), \
    #            stage[0].branches[b][0].conv1
    #        yield 'stg{}_br{}_conv2'.format(s, b), \
    #            stage[0].branches[b][0].conv2
    # for m in net:
    #     if isinstance(m, nn.Conv2d):
    #         save_afm(m, dir=dir)
    
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
        mask = disp_L < args.maxdisp
        mask.detach_()
        outputs = model(imgL, imgR)
        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(stages):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])
        
        if batch_idx % args.print_freq == 0:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    
    model.eval()
    add_all_afm_hooks(model)
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        start = time.time()
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        mask = disp_L < args.maxdisp
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
#                 output = output[:, 4:, :]
#                 result = (np.array(output.cpu())[0]).astype(np.uint8)
#                 imageio.imwrite("result-id{}-stage{}.png".format(batch_idx, x), result)
#                 print("Mask matched")
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())
            
#             gt = (np.array(disp_L[0])*255).astype(np.uint8)
#             imageio.imwrite("gt-id{}.png".format(batch_idx), gt)
        end = time.time()
#         print(end-start)
        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])
        
        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    save_all_afm(model)
    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)

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
