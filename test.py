import torch
import utility
import data
import model
from option import args
import os
from importlib import import_module
import math
import lpips
import sys
import numpy as np
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm

torch.manual_seed(args.seed)

def calc_psnr_ssim(hr, sr, shave, rgb_range):

    y_sr = 1.0*sr/rgb_range
    y_sr[:, 0, :, :] = y_sr[:, 0, :, :] * 65.738/256
    y_sr[:, 1, :, :] = y_sr[:, 1, :, :] * 129.057/256
    y_sr[:, 2, :, :] = y_sr[:, 2, :, :] * 25.064/256

    y_hr = 1.0*hr/rgb_range
    y_hr[:, 0, :, :] = y_hr[:, 0, :, :] * 65.738/256
    y_hr[:, 1, :, :] = y_hr[:, 1, :, :] * 129.057/256
    y_hr[:, 2, :, :] = y_hr[:, 2, :, :] * 25.064/256

    y_sr = torch.sum(y_sr, dim=1)
    y_hr = torch.sum(y_hr, dim=1)

    y_sr = y_sr[:, shave:-shave, shave:-shave]
    y_hr = y_hr[:, shave:-shave, shave:-shave]

    diff = y_sr - y_hr
    mse = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)

    y_sr = y_sr.squeeze(0).cpu().numpy()
    y_hr = y_hr.squeeze(0).cpu().numpy()

    ssim = structural_similarity(y_sr, y_hr, data_range=y_sr.max()-y_sr.min())

    return psnr, ssim


def normalize_tensor(t, dim=1):
    normed_tensors = []
    for i in range(t.size()[dim]):
        new_t = t[:, i, :, :]
        new_t_norm = (new_t - new_t.min())/(new_t.max()-new_t.min())
        normed_tensors.append(2*new_t_norm - 1)
    
    tt = torch.cat(normed_tensors, dim=0)

    return tt.unsqueeze(0)


def test():
    
    scale  = args.scale[0]

    if args.data_test[0] in ['Set5', 'Set14', 'B100', 'Urban100']:
        module_test = import_module('data.benchmark')
        test_dataset = getattr(module_test, 'Benchmark')(args, name=args.data_test[0], train=False)
    else:
        print('Specified dataset is not yet supported.')
        sys.exit(0)

    loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    model_ = model.Model(args, None)
    device = torch.device('cpu' if args.cpu else 'cuda')

    print('Total params: %.2fM' % (sum(p.numel() for p in model_.parameters())/1000000.0))

    if args.save_test_results:
        model_testdir = 'results_' + args.model.lower()
        if not os.path.isdir(model_testdir):
            os.mkdir(model_testdir)

        sr_results_dir = model_testdir + '/' + args.data_test[0] + '_SR'
        if not os.path.isdir(sr_results_dir):
            os.mkdir(sr_results_dir)
    
    model_.eval()
    with torch.no_grad():

        psnr = 0
        ssim = 0
        calc_lpips = args.lpips
        
        if calc_lpips == True:
            loss_fn_alex = lpips.LPIPS(net='alex').to(device)
            loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

            lpips_alex = 0
            lpips_vgg = 0
        
        for i, (lr, hr, img_name) in enumerate(tqdm(loader)):
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model_(lr, 0)
            sr = utility.quantize(sr, args.rgb_range)

            psnr_i, ssim_i = calc_psnr_ssim(hr, sr, args.scale[0], args.rgb_range)
            psnr += psnr_i
            ssim += ssim_i

            if calc_lpips == True:
                sr_norm = normalize_tensor(sr)
                hr_norm = normalize_tensor(hr)
                
                alex_loss = loss_fn_alex(sr_norm, hr_norm)
                lpips_alex += alex_loss.item()

                vgg_loss = loss_fn_vgg(sr_norm.to(device), hr_norm.to(device))
                lpips_vgg += vgg_loss.item()
            
            if args.save_test_results:
                normalized = sr.data.mul(255 / args.rgb_range)
                normalized = normalized.view(3, int(sr.shape[2]), int(sr.shape[3]))
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                # print('Saving %s.'%img_name)
                imageio.imwrite(sr_results_dir + '/{}.png'.format(img_name[0]), ndarr)


        
        print('%s Results.'%args.data_test[0])
        print('PSNR/SSIM: %.2f/%.3f'%(psnr/len(loader), ssim/len(loader)))

        if calc_lpips == True:
            print('LPIPS VGGNet/AlexNet: %.4f/%.4f'%(lpips_vgg/len(loader), lpips_alex/len(loader)))


if __name__ == '__main__':
    test()