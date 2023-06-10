import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

# load('model/model_checkpoint_00100.pt')
load('models/model2/model_checkpoint_01000.pt')

net.eval()

dwt = common.DWT()
iwt = common.IWT()


with torch.no_grad():
    psnr_s = []
    psnr_c = []
    net.eval()
    for x in datasets.testloader:
        # print(x.shape)
        x = x.to(device)
        cover = x[:,3:,:,:]
        secret = x[:,:3,:,:]

        cover_input = dwt(cover)
        secret_input = dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        # print()
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        steg = iwt(output_steg)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        output_z = gauss_noise(output_z.shape)

        #################
        #   backward:   #
        #################
        output_steg = output_steg.cuda()
        output_rev = torch.cat((output_steg, output_z), 1)
        output_image = net(output_rev, rev=True)
        secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)

        secret_rev = secret_rev.cpu().numpy().squeeze() * 255
        np.clip(secret_rev, 0, 255)
        secret = secret.cpu().numpy().squeeze() * 255
        np.clip(secret, 0, 255)
        cover = cover.cpu().numpy().squeeze() * 255
        np.clip(cover, 0, 255)
        steg = steg.cpu().numpy().squeeze() * 255
        np.clip(steg, 0, 255)
        psnr_temp = computePSNR(secret_rev, secret)
        psnr_s.append(psnr_temp)
        psnr_temp_c = computePSNR(cover, steg)
        psnr_c.append(psnr_temp_c)

            # writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
            # writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
        print("PSNR_S, average psnr:", np.mean(psnr_s))
        print("PSNR_C, average psnr:", np.mean(psnr_c))

        # torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        # torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        # torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
        # torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)




