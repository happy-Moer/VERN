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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def computeSSIM(img1, img2):
    ssim_value = ssim(secret_rev, secret, channel_axis=0, data_range=secret.max() - secret.min())
    return ssim_value

def computeRMSE(img1, img2):
    num_channels, _, _ = img1.shape
    rmse_sum = 0.0
    for j in range(num_channels):
        mse = mean_squared_error(img1[j], img2[j])
        rmse = np.sqrt(mse)
        rmse_sum += rmse
    return rmse_sum / num_channels  # average over channels

def computeMAE(img1, img2):
    num_channels, _, _ = img1.shape
    mae_sum = 0.0
    for j in range(num_channels):
        mae_sum += mean_absolute_error(img1[j], img2[j])
    return mae_sum / num_channels  # average over channels

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

# load('models/model/model_checkpoint_00100.pt')
# load('models/model.pt')
load('models/model2/model_checkpoint_01000.pt')

net.eval()

dwt = common.DWT()
iwt = common.IWT()

import time

with torch.no_grad():
    psnr_s = []
    psnr_c = []
    ssim_s = []
    ssim_c = []
    rmse_s = []
    rmse_c = []
    mae_s = []
    mae_c = []
    encryption_time_l = []
    Decryption_time_l = []

    net.eval()
    for x in datasets.testloader:
   

        # print(x.shape)
        x = x.to(device)
        cover = x[:,3:,:,:]
        secret = x[:,:3,:,:]

        start_time = time.time()
        
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

        encryption_time = time.time() - start_time

        #################
        #   backward:   #
        #################
        start_time = time.time()

        output_steg = output_steg.cuda()
        output_rev = torch.cat((output_steg, output_z), 1)
        output_image = net(output_rev, rev=True)
        secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        decryption_time = time.time() - start_time

        secret_rev = secret_rev.cpu().numpy().squeeze() * 255
        np.clip(secret_rev, 0, 255)
        secret = secret.cpu().numpy().squeeze() * 255
        np.clip(secret, 0, 255)
        cover = cover.cpu().numpy().squeeze() * 255
        np.clip(cover, 0, 255)
        steg = steg.cpu().numpy().squeeze() * 255
        np.clip(steg, 0, 255)

        # print(secret_rev.shape)
        # print(secret.shape)
        encryption_time_l.append(encryption_time)
        Decryption_time_l.append(decryption_time)
        
        psnr_s.append(computePSNR(secret_rev, secret))
        psnr_c.append(computePSNR(cover, steg))
        ssim_s.append(computeSSIM(secret_rev, secret))
        ssim_c.append(computeSSIM(cover, steg))
        rmse_s.append(computeRMSE(secret_rev, secret))
        rmse_c.append(computeRMSE(cover, steg))
        mae_s.append(computeMAE(secret_rev, secret))
        mae_c.append(computeMAE(cover, steg))
    
    # print("psnr_s")
    # for item in psnr_s:
    #     print("{:.2f}".format(item), end=',')
    # print()
    # print()

    # print("psnr_c")
    # for item in psnr_c:
    #     print("{:.2f}".format(item), end=',')
    # print()
    # print()

    # print("ssim_s")
    # for item in ssim_s:
    #     print("{:.10f}".format(item), end=',')
    # print()
    # print()
    
    # print("ssim_c")
    # for item in ssim_c:
    #     print("{:.10f}".format(item), end=',')
    # print()
    # print()
    
    # print("rmse_s")
    # for item in rmse_s:
    #     print("{:.2f}".format(item), end=',')
    # print()
    # print()

    # print("rmse_c")
    # for item in rmse_c:
    #     print("{:.2f}".format(item), end=',')
    # print()
    # print()

    # print("mae_s")
    # for item in mae_s:
    #     print("{:.2f}".format(item), end=',')
    # print()
    # print()

    # print("mae_c")
    # for item in mae_c:
    #     print("{:.2f}".format(item), end=',')
    # print()
    # print()
    # print(len(psnr_s))
    print(f"Encryption time: {np.mean(encryption_time_l):.4f} seconds")
    std_value = np.std(encryption_time_l)
    print(std_value)
    std_value = np.max(encryption_time_l)
    print(std_value)
    print(f"Decryption time: {np.mean(Decryption_time_l):.4f} seconds")
    std_value = np.std(Decryption_time_l)
    print(std_value)
    std_value = np.max(Decryption_time_l)
    print(std_value)

    print("PSNR_S, average psnr:", np.mean(psnr_s))
    print("PSNR_C, average psnr:", np.mean(psnr_c))
    print("SSIM_S, average ssim:", np.mean(ssim_s))
    print("SSIM_C, average ssim:", np.mean(ssim_c))
    print("RMSE_S, average rmse:", np.mean(rmse_s))
    print("RMSE_C, average rmse:", np.mean(rmse_c))
    print("MAE_S, average mae:", np.mean(mae_s))
    print("MAE_C, average mae:", np.mean(mae_c))









