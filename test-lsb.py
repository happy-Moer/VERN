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

def hide_4lsb(cover, secret):
    # Ensure cover and secret tensors are in the correct range [0, 255]
    cover = (cover * 255).int()
    secret = (secret * 255).int()

    # Prepare the cover for LSB by zeroing its last 4 bits
    cover = torch.bitwise_and(cover, 240)

    # Prepare the secret for LSB by shifting its values to match the last 4 bits
    secret = torch.bitwise_and(secret, 240)
    secret = torch.right_shift(secret, 4)

    # Combine the cover and secret
    stego = torch.bitwise_or(cover, secret)

    # Extract the hidden secret
    hidden_secret = torch.left_shift(torch.bitwise_and(stego, 15), 4)

    # Ensure the output tensors are in the correct range [0, 1]
    stego = stego.float() / 255.0
    hidden_secret = hidden_secret.float() / 255.0

    return stego, hidden_secret

def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    # check if the shapes are not aligned
    if origin.shape != pred.shape:
        # reshape origin and pred to make sure they have the same shape
        origin = np.transpose(origin, (0, 2, 3, 1))
        pred = np.transpose(pred, (0, 2, 3, 1))
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)
# Test the function
psnr_s = []
psnr_c = []
ssim_s = []
ssim_c = []
rmse_s = []
rmse_c = []
mae_s = []
mae_c = []
with torch.no_grad():
    for x in datasets.testloader:
        x = x.to(device)
        cover = x[:,3:,:,:]
        secret = x[:,:3,:,:]

        # Call the LSB hiding function
        stego, hidden_secret = hide_4lsb(cover, secret)

        print(secret.shape)

        psnr_c.append(computePSNR(cover.cpu(), stego.cpu()))
        psnr_s.append(computePSNR(secret.cpu(), hidden_secret.cpu()))
        
        ssim_c.append(computeSSIM(cover.cpu(), stego.cpu()))
        ssim_s.append(computeSSIM(secret.cpu(), hidden_secret.cpu()))

        rmse_c.append(computeRMSE(cover.cpu(), stego.cpu()))
        rmse_s.append(computeRMSE(secret.cpu(), hidden_secret.cpu()))

        mae_c.append(computeMAE(cover.cpu(), stego.cpu()))
        mae_s.append(computeMAE(secret.cpu(), hidden_secret.cpu()))

        print("PSNR_S, average psnr:", np.mean(psnr_s))
        print("PSNR_C, average psnr:", np.mean(psnr_c))
        print("SSIM_S, average ssim:", np.mean(ssim_s))
        print("SSIM_C, average ssim:", np.mean(ssim_c))
        print("RMSE_S, average rmse:", np.mean(rmse_s))
        print("RMSE_C, average rmse:", np.mean(rmse_c))
        print("MAE_S, average mae:", np.mean(mae_s))
        print("MAE_C, average mae:", np.mean(mae_c))
        




