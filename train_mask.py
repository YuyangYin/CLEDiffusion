import colorsys
import os
from typing import Dict, List
import PIL
import lpips as lpips
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import torch
import torch.optim as optim
#from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import albumentations as A
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet,UNet_Mask
from Scheduler import GradualWarmupScheduler
from loss import Myloss
import numpy as np
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM    
import torch.utils.data as data
import glob
import random
from albumentations.pytorch import ToTensorV2
import lpips
import time
import argparse

class load_data(data.Dataset):
    def __init__(self, input_data_low, input_data_high):
        self.input_data_low = input_data_low
        self.input_data_high = input_data_high
        print("Total training examples:", len(self.input_data_high))
        self.transform=A.Compose(
            [
                A.RandomCrop(height=128, width=128),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )

        self.mask_path='./data/mask/'
    
    def __len__(self):
        return len(self.input_data_low)

    def __getitem__(self, idx):
        rand_int = torch.randint(low=0, high=500, size=(1,)).item()
        mask=self.mask_path+str(rand_int)+'.png'
        mask=cv2.imread(mask)/255.
        mask=torch.from_numpy(mask)
        mask=mask.permute(2, 0, 1)
        
        seed = torch.random.seed()

        data_low = cv2.imread(self.input_data_low[idx])
        data_low=data_low[:,:,::-1].copy()
        random.seed(1)
        data_low = self.transform(image=data_low)["image"]
        data_low=data_low/255.0
        data_low2 = data_low
        data_low2 = data_low2 * 2 - 1
        data_low=np.power(data_low,0.25)

         #mean and var of lol training dataset. If you change dataset, please change mean and var.
        mean=torch.tensor([0.4350, 0.4445, 0.4086])  
        var=torch.tensor([0.0193, 0.0134, 0.0199])
        data_low=(data_low-mean.view(3,1,1))/var.view(3,1,1)
        data_low=data_low/20


        data_max_r=data_low[0].max()
        data_max_g = data_low[1].max()
        data_max_b = data_low[2].max()
        color_max=torch.zeros((data_low.shape[0],data_low.shape[1],data_low.shape[2]))
        color_max[0,:,:]=data_max_r*torch.ones((data_low.shape[1],data_low.shape[2]))    
        color_max[1,:, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2]))
        color_max[2,:, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2]))
        data_color=data_low/(color_max+ 1e-6)

        data_high = cv2.imread(self.input_data_high[idx])
        data_high=data_high[:,:,::-1].copy()
        random.seed(1)
        data_high = self.transform(image=data_high)["image"]/255.0
        data_high=data_high*2-1
        
        brighness_leve_high = data_high.mean([0, 1, 2])
        brighness_leve_high = torch.round(brighness_leve_high * 10)
        data_high=mask*data_high+(1-mask)*data_low2
        data_high=data_high.float()

        data_blur = data_low.permute(1, 2, 0).numpy() * 255.0
        data_blur = cv2.blur(data_blur, (5, 5))
        data_blur = data_blur * 1.0 / 255.0
        data_blur = torch.Tensor(data_blur).float().permute(2, 0, 1)

        mask = mask[0:1,:, :]
        return [data_low, data_high,data_color,data_blur,mask,brighness_leve_high]



class load_data_test(data.Dataset):
    def __init__(self, input_data_low, input_data_high):
        self.input_data_low = input_data_low
        self.input_data_high = input_data_high
        print("Total training examples:", len(self.input_data_high))
        self.transform=A.Compose(
            [
                ToTensorV2(),
            ]
        )


    def __len__(self):
        return len(self.input_data_low)

    def __getitem__(self, idx):
        seed = torch.random.seed()

        data_low = cv2.imread(self.input_data_low[idx])
        data_low=data_low[:,:,::-1].copy()
        random.seed(1)
        data_low=data_low/255.0

        data_low=np.power(data_low,0.25)
        data_low = self.transform(image=data_low)["image"]
        mean=torch.tensor([0.4350, 0.4445, 0.4086])
        var=torch.tensor([0.0193, 0.0134, 0.0199])
        data_low=(data_low-mean.view(3,1,1))/var.view(3,1,1)
        data_low=data_low/20


        data_max_r=data_low[0].max()
        data_max_g = data_low[1].max()
        data_max_b = data_low[2].max()
        color_max=torch.zeros((data_low.shape[0],data_low.shape[1],data_low.shape[2]))
        color_max[0,:,:]=data_max_r*torch.ones((data_low.shape[1],data_low.shape[2]))    
        color_max[1,:, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2]))
        color_max[2,:, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2]))
        data_color=data_low/(color_max+ 1e-6)
        #data_color = self.transform(data_color)
        #data_color=torch.from_numpy(data_color).float()
        #data_color=data_color.permute(2,0,1)

        data_high = cv2.imread(self.input_data_high[idx])
        data_high=data_high[:,:,::-1].copy()
        #data_high = Image.fromarray(data_high)
        random.seed(1)
        data_high = self.transform(image=data_high)["image"]/255.0
        data_high=data_high*2-1

        #normalization
        #data_high=data_high**0.25

        data_blur = data_low.permute(1, 2, 0).numpy() * 255.0
        data_blur = cv2.blur(data_blur, (5, 5))
        data_blur = data_blur * 1.0 / 255.0
        data_blur = torch.Tensor(data_blur).float().permute(2, 0, 1)


        return [data_low, data_high,data_color,data_blur,self.input_data_low[idx]]


def getSnrMap(data_low,data_blur):
    data_low = data_low[:, 0:1, :, :] * 0.299 + data_low[:, 1:2, :, :] * 0.587 + data_low[:, 2:3, :, :] * 0.114
    data_blur = data_blur[:, 0:1, :, :] * 0.299 + data_blur[:, 1:2, :, :] * 0.587 + data_blur[:, 2:3, :, :] * 0.114
    noise = torch.abs(data_low - data_blur)

    mask = torch.div(data_blur, noise + 0.0001)

    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
    mask_max = mask_max.view(batch_size, 1, 1, 1)
    mask_max = mask_max.repeat(1, 1, height, width)
    mask = mask * 1.0 / (mask_max + 0.0001)

    mask = torch.clamp(mask, min=0, max=1.0)
    mask = mask.float()
    return mask

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_color_map(im):
    return im / (rgb2gray(im)[..., np.newaxis] + 1e-6) * 100
    # return im / (np.mean(im, axis=-1)[..., np.newaxis] + 1e-6) * 100


def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calculate_ssim(img1, img2):
    score, _ = SSIM(img1, img2, full=True)
    return score

def train(config: Dict):
    if config.DDP==True:
        local_rank = int(os.getenv('LOCAL_RANK', -1))
        print('locak rank:',local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    
    train_low_path=config.dataset_path+r'our485/low/*.png'    
    train_high_path=config.dataset_path+r'our485/high/*.png'
    datapath_train_low = glob.glob(train_low_path)
    datapath_train_high = glob.glob(train_high_path)
    dataload_train=load_data(datapath_train_low, datapath_train_high)
    if config.DDP == True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataload_train)
        dataloader= DataLoader(dataload_train, batch_size=config.batch_size,sampler=train_sampler)
    else:
        dataloader = DataLoader(dataload_train, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                drop_last=True, pin_memory=True)
        
    net_model = UNet_Mask(T=config.T, ch=config.channel, ch_mult=config.channel_mult, attn=config.attn,
                     num_res_blocks=config.num_res_blocks, dropout=config.dropout)

    if config.pretrained_path is not None:
        ckpt = torch.load(os.path.join(
                config.pretrained_path), map_location='cpu')
        checkpoint = torch.load({k.replace('module.', ''): v for k, v in ckpt.items()})
        state_dict = checkpoint['state_dict']
        state_dict['conv1.weight'][:, :10, :, :] = state_dict['conv1.weight']
        state_dict['conv1.weight'][:, 10, :, :] = torch.zeros_like(state_dict['conv1.weight'][:, 0, :, :])
        state_dict['conv1.in_channels'] = 11

    if config.DDP == True:
        net_model = DDP(net_model.cuda(), device_ids=[local_rank], output_device=local_rank,)
    else:
        net_model=torch.nn.DataParallel(net_model,device_ids=config.device_list)
        device=config.device_list[0]
        net_model.to(device)


    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=config.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=config.multiplier, warm_epoch=config.epoch // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, config.beta_1, config.beta_T, config.T,).to(device)


    log_savedir=config.output_path+'/logs/'
    if not os.path.exists(log_savedir):
        os.makedirs(log_savedir)
    writer = SummaryWriter(log_dir=log_savedir)

    ckpt_savedir=config.output_path+'/ckpt/'
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)

    save_txt= config.output_path + 'res.txt'
    
    num=0
    for e in range(config.epoch):
        if config.DDP == True:
           dataloader.sampler.set_epoch(e)

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data_low, data_high, data_color, data_blur, mask,brighness_leve_high in tqdmDataLoader:
                data_high = data_high.to(device)
                data_low = data_low.to(device)
                data_color=data_color.to(device)
                data_blur=data_blur.to(device)
                snr_map = getSnrMap(data_low, data_blur)
                mask=mask.to(device)
                brighness_leve_high=brighness_leve_high.to(device)
                
                data_concate=torch.cat([data_color, snr_map,mask], dim=1)
                optimizer.zero_grad()
                [loss, mse_loss, col_loss,exp_loss,ssim_loss,vgg_loss] = trainer(data_high, data_low,data_concate,e,brighness_leve_high)
                loss = loss.mean()
                mse_loss = mse_loss.mean()
                ssim_loss=ssim_loss.mean()
                vgg_loss = vgg_loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "mse_loss":mse_loss.item(),
                    "exp_loss":exp_loss.item(),
                    "col_loss":col_loss.item(),
                    'ssim_loss':ssim_loss.item(),
                    'vgg_loss':vgg_loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    "num":num+1
                })
                loss_num=loss.item()
                mse_num=mse_loss.item()
                exp_num=exp_loss.item()
                col_num=col_loss.item()
                ssim_num = ssim_loss.item()
                vgg_num=vgg_loss.item()
                writer.add_scalars('loss', {"loss_total":loss_num,
                                             "mse_loss":mse_num,
                                             "exp_loss":exp_num,
                                            'ssim_loss':ssim_num,
                                             "col_loss":col_num,
                                            "vgg_loss":vgg_num,
                                              }, num)
                num+=1
        warmUpScheduler.step()
        
        if e % 50 == 0 :
            if config.DDP == True:
                if dist.get_rank() == 0:
                    torch.save(net_model.state_dict(), os.path.join(
                        ckpt_savedir, 'ckpt_' + str(e) + "_.pt"))
            elif config.DDP == False:
                torch.save(net_model.state_dict(), os.path.join(
                    ckpt_savedir, 'ckpt_' + str(e) + "_.pt"))



if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    modelConfig = {

        "DDP": False,
        "state": "train", # or eval
        "epoch": 5000,
        "batch_size":16 ,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "device_list": [3],
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
    }


    parser.add_argument('--dataset_path', type=str, default="./data/LOL/")
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval
    parser.add_argument('--output_path', type=str, default="./output_mask/")  #or eval

    config = parser.parse_args()
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    print(config)
    train(config)
    #Test_for_one(modelConfig,epoch=14000)