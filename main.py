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
from Diffusion.Model import UNet
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

        data_high = cv2.imread(self.input_data_high[idx])
        data_high=data_high[:,:,::-1].copy()
        #data_high = Image.fromarray(data_high)
        random.seed(1)
        data_high = self.transform(image=data_high)["image"]/255.0
        data_high=data_high*2-1

        data_blur = data_low.permute(1, 2, 0).numpy() * 255.0
        data_blur = cv2.blur(data_blur, (5, 5))
        data_blur = data_blur * 1.0 / 255.0
        data_blur = torch.Tensor(data_blur).float().permute(2, 0, 1)


        return [data_low, data_high,data_color,data_blur]



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

#rgb_to_grayscale_channel_multiplication
def rgb_to_grayscale_channel_multiplication(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calculate_ssim(img1, img2):
    score, _ = SSIM(img1, img2, full=True)
    return score

def train(modelConfig: Dict):
    if modelConfig["DDP"]==True:
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--local_rank", default=-1)
        # FLAGS = parser.parse_args()
        # local_rank = FLAGS.local_rank
        local_rank = int(os.getenv('LOCAL_RANK', -1))
        print('locak rank:',local_rank)
        #local_rank=int(local_rank)+3
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    datapath_train_low = glob.glob(r'/home/pdi/Documentos/CLEDiffusion-final/CLEDiffusion-final/CLEDiffusion-main/data/SUIM/train_val*.jpg')
    datapath_train_high = glob.glob(r'/home/pdi/Documentos/CLEDiffusion-final/CLEDiffusion-final/CLEDiffusion-main/data/SUIM//*.jpg')

    dataload_train=load_data(datapath_train_low, datapath_train_high)
    if modelConfig["DDP"] == True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataload_train)
        dataloader= DataLoader(dataload_train, batch_size=modelConfig["batch_size"],sampler=train_sampler)
    else:
        dataloader = DataLoader(dataload_train, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4,
                                drop_last=True, pin_memory=True)
    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])

    if modelConfig["training_load_weight"] is not None:
        ckpt = torch.load(os.path.join(
                modelConfig["load_weight"]), map_location='cpu')
        net_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        get_epoch = int(modelConfig["load_weight"].split('_')[1])+1
        print('load model from epoch:',get_epoch)
    else:
        get_epoch=0

    if modelConfig["DDP"] == True:
        net_model = DDP(net_model.cuda(), device_ids=[local_rank], output_device=local_rank,)
    else:
        net_model=torch.nn.DataParallel(net_model,device_ids=modelConfig['device_list'])
        device=modelConfig['device_list'][0]
        net_model.to(device)


    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],).to(device)


    log_savedir=modelConfig["ouput_save_dir"]+'logs/'
    if not os.path.exists(log_savedir):
        os.makedirs(log_savedir)
    writer = SummaryWriter(log_dir=log_savedir)

    ckpt_savedir=modelConfig["ouput_save_dir"]+modelConfig["save_weight_dir"]
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)

    save_txt= modelConfig["ouput_save_dir"] + 'res.txt'

    num=0

    for e in range(get_epoch,modelConfig["epoch"]):
       # print(e)
        if modelConfig["DDP"] == True:
           dataloader.sampler.set_epoch(e)


        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:

            for data_low, data_high, data_color, data_blur in tqdmDataLoader:

                data_high = data_high.to(device)
                data_low = data_low.to(device)
                data_color=data_color.to(device)
                data_blur=data_blur.to(device)
                snr_map = getSnrMap(data_low, data_blur)
                data_color=torch.cat([data_color, snr_map], dim=1)
                optimizer.zero_grad()
                [loss, mse_loss, col_loss,exp_loss,ssim_loss,l1round2_loss,vgg_loss] = trainer(data_high, data_low,data_color,e)
                loss = loss.mean()
                mse_loss = mse_loss.mean()
                ssim_loss=ssim_loss.mean()
                vgg_loss = vgg_loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
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
                             
                                              }, num)
                num+=1
        warmUpScheduler.step()

        if e % 50 == 0 :
            if modelConfig["DDP"] == True:
                if dist.get_rank() == 0:
                    torch.save(net_model.state_dict(), os.path.join(
                        modelConfig["ouput_save_dir"],
                        modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

            elif modelConfig["DDP"] == False:
                torch.save(net_model.state_dict(), os.path.join(
                    modelConfig["ouput_save_dir"],
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

        if e%50==0:
            avg_psnr,avg_ssim=Test(modelConfig,e)
            write_data = 'epoch: {}  psnr: {:.4f} ssim: {:.4f}\n'.format(e, avg_psnr,avg_ssim)
            f = open(save_txt, 'a+')
            f.write(write_data)
            f.close()


def Test(modelConfig: Dict,epoch):
    # load model and evaluate
    device = modelConfig['device_list'][0]
    datapath_test_low = glob.glob(r'/home/yyy/data/Dataset/LOL/eval15/low/*.jpg')
    datapath_test_high = glob.glob(r'/home/yyy/data/Dataset/LOL/eval15/high/*.jpg')
    dataload_test = load_data_test(datapath_test_low,datapath_test_high)
    dataloader = DataLoader(dataload_test, batch_size=1, num_workers=4,
                            drop_last=True, pin_memory=True)


    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                 attn=modelConfig["attn"],
                 num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
    ckpt_path=modelConfig["ouput_save_dir"]+modelConfig["save_weight_dir"]+'ckpt_'+str(epoch)+'_.pt'
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")


    sample_savedir=modelConfig["ouput_save_dir"]+'result/epoch'+str(epoch)+'ddim100/'
    #sample_savedir+='31200_ddim100/'  #epoch
    if not os.path.exists(sample_savedir):
        os.makedirs(sample_savedir)
    save_dir=sample_savedir



    save_txt_name =sample_savedir + 'res.txt'
    f = open(save_txt_name, 'w+')
    f.close()

    image_num = 0
    psnr_list = []
    ssim_list = []
    lpips_list=[]


    model.eval()
    sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    loss_fn_vgg=lpips.LPIPS(net='vgg')


    with torch.no_grad():
        with tqdm( dataloader, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for data_low, data_high, data_color,data_blur,filename in tqdmDataLoader:
                # or data_low in tqdmDataLoader:
                #     if image_num<=10:
                #         image_num += 1
                #         continue
                    name=filename[0].split('/')[-1]
                    print('name:',name)
                    gt_images = data_high.to(device)
                    lowlight_image = data_low.to(device)
                    data_color = data_color.to(device)
                    data_blur=data_blur.to(device)
                    snr_map = getSnrMap(lowlight_image, data_blur)
                    data_color=torch.cat([data_color, snr_map], dim=1)

                    #for i in range(-10, 10,1): 
                        # light_high = torch.ones([1]) * i*0.1
                        # light_high = light_high.to(device)
                    time_start = time.time()
                    sampledImgs = sampler(lowlight_image, data_color,gt_images,ddim=True,
                                          unconditional_guidance_scale=1,ddim_step=100)
                    time_end=time.time()
                    #print('time cost:', time_end - time_start)

                    sampledImgs=(sampledImgs+1)/2
                    gt_images=(gt_images+1)/2
                    lowlight_image=(lowlight_image+1)/2
                    res_Imgs=np.clip(sampledImgs.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1] 
                    gt_img=np.clip(gt_images.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]
                    low_img=np.clip(lowlight_image.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]


                    # compute psnr
                    psnr = PSNR(res_Imgs, gt_img)
                    res_Imgs = (res_Imgs * 255)
                    gt_img = (gt_img * 255)
                    low_img = (low_img * 255)
                    #ssim = SSIM(res_Imgs, gt_img, channel_axis=2,data_range=255)
                    res_gray = rgb_to_grayscale_channel_multiplication(res_Imgs)
                    gt_gray = rgb_to_grayscale_channel_multiplication(gt_img)

                    ssim_score = calculate_ssim(res_gray, gt_gray)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim_score)
                    #print('pic', str(image_num),'light:',str(i), 'orgin: ', 'psnr:', psnr, '  ssim:', ssim)

                    # show result
                    # output = np.concatenate([low_img, gt_img, res_Imgs], axis=1) / 255
                    #output = np.concatenate([low_img, gt_img, res_Imgs, res_trick], axis=1) / 255
                    # plt.axis('off')
                    # plt.imshow(output)
                    # plt.show()

                    # save_path = save_dir + name
                    # cv2.imwrite(save_path, output * 255)
                    save_path =save_dir+ str(image_num)+'.png'
                    cv2.imwrite(save_path, res_Imgs)
                    image_num+=1
            # show avg result
                avg_psnr = sum(psnr_list) / len(psnr_list)
                avg_ssim = sum(ssim_list) / len(ssim_list)
                print('psnr_orgin_avg:', avg_psnr)
                print('ssim_orgin_avg:', avg_ssim)

                f = open(save_txt_name, 'w+')
                f.write('\npsnr_orgin :')
                f.write(str(psnr_list))
                f.write('\nssim_orgin :')
                f.write(str(ssim_list))

                f.write('\npsnr_orgin_avg:')
                f.write(str(avg_psnr))
                f.write('\nssim_orgin_avg:')
                f.write(str(avg_ssim))
                f.close()
                return avg_psnr,avg_ssim


#old vision
if __name__== "__main__" :
    modelConfig = {
        ##1 resize   2 randomcrop
        "DDP": False,
        "state": "train", # or eval
        "epoch": 60001,
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
        "device": "cuda:0", #MODIFIQUEI
        "device_list": [1],
        #"device_list": [3,2,1,0],
        "training_load_weight": True,

        "ouput_save_dir": "./output/",
        "save_weight_dir":"Checkpoint/",
        "load_weight": "/data/users/yyy/CnnDiffusion4.15/CnnDiffusion4.15/output/Checkpoint/ckpt_1850_.pt",   #5000*8+2600*16

        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
        }

    train(modelConfig)
    #Test_for_one(modelConfig,epoch=14000)