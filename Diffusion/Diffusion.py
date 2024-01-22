from typing import Dict
from tensorboardX import SummaryWriter
import cv2
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#from matplotlib.animation import FuncAnimation
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from loss import Myloss
from kornia.losses import ssim_loss
import numpy as np
import lpips


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    # t (80,)
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)  #gather用来取索引   取出80个内容
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))  #view和reshape一样   [80]+[1,1,1]=[80,1,1,1]


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,Pre_train=None):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        #self.alphas_bar=alphas_bar
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.L_color=None
        self.num=0
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')



    def forward(self, gt_images, lowlight_image, data_concate,epoch,brighness_leve_high=None):
        """
        Algorithm 1.
        """

        t = torch.randint(self.T, size=(gt_images.shape[0],), device=gt_images.device)  # (80,)  T设置为1000
        noise = torch.randn_like(gt_images)  # (80,3,32,32)
        y_t = (
                extract(self.sqrt_alphas_bar, t, gt_images.shape) * gt_images +
                extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise)

        input = torch.cat([lowlight_image, data_concate, y_t], dim=1).float()
        light_high = gt_images.mean([1, 2, 3])  # b*1
        light_high=torch.clip(light_high,-10,10)

        #train mask CLE diffusion
        if brighness_leve_high!=None:
            light_high=brighness_leve_high.int()

        if torch.rand(1) < 0.02:
            noise_pred = self.model(input, t, light_high,context_zero=True)
        else:
            noise_pred = self.model(input, t, light_high)



        # los
        loss = 0
        mse_loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss += mse_loss

        y_0_pred = 1 / extract(self.sqrt_alphas_bar, t, gt_images.shape) * (
                    y_t - extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise_pred).float()
        # plt.axis('off')
        # plt.imshow(np.clip(y_0_pred.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1))
        # plt.show()

        col_loss = 0
        col_loss_weight=100
        if epoch<20:
            col_loss_weight=0
        col_loss = Myloss.color_loss(y_0_pred, gt_images) * col_loss_weight
        loss+=col_loss

        exposure_loss=0
        exposure_loss_weight=20
        if epoch<20:
            exposure_loss_weight=0
        exposure_loss = Myloss.light_loss(y_0_pred, gt_images) * exposure_loss_weight
        loss+=exposure_loss

        ssimLoss = ssim_loss(y_0_pred, gt_images, window_size=11)
        ssimLoss*=2.83
        loss+=ssimLoss
        #ssimLoss=0

        vgg_loss=0
        vgg_loss_wight=50
        # print('y_0_pred:',  y_0_pred.dtype)
        # print('gt_images:',  gt_images.dtype)
        vgg_loss = self.loss_fn_vgg(gt_images, y_0_pred)*vgg_loss_wight
        loss+=vgg_loss

        return [loss, mse_loss, col_loss, exposure_loss,ssimLoss,vgg_loss]

    # def forward(self, gt_images,lowlight_image,snr_map):
    #     """
    #     Algorithm 1.
    #     """
    #     t = torch.randint(self.T, size=(gt_images.shape[0], ), device=gt_images.device)  #(80,)  T设置为1000
    #     noise = torch.randn_like(gt_images)  #(80,3,32,32)
    #     y_t = (
    #         extract(self.sqrt_alphas_bar, t, gt_images.shape) * gt_images +
    #         extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise)
    #     input=torch.cat([lowlight_image, snr_map,y_t], dim=1)
    #
    #     noise_pred=self.model(input, t)
    #
    #     #los
    #     loss = 0
    #     mse_loss = F.mse_loss(noise_pred, noise, reduction='none')
    #     loss+=mse_loss
    #
    #     y_0_pred=1/extract(self.sqrt_alphas_bar, t, gt_images.shape)*(y_t-extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape)*noise_pred)
    #     # plt.axis('off')
    #     # plt.imshow(np.clip(y_0_pred.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1))
    #     # plt.show()
    #
    #     col_loss=0
    #     if self.L_color is not None:
    #         col_loss=torch.mean(self.L_color(y_0_pred))
    #         col_loss=col_loss*0.001
    #         loss+=col_loss
    #        # print('\nmse_loss:',torch.mean(mse_loss).item(),'  col_loss:',col_loss.item())
    #     exposure_loss=0
    #     if self.exp_loss is not None:
    #         exposure_loss=torch.mean(self.exp_loss(y_0_pred))*0.01
    #         loss+=exposure_loss
    #
    #     return [loss,mse_loss,col_loss,exposure_loss]
    #     #return mse_loss



class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.sqrt_alphas_bar=alphas_bar
        self.sqrt_one_minus_alphas_bar=torch.sqrt(1. - alphas_bar)
        self.alphas_bar=alphas_bar
        self.one_minus_alphas_bar=(1. - alphas_bar)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))


        self.grad_coeff = None
        self.every_mins = 0




    def predict_xt_prev_mean_from_eps(self,  t, eps,y_t):
        assert y_t.shape == eps.shape
        return (
            extract(self.coeff1, t, y_t.shape) * y_t -
            extract(self.coeff2, t, y_t.shape) * eps
        )

    def p_mean_variance(self, input, t,y_t,brightness_level):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, input.shape)
        eps = self.model(input, t,brightness_level)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(t, eps,y_t)

        return xt_prev_mean, var


    def forward(self, lowlight_image,data_concate,brightness_level,ddim=False,unconditional_guidance_scale=1,ddim_step=None):

        if ddim==False:
            device=lowlight_image.device
            noise = torch.randn_like(lowlight_image).to(device)
            y_t = noise
            for time_step in reversed(range(self.T)):
                t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step  
                input = torch.cat([lowlight_image, data_concate, y_t], dim=1).float()
                mean, var= self.p_mean_variance(input, t,y_t,brightness_level)
                if time_step > 0:
                    noise = torch.randn_like(y_t)
                else:
                    noise = 0
                y_t = mean + torch.sqrt(var) * noise
                #assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

            y_0 = y_t
            return torch.clip(y_0, -1, 1)

        else:
            device = lowlight_image.device
            noise = torch.randn_like(lowlight_image).to(device)
            y_t = noise
 
            step=1000/ddim_step
            step=int(step)
            seq = range(0, 1000, step)
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(y_t.shape[0]) * i).to(device).long()
                next_t = (torch.ones(y_t.shape[0]) * j).to(device).long()
                at = extract(self.alphas_bar.to(device), (t + 1).long(), y_t.shape)
                at_next = extract(self.alphas_bar.to(device), (next_t + 1).long(), y_t.shape)
                input = torch.cat([lowlight_image, data_concate, y_t], dim=1).float()
                eps = self.model(input, t, brightness_level)

                #classifier free guide
                if unconditional_guidance_scale!=1:  
                    eps_unconditional = self.model(input, t, brightness_level,context_zero=True)
                    eps = eps_unconditional + unconditional_guidance_scale * (eps -eps_unconditional)


                y0_pred = (y_t - eps * (1 - at).sqrt()) / at.sqrt()
                eta=0
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                y_t = at_next.sqrt() * y0_pred + c1 * torch.randn_like(lowlight_image) + c2 * eps
            y_0 = y_t
            return torch.clip(y_0, -1, 1)
