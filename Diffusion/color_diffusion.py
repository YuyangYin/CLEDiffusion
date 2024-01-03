from typing import Dict
from tensorboardX import SummaryWriter
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from loss import Myloss
import numpy as np
from focal_frequency_loss import FocalFrequencyLoss as FFL



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
    def __init__(self, model, beta_1, beta_T, T,col_loss=None,exp_loss=None):
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
        if col_loss is not None:
            self.L_color = Myloss.L_color()
        self.exp_loss=None
        if exp_loss is not None:
            self.exp_loss=Myloss.L_exp(16,0.5)
        self.ffl_loss = Myloss.FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        self.writer = SummaryWriter(log_dir='logs/2.7')
        self.num=0

    def forward(self, gt_images,lowlight_image,color_map,snr_map):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(gt_images.shape[0], ), device=gt_images.device)  #(80,)  T设置为1000
        noise = torch.randn_like(gt_images)  #(80,3,32,32)
        y_t = (
            extract(self.sqrt_alphas_bar, t, gt_images.shape) * gt_images +
            extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise)
        input=torch.cat([lowlight_image, color_map,y_t], dim=1)

        noise_pred=self.model(input, t)
        
        #los
        loss = 0
        mse_loss = F.mse_loss(noise_pred, noise, reduction='none') 
        loss+=mse_loss

        y_0_pred=1/extract(self.sqrt_alphas_bar, t, gt_images.shape)*(y_t-extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape)*noise_pred)
        # plt.axis('off')
        # plt.imshow(np.clip(y_0_pred.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1))
        # plt.show()

        col_loss=0
        if self.L_color is not None:
            col_loss=torch.mean(self.L_color(y_0_pred))
            col_loss=col_loss*0.001
            loss+=col_loss
           # print('\nmse_loss:',torch.mean(mse_loss).item(),'  col_loss:',col_loss.item())
        exposure_loss=0
        if self.exp_loss is not None:
            exposure_loss=torch.mean(self.exp_loss(y_0_pred))*0.01
            loss+=exposure_loss

        return [loss,mse_loss,col_loss,exposure_loss]
        #return mse_loss



class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.sqrt_alphas_bar=alphas_bar
        self.sqrt_one_minus_alphas_bar=torch.sqrt(1. - alphas_bar)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))



    def predict_xt_prev_mean_from_eps(self,  t, eps,y_t):
        assert y_t.shape == eps.shape
        return (
            extract(self.coeff1, t, y_t.shape) * y_t -
            extract(self.coeff2, t, y_t.shape) * eps
        )

    def p_mean_variance(self, input, t,y_t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, input.shape)

        eps = self.model(input, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(t, eps,y_t)

        return xt_prev_mean, var


    def forward(self, lowlight_image,snr_map):
        """
        Algorithm 2. sdedit
        """
        device=lowlight_image.device
        noise = torch.randn_like(lowlight_image).to(device)
        y_t = noise

        for time_step in reversed(range(self.T)):

            t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step  #(80,)和time_step值一样的
            input=torch.cat([lowlight_image, snr_map,y_t], dim=1)
            mean, var= self.p_mean_variance(input, t,y_t)
            
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(y_t)
            else:
                noise = 0
            y_t = mean + torch.sqrt(var) * noise
            #assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."

        y_0 = y_t
        return torch.clip(y_0, -1, 1)
    
    # def forward(self, noisyImage,lowlight_image,gt_image=None):
    #
    #     """
    #     Algorithm 2.
    #     """
    #     if self.sdeidt is not None:
    #         self.T=self.sdeidt
    #
    #     y_t = noisyImage
    #
    #     #进行系数的循环搜索
    #     if self.samlpe_config['coeff_loop']==True:
    #         self.grad_coeff = self.every_mins*self.mins_num_count
    #         self.mins_num_count -= 1
    #         #重置计算轮次
    #         if self.mins_num_count==self.samlpe_config['sample_num']//(-2)-1:
    #             self.mins_num_count=self.samlpe_config['sample_num']//2
    #
    #     print('grad_coeff:',self.grad_coeff)
    #
    #     y_t_list=[]
    #     #self.T=0
    #     for time_step in reversed(range(self.T)):
    #     #for time_step in range(2):
    #         #print('time',time_step)
    #         t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step  #(80,)和time_step值一样的
    #         mean, var= self.p_mean_variance(torch.cat([lowlight_image, y_t], dim=1), t,y_t,gt_image)
    #         # no noise when t == 0
    #         if time_step > 0:
    #             noise = torch.randn_like(y_t)
    #         else:
    #             noise = 0
    #         y_t = mean + torch.sqrt(var) * noise
    #         y_append=y_t.detach().cpu().numpy()[0].transpose(1, 2, 0)
    #
    #         if self.samlpe_config['animation'] == True:
    #             y_t_list.append(y_append)
    #
    #     #animate
    #     if self.samlpe_config['animation']==True:
    #         fig, axs = plt.subplots()
    #         ims=[]
    #         for i in range(1000):
    #             ims.append([axs.imshow(y_t_list[i])])
    #         ani=animation.ArtistAnimation(fig,ims,interval=5,repeat_delay=1000,blit=True)
    #         save_dir = 'output/' + 'loop_grad/1.29night/t{}_sde{}_coeff{}_avg{}_y0{}/'.format(
    #             self.samlpe_config['time_after'],
    #             self.samlpe_config['sdedit'],
    #             self.samlpe_config['coeff'] ,
    #             self.samlpe_config['avg_type'],
    #             self.samlpe_config['y_0_pred'])
    #         save_name=save_dir+str(self.grad_coeff)+'res.gif'
    #         ani.save(save_name,dpi=96,fps=200)
    #         plt.show()
    #         #ln, = axs.plot([], [], animated=False)
    #
    #     if torch.isnan(y_t).int().sum() !=0 :
    #         print('nan in tensor')
    #         #assert torch.isnan(y_t).int().sum() == 0, "nan in tensor."
    #
    #     y_0 = y_t
    #     return torch.clip(y_0, -1, 1)

   