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
from focal_frequency_loss import FocalFrequencyLoss as FFL
import lpips
from loss.HWMNet import HWMNet

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

        self.ffl_loss = Myloss.FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        self.writer = SummaryWriter(log_dir='logs/2.7')
        self.num=0
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')


        self.HWMNet=None
        # if Pre_train!=None:
        #     self.HWMNet = HWMNet(in_chn=3, wf=96, depth=4)
        #     checkpoint = torch.load('/raid/data_stuyiny/HWMNet-main/LOL_enhancement_HWMNet.pth', map_location='cpu')
        #     self.HWMNet.load_state_dict(checkpoint["state_dict"])
        #     self.HWMNet.eval()




    def forward(self, gt_images, lowlight_image, color_map,epoch):
        """
        Algorithm 1.
        """
        if self.HWMNet!=None:
            mul = 16
            h, w = lowlight_image.shape[2], lowlight_image.shape[3]
            H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
            padh = H - h if h % mul != 0 else 0
            padw = W - w if w % mul != 0 else 0
            hwm_input = F.pad(lowlight_image, (0, padw, 0, padh), 'reflect')
            HWM_result=self.HWMNet(hwm_input)
            HWM_result = HWM_result[:, :, :h, :w]
            color_map=torch.cat([color_map,HWM_result],dim=1)



        t = torch.randint(self.T, size=(gt_images.shape[0],), device=gt_images.device)  # (80,)  T设置为1000
        noise = torch.randn_like(gt_images)  # (80,3,32,32)
        y_t = (
                extract(self.sqrt_alphas_bar, t, gt_images.shape) * gt_images +
                extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise)
        #input = torch.cat([lowlight_image, color_map, y_t], dim=1)
        input = torch.cat([lowlight_image, color_map, y_t], dim=1).float()
        light_high = gt_images.mean([1, 2, 3])  # b*1
        #light_high = torch.ceil(light_high * 10)
        #light_high = torch.round(light_high * 10)
        light_high=torch.clip(light_high,-10,10)

        if torch.rand(1) < 0.02:
            noise_pred = self.model(input, t, light_high,context_zero=True)
        else:
            noise_pred = self.model(input, t, light_high)


        # los
        loss = 0
        mse_loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss += mse_loss

        y_0_pred = 1 / extract(self.sqrt_alphas_bar, t, gt_images.shape) * (
                    y_t - extract(self.sqrt_one_minus_alphas_bar, t, gt_images.shape) * noise_pred)
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
        vgg_loss = self.loss_fn_vgg(gt_images, y_0_pred)*vgg_loss_wight
        loss+=vgg_loss


        #round 2 loss
        # t2 = torch.randint(self.T//2, size=(gt_images.shape[0],), device=gt_images.device)
        # noise = torch.randn_like(gt_images)
        # y_t_roudn2 = (
        #         extract(self.sqrt_alphas_bar, t2, y_0_pred.shape) * y_0_pred +
        #         extract(self.sqrt_one_minus_alphas_bar, t2, y_0_pred.shape) * noise)
        # input = torch.cat([lowlight_image, color_map, y_t_roudn2], dim=1)
        # noise_pred = self.model(input, t, light_high)

        # y0_mask=[1 if tt<300 else 0 for tt in t]
        # y0_mask=torch.Tensor(y0_mask).to(gt_images.device)
        # y0_mask=t<300
        # y0_mask=y0_mask.float()
        # y0_mask=y0_mask.reshape(y0_mask.shape[0],1,1,1)
        # y_0_pred_round2 = 1 / extract(self.sqrt_alphas_bar, t2, gt_images.shape) * (
        #         y_t_roudn2 - extract(self.sqrt_one_minus_alphas_bar, t2, gt_images.shape) * noise_pred)
        # y_0_pred_round2=y0_mask*y_0_pred_round2
        # gt_images=y0_mask*gt_images
        # l1_round2=F.l1_loss(gt_images,y_0_pred_round2)
        # l1_round2=l1_round2*5
        # loss+=l1_round2
        l1_round2=0
        return [loss, mse_loss, col_loss, exposure_loss,ssimLoss,l1_round2,vgg_loss]

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

    def p_mean_variance(self, input, t,y_t,gt_image=None,light_high=None):
        # below: only log_variance is used in the KL computations
        device=input.device
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, input.shape)
        if light_high==None:
            light_high = gt_image.mean([1, 2, 3])  # b*1
            #light_high = torch.round(light_high * 10)
        eps = self.model(input, t,light_high)

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(t, eps,y_t)

        return xt_prev_mean, var


    def forward(self, lowlight_image,snr_map,gt_img=None,light_high=None,ddim=False,unconditional_guidance_scale=1,ddim_step=None):
        """
        Algorithm 2. sdedit
        """
        if ddim==False:
            device=lowlight_image.device
            noise = torch.randn_like(lowlight_image).to(device)
            y_t = noise

            # if self.sdeidt is not None:
            #     self.T=self.sdeidt
            #
            # if self.HWMNet!=None:
            #     mul = 16
            #     h, w = lowlight_image.shape[2], lowlight_image.shape[3]
            #     H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
            #     padh = H - h if h % mul != 0 else 0
            #     padw = W - w if w % mul != 0 else 0
            #     hwm_input = F.pad(lowlight_image, (0, padw, 0, padh), 'reflect')
            #     HWM_result=self.HWMNet(hwm_input)
            #     HWM_result = HWM_result[:, :, :h, :w]
            #     snr_map=torch.cat([snr_map,HWM_result],dim=1)
            #
            #
            # #进行系数的循环搜索
            # if self.samlpe_config['coeff_loop']==True:
            #     self.grad_coeff = self.every_mins*self.mins_num_count
            #     self.mins_num_count -= 1
            #     #重置计算轮次
            #     if self.mins_num_count==self.samlpe_config['sample_num']//(-2)-1:
            #         self.mins_num_count=self.samlpe_config['sample_num']//2

            #print('grad_coeff:',self.grad_coeff)
            # self.T = 10
            # t = self.T * torch.ones(gt_img.shape[0],dtype=torch.int64,device=gt_img.device)

            # y_t = (
            #         extract(self.sqrt_alphas_bar.to(device), t, gt_img.shape) * gt_img +
            #         extract(self.sqrt_one_minus_alphas_bar.to(device), t, gt_img.shape) * noise)
            for time_step in reversed(range(self.T)):
                #print(time_step)
                t = y_t.new_ones([y_t.shape[0], ], dtype=torch.long) * time_step  #(80,)和time_step值一样的
                input = torch.cat([lowlight_image, snr_map, y_t], dim=1).float()
                mean, var= self.p_mean_variance(input, t,y_t,gt_img,light_high)

                # no noise when t == 0
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
            if ddim_step==None:
                step=40  #25步采样
            else:
                step=1000/ddim_step
                step=int(step)
            seq = range(0, 1000, step)
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(y_t.shape[0]) * i).to(device).long()
                next_t = (torch.ones(y_t.shape[0]) * j).to(device).long()
                at = extract(self.alphas_bar.to(device), (t + 1).long(), y_t.shape)
                at_next = extract(self.alphas_bar.to(device), (next_t + 1).long(), y_t.shape)
                input = torch.cat([lowlight_image, snr_map, y_t], dim=1).float()
                if light_high == None:
                    light_high = gt_img.mean([1, 2, 3])  # b*1
                    #light_high = torch.round(light_high * 10)
                eps = self.model(input, t, light_high)

                if unconditional_guidance_scale!=1:
                    eps_unconditional = self.model(input, t, light_high,context_zero=True)
                    eps = eps_unconditional + unconditional_guidance_scale * (eps -eps_unconditional)


                y0_pred = (y_t - eps * (1 - at).sqrt()) / at.sqrt()
                eta=0
                c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                y_t = at_next.sqrt() * y0_pred + c1 * torch.randn_like(lowlight_image) + c2 * eps
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

   