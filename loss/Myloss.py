import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def color_loss(output, gt,mask=None):
    img_ref = F.normalize(output, p = 2, dim = 1)
    ref_p = F.normalize(gt, p = 2, dim = 1)
    if mask!=None:
        img_ref=mask*img_ref
        ref_p*=mask
    loss_cos = 1 - torch.mean(F.cosine_similarity(img_ref, ref_p, dim=1))
    # loss_cos = self.mse(img_ref, ref_p)
    return loss_cos

def light_loss(output,gt,mask=None):
    #output = torch.mean(output, 1, keepdim=True)
    #gt=torch.mean(gt,1,keepdim=True)
    output =output[:, 0:1, :, :] * 0.299 + output[:, 1:2, :, :] * 0.587 + output[:, 2:3, :, :] * 0.114
    gt = gt[:, 0:1, :, :] * 0.299 + gt[:, 1:2, :, :] * 0.587 + gt[:, 2:3, :, :] * 0.114
    if mask != None:
        output*=mask
        gt*=mask
    loss=F.l1_loss(output,gt)
    return loss



class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x ):
        device=x.device
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).to(device),2))
        return d
