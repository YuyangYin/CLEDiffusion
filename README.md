# **(MM2023)CLE Diffusion: Controllable Light Enhancement Diffusion Model**
Authors: Yuyang Yin, Dejia Xu, Chuangchuang Tan, Ping Liu, Yao Zhao, Yunchao Wei

Institution: Beijing Jiaotong University, UT Austin, A*Star

# Paper Link:

[[2308.06725\] CLE Diffusion: Controllable Light Enhancement Diffusion Model (arxiv.org)](https://arxiv.org/abs/2308.06725)




Project Page：

[CLE Diffusion: Controllable Light Enhancement Diffusion Model(ACM MM 2023) (yuyangyin.github.io)](https://yuyangyin.github.io/CLEDiffusion/)


# Data
Download LOL dataset from [LOL](https://daooshee.github.io/BMVC2018website/). Put the dataset under '/data/LOL' file.

The code also supports other dataset.

# Checkpoint
[Pretrianed model](https://drive.google.com/file/d/1uf8Sj1LUduWs6TALM77wxapMAmoGIaEY/view?usp=sharing) on LOL dataset 

# Setup
```python
pip install -r requirements.txt
```

# Usage
Our code structure is based on the original implementation of DDPM. We test the code on one RTX3090. Increasing the size of the U-Net may lead to better results.
```python
python train.py   #train from scratch, you can change setting in modelConfig 
python train.py --pretrained_path ckpt/lol.pt  
python test.py --pretrained_path ckpt/lol.pt  
```

# Mask CLE Diffusion
Mask CLE Diffusion finetunes lol checkpoint. In our experiments, lol checkpoint is better than mit-adobe-5K checkpoint.

We show some inference cases in 'data/Mask_CLE_cases'. Welcome to use your cases to test the performance.

```python
python mask_generation.py   #generate masks for training
python train_mask.py --pretrained_path ckpt/lol.pt  #finetune Mask CLE Diffusion
python test_mask.py --pretrained_path ckpt/mask_cle.pt --input_path data/Mask_CLE_cases/opera.png --mask_path data/Mask_CLE_cases/opera_mask.png
```




# To Do List

- [x] LOL dataset outputs

- [x] release training and testing code

- [x] release Mask CLE Diffusion code

- [ ] release mit-5k training code and checkpoints

- [ ] update lpips and LI-lpips metrics



# Acknowledgement
This work is mainly built on [DenoisingDiffusionProbabilityModel-ddpm](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-). Thanks a lot to authors for sharing!

# Citation



```
@inproceedings{yin2023cle,
  title={CLE Diffusion: Controllable Light Enhancement Diffusion Model},
  author={Yin, Yuyang and Xu, Dejia and Tan, Chuangchuang and Liu, Ping and Zhao, Yao and Wei, Yunchao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8145--8156},
  year={2023}
}
```

If you have any problems, please feel free to create a new issue or email me(yuyangyin@bjtu.edu.cn)..
