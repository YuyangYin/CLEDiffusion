# **CLE Diffusion: Controllable Light Enhancement Diffusion Model**

# Paper Link:

[[2308.06725\] CLE Diffusion: Controllable Light Enhancement Diffusion Model (arxiv.org)](https://arxiv.org/abs/2308.06725)



Project Page：

[CLE Diffusion: Controllable Light Enhancement Diffusion Model(ACM MM 2023) (yuyangyin.github.io)](https://yuyangyin.github.io/CLEDiffusion/)


# Data
Download LOL dataset from [LOL](https://daooshee.github.io/BMVC2018website/).

Change your own dataset path in four places(datapath_train_low,datapath_train_high,datapath_test_low,datapath_test_high) in main.py .

The code also supports other dataset.


# Usage
```python
python main.py   #train from scratch, you can change setting in modelConfig 
```




# To Do List

- [x] LOL dataset outputs

- [x] release train and test code

- [ ] release Mask CLE Diffusion code



Due to the recent busy schedule with CVPR's deadline, we will do our best to release the code shortly after the CVPR submission deadline. Thanks for your attention.



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

