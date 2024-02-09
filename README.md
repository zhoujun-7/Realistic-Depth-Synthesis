# Realistic Depth Image Synthesis for 3D Hand Pose Estimation
This is the official implementation for the paper, "Realistic Depth Image Synthesis for 3D Hand Pose Estimation".

```
@ARTICLE{10310128,
  author={Zhou, Jun and Xu, Chi and Ge, Yuting and Cheng, Li},
  journal={IEEE Transactions on Multimedia}, 
  title={Realistic Depth Image Synthesis for 3D Hand Pose Estimation}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2023.3330522}}

```


## Installation
```
conda env create -f env_Sty2.yaml
```



## Download

Download the checkpoint for nyu style image at [here](
https://1drv.ms/f/c/1ed9576d2145b823/EoVwSqY6ji1JvjEi1mUvR5gBhkRr51PBPBM0WbdtuxmXow?e=RifYwH).

Download the noise-free image at [here](
https://1drv.ms/f/c/1ed9576d2145b823/EoVwSqY6ji1JvjEi1mUvR5gBhkRr51PBPBM0WbdtuxmXow?e=RifYwH).

More noise-free image can be synthesized following this [work](https://github.com/anilarmagan/HANDS19-Challenge-Toolbox).

Finally, the directories look like
```
├── ckpt
│   └── nyu_style.pkl
├── data
│   ├── 2-0_Is
│   └── mean_std.npy
├── Dataset_utils
├── Dnnlib
├── env_Sty2.yaml
├── generate_nyu.py
├── Metrics
├── Networks
└── Torch_utils
```

## Generation Realistic Depth Images
```
python generate_nyu.py
```


## Acknowledgement

We thanks [HANDS19-Challenge-Toolbox](https://github.com/anilarmagan/HANDS19-Challenge-Toolbox) for providing a simple way to synthesis noise-free images.

We thank the following repos providing helpful components/functions in our work.

[StyleGAN2](https://github.com/NVlabs/stylegan2)
[A2J](https://github.com/zhangboshen/A2J)
[DCL](https://github.com/Jhonve/DCL-DepthSynthesis)
[Noise Simulator](https://github.com/ShudaLi/rgbd_simulation)

