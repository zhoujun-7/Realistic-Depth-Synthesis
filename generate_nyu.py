
import os
import os.path as osp
import pickle
from typing import List

import natsort
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from tqdm import tqdm

import Dnnlib
from Dataset_utils.Sty2Dataset import tensor2img

_device = torch.device('cuda:0')



def depthToPointcloud(depth:np.ndarray):
    (_h, _w) = depth.shape
    scale = (_h + _w) * 6
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * scale
    points = np.zeros([_h*_w, 3], dtype=np.float32)
    idx = 0
    for y in range(_h):
        for x in range(_w):
            points[idx, :] = np.array([x, y, depth[y,x]])
            idx += 1
    return points


def cv2_link_Depthjoints(img, joints:np.ndarray): # N*2 / N*3
    img = img.copy()
    line_order = np.array([[13, 11], [13, 12], [13, 10], [13, 7], [13, 5], [13, 3], [13, 1], [10, 9], 
                [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=int)
    line_color = (60,179,113)
    circle_color = (0, 0, 255)
    for i in range(13):
        ptStart = (int(joints[line_order[i,0], 1]), int(joints[line_order[i,0], 0]))
        ptEnd   = (int(joints[line_order[i,1], 1]), int(joints[line_order[i,1], 0]))
        thinkness = 2
        img = cv2.line(img, ptStart, ptEnd, line_color, thinkness)
        img = cv2.circle(img, ptStart, radius=2, color=circle_color, thickness=thinkness)
        img = cv2.circle(img, ptEnd, radius=2, color=circle_color, thickness=thinkness)

    return img


def show_depth_2D(depth:np.ndarray, ax:Axes):
    # depth: H*W
    ax.imshow(depth, cmap='gray')


def show_depth_3D(depth:np.ndarray, ax:Axes3D):
    # depth: H*W
    points = depthToPointcloud(depth)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5)
    ax.view_init(elev=-90, azim=-90)


def construct_Generator(model_pkl:str='.'):
    with open(model_pkl, 'rb') as f:
        params = pickle.load(f)
    f.close()
    G = Dnnlib.construct_class_by_name(**params['net_kwargs']['G']) 
    G.load_state_dict(params['G_ema'])
    G.eval().to(device=_device)
    return G


def Gout_toDepth(Gout:torch.Tensor):
    Gout = Gout.clamp(-1., 1.)    
    depth = Gout[0,0,...].detach().cpu().numpy()
    return depth


MEAN, STD = np.load('./data/mean_std.npy')
def imreadFilenames_toGin(filenames:List[str]):
    xs = []
    for name in filenames:
        x = np.load(name)
        x = torch.from_numpy(x.copy()).to(dtype=torch.float32)[None,...] 
        x = (x - MEAN) / STD / 5.3185
        xs.append(x)
    Gin = torch.stack(xs, dim=0)
    Gin = F.interpolate(Gin, size=[128,128], mode='nearest')
    return  Gin.to(device=_device)


def saveGout_withFilenames(filenames:List[str], Gout:torch.Tensor):
    assert len(filenames) == Gout.shape[0], 'Length of filenames not match the batch of Tensor'
    Gout = Gout.detach().cpu()
    Gout = F.interpolate(Gout, size=[176,176], mode='nearest')
    for idx,name in enumerate(filenames):
        x = Gout[idx].clamp(-1., 1.).numpy()[0]
        x = (x * 5.3185 * STD ) + MEAN
        x = x.astype(np.float16)
        np.save(name, x)
    return True



def idx_toGin(idx:int, data_root:str='.'):
    data_path = os.path.join(data_root, f'{idx}.npy')

    # [img, syn] = np.load(data_path)
    # img, syn = img[:,:,0], syn[:,:,0]
    # syn = np.load(data_path)
    # syn = syn[:,:,0]
    # Gin = torch.from_numpy(syn.copy()).to(dtype=torch.float32)
    # Gin = Gin[None, None, ...].to(device=_device)
    # return img, syn, Gin


    syn = np.load(data_path)
    Gin = torch.from_numpy(syn.copy()).to(dtype=torch.float32)
    Gin = Gin[None, None, ...].to(device=_device)
    return syn, Gin

    # syn = np.load(data_path) / 5.3185
    # Gin = torch.from_numpy(syn.copy()).to(dtype=torch.float32)
    # Gin = Gin[None, None, ...].to(device=_device)
    # return syn, Gin




""" ----------------------------------- """
def z_transformProcess(idx:int, model_pkl:str, data_root:str):

    G = construct_Generator(model_pkl=model_pkl)
    img, syn, Gin = idx_toGin(idx=idx, data_root=data_root)
    # Gin = torch.ones_like(Gin).to(device=_device)

    fig = plt.figure(figsize=(12,4))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    show_depth_2D(img, ax=ax0)
    show_depth_2D(syn, ax=ax1)

    # z1 = torch.randn([1, 128]).to(device=_device)
    # z2 = torch.randn([1, 128]).to(device=_device)
    # w1 = G.mapping(z1)
    # w2 = G.mapping(z2)
    # interp = (w2-w1)/50
    
    for i in range(50):
        # wx = w1 + interp * i

        z = torch.randn([1, 128]).to(device=_device)
        wx = G.mapping(z)
        out = G.synthesis(Gin, wx)
        print(out.mean(), out.std())
        out = Gout_toDepth(out)
        show_depth_2D(out, ax=ax2)
        plt.pause(0.1)
        ax2.cla()



""" ----------------------------------- """
def z_transformProcess_gif(idx:int, model_pkl:str, data_root:str):

    G = construct_Generator(model_pkl=model_pkl)
    img, syn, Gin = idx_toGin(idx=idx, data_root=data_root)
    # Gin = torch.ones_like(Gin).to(device=_device)
    syn = (syn + 1) * 127.5
    
    for i in range(50):

        z = torch.randn([1, 128]).to(device=_device)
        wx = G.mapping(z)
        out = G.synthesis(Gin, wx)
        out = Gout_toDepth(out)
        out = (out + 1) * 127.5
        out = np.hstack([syn, out])

        cv2.imwrite(f'_CacheDir/_videoImgs/{i:03d}.png', out)

    os.popen('cd /root/Workspace/HandGeneration_v2/_CacheDir/_videoImgs && \
        ffmpeg -y -f image2 -r 10 -pattern_type glob -i \'*.png\' out.mp4 && \
        ffmpeg -y -i out.mp4 -vf fps=10,scale=320:-1:flags=lanczos,palettegen palette.png && \
        ffmpeg -y -i out.mp4 -i palette.png -filter_complex \"fps=10, scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse\" out.gif && \
        rm *.png && \
        cd')

    # Note: 可在 windows 下播放的 mp4 视频设置及 ffmpeg 命令基本形式：
    # ffmpeg -y -f image2 -r 30 -pattern_type glob -i '*.png' -pix_fmt yuv420p -c:v libx264 out.mp4  
    # ffmpeg [input options] -i input [output options] output


""" ---------------------------------------- """
def show_3d_results(idx:int, model_pkl:str, data_root:str):

    G = construct_Generator(model_pkl=model_pkl)
    img, syn, Gin = idx_toGin(idx=idx, data_root=data_root)

    fig = plt.figure(figsize=(12,4))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(133, projection='3d')

    show_depth_2D(syn, ax=ax0)
    show_depth_3D(img, ax=ax1)

    z = torch.randn([1, 128]).to(device=_device)
    w = G.mapping(z)
    out = G.synthesis(Gin, w)
    out = Gout_toDepth(out)

    show_depth_3D(out, ax=ax2)
    plt.show()



""" -------------------------------------------------------------- """
def generating_depthHands(model_pkl:str, data_root:str, save_dir:str):

    G = construct_Generator(model_pkl=model_pkl)

    length = len(os.listdir(data_root))
    for idx in tqdm(range(length)):
        img, syn, Gin = idx_toGin(idx, data_root=data_root)
        
        z = torch.randn([1, 128]).to(device=_device)
        w = G.mapping(z)
        out = G.synthesis(Gin, w)
        out = Gout_toDepth(out)

        np.save(os.path.join(save_dir, f'{idx}.npy'), out)





""" -------------------------------------------------------------- """
def generate_test(idx:int, model_pkl:str, data_root:str):
    
    G = construct_Generator(model_pkl=model_pkl)

    fig = plt.figure(figsize=(15,5))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133, projection='3d')

    syn, Gin = idx_toGin(idx, data_root=data_root)
    show_depth_2D(syn, ax=ax0)

    z = torch.randn([1, 128]).to(device=_device)
    w = G.mapping(z)
    out = G.synthesis(Gin, w)
    out = Gout_toDepth(out)

    show_depth_2D(out, ax1)
    show_depth_3D(out, ax2)
    plt.show()



""" ----------------------------------------------------------- """
def generate_z_dynamic(idx:int, model_pkl:str, data_root:str):
    
    G = construct_Generator(model_pkl=model_pkl)

    fig = plt.figure(figsize=(10,5))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    syn, Gin = idx_toGin(idx, data_root=data_root)
    show_depth_2D(syn, ax=ax0)

    for i in range(50):
        z = torch.randn([1, 128]).to(device=_device)
        w = G.mapping(z)
        out = G.synthesis(Gin, w)
        out = Gout_toDepth(out)

        show_depth_2D(out, ax1)
        plt.pause(0.1)
        ax1.cla()



""" -------------------------------------------------------------- """
def generate_samples(model_pkl:str, data_root:str, save_dir:str):

    batch = 32
    G = construct_Generator(model_pkl=model_pkl)
    os.makedirs(save_dir, exist_ok=True)

    _inputs_names = os.listdir(data_root)
    _length = len(os.listdir(data_root))

    for idx in tqdm(range(0, _length, batch)):
        if idx + batch > _length:
            batch = _length - idx
        
        names = []; save_names = []
        for idx_imread in range(idx, idx+batch, 1):
            names.append(osp.join(data_root, _inputs_names[idx_imread]))
            save_names.append(osp.join(save_dir, _inputs_names[idx_imread]))
            Gin = imreadFilenames_toGin(names)

        z = torch.randn([batch, 128]).to(device=_device)
        w = G.mapping(z)
        out = G.synthesis(Gin, w)

        saveGout_withFilenames(save_names, out)

 

if __name__ == '__main__':

    args = Dnnlib.EasyDict()
    # args.idx = 500
    args.model_pkl = './ckpt/nyu_style.pkl'
    args.data_root = './data/2-0_Is'
    args.save_dir = './out'

    generate_samples(**args)
    exit()


    filenames = os.listdir(args.save_dir)
    filenames = natsort.natsorted(filenames)
    dir_length = len(filenames)
    # joints2d = np.load('/root/PersonalData/video_cache/jointsLabel.npy')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in tqdm(range(dir_length)):
        name = filenames[idx]

        ori = np.load(osp.join(args.data_root, name))
        ori = ((ori + 1) * 127.5).astype(np.uint8)
        ori = np.stack([ori, ori, ori], axis=2)

        out = np.load(osp.join(args.save_dir, name))
        out = ((out + 1) * 127.5).astype(np.uint8)
        out = np.stack([out, out, out], axis=2)
        
        # out2 = cv2_link_Depthjoints(out, joints2d[idx])
        img = np.hstack([ori, out])
        # cv2.imwrite(os.path.join('/root/PersonalData/video_cache', f'{idx:04d}.png'), img)
        ax.imshow(img, cmap='gray')
        plt.pause(0.5)
        ax.cla()


