import os
import sys
sys.path.append(os.getcwd())

import os
import json
import os
import os.path as osp
from typing import Optional

import cv2
import Dnnlib
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# ---------------------------------------------
def img2tensor(img):
    """ np.ndarray(HWC) -> torch.Tensor(CHW), 0~255 -> -1~1"""

    img = img.astype(np.float32)
    tensor = torch.from_numpy(img).permute(2,0,1) # HWC -> CHW
    tensor = tensor / 127.5 - 1

    return tensor



# ---------------------------------------------
def tensor2img(tensor):
    """torch.Tensor(CHW) -> np.ndarray(HWC), -1~1 -> 0~255. """
    
    tensor = tensor.detach().cpu().numpy()
    img = (tensor + 1) * 127.5
    img = np.transpose(img, (1,2,0))
    img = np.clip(img, 0, 255).astype(int)

    return img
    

# ---------------------------------------------
class RGBDataset(Dataset):
    
    def __init__(self,
        data_name: str='.',
        data_root: str='.',
        # **Kwargs,
    ) -> None:
        super().__init__()

        self._name = data_name
        self._data_root = data_root

        with open(osp.join(data_root, 'dataset.json'), 'rb') as jsfile:
            self.dataJson = json.load(jsfile)
        jsfile.close()
        assert 'fname' in self.dataJson_keys, 'Lack Necessary Infomations about Image\'s Name.'

        self._length = len(self.dataJson)
        
    #
    def __len__(self):
        return self._length

    #
    def __getitem__(self, idx):
        return self._load_img(idx)

    #
    def _load_img(self, idx):
        assert idx < self._length
        
        fname = self.dataJson[idx]['fname']
        img = cv2.imread(osp.join(self._data_root, fname), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img2tensor(img)

        return img

    #
    @property
    def dataJson_keys(self):
        keys = self.dataJson[0].keys()
        return keys

    #
    @property
    def name(self):
        return self._name

    #
    @property
    def image_shape(self):
        img = self._load_img(0)
        return img.shape
    
    #
    @property
    def image_channels(self):
        return self.image_shape[0]
    
    #
    @property
    def image_resolution(self):
        return self.image_shape[1]





# --------------------------------------------
class DepthDataset(Dataset):
    
    def __init__(self,
        data_name: str='.',
        data_root: str='.',
    ) -> None:
        super().__init__()

        self._name = data_name
        self._images_root = osp.join(data_root, '1-0_Ir')
        self._syns_root = osp.join(data_root, '2-0_Is')
        self._img_names = os.listdir(self._images_root)
        self._nmap_names = os.listdir(self._syns_root)
        img_length = len(self._img_names)
        nmap_length = len(self._nmap_names)
        self._length = min(img_length, nmap_length)

        self.img_fetch_orders = np.arange(0, self._length)
        self.syn_fetch_orders = np.arange(0, self._length)
        rnd = np.random.RandomState(321)
        rnd.shuffle(self.img_fetch_orders)
        rnd.shuffle(self.syn_fetch_orders)
        
        self.mean, self.std = np.load('/root/Workspace/A2J_re/nyu/data/mean_std.npy')

    #
    def __len__(self):
        return self._length

    #
    def __getitem__(self, idx):
        return self._load_img_syn(idx)

    @staticmethod
    def augment(img, syn, thresh=145):
        def get_r_s(hand):
            hand_max, hand_min = hand.max(), hand.min()
            R = 250 / (hand_max - hand_min) - 0.5
            R = min(1.5, R)
            r = np.random.rand()*R + 0.5

            hand_r = hand * r
            shift_deep = 150 - hand_r.max()
            shift_shallow = -150 - hand_r.min()

            s = np.random.rand()*(shift_deep - shift_shallow) + shift_shallow

            return r, s

        hand = img[img < thresh].copy()
        r, s = get_r_s(hand)

        img[img < thresh] = img[img < thresh] * r + s
        syn[syn < thresh] = syn[syn < thresh] * r + s
        img = np.clip(img, -150, 150)
        syn = np.clip(syn, -150, 150)

        return img, syn

    #
    def _load_img_syn(self, idx):
        assert idx < self._length

        img_idx = self.img_fetch_orders[idx]
        syn_idx = self.syn_fetch_orders[idx]

        # syn_idx = img_idx # note:  matched inputs

        img = np.load(osp.join(self._images_root, self._img_names[img_idx]))
        syn = np.load(osp.join(self._syns_root, self._nmap_names[syn_idx]))

        # if True:
        #     img, syn = self.augment(img, syn)


        img = torch.from_numpy(img).to(dtype=torch.float32)[None,...]
        syn = torch.from_numpy(syn).to(dtype=torch.float32)[None,...]

        img = (img - self.mean) / self.std / 5.3185
        syn = (syn - self.mean) / self.std / 5.3185


        img = F.interpolate(img[None], size=(128,128), mode='nearest')[0] # note: resize to 128
        syn = F.interpolate(syn[None], size=(128,128), mode='nearest')[0]

        return img, syn     # note:  add style
        # return syn, img     # note:  remove noise


    #
    def get_paired_ImgNmap(self, idx):
        assert idx < self._length

        idx = self.img_fetch_orders[idx]

        img = np.load(osp.join(self._images_root, self._img_names[idx]))
        syn = np.load(osp.join(self._syns_root, self._nmap_names[idx])) 
        
        img = torch.from_numpy(img).to(dtype=torch.float32)[None,...]
        syn = torch.from_numpy(syn).to(dtype=torch.float32)[None,...]

        img = (img - self.mean) / self.std / 5.3185
        syn = (syn - self.mean) / self.std / 5.3185

        img = F.interpolate(img[None], size=(128,128), mode='nearest')[0] # note: resize to 128
        syn = F.interpolate(syn[None], size=(128,128), mode='nearest')[0]

        return img, syn     # note:  add style
        # return syn, img     # note:  remove noise

    
    #
    @property
    def name(self):
        return self._name

    #
    @property
    def image_shape(self):
        img = self._load_img_syn(0)[0]
        return img.shape
    
    #
    @property
    def image_channels(self):
        return self.image_shape[0]
    
    #
    @property
    def image_resolution(self):
        return self.image_shape[1]


class OtherDepthDataset(DepthDataset):
    def __init__(self,
        data_name: str='.',
        data_root: str='.',
        dir_name:  str='.'
    ) -> None:
        self._name = data_name
        self._images_root = dir_name
        self._syns_root = osp.join(data_root, '2-0_Is')
        self._img_names = os.listdir(self._images_root)
        self._nmap_names = os.listdir(self._syns_root)
        img_length = len(self._img_names)
        nmap_length = len(self._nmap_names)
        self._length = min(img_length, nmap_length)

        self.img_fetch_orders = np.arange(0, self._length)
        self.syn_fetch_orders = np.arange(0, self._length)
        rnd = np.random.RandomState(321)
        rnd.shuffle(self.img_fetch_orders)
        rnd.shuffle(self.syn_fetch_orders)
        
        self.mean, self.std = np.load('/root/Workspace/A2J_re/nyu/data/mean_std.npy')





# 72757 102858
# --------------------------------------------
class DepthDataset_v2(Dataset):
    
    def __init__(self,
        data_name: str='.',
        data_root: str='.',
    ) -> None:
        super().__init__()

        self._name = data_name
        self.data_root = data_root

        self.nyu_syn_root = '/root/PersonalData/Depth_data/NYU_data'
        self.nyu_img_root = '/root/PersonalData/Depth_data/NYU_data'

        self.a2j_syn_root = '/root/PersonalData/Depth_data/A2J_data/image'
        self.a2j_img_root = '/root/PersonalData/Depth_data/patch'

        img_length = 102858
        nmap_length = 102858
        self._length = min(img_length, nmap_length)

        self.img_fetch_orders = np.arange(0, self._length)
        self.syn_fetch_orders = np.arange(0, self._length)
        rnd = np.random.RandomState(321)
        rnd.shuffle(self.img_fetch_orders)
        rnd.shuffle(self.syn_fetch_orders)

    #
    def __len__(self):
        return self._length

    #
    def __getitem__(self, idx):
        return self._load_img_syn(idx)

    #
    def _load_img_syn(self, idx):
        assert idx <= self._length

        img_idx = self.img_fetch_orders[idx]
        syn_idx = self.syn_fetch_orders[idx]

        if img_idx <= 72756:
            img = np.load(osp.join(self.nyu_img_root, f'{img_idx}.npy'))[0] # H*W*1, -1~1
        else:
            img = np.load(osp.join(self.a2j_img_root, f'{img_idx}.npy')) / 5.3185 # 176 * 176, -1~1
            img = cv2.resize(img, (128, 128))[..., None]

        if syn_idx <= 72756:
            syn = np.load(osp.join(self.nyu_syn_root, f'{syn_idx}.npy'))[1] # H*W*1, -1~1
        else:
            syn = np.load(osp.join(self.a2j_syn_root, f'{syn_idx}.npy')) / 5.3185 # 128 * 128
            syn = syn[..., None]


        img = torch.from_numpy(img).to(dtype=torch.float32)
        syn = torch.from_numpy(syn).to(dtype=torch.float32)

        img = img.permute(2,0,1)
        syn = syn.permute(2,0,1)

        return img, syn

    #
    def get_paired_ImgNmap(self, idx):
        assert idx <= self._length

        idx = self.img_fetch_orders[idx]

        if idx <= 72756:
            img = np.load(osp.join(self.nyu_img_root, f'{idx}.npy'))[0] # H*W*1, -1~1
            syn = np.load(osp.join(self.nyu_syn_root, f'{idx}.npy'))[1] # H*W*1, -1~1
        else:
            syn = np.load(osp.join(self.a2j_syn_root, f'{idx}.npy')) / 5.3185
            syn = syn[..., None]
            img = np.zeros_like(syn)
        
        img = torch.from_numpy(img).to(dtype=torch.float32)
        syn = torch.from_numpy(syn).to(dtype=torch.float32)

        img = img.permute(2,0,1)
        syn = syn.permute(2,0,1)

        return img, syn

    
    #
    @property
    def name(self):
        return self._name

    #
    @property
    def image_shape(self):
        img = self._load_img_syn(0)[0]
        return img.shape
    
    #
    @property
    def image_channels(self):
        return self.image_shape[0]
    
    #
    @property
    def image_resolution(self):
        return self.image_shape[1]







if __name__ == '__main__':

    import os
    import sys
    sys.path.append(os.getcwd())
    import matplotlib.pyplot as plt

    a = DepthDataset(data_name='Depth', 
                     data_root='/root/PersonalData/Depth_data/NYU_data_2/BGCT')
    print(len(a))
    print(a.image_channels)
    print(a.image_resolution)
    print(a.image_shape)

    x,y = a[1111]
    plt.imshow(x[0], cmap='gray')
    plt.show()
    plt.imshow(y[0], cmap='gray')
    plt.show()
    print(x.shape, y.shape)


