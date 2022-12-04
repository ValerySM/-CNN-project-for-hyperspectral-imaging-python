import scipy.io
import torch
from torch.utils.data import Dataset
import scipy.io
import torchvision.transforms as transforms
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, paths_list, isTrain=True, transform=None):
        self.paths_list = paths_list
        self.pixels = {}
        self.transform = transform
        # self.tran_mask = transforms.CenterCrop(145)
        self.max_k = 0
        idx = 0
        # Create dictionary of index : (name of image, transformed pytorch tensor)
        for img_path in self.paths_list:
            HSimage_temp = scipy.io.loadmat(img_path)
            HSname = list(HSimage_temp.keys())[-1]
            if "_gt" in str(HSname):
                continue

            HSimage_temp = torch.from_numpy(np.asarray(HSimage_temp[HSname], dtype=np.float32)).permute(2, 0, 1)
            HSimage = HSimage_temp[:220,:,:]
            HS_min = torch.min(HSimage)
            HS_max = torch.max(HSimage)
            HSimage = (HSimage - HS_min)/(HS_max - HS_min)
            if isTrain:
                HSmask_temp = scipy.io.loadmat(f'./Datasets/{HSname}_gt')
            else:
                HSmask_temp = scipy.io.loadmat(f'./test/{HSname}_gt')
            HSmask_name = list(HSmask_temp.keys())[-1]
            HSmask_temp = torch.from_numpy(np.asarray(HSmask_temp[HSmask_name], dtype=np.float32))
            HSmask = HSmask_temp
            if self.max_k < len(torch.unique(HSmask)):
                self.max_k = len(torch.unique(HSmask))
            for x in range(HSimage.shape[1]):
                for y in range(HSimage.shape[2]):
                    self.pixels[idx] = (HSimage[:, x, y], HSmask[x, y])
                    idx += 1

    def __len__(self):
        return len(self.pixels)

    def get_max(self):
        return self.max_k
    def __getitem__(self, idx):
        pixel = self.pixels[idx][0]
        label = self.pixels[idx][1]
        return pixel, label

