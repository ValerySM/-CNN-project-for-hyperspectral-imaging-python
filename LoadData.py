import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from CustomImageDataset import CustomImageDataset
import glob


def load_data(path, batch_size,loader_name, shuffle=True, isTrain=True):
    # Pre-processing
    HSimagelist = glob.glob(path)
    dataset = CustomImageDataset(HSimagelist,isTrain=isTrain)
    k = dataset.get_max()

    # Data Loader (Input Pipeline)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    torch.save(loader, f'{loader_name}.pth')
    return k

def create_loader():
    k_test = load_data("./test/*", batch_size=145, loader_name="test_loader", shuffle=False, isTrain=False)
    k_train = load_data("./Datasets/*", batch_size=b, loader_name="train_loader")
    return k_test if k_test > k_train else k_train