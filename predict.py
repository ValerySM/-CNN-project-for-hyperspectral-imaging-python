import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

def predict(model_path='linear_model.pkl', test_modul_path='test_loader.pth'):
    cnn = torch.load(model_path)
    test_loader = torch.load(test_modul_path)
    cnn.eval()
    pred_list = []
    mask_list = []
    for i, (pixels, labels) in enumerate(test_loader):
        # Forward + Backward + Optimize
        prediction = cnn(pixels)
        _, prediction = torch.max(prediction.data, 1)
        pred_list.append(prediction)
        mask_list.append(labels)

    picture_mask = torch.vstack(mask_list)
    picture_pred = torch.vstack(pred_list)
    plt.imshow(picture_pred.detach().numpy())
    plt.savefig(f'./prediction_{i}.png')
    plt.imshow(picture_mask.detach().numpy())
    plt.savefig(f'./mask_{i}.png')

