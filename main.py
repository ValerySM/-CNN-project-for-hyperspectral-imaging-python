import os
import pandas as pd
from LoadData import create_loader
from train_module import train_model
from predict import predict
###
### Best HP: batch size: 50, epochs:10, lr=0.001, layer1dp=0, layer1dp=0.3, layer1dp=0.3


if __name__ == "__main__":
    b_size = 50
    l_rate = 1e-17
    epochs = 5

    k_max = create_loader()
    if not os.path.exists("./linear_model.pkl"):
        train_model(k_max, epochs, l_rate, "train_loader.pth")

    else:
        predict()
