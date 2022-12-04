import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(220, 128),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, k),
            nn.ReLU()
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
