from torch.autograd import Variable
from CNN import CNN
import torch
import torch.nn as nn
from CustomImageDataset import CustomImageDataset


# Hyper Parameters
def train_model(k, num_epochs, learning_rate, train_loader_path):
    # convert all the weights tensors to cuda()
    train_loader = torch.load(train_loader_path)
    cnn = CNN(k)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    cnn.train()
    for epoch in range(num_epochs):
        for i, (pixels, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                pixels = pixels.cuda()
                labels = labels.cuda()
            # Forward + Backward + Optimize
            prediction = cnn(pixels)
            # _, prediction = torch.max(prediction.data, 1)
            prediction = prediction.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.long)
            prediction = Variable(prediction.data, requires_grad=True)
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1,
                         len(train_loader) // 50, loss.data))

        torch.save(cnn, 'linear_model.pkl')
