import torch
import torchvision
from torchvision.transforms import v2
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
import time


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def imshow(img):
    # Use imshow(torchvision.utils.make_grid(images)) to show an image grid
    # Use print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE))) to print their respective classes
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_model(model, PATH=None):
    if PATH is None:
        import glob
        PATH = os.path.join('Models', '*.pth')
        list_of_files = glob.glob(PATH)
        latest_file = max(list_of_files, key=os.path.getctime)
    try:
        model.load_state_dict(torch.load(latest_file))    
    except:
        print("==============================================")
        print("NO EXISTING MODEL FOUND")
        print("==============================================")
    
    print("==============================================")
    print("Existing model found - continuing training!")
    print("==============================================")

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    SAVE_PATH = './Models/model{}.pth'.format(int(time.time()))
    KAGGLE = "Kaggle"
    EPOCHS = 2
    PRINTLOSS = 25
    BATCH_SIZE = 100
    device = torch.device("cuda:0")

    transforms = v2.Compose([
        v2.ToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mushroom_jpegs = torchvision.datasets.ImageFolder(root=KAGGLE, transform=transforms)
    
    jpeg_amount = len(mushroom_jpegs)  
    test_jpeg_amount = int(0.2 * jpeg_amount)  
    test_set = data.Subset(mushroom_jpegs, range(test_jpeg_amount))
    train_set = data.Subset(mushroom_jpegs, range(test_jpeg_amount, jpeg_amount))

    test_set_loader  = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    train_set_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

    classes = next(os.walk(KAGGLE))[1] # List folder names - ergo list mushroom family

    dataiter = iter(train_set_loader)
    images, labels = next(dataiter)

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    load_model(net)
    for epoch in range(EPOCHS):

        print("Loop begins, epoch: {}".format(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_set_loader, 0):
            inputs, labels = data # data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % PRINTLOSS == PRINTLOSS-1:    # Print every #printloss mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINTLOSS:.3f}')
                running_loss = 0.0

    print('Finished Training - Saving to {}'.format(SAVE_PATH)) 
    torch.save(net.state_dict(), SAVE_PATH)
