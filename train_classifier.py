import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile

import cnnetwork as cnn


def imshow(img):
    # Use imshow(torchvision.utils.make_grid(images)) to show an image grid
    # Use print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE))) to print their respective classes
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_model(model, PATH=None):
    
    try:
        if PATH is None:
            import glob
            PATH = os.path.join('Models', '*.pth')
            list_of_files = glob.glob(PATH)
            latest_file = max(list_of_files, key=os.path.getctime)
            PATH = latest_file
        model.load_state_dict(torch.load(PATH)) 
        print("==============================================")
        print("Existing model found - continuing training!")
        print("==============================================")
    except:
        print("==============================================")
        print("NO EXISTING MODEL FOUND")
        print("==============================================")
    
    


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    SAVE_PATH = './Models/model{}.pth'.format(int(time.time()))
    CHECKPOINT_PATH = './Checkpoints/cpmodel{}.pth'.format(int(time.time()))
    KAGGLE = "Kaggle"
    EPOCHS = 80
    PRINTLOSS = 6
    BATCH_SIZE = 100
    device = torch.device("cuda:0")

    transforms = v2.Compose([
        v2.ToTensor(),
        v2.CenterCrop([1000,800]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mushroom_jpegs = torchvision.datasets.ImageFolder(root=KAGGLE, transform=transforms)
    train_set_loader = DataLoader(mushroom_jpegs, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

    # classes = next(os.walk(KAGGLE))[1] # List folder names - ergo list mushroom family

    dataiter = iter(train_set_loader)
    images, labels = next(dataiter)

    net = cnn.CNN().to(device)
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
        if epoch%2==1:
            torch.save(net.state_dict(), CHECKPOINT_PATH)
        torch.cuda.empty_cache()
    print('Finished Training - Saving to {}'.format(SAVE_PATH)) 
    torch.save(net.state_dict(), SAVE_PATH)


if __name__ == '__main__':
    main()
