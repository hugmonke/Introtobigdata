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
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    
    KAGGLE = "Kaggle"
    BATCH_SIZE = 10

    transforms = v2.Compose([
        v2.ToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mushroom_jpegs = torchvision.datasets.ImageFolder(root=KAGGLE, transform=transforms)
    n = len(mushroom_jpegs)  
    n_test = int(0.2 * n)  
    test_set = data.Subset(mushroom_jpegs, range(n_test))
    test_set_loader  = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

    classes = next(os.walk(KAGGLE))[1] # Lists folder names - ergo lists mushroom family
    
    # print(classes)

    dataiter = iter(test_set_loader)
    images, labels = next(dataiter)
    
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
    net = CNN()
    try:
        LOAD_PATH = './Models/model{}.pth'.format(int(time.time()))
    except:
        try:
            import glob
            LOAD_PATH = os.path.join('Models', '*.pth')
            list_of_files = glob.glob(LOAD_PATH)
            latest_file = max(list_of_files, key=os.path.getctime)
            net.load_state_dict(torch.load(latest_file, weights_only=True))
        except:
            import sys
            sys.exit("NO MODEL FOUND")

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(BATCH_SIZE)))

    one = [f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)]
    two = [f'{classes[predicted[j]]:5s}'for j in range(BATCH_SIZE)]
    match = 0
    for i in range(len(one)):
        if one[i] == two[i]:
            match += 1
    print("Match:", match, "out of ", len(one), " RATIO: ", match/len(one))



    # #path = os.path.join(KAGGLE, fungi_family[0])
    # #directory = os.fsencode(path)
    # #print(directory)
    # # for file in os.listdir(directory):
    # #     filename = os.fsdecode(file)
    # #     if filename.endswith('.jpg'):
    # #         image = Image.open(os.path.join(path, filename)).convert("RGB")
    # #         image.show()
    # #         out = transforms(image)
    # #         out.show()