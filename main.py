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
ImageFile.LOAD_TRUNCATED_IMAGES = True
SAVE_PATH = './test_net_2803.pth'
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

def load_model(model, path=SAVE_PATH):
    model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    KAGGLE = "Kaggle"
    transforms = v2.Compose([
        v2.ToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mushroom_jpegs = torchvision.datasets.ImageFolder(root=KAGGLE, transform=transforms)

    BATCH_SIZE = 100
    n = len(mushroom_jpegs)  
    n_test = int(0.2 * n)  
    test_set = data.Subset(mushroom_jpegs, range(n_test))
    train_set = data.Subset(mushroom_jpegs, range(n_test, n))

    test_set_loader  = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    train_set_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)

    classes = next(os.walk(KAGGLE))[1] # Lists folder names - ergo lists mushroom family
    
    # print(classes)
    dataiter = iter(train_set_loader)
    images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))

    # device = torch.device("cuda:0") #
    # net = CNN().to(device)
    # criterion = nn.CrossEntropyLoss()
    # print("Setting optimizer...")

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # load_model(net)
    # printloss = 25
    # for epoch in range(100):  # loop over the dataset multiple times
    #     print("Loop begins, epoch: {}".format(epoch))
    #     running_loss = 0.0
    #     for i, data in enumerate(train_set_loader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         # print statistics
    #         running_loss += loss.item()

    #         if i % printloss == printloss-1:    # print every #printloss mini-batches
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / printloss:.3f}')
    #             running_loss = 0.0

    # print('Finished Training')

    
    #torch.save(net.state_dict(), SAVE_PATH)

    dataiter = iter(train_set_loader)
    images, labels = next(dataiter)


    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))



    net = CNN()
    net.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(BATCH_SIZE)))

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
