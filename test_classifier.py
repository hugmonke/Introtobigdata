import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile

from cnnetwork import CNN


def imshow(img):
    # Use imshow(torchvision.utils.make_grid(images)) to show an image grid
    # Use print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE))) to print their respective classes
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    LOAD_PATH = 'Checkpoints/cpmodel1743508719.pth'
    KAGGLE_TEST = "KaggleTest"
    BATCH_SIZE = 100

    transforms = v2.Compose([
        v2.ToTensor(),
        v2.CenterCrop([1000,800]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_mushroom_jpegs = torchvision.datasets.ImageFolder(root=KAGGLE_TEST, transform=transforms)
    test_set_loader  = DataLoader(test_mushroom_jpegs, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

    classes = next(os.walk(KAGGLE_TEST))[1]
    print(classes)

    dataiter = iter(test_set_loader)
    images, labels = next(dataiter)
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))


    net = CNN()
    
    try:
        net.load_state_dict(torch.load(LOAD_PATH, weights_only=True))
    except:
        print("Given load path is incorrect. Trying latest file.")
        try:
            import glob
            LOAD_PATH = os.path.join('Checkpoints', '*.pth')
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

if __name__ == '__main__':
    main()

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
