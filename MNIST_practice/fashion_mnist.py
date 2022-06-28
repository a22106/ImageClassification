import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.models import resnet
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
    
learning_rate = 3e-4
training_epochs = 15
batch_size = 10


# fashion mnist data
fashion_train = dsets.FashionMNIST(root="data/FashionMNIST/train",
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)
fashion_test = dsets.FashionMNIST(root="data/FashionMNIST/test",
                                         train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)

fashion_train_loader = DataLoader(fashion_train,
                                  batch_size=10,
                                  shuffle=True,
                                  drop_last= True)
fashion_test_loader = DataLoader(fashion_test,
                                 batch_size=10,
                                 shuffle=True,
                                 drop_last= True)

fashion_train_imgs, fashion_train_labels = next(iter(fashion_train_loader))
print(fashion_train_imgs.shape)
print(fashion_train_labels)

plt.imshow( fashion_train_imgs[0])