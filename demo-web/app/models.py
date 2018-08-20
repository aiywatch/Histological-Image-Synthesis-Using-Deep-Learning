import torch
import torch.nn as nn
import torch.nn.functional as F

import os, time, pickle, argparse#, util
import torch.optim as optim
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread

import matplotlib.pyplot as plt

import scipy
from skimage.color import gray2rgb
import numpy as np
from skimage.transform import resize

class generator(nn.Module):
  # initializers
  def __init__(self, d=64):
    super(generator, self).__init__()
    # Unet encoder
    self.conv1 = nn.Conv2d(2, d, 4, 2, 1)
    self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
    self.conv2_bn = nn.BatchNorm2d(d * 2)
    self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
    self.conv3_bn = nn.BatchNorm2d(d * 4)
    self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
    self.conv4_bn = nn.BatchNorm2d(d * 8)
    self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
    self.conv5_bn = nn.BatchNorm2d(d * 8)
    self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
    self.conv6_bn = nn.BatchNorm2d(d * 8)
    self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
    self.conv7_bn = nn.BatchNorm2d(d * 8)
    self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
#     self.conv8_bn = nn.BatchNorm2d(d * 8)
    

    # Unet decoder
    self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
    self.deconv1_bn = nn.BatchNorm2d(d * 8)
    self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
    self.deconv2_bn = nn.BatchNorm2d(d * 8)
    self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
    self.deconv3_bn = nn.BatchNorm2d(d * 8)
    self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
    self.deconv4_bn = nn.BatchNorm2d(d * 8)
    self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
    self.deconv5_bn = nn.BatchNorm2d(d * 4)
    self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
    self.deconv6_bn = nn.BatchNorm2d(d * 2)
    self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
    self.deconv7_bn = nn.BatchNorm2d(d)
    self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, input):
    e1 = self.conv1(input)
    e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
    e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
    e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
    e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
    e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
    e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
    e8 = self.conv8(F.leaky_relu(e7, 0.2))
    # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
    

  
    d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
    d1 = torch.cat([d1, e7], 1)
    d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
    d2 = torch.cat([d2, e6], 1)
    d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
    d3 = torch.cat([d3, e5], 1)
    d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
    # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
    d4 = torch.cat([d4, e4], 1)
    d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
    d5 = torch.cat([d5, e3], 1)
    d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
    d6 = torch.cat([d6, e2], 1)
    d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
    d7 = torch.cat([d7, e1], 1)
    d8 = self.deconv8(F.relu(d7))
    o = F.tanh(d8)

    return o

  
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
  
class discriminator(nn.Module):
  # initializers
  def __init__(self, d=64):
    super(discriminator, self).__init__()
    self.conv1 = nn.Conv2d(5, d, 4, 2, 1)
    self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
    self.conv2_bn = nn.BatchNorm2d(d * 2)
    self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
    self.conv3_bn = nn.BatchNorm2d(d * 4)
    self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
    self.conv4_bn = nn.BatchNorm2d(d * 8)
    self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, input, label):
    x = torch.cat([input, label], 1)
    x = F.leaky_relu(self.conv1(x), 0.2)
    x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
    x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
    x = F.sigmoid(self.conv5(x))
    
    
    return x

def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()


class GenNucleiDataset(Dataset):

  def __init__(self, root_dir, transform=None, mode='train'):
    self.root_dir = root_dir
    self.transform = transform
    self.file_names = next(os.walk(root_dir))[2]

  def __len__(self):
    return len(self.file_names)

  def __getitem__(self, idx):
    mask_path = os.path.join(self.root_dir, self.file_names[idx])
    
    mask = imread(mask_path)
    mask = resize(mask, (256, 256))
    mask = torch.Tensor(mask).view(-1, 256, 256)
    
    return mask




