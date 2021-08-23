import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d


class small_network(nn.Module):
  def __init__(self, iteration):
      super(small_network, self).__init__()
      self.d_1024_7 = BinarizeLinear((2048 * 4)+(128*(iteration-1)), 1024, bias=False)
      self.bn_7 = nn.BatchNorm1d(1024)        
      self.d_1024_8 = BinarizeLinear(1024, 128, bias=False)
      self.bn_8 = nn.BatchNorm1d(128)
      self.d_10_9 = BinarizeLinear(128, 10, bias=False)
      self.bn_9 = nn.BatchNorm1d(10)
      self.relu = nn.ReLU()
      self.hardtanh = nn.Hardtanh(inplace=True)
      self.logsoft = nn.LogSoftmax()

  def forward(self, input):
      #x = nn.Flatten()(input)
      #x = input.view(-1, 512 * 4 * 4)
      #print(x)
      x = self.d_1024_7(input)
      x = self.bn_7(x)
      x = self.hardtanh(x)

      x = self.d_1024_8(x)
      x = self.bn_8(x)
      x = self.hardtanh(x)
      hook = x

      x = self.d_10_9(x)
      x = self.bn_9(x)
      x = self.logsoft(x)
      
      return x, hook



def Small_Network(iteration):
    return small_network(iteration).cuda()
    