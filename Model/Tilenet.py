'''Modified ResNet-18 in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import LayerNorm 


class BatchNorm2d(nn.Module):
    def __init__(self,size):
        super(BatchNorm2d,self).__init__()
        self.norm = LayerNorm(size)
    def forward(self,x):
        if len(x.size())==4:
           a,b,c,d = x.size()
           return self.norm(x.view(x.size(0),x.size(1),-1)).view(a,b,c,d)
        return self.norm(x)
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes,batchnorm,stride=1):
        super(ResidualBlock, self).__init__()
        self.inplanes = in_planes
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = BatchNorm2d(batchnorm[0])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = BatchNorm2d(batchnorm[1])
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                BatchNorm2d(batchnorm[2]))

    def forward(self, x):
        #import pdb; pdb.set_trace()
        out = F.relu(self.bn1(self.conv1(x)))
        #import pdb; pdb.set_trace()
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512):
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0],[64,64,64],stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1],[16,16,16],stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], [4,4,4],stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3],[4,4,4],stride=1)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],[4,4,4],stride=1)

    def _make_layer(self, planes, num_blocks,batchnorm,stride,no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes,batchnorm=batchnorm, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.2, l2=0.01):
        l_n = torch.sqrt(((z_p - z_n) ** 2).mean())
        l_d = - torch.sqrt(((z_p - z_d) ** 2).mean())
        loss = l_n + l_d + margin
        orig_loss = loss
        return_nan = loss <= 0.0
        if return_nan:
            print(loss)
        #print(loss.item(),"loss1",l_n,l_d,margin,"l_n,l_d,margin")
        #l_n = torch.mean(l_n)
        #l_d = torch.mean(l_d)
        #l_nd = torch.mean(l_n + l_d
        #loss = torch.mean(loss)
        #print(torch.mean(loss),'mean')
        if l2 != 0:
            l2_val = l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
            loss += l2_val
        return loss,return_nan, l2_val,orig_loss #, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.2, l2=0.01):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)


def make_tilenet(in_channels=13, z_dim=512):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim)
