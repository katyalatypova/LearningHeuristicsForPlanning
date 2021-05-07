import torch
import torch.nn as nn
import numpy as np

north = np.array([[0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 0]])
south = np.array([[0, 0, 0],
                  [0, -1, 0],
                  [0, 1, 0]])
east = np.array([[0, 0, 0],
                 [0, -1, 1],
                 [0, 0, 0]])
west = np.array([[0, 0, 0],
                 [1, -1, 1],
                 [0, 0, 0]])
ne = np.array([[0, 0, 1],
               [0, -1, 1],
               [0, 0, 0]])
nw = np.array([[1, 0, 0],
               [0, -1, 1],
               [0, 0, 0]])
se = np.array([[0, 0, 0],
               [0, -1, 0],
               [0, 0, 1]])
sw = np.array([[0, 0, 0],
               [0, -1, 0],
               [1, 0, 0]])

FILTERS = {
    'n': torch.FloatTensor(north).unsqueeze(0).unsqueeze(0),
    's': torch.FloatTensor(south).unsqueeze(0).unsqueeze(0),
    'e': torch.FloatTensor(east).unsqueeze(0).unsqueeze(0),
    'w': torch.FloatTensor(west).unsqueeze(0).unsqueeze(0),
    'ne': torch.FloatTensor(ne).unsqueeze(0).unsqueeze(0),
    'nw': torch.FloatTensor(nw).unsqueeze(0).unsqueeze(0),
    'se': torch.FloatTensor(se).unsqueeze(0).unsqueeze(0),
    'sw': torch.FloatTensor(sw).unsqueeze(0).unsqueeze(0)
}


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=ksize // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, ksize, padding=ksize // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.block1 = UNetBlock(2, 64, 3)
        self.block1_pair = UNetBlock(128, 64, 3)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.block2 = UNetBlock(64, 128, 3)
        self.block2_pair = UNetBlock(256, 128, 3)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.block3 = UNetBlock(128, 256, 3)
        self.block3_pair = UNetBlock(512, 256, 3)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.block4 = UNetBlock(256, 512, 3)
        self.block4_pair = UNetBlock(1024, 512, 3)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.block5 = UNetBlock(512, 1024, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1x1 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        assert x.shape[-1] % 16 == 0 and x.shape[-2] % 16 == 0, 'Wrong image size'
        out1 = self.block1(x)
        out2 = self.block2(self.pool(out1))
        out3 = self.block3(self.pool(out2))
        out4 = self.block4(self.pool(out3))
        out5 = self.block5(self.pool(out4))

        up5 = torch.cat((self.upconv4(out5), out4), 1)
        up5 = self.block4_pair(up5)

        up4 = torch.cat((self.upconv3(up5), out3), 1)
        up4 = self.block3_pair(up4)

        up3 = torch.cat((self.upconv2(up4), out2), 1)
        up3 = self.block2_pair(up3)

        up2 = torch.cat((self.upconv1(up3), out1), 1)
        up2 = self.block1_pair(up2)
        return self.conv1x1(up2)


class SmallUNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.block1 = UNetBlock(2, 64, 3)
        self.block1_pair = UNetBlock(128, 64, 3)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.block2 = UNetBlock(64, 128, 3)
        self.block2_pair = UNetBlock(256, 128, 3)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.block3 = UNetBlock(128, 256, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1x1 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        assert x.shape[-1] % 16 == 0 and x.shape[-2] % 16 == 0, 'Wrong image size'
        out1 = self.block1(x)
        out2 = self.block2(self.pool(out1))
        out3 = self.block3(self.pool(out2))

        up3 = torch.cat((self.upconv2(out3), out2), 1)
        up3 = self.block2_pair(up3)

        up2 = torch.cat((self.upconv1(up3), out1), 1)
        up2 = self.block1_pair(up2)
        return self.conv1x1(up2)


def gradient_loss(output, target_map, device, valid_actions=('n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se')):
    global FILTERS
    loss = 0
    for action in valid_actions:
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        conv.weight = torch.nn.Parameter(FILTERS[action], requires_grad=False)
        conv.to(device)
        conv_with_output = conv(output)
        conw_with_target_map = conv(target_map)
        loss += torch.abs(conv_with_output - conw_with_target_map).sum()
    return loss / len(valid_actions)


def piece_loss(output, target_map, minimal_cost, alpha1=2, alpha2=10):
    abs_loss = torch.abs(output - target_map)
    return torch.mean(
        abs_loss * alpha1 * (output < minimal_cost) + alpha2 * abs_loss * (target_map < output) + abs_loss * (
                target_map >= output) * (output >= minimal_cost))


def loss(output, target_map, minimal_cost, device, alpha, alpha1, alpha2,
         valid_actions=('n', 's', 'e', 'w', 'nw', 'ne', 'sw', 'se')):
    # global device, alpha, alpha1, alpha2
    return alpha * gradient_loss(output, target_map, device, valid_actions) + piece_loss(output, target_map,
                                                                                         minimal_cost, alpha1, alpha2)
