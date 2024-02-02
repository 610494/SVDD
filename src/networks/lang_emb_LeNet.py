import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class LangEmbLeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 256
        self.pool = nn.MaxPool1d(2)

        ## Fully connected layer 0
        self.fc0 = nn.Linear(20, 512, bias=False)
        ## Convolutional layer 1
        self.conv1 = nn.Conv1d(1, 16, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm1d(16, eps=1e-04, affine=False) # Batch normalization
        ## Convolutional layer 2
        self.conv2 = nn.Conv1d(16, 32, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(32, eps=1e-04, affine=False) # Batch normalization
        ## Convolutional layer 3
        self.conv3 = nn.Conv1d(32, 64, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm1d(64, eps=1e-04, affine=False) # Batch normalization
        ## Fully connected layer 1
        self.fc1 = nn.Linear(64 * 64, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.fc0(x)
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        print(f"************\n[networks/lang_emb_LeNet.py] LangEmbLeNet.forward \n************")
        return x

class LangEmbLeNetAutoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 256
        self.pool = nn.MaxPool1d(2)

        # Encoder (must match the Deep SVDD network above)
        self.fc0 = nn.Linear(20, 512, bias=False)
        self.conv1 = nn.Conv1d(1, 16, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(16, 32, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.conv3 = nn.Conv1d(32, 64, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64 * 64, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose1d(int(self.rep_dim / 64), 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose1d(64, 32, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose1d(32, 16, 5, bias=False, padding=2)
        self.bn6 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose1d(16, 1, 5, bias=False, padding=2)
        self.fc2 = nn.Linear(512, 20, bias=False)

    def forward(self, x):
        x = self.fc0(x)
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / 64), 64)
        x = F.elu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.elu(self.bn4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.elu(self.bn5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.elu(self.bn6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        print(f"************\n[networks/lang_emb_LeNet.py] LangEmbLeNetAutoencoder.forward \n************")

        return x