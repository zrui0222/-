import torch
import torch.nn as nn


__all__ = ["csinet"]

Nc = 32  # The number of subcarriers
Nt = 32  # The number of transmit antennas
N_channel = 2  # Real, Imaginary
img_total = Nc * Nt * N_channel
batch_size = 100


class Encoder(nn.Module):
    # input: (batch_size, Nc, Nt) channel matrix
    # output: (batch_size, encoded_dim) codeword
    # CSI_NET
    def __init__(self, encoded_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.fc = nn.Linear(in_features=2 * Nc * Nt, out_features=encoded_dim)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, test=False):
        out = self.conv_block(x)
        out = torch.reshape(out, (batch_size, -1))
        out = self.fc(out)

        return out


class Refinenet(nn.Module):
    # input: (batch_size, 2, Nc, Nt)
    # output: (batch_size, 2, Nc, Nt)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )

    def forward(self, x):
        skip_connection = x
        out = self.conv1(x)
        # out.shape = (batch_size, 8, Nc, Nt)
        out = self.conv2(out)
        # out.shape = (batch_size, 16, Nc, Nt)
        out = self.conv3(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = out + skip_connection

        return out


class Decoder(nn.Module):
    # input: (batch_size, encoded_dim) codeword
    # output: (batch_size, Nc, Nt) reconstructed channel matrix
    # CSI_NET
    def __init__(self, encoded_dim, test=False):
        super().__init__()
        self.fc = nn.Linear(in_features=encoded_dim, out_features=2 * Nc * Nt)
        self.refine1 = Refinenet()
        self.refine2 = Refinenet()
        self.test = test

    def forward(self, x, test=False):
        out = self.fc(x)
        out = torch.reshape(out, (batch_size, 2, Nc, Nt))
        out = self.refine1(out)
        out = self.refine2(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(encoded_dim)  # 定义编码函数
        self.decoder = Decoder(encoded_dim)  # 定义解码函数

    def forward(self, x):
        feature = self.encoder(x)    # 调用编码
        out = self.decoder(feature)  # 调用解码
        return out


def csinet(encoded_dim=32):
    model = AutoEncoder(encoded_dim=32)
    return model
