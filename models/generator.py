import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=31, stride=1, padding=15)
        self.in1 = nn.InstanceNorm1d(channels, affine=True)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=31, stride=1, padding=15)
        self.in2 = nn.InstanceNorm1d(channels, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out = out + residual

        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)

        padding = kernel_size // 2

        self.conv2d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):

        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv2d(x)

        return x

class Generator_speech(nn.Module):
    def __init__(self):
        super(Generator_speech, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=31, stride=1, padding=15)
        self.in1 = nn.InstanceNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15)
        self.in2 = nn.InstanceNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15)
        self.in3 = nn.InstanceNorm1d(64)

        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.resblock4 = ResidualBlock(64)


        self.up1 = UpsampleConvLayer(64, 32, kernel_size=31, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm1d(32)
        self.up2 = UpsampleConvLayer(32, 16, kernel_size=31, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm1d(16)


        self.conv4 = nn.Conv1d(16, 1, kernel_size=31, stride=1, padding=15)
        self.in6 = nn.InstanceNorm1d(16)


    def forward(self, x):

        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = F.relu(self.in4(self.up1(x)))
        x = F.relu(self.in5(self.up2(x)))

        x = self.in6(self.conv4(x)) # remove relu for better performance and when input is [-1 1]

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2,
                              padding=15)  # 8192
        self.enc1_nl = nn.PReLU()
        # (in_channels,out_channels, kernel_size,stride, padding)
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # 4096
        self.enc2_nl = nn.PReLU()

        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # 2048
        self.enc3_nl = nn.PReLU()

        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # 1024
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # 512
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # 256
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # 128
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # 64
        self.enc8_nl = nn.PReLU()
        # (in_channels, out_channels, kernel_size, stride, padding)
        self.dec7 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # 128
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # 256
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # 512
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # 1024
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # 2048
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # 4096
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # 8192
        self.dec1_nl = nn.PReLU()
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # 16384
        self.dec_tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))

        c = self.enc8_nl(e8)

        d7 = self.dec7(c)
        d7_c = self.dec7_nl(torch.cat((d7, e7), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e6), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out

class Generator_pert(nn.Module):
    def __init__(self):
        super(Generator_pert, self).__init__()

        encoder_lis = [
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2,
                      padding=15),  # 8192
            nn.InstanceNorm1d(16),
            nn.PReLU(),

            nn.Conv1d(16, 32, 32, 2, 15),  # 4096
            nn.InstanceNorm1d(32),
            nn.PReLU(),

            nn.Conv1d(32, 64, 32, 2, 15),  # 1024
            nn.InstanceNorm1d(64),
            nn.PReLU(),

            nn.Conv1d(64, 128, 32, 2, 15),  # 256
            nn.InstanceNorm1d(128),
            nn.PReLU(),

            nn.Conv1d(128, 256, 32, 2, 15),  # 64
            nn.PReLU(),
            nn.InstanceNorm1d(256)
        ]



        decoder_lis = [
            nn.ConvTranspose1d(256, 128, 32, 2, 15),  # 128
            nn.InstanceNorm1d(128),
            nn.PReLU(),

            nn.ConvTranspose1d(128, 64, 32, 2, 15),  # 512
            nn.InstanceNorm1d(64),
            nn.PReLU(),

            nn.ConvTranspose1d(64, 32, 32, 2, 15),  # 2048
            nn.InstanceNorm1d(32),
            nn.PReLU(),

            nn.ConvTranspose1d(32, 16, 32, 2, 15), # 4096
            nn.InstanceNorm1d(16),
            nn.PReLU(),

            nn.ConvTranspose1d(16, 1, 32, 2, 15),  # 16384
            nn.Tanh()
        ]
        self.encoder = nn.Sequential(*encoder_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        # (in_channels, out_channels, kernel_size, stride, padding)
        # self.dec7 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # 128
        # self.dec7_nl = nn.PReLU()
        # self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # 512
        # self.dec5_nl = nn.PReLU()
        # self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # 2048
        # self.dec3_nl = nn.PReLU()
        # self.dec2 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # 4096
        # self.dec2_nl = nn.PReLU()
        # self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # 16384
        # self.dec_tanh = nn.Tanh()
        # self.init_weights()


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        # e1 = self.in1(self.enc1(x))
        # e2 = self.in2(self.enc2(self.enc1_nl(e1)))
        # e4 = self.in3(self.enc4(self.enc2_nl(e2)))
        # e6 = self.in4(self.enc6(self.enc4_nl(e4)))
        # e8 = self.in5(self.enc8(self.enc6_nl(e6)))
        #
        # c = self.enc8_nl(e8)
        # UNet
        # d7 = self.dec7(c)
        # d7_c = self.dec7_nl(torch.cat((d7, e6), dim=1))
        # d5 = self.dec5(d7_c)
        # d5_c = self.dec5_nl(torch.cat((d5, e4), dim=1))
        # d3 = self.dec3(d5_c)
        # d3_c = self.dec3_nl(torch.cat((d3, e2), dim=1))
        # d2 = self.dec2(d3_c)
        # d2_c = self.dec2_nl(torch.cat((d2, e1), dim=1))

        #
        out1 = self.encoder(x)
        out = self.decoder(out1)
        return out