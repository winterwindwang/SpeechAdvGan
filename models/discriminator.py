import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    def __init__(self, dropout_drop=0.5):
        super(Discriminator, self).__init__()
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31, stride=2, padding=15)
        self.vbn1 = nn.BatchNorm1d(32)
        self.lrelu1 = nn.LeakyReLU(negative_slope)

        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)
        self.vbn2 = nn.BatchNorm1d(64)
        self.lrelu2 = nn.LeakyReLU(negative_slope)

        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)
        self.dropout1 = nn.Dropout(dropout_drop)
        self.vbn3 = nn.BatchNorm1d(64)
        self.lrelu3 = nn.LeakyReLU(negative_slope)

        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15)
        self.vbn4 = nn.BatchNorm1d(128)
        self.lrelu4 = nn.LeakyReLU(negative_slope)

        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)
        self.vbn5 = nn.BatchNorm1d(128)
        self.lrelu5 = nn.LeakyReLU(negative_slope)

        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)
        self.dropout2 = nn.Dropout(dropout_drop)
        self.vbn6 = nn.BatchNorm1d(256)
        self.lrelu6 = nn.LeakyReLU(negative_slope)

        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)
        self.vbn7 = nn.BatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU(negative_slope)

        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)
        self.vbn8 = nn.BatchNorm1d(512)
        self.lrelu8 = nn.LeakyReLU(negative_slope)

        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)
        self.dropout3 = nn.Dropout(dropout_drop)
        self.vbn9 = nn.BatchNorm1d(512)
        self.lrelu9 = nn.LeakyReLU(negative_slope)

        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)
        self.vbn10 = nn.BatchNorm1d(1024)
        self.lrelu10 = nn.LeakyReLU(negative_slope)

        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)
        self.vbn11 = nn.BatchNorm1d(2048)
        self.lrelu11 = nn.LeakyReLU(negative_slope)

        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)
        self.fully_connected_classfier = nn.Linear(in_features=8, out_features=2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.vbn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.vbn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.vbn3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.vbn4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        x = self.vbn5(x)
        x = self.lrelu5(x)
        x = self.conv6(x)
        x = self.dropout2(x)
        x = self.vbn6(x)
        x = self.lrelu6(x)
        x = self.conv7(x)
        x = self.vbn7(x)
        x = self.lrelu7(x)
        x = self.conv8(x)
        x = self.vbn8(x)
        x = self.lrelu8(x)
        x = self.conv9(x)
        x = self.dropout3(x)
        x = self.vbn9(x)
        x = self.lrelu9(x)
        x = self.conv10(x)
        x = self.vbn10(x)
        x = self.lrelu10(x)
        x = self.conv11(x)
        x = self.vbn11(x)
        x = self.lrelu11(x) # remove this layer can reduce the number of parameter from 97472202 to 32453322
        x = self.conv_final(x)
        x = self.lrelu_final(x)
        x = torch.squeeze(x)
        res = self.fully_connected(x)
        logit = self.fully_connected_classfier(x)
        return res, logit
