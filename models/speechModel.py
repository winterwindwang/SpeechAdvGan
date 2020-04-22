import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

class SpeechModels(nn.Module):
    def __init__(self, n_labels=10):
        super(SpeechModels, self).__init__()
        n_featmaps1 = 64
        dropout_prob = 0.5
        conv1_size = (20, 8)
        conv1_pool = (2, 2)
        conv1_stride = (1, 1)
        width = 101
        height = 40
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        tf_variant = True
        self.tf_variant = tf_variant
        if tf_variant:
            truncated_normal(self.conv1.weight.data)
            self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(conv1_pool)
        x = Variable(torch.zeros(1, 1, height, width), volatile=True)
        x = self.pool1(self.conv1(x))
        # conv_net_size = x.view(1, -1).size(1)
        # last_size = conv_net_size
        conv2_size = (10, 4)
        conv2_pool = (1, 1)
        conv2_stride = tuple((1, 1))
        n_featmaps2 = 64
        self.conv2 = nn.Conv2d(n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride)
        if tf_variant:
            truncated_normal(self.conv2.weight.data)
            self.conv2.bias.data.zero_()
        self.pool2 = nn.MaxPool2d(conv2_pool)
        x = self.pool2(self.conv2(x))
        conv_net_size = x.view(1, -1).size(1)
        last_size = conv_net_size
        # if not tf_variant:
        #     self.lin = nn.Linear(conv_net_size, 32)
        self.output = nn.Linear(last_size, n_labels)
        if tf_variant:
            truncated_normal(self.output.weight.data)
            self.output.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))  # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))  # shape: (batch, o1, i2, o2)
        x = self.dropout(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # shape: (batch, o3)
        if hasattr(self, "lin"):
            x = self.lin(x)
        return self.output(x)

if __name__ == '__main__':
    x = np.random.randn(40,101).astype(dtype=np.float32)
    x = torch.from_numpy(x)  # the input is (batch_size, height, width)
    model = SpeechModels()
    print(model(x.unsqueeze(dim=0)))