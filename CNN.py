import torch
from torch import nn
import torch.nn.functional as F
from ResBlock import ResBlock


class CNN(nn.Module):
    def __init__(self, window_size):
        super(CNN, self).__init__()
        output_channel_list = [8, 16, 32, 64]
        self.res_block1 = ResBlock(input_channel=1, output_channel=output_channel_list[0])
        self.res_block2 = ResBlock(input_channel=output_channel_list[0], output_channel=output_channel_list[1])
        self.res_block3 = ResBlock(input_channel=output_channel_list[1], output_channel=output_channel_list[2])
        self.res_block4 = ResBlock(input_channel=output_channel_list[2], output_channel=output_channel_list[3])
        self.avgpooling = nn.AvgPool1d(kernel_size=3)
        self.maxpooling = nn.MaxPool1d(kernel_size=3)
        self.flatten = torch.nn.Flatten()
        # print("<<<<<<<<<<",window_size)
        self.linear = nn.Linear(in_features=window_size//3//3 * output_channel_list[-1], out_features=1)

    def forward(self, x):
        # print("<<<<<",x.shape)
        out = self.res_block1(x)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.avgpooling(out)
        out = self.maxpooling(out)
        # print("out.shape",out.shape)
        out = self.flatten(out)
        out = self.linear(out)
        # out = F.tanh(out)
        return out

    def loss_function(self, x, y) -> torch.Tensor:
        return F.mse_loss(x, y)
