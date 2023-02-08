import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(output_channel),
        )
        self.upsample = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(output_channel),
        )

    def forward(self, x):
        # print("<<<<<",x.shape)
        out = self.block(x)
        out += self.upsample(x)
        out = F.relu(out)
        return out
