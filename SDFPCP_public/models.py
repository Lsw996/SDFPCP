import torch.nn as nn
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()

        self.conv_1 = nn.Conv1d(1, 128, kernel_size=7, dilation=1, padding=3)
        self.conv_2 = nn.Conv1d(128, 64, kernel_size=3, dilation=2, padding=2)
        self.conv_3 = nn.Conv1d(64, 32, kernel_size=3, dilation=4, padding=4)
        self.conv_4 = nn.Conv1d(32, 16, kernel_size=3, dilation=8, padding=8)
        self.dense_1 = nn.Linear(16544, 128)
        self.dense_2 = nn.Linear(128, 2)

    def forward(self, x):

        x = self.conv_1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_3(x)
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_4(x)
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)

        return x
