import torch.nn as nn

class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet,self).__init__()
        self.conv1d_l1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1d_l2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv1d_l3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.maxpool   = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout   = nn.Dropout(p=0.5) 
        self.dense     = nn.Linear(in_features=589056,out_features=32)

    def forward(self,x):
        x = self.conv1d_l1(x)
        x = self.bn1(x)
        x = self.conv1d_l2(x)
        x = self.bn2(x)
        x = self.conv1d_l3(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        x = x.reshape(x.size(0),-1)
        x = self.dense(x)
        return x
