import torch.nn as nn

#Feature extractor from the input vectors 
#Use triplet training with semi-hard triplet mining for superior results

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
        x_in = self.conv1d_l1(x)
        x_in = self.bn1(x_in)
        x_in = self.conv1d_l2(x_in)
        x_in = self.bn2(x_in)
        x_in = self.conv1d_l3(x_in)
        x_in = self.bn3(x_in)
        x_in = self.maxpool(x_in)
        x_in = self.dropout(x_in)
        x_in = x_in.permute(0,2,1)
        x_in = x_in.reshape(x_in.size(0),-1)
        x_in = self.dense(x_in)
        return x_in
