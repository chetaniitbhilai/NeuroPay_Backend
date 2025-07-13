import torch
from .DecisionNetwork import DecisionNetwork
from .SiameseNet import SiameseNet

class CompleteModel():
    def __init__(self,siamesePath='./Gesture_models/best_model.pth',decisionPath='./Gesture_models/decision_model.pth'):
        self.siameseNet = SiameseNet()
        # remove map loaction for gpu 
        self.siameseNet.load_state_dict(torch.load(siamesePath, map_location='cpu'))
        self.siameseNet.eval()

        self.decisionNet = DecisionNetwork()
        self.decisionNet.load_state_dict(torch.load(decisionPath, map_location='cpu'))
        self.decisionNet.eval()  # <--- Add this

    
    def computeEmbedding(self,x1,x2):
        embedding1 = self.siameseNet(x1)
        embedding2 = self.siameseNet(x2)
        distanceVector = torch.abs(embedding1-embedding2)
        return distanceVector
    
    def makeDecision(self,distanceVector):
        prediction = self.decisionNet(distanceVector)
        return True if prediction.item() > 0.6 else False
    
    def padOrTrim(self,x,target_len=9216):
        if (x.numel() < target_len):
            padding = torch.zeros(target_len - x.numel(),dtype = x.dtype,device = x.device)
            return torch.cat((x,padding),dim=0)
        elif x.numel() > target_len:
            return x[:target_len]
        return x
    
    def solve(self,x1_list,x2_list):
        # convert to tensors
        x1 = torch.tensor(x1_list).float().flatten()
        x2 = torch.tensor(x2_list).float().flatten()

        x1 = self.padOrTrim(x1.flatten()).view(1,-1).unsqueeze(0)
        x2 = self.padOrTrim(x2.flatten()).view(1,-1).unsqueeze(0)

        device = next(self.siameseNet.parameters()).device
        x1,x2 = x1.to(device).float(),x2.to(device).float()

        distanceVec = self.computeEmbedding(x1,x2)
        result = self.makeDecision(distanceVec)
        return result