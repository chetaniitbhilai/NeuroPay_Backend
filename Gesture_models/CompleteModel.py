from pathlib import Path
import torch
from .DecisionNetwork import DecisionNetwork
from .SiameseNet import SiameseNet

class CompleteModel:
    def __init__(self, siamesePath='./Gesture_models/best_model.pth',decisionPath='./Gesture_models/latest_model_epoch_46.pth'):
        self.siameseNetwork = SiameseNet()
        self.decisionNetwork = DecisionNetwork()
        self.siameseModelPath = Path("./Gesture_models/encoder_checkpoint_epoch_40.pth")
        self.temp = torch.load(self.siameseModelPath, map_location='cpu')
        self.decisionModelPath = decisionPath
        self.siameseNetwork.load_state_dict(self.temp['model_state_dict'])  
        self.decisionNetwork.load_state_dict(torch.load(self.decisionModelPath, map_location='cpu'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.siameseNetwork.to(self.device)
        self.decisionNetwork.to(self.device)
        self.siameseNetwork.eval()
        self.decisionNetwork.eval()

    def computeEmbedding(self, x1, x2):
        embedding1 = self.siameseNetwork(x1)
        embedding2 = self.siameseNetwork(x2)
        distanceVector = torch.abs(embedding1 - embedding2)
        return distanceVector

    def makeDecision(self, distanceVector):
        prediction = self.decisionNetwork(distanceVector)
        prediction = torch.sigmoid(prediction)
        print(f"🔍 Prediction: {prediction.item()}")
        return False if prediction.item() > 0.60 else True

    def padOrTrim(self, x, target_len=9216):
        if x.numel() < target_len:
            padding = torch.zeros(target_len - x.numel(), dtype=x.dtype, device=self.device)
            return torch.cat((x, padding), dim=0)
        elif x.numel() > target_len:
            return x[:target_len]
        return x

    def solve(self, x1_list, x2_list):

        x1 = torch.tensor(x1_list).float().flatten()
        x2 = torch.tensor(x2_list).float().flatten()

        x1, x2 = x1.to(self.device).float(), x2.to(self.device).float()
        x1 = self.padOrTrim(x1.flatten()).view(1, -1).unsqueeze(0)
        x2 = self.padOrTrim(x2.flatten()).view(1, -1).unsqueeze(0)
        distanceVec = self.computeEmbedding(x1, x2)
        result = self.makeDecision(distanceVec)
        return result


# class CompleteModel():
#     def __init__(self,siamesePath='./Gesture_models/best_model.pth',decisionPath='./Gesture_models/decision_model.pth'):
#         self.siameseNet = SiameseNet()
#         # remove map loaction for gpu 
#         self.siameseNet.load_state_dict(torch.load(siamesePath, map_location='cpu'))
#         self.siameseNet.eval()

#         self.decisionNet = DecisionNetwork()
#         self.decisionNet.load_state_dict(torch.load(decisionPath, map_location='cpu'))
#         self.decisionNet.eval()  # <--- Add this

    
#     def computeEmbedding(self,x1,x2):
#         embedding1 = self.siameseNet(x1)
#         embedding2 = self.siameseNet(x2)
#         distanceVector = torch.abs(embedding1-embedding2)
#         return distanceVector
    
#     def makeDecision(self,distanceVector):
#         prediction = self.decisionNet(distanceVector)
#         return True if prediction.item() > 0.6 else False
    
#     def padOrTrim(self,x,target_len=9216):
#         if (x.numel() < target_len):
#             padding = torch.zeros(target_len - x.numel(),dtype = x.dtype,device = x.device)
#             return torch.cat((x,padding),dim=0)
#         elif x.numel() > target_len:
#             return x[:target_len]
#         return x
    
#     def solve(self,x1_list,x2_list):
#         # convert to tensors
#         x1 = torch.tensor(x1_list).float().flatten()
#         x2 = torch.tensor(x2_list).float().flatten()

#         x1 = self.padOrTrim(x1.flatten()).view(1,-1).unsqueeze(0)
#         x2 = self.padOrTrim(x2.flatten()).view(1,-1).unsqueeze(0)

#         device = next(self.siameseNet.parameters()).device
#         x1,x2 = x1.to(device).float(),x2.to(device).float()

#         distanceVec = self.computeEmbedding(x1,x2)
#         result = self.makeDecision(distanceVec)
#         return result