import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import cv2
import torch.nn.functional as F
from torchvision import  models, transforms
import torchvision

batch_size=100
LR=1e-5
epochs=500
h,w=224,224
mean = [0.15854794283031065, 0.15854794283031065, 0.15854794283031065]
std = [0.22935350279045194, 0.22935350279045194, 0.22935350279045194]

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
def check_list(words):
    # Check if the list contains any item other than 'normal'
    other_items = [word for word in words if word != 'normal']

    if other_items:
        # Return all items if there are other items than 'normal'
        return other_items
    else:
        # Return ['normal'] if there are no other items
        return ['normal']
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#see the results
defects=['lop','lof','por','u','c']
models=[]
for i,defect in enumerate(defects):
    PATH=defect+'.pth'
    checkpoint=torch.load(PATH)
    models.append(torchvision.models.resnet18(pretrained=True)) # Using pre-trained for demo purpose

    for parameter in models[i].parameters():
            parameter.requires_grad = False 

    #feat_out=resnet50.fc.out_features
    num_feat=models[i].fc.in_features
    #model.fc=nn.Sequential(
    #     nn.Dropout(dropout_rate),
    #     nn.Linear(model.fc.in_features, 2)

    models[i].fc=nn.Linear(num_feat,2)

    models[i].load_state_dict(checkpoint['model_state_dict'])

    models[i].to(device)
    models[i].eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(h),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


video='static/videos/weld_vision.UBCDefect_Pipe1i_LOP.mp4'
cap=cv2.VideoCapture(video)
preds=[]
while(True):

    ret,frame1=cap.read()
    if ret:

        frame = preprocess(frame1)
        # Add batch dimension (model expects batches of images)
        frame = frame.unsqueeze(0)
        
        # Transfer to the device
        frame = frame.to(device)

        # Model inference
        outputs=[]
        for i in range(len(models)):
            output = models[i](frame)
            _, pred = torch.max(output, 1)
            outputs.append(int(pred.item()==0)*'normal'+int(pred.item()==1)*defects[i])
            preds.append(check_list(outputs))
                          
    else:
        break
cap.release()
print(preds)


