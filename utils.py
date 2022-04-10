import os
import random
import torch
import numpy as np
from torch.utils import data
import cv2
import torch.nn as nn
import torchvision
import pandas as pd
import monai
from sklearn.model_selection import train_test_split

def set_random_seed():
    seed = 73
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_model(model_name, weight=None):
    if model_name == 'inceptionv3':
        model = torchvision.models.inception_v3(pretrained=True,aux_logits=False)
        model.dropout = nn.Dropout(0.4)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.features.norm5 = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Sequential( nn.Dropout(p=0.4), nn.Linear(model.classifier.in_features, 2))
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'efficientnet_b2':
        model = torchvision.models.efficientnet_b2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'regnet':
        model = torchvision.models.regnet_y_1_6gf(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'resnext':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    else:
        pass

class Ensemble(nn.Module):
    def __init__(self, *models):
        super(Ensemble, self).__init__()
        self.models = []
        for model in models:
            self.models.append(model)
        self.linear = nn.Linear(2*len(models),2)
        
       
    def forward(self, x):
        out = []
        for model in self.models:
            out.append(model(x))
            
        out = self.linear(torch.cat(out,1))
        
        return out

def get_loss(loss_name):
    if loss_name == 'CE':
        criterion = nn.CrossEntropyLoss()
        return criterion
    elif loss_name == 'BCE':
        criterion = nn.BCELoss()
        return criterion
    elif loss_name == 'FL':
        criterion = monai.losses.FocalLoss(to_onehot_y=True)
        return criterion
    else:
        pass

def get_optimizer(optimizer_name, param,lr, momentum,weight_decay):
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(param, lr=lr, weight_decay=weight_decay)
        return optimizer
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(param, lr=lr, weight_decay=weight_decay)
        return optimizer
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(param, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    else:
        return None

def get_scheduler(scheduler_name, optimizer, max_epoch):
    if scheduler_name == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=28, gamma=0.1)
        return scheduler
    if scheduler_name == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-6, 5e-2, step_size_up=50, step_size_down=None)
        return scheduler
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
        return scheduler
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(max_epoch/2))
        return scheduler


def split_data():
    csv = pd.read_csv('/home/u/woody8657/tmp/C426_G1_01_RADNCLREPORT.csv')
    img_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/'
    imgs = os.listdir(img_dir)
    patients = [img[0:13] for img in imgs]
    Pneumothorax = []
    for patient in patients:
        Pneumothorax.append(csv.iloc[csv[csv["PERSONID2"]==patient].index.values,1].values[0])
    label_data = [[patients[i], Pneumothorax[i]] for i in range(len(patients))]
    print(len(label_data))
    # split data
    set_random_seed() 
    train, val = train_test_split(label_data, test_size=0.2)
    val, test = train_test_split(val, test_size=0.5)
    
    return train, val, test
    


class Dataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.img_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/'
        self.data = data
        self.transform = transform
        
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.img_dir,self.data[index][0])+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 
            

        label = self.data[index][1]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.data)