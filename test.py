import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import torchvision
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,classification_report, confusion_matrix, plot_confusion_matrix
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from utils import *

class myEnsemble(nn.Module):
    def __init__(self, *models):
        super(myEnsemble, self).__init__()
        self.models = models
       
        # for model in models:
        #     self.models.append(model)
       
    def forward(self, x):
        out = []
        for model in self.models:
            out.append(model(x))
        
        
        out = sum(out) / len(out)
        
        return out

def main(opt):
    # data
    csv = pd.read_csv('/home/u/woody8657/tmp/C426_G1_01_RADNCLREPORT.csv')
    img_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/'
    imgs = os.listdir(img_dir)
    patients = [img[0:13] for img in imgs]
    Pneumothorax = []
    for patient in patients:
        Pneumothorax.append(csv.iloc[csv[csv["PERSONID2"]==patient].index.values,1].values[0])
    label_data = [[patients[i], Pneumothorax[i]] for i in range(len(patients))]
    print(len(label_data))
    # spli data
    set_random_seed() 
    train, val = train_test_split(label_data, test_size=0.2)
    val, test = train_test_split(val, test_size=0.5)
    
    positive_train = sum([train[i][1] for i in range(len(train))])
    positive_val = sum([val[i][1] for i in range(len(val))])
    positive_test = sum([test[i][1] for i in range(len(test))])
    print(f"{positive_train} / {len(train)} of train set has Pneumothorax")
    print(f"{positive_val} / {len(val)} of valid set has Pneumothorax")
    print(f"{positive_test} / {len(test)} of test set has Pneumothorax")
  
    # Data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.img_size,opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    
    # Dataset
    test_dataset = Dataset(test,transform=test_transform)
    
    # Dataloader
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = get_model('inceptionv3')
    model1.load_state_dict(torch.load('/home/u/woody8657/runs/2cls_new/training_2021-11-04_18:49:34/best.ckpt'))
    # tmp = torch.load('/home/u/woody8657/projs/Pneumothorax-detection/2cls/ray_logs/RS/wrap_2021-11-14_16-33-54/wrap_9a357_00010_10_batch_size=48,epochs=100,img_size=512,loss=CE,lr=0.00028543,model=inceptionv3,optimizer=AdamW,scheduler=Cosine_2021-11-15_13-24-29/final.ckpt')["state_dict"]
    # tmp = dict((key.replace('network.', ''), value) for (key, value) in tmp.items())
    # model1.load_state_dict(tmp)
    model1.to(device)
    model1.eval()
    model2 = get_model('regnet')
    model2.load_state_dict(torch.load('/home/u/woody8657/runs/2cls_new/training_2021-11-04_14:06:28/best.ckpt'))
    # tmp = torch.load('/home/u/woody8657/projs/Pneumothorax-detection/2cls/ray_logs/RS/wrap_2021-11-18_02-12-31/wrap_ee954_00000_0_batch_size=12,epochs=100,img_size=512,loss=CE,lr=0.00015628,model=regnet,optimizer=AdamW,scheduler=CosineAnneal_2021-11-18_02-12-32/lightning_logs/version_0/checkpoints/epoch=31-step=5503.ckpt')["state_dict"]
    # tmp = dict((key.replace('network.', ''), value) for (key, value) in tmp.items())
    # model2.load_state_dict(tmp)
    model2.to(device)
    model2.eval()
    model3 = get_model('resnext')
    model3.load_state_dict(torch.load('/home/u/woody8657/runs/2cls_new/training_2021-11-04_16:20:46/best.ckpt'))
    # tmp = torch.load('/home/u/woody8657/projs/Pneumothorax-detection/2cls/ray_logs/RS/wrap_2021-11-18_02-12-31/wrap_ee954_00003_3_batch_size=12,epochs=100,img_size=512,loss=CE,lr=0.00012366,model=resnext,optimizer=AdamW,scheduler=None,weight_2021-11-18_03-54-47/lightning_logs/version_0/checkpoints/epoch=99-step=17199.ckpt')["state_dict"]
    # tmp = dict((key.replace('network.', ''), value) for (key, value) in tmp.items())
    # model3.load_state_dict(tmp)
    model3.to(device)
    model3.eval()
    model4 = get_model('densenet121')
    model4.load_state_dict(torch.load('/home/u/woody8657/runs/2cls_new/training_2021-11-04_16:59:54/best.ckpt'))
    model4.to(device)
    model4.eval()
    model = myEnsemble(model1, model3,model2)
    model = Ensemble(get_model('inceptionv3', '/home/u/woody8657/runs/2cls_new/training_2021-11-04_18:49:34/best.ckpt').to(device), 
                    get_model('resnext', '/home/u/woody8657/runs/2cls_new/training_2021-11-04_16:20:46/best.ckpt').to(device),
                    get_model('regnet', '/home/u/woody8657/runs/2cls_new/training_2021-11-04_14:06:28/best.ckpt').to(device))
    tmp = torch.load('/home/u/woody8657/projs/Pneumothorax-detection/2cls/lightning_logs_ensemble/version_7/checkpoints/epoch=99-step=18699.ckpt')['state_dict']
    print(tmp.keys())
    tmp = dict((key.replace('network.', ''), value) for (key, value) in tmp.items())
    raise
    model.load_state_dict(tmp)
    model.to(device)

    
    # pl_weight = torch.load('/home/u/woody8657/projs/Pneumothorax-detection/2cls/lightning_logs_ensemble/version_17/checkpoints/epoch=39-step=2599.ckpt')['state_dict']
    # weight = {}
    # for key in pl_weight.keys():
    #     weight[key.replace('network.', '')] = pl_weight[key]
    # model.load_state_dict(weight)

    criterion = nn.CrossEntropyLoss()
    # These are used to record information in validation.
    test_loss = []
    test_accs = []
    f = nn.Softmax(dim=1)
    prob_all = []
    label_all = []
    pred = []
    for images, labels  in tqdm.tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            prob = f(outputs)
        prob_all.extend(prob[:,1].cpu().numpy())
        label_all.extend(labels.cpu().numpy())
        pred.extend(outputs.argmax(dim=-1).cpu().numpy())
        # We can still compute the loss (but not the gradient).-
        loss = criterion(outputs, labels)
        # Compute the accuracy for current batch.
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        # Record the loss and accuracy.
        test_loss.append(loss.item())
        test_accs.append(acc)
        
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    test_loss = sum(test_loss) / len(test_loss)
    print(f"AUC = {roc_auc_score(label_all, prob_all)}")
    tn, fp, fn, tp = confusion_matrix(label_all, pred).ravel()
    p = tp / (tp + fp)
    print(f"tp {tp}")
    print(f"fp {fp}")
    print(f"fn {fn}")
    print(f"tn {tn}")
    r = tp / (tp + fn)
    F1 = 2 * r * p / (r + p)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print(f'Test_Acc: {acc}')
    print(f'Test_Percision: {p}')
    print(f'Test_Recall: {r}')
    print(f'Test_F1_score: {F1}')

    y_pred, y_true = pred, label_all


    cf_matrix = confusion_matrix(y_true, y_pred)                                
  
    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)                   
    class_names = ['None', 'Pneumothorax']
    print(class_names)
    print(per_cls_acc)                                                          
    print("Plot confusion matrix")


    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)   
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=512, help='image sizes')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for per GPUs')
    parser.add_argument('--weight', default='/home/u/woody8657/projs/Pneumothorax-detection/2cls/model.ckpt', help='save to tensorboard')
    opt = parser.parse_args()
    
    main(opt)

