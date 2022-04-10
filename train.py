import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import torchvision
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,classification_report
import tqdm
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from time import gmtime, strftime
from utils import *



def train(opt):
    # data
    if __name__ != '__main__':
        torch.cuda.empty_cache()
        # tune.utils.wait_for_gpu()
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
    positive_train = sum([train[i][1] for i in range(len(train))])
    positive_val = sum([val[i][1] for i in range(len(val))])
    positive_test = sum([test[i][1] for i in range(len(test))])
    print(f"{positive_train} / {len(train)} of train set has Pneumothorax")
    print(f"{positive_val} / {len(val)} of valid set has Pneumothorax")
    print(f"{positive_test} / {len(test)} of test set has Pneumothorax")
  
    # Data augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt['img_size'],opt['img_size'])),
        transforms.RandomRotation(np.pi/24),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt['img_size'],opt['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

  
    # Dataset
    train_dataset = Dataset(train, img_dir, transform=transform)
    val_dataset = Dataset(val, img_dir, transform=test_transform)
    
    
    # Dataloader
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
    
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(opt['model'])
    model.to(device)
    # model =nn.DataParallel(model)

    # loss
    criterion = get_loss(opt['loss'])
    
    # optimizer
    optimizer = get_optimizer(opt['optimizer'], model.parameters(), opt['lr'], 0, 0)
    scheduler = get_scheduler(opt['scheduler'], optimizer, opt['epochs'])


    train_losses = []
    valid_losses = []
    if  __name__ == "__main__":
        run_folder = '/home/u/woody8657/runs/2cls_new/training_'+strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        writer = SummaryWriter(run_folder)
    set_random_seed() 
    num_epochs = opt['epochs']
    the_last_loss = 100
    patience = 8
    trigger_times = 0
    best_auc = 0
    for epoch in range(num_epochs):
        # -----training-----
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        f = nn.Softmax(dim=1)
        prob_all = []
        label_all = []
        pred = []
        for images, labels  in tqdm.tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            #print(model.is_cuda)                                                                                                                                         
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            prob = f(outputs)
            prob_all.extend(prob[:,1].cpu().detach().numpy())
            label_all.extend(labels.cpu().detach().numpy())
            pred.extend(outputs.argmax(dim=-1).cpu().detach().numpy())
      
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            
            # Compute the accuracy for current batch.
            acc = (outputs.argmax(dim=-1) == labels).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

    
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        fpr,tpr,_= roc_curve(label_all,prob_all[:len(label_all)])
        area_under_curve=auc(fpr,tpr)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f},  auc = {area_under_curve:.5f}")
        if  __name__ == "__main__":
            writer.add_scalar('Loss/train', train_loss, epoch)   
            writer.add_scalar('aucuracy/train', train_acc, epoch) 
            writer.add_scalar('AUC/train', area_under_curve, epoch) 
            writer.add_scalar('others/lr', scheduler.get_last_lr()[0], epoch) 
        else:
            tune.report(Loss_train=train_loss)
            tune.report(aucuracy_train=train_acc)
            tune.report(AUC_train=area_under_curve)
            tune.report(lr=scheduler.get_last_lr()[0])
        scheduler.step()
        # -----validation-----
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        f = nn.Softmax(dim=1)
        prob_all = []
        label_all = []
        pred = []
        for images, labels  in tqdm.tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                prob = f(outputs)
            prob_all.extend(prob[:,1].unsqueeze(1).cpu().numpy())
            label_all.extend(labels.cpu().numpy())
            pred.extend(outputs.argmax(dim=-1).cpu().numpy())
            # We can still compute the loss (but not the gradient).-
    
            loss = criterion(outputs, labels)
            # Compute the accuracy for current batch.
            acc = (outputs.argmax(dim=-1) == labels).float().mean()
            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_losses.append(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        fpr,tpr,_= roc_curve(label_all,prob_all[:len(label_all)])
        area_under_curve=auc(fpr,tpr) 
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f},  auc = {area_under_curve:.5f}")
        if  __name__ == "__main__":
            writer.add_scalar('Loss/valid', valid_loss, epoch)   
            writer.add_scalar('aucuracy/valid', valid_acc, epoch) 
            writer.add_scalar('AUC/valid', area_under_curve, epoch) 
        else:
            tune.report(Loss_valid=valid_loss)
            tune.report(aucuracy_valid=valid_acc)
            tune.report(AUC_valid=area_under_curve)

        the_current_loss = valid_loss
        if the_current_loss > the_last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                if  __name__ == "__main__":
                    writer.add_hparams(opt,{'hparam/AUC': best_auc})
                    writer.close()
                raise

        else:
            print('trigger times: 0')
            trigger_times = 0

        the_last_loss = the_current_loss
           

        if area_under_curve > best_auc:
            best_auc = area_under_curve
            if  __name__ == "__main__":
                torch.save(model.state_dict(), os.path.join(run_folder, 'best.ckpt'))
            print('saving model with auc {:.3f}'.format(best_auc))
        if  __name__ == "__main__":
            torch.save(model.state_dict(), os.path.join(run_folder, 'last.ckpt'))
 
    if  __name__ == "__main__":
        writer.add_hparams(opt,{'hparam/AUC': best_auc})
        writer.close()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img-size', type=int, default=512, help='image sizes')
    # parser.add_argument('--batch-size', type=int, default=32, help='batch size for per GPUs')
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--tensorboard-name', default='runs/2cls/test_arg', help='save to tensorboard')
    # opt = parser.parse_args()
    # opt = (vars(opt))
    opt = {
        'model': 'inceptionv3',
        'loss': 'CE',
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingWarmRestarts',
        'img_size': 512,
        'batch_size': 48,
        'epochs': 100,
        'lr': 0.001,
    }
    
    train(opt)