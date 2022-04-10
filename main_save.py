import pytorch_lightning as pl
from torch.utils import data

import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import roc_curve,auc

from  utils import *

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
    
        # define network, loss, dataset
        self.network = get_model(config['model'])
        self.loss = get_loss(config['loss'])
        self.train_data, self.val_data, self.test_data = split_data()
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'],config['img_size'])),
        transforms.RandomRotation(np.pi/24),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'],config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = Dataset(self.train_data, transform=self.transform)
        self.val_dataset = Dataset(self.val_data, transform=self.test_transform)
        
        # initialize some variables
        self.best_val_loss = None
        self.best_auc = 0
        self.best_epoch = None
        
    def train_dataloader(self):
        train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = data.DataLoader(dataset=self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
        return val_loader
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.config['optimizer'], self.parameters(), self.config['lr'], 0, 0)
        self.lr_scheduler = get_scheduler(self.config['scheduler'], optimizer, self.config['epochs'])
        return optimizer
    
    def forward(self, x):
        return self.network(x)    
    
    def training_step(self, batch, batch_idx):
        prob_all = []
        label_all = []
        pred = []
        
        images, labels = batch
        outputs = self.network(images)
        loss = self.loss(outputs, labels)
        prob = F.softmax(outputs,dim=1)
        prob_all.extend(prob[:,1].cpu().detach().numpy())
        label_all.extend(labels.cpu().detach().numpy())
        pred.extend(outputs.argmax(dim=-1).cpu().detach().numpy())
        acc = (outputs.argmax(dim=-1) == labels).float().mean()

        
        return {'loss': loss,'acc': acc, 'prob': prob_all, 'label': label_all, 'pred': pred}
        
    def training_epoch_end(self, outputs):
        s = 0
        acc = 0
        prob_all = []
        label_all = []
        pred = []
        for t in outputs:
            s += t['loss']
            acc += t['acc']
            prob_all += t['prob']
            label_all += t['label']
            pred += t['pred']
        s /= len(outputs)
        acc /= len(outputs)
        fpr,tpr,_= roc_curve(label_all,prob_all[:len(label_all)])
        area_under_curve=auc(fpr,tpr)
        
        
        self.log('step',self.trainer.current_epoch)
        if self.lr_scheduler:
            self.lr_scheduler.step()
            self.log('lr', self.lr_scheduler.get_last_lr()[0], on_step=False, on_epoch=True)
        else:
            self.log('lr', self.config['lr'], on_step=False, on_epoch=True)
        
        self.log('train/loss', s, on_step=False, on_epoch=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True)
        self.log('train/AUC', area_under_curve, on_step=False, on_epoch=True)
     
        
    
    def validation_step(self, batch, batch_idx):
        prob_all = []
        label_all = []
        pred = []

        images, labels = batch
        outputs = self.network(images)
        loss = self.loss(outputs, labels)
        prob = F.softmax(outputs,dim=1)
        prob_all.extend(prob[:,1].cpu().detach().numpy())
        label_all.extend(labels.cpu().detach().numpy())
        pred.extend(outputs.argmax(dim=-1).cpu().detach().numpy())
        acc = (outputs.argmax(dim=-1) == labels).float().mean()

        
        return {'loss': loss,'acc': acc, 'prob': prob_all, 'label': label_all, 'pred': pred}
     
    def validation_epoch_end(self, outputs):
        s = 0
        acc = 0
        prob_all = []
        label_all = []
        pred = []
        for t in outputs:
            s += t['loss']
            acc += t['acc']
            prob_all += t['prob']
            label_all += t['label']
            pred += t['pred']
        s /= len(outputs)
        acc /= len(outputs)
        fpr,tpr,_= roc_curve(label_all,prob_all[:len(label_all)])
        area_under_curve=auc(fpr,tpr)
        
        if area_under_curve > self.best_auc:
            self.best_epoch = self.current_epoch
            self.best_auc = area_under_curve
            
        
         
        self.log('step',self.trainer.current_epoch)
        self.log('val/loss', s, on_step=False, on_epoch=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True)
        self.log('val/AUC', area_under_curve, on_step=False, on_epoch=True)      
        self.log('hp_metric', self.best_auc, on_step=False, on_epoch=True)
        self.log('best_epoch', self.best_epoch, on_step=False, on_epoch=True)

if __name__ == '__main__':
    # arg parser
    #### define dataset

    opt = {
        'model': 'inceptionv3',
        'loss': 'FL',
        'optimizer': 'SGD',
        'scheduler': '',
        'img_size': 512,
        'batch_size': 40,
        'epochs': 100,
        'lr': 0.0005,
    }
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=opt['epochs'],
        check_val_every_n_epoch=1
    )
    trainer.fit(LitModel(opt))