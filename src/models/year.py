from pytorch_lightning import LightningModule
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np

class ensemble_dataset(Dataset):
    """Generate dataset for learning ensemble among years"""
    def __init__(self, data_dict, labels=None):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.labels = labels
        
    def __len__(self):
        return len(list(self.data_dict.keys()))
    
    def __getitem__(self, index):
        year_results = self.data_dict[self.keys[index]]
        year_stack = torch.tensor(np.concatenate(year_results))
        if not isinstance(type(self.labels), type(None)):
            label = torch.tensor(self.labels[index])
            return year_stack.unsqueeze(0), label
        else:
            return year_stack.unsqueeze(0)
    
class year_ensemble(LightningModule):
    def __init__(self, train_dict, train_labels, val_dict, val_labels, config, classes, years):
        super().__init__()
        self.config = config
        self.train_ds = ensemble_dataset(train_dict,labels=train_labels)
        self.val_ds = ensemble_dataset(val_dict,labels=val_labels)
        self.fc1 = torch.nn.Linear(in_features=(1,classes * years), out_features=classes* 2)
        self.fc2 = torch.nn.Linear(in_features=classes * 2, out_features=classes)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        
        return optimizer
        

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def training_step(self, batch):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y) 
        self.log("train_ensemble_loss", loss)
        
        return loss
    

    def validation_step(self, batch):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y) 
        self.log("val_ensemble_loss", loss)
        
        return loss
    
    