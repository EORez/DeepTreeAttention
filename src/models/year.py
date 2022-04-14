from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ensemble_dataset(Dataset):
    """Generate dataset for learning ensemble among years"""
    def __init__(self, data_dict, labels=None):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.labels = labels
        
    def __len__(self):
        return len(list(self.data_dict.keys()))
    
    def __getitem__(self, index):
        #Get last three years, long term we need an aggregator for variable length.
        try:
            year_results = self.data_dict[self.keys[index]][-3:]
        except:
            "Cannot gen data of size {} with elements {}".format(len(self.data_dict[self.keys[index]]), self.data_dict[self.keys[index]])
        year_stack = torch.tensor(np.vstack(year_results))
        if not isinstance(type(self.labels), type(None)):
            label = torch.tensor(self.labels[index])
            return year_stack, label
        else:
            return year_stack
    
class year_ensemble(LightningModule):
    def __init__(self, train_dict, train_labels, val_dict, val_labels, config, classes, years):
        super().__init__()
        self.config = config
        self.train_ds = ensemble_dataset(train_dict,labels=train_labels)
        self.val_ds = ensemble_dataset(val_dict,labels=val_labels)
        self.fc1 = torch.nn.Linear(in_features=classes, out_features=classes)
        
    def forward(self,x):
        x = x.sum(axis=1)
        x = self.fc1(x)
        
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        
        return optimizer
        

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["workers"],
            shuffle=True
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
    

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y) 
        self.log("val_ensemble_loss", loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        scores = F.softmax(y_hat, dim=1)
        
        return scores 
    
def run_ensemble(model, config, logger=None):
    """Train and predict an ensemble model"""
    trainer = Trainer(gpus=config["gpus"], max_epochs=config["ensemble_epochs"], logger=logger, checkpoint_callback=False)
    trainer.fit(model)
    gather = trainer.predict(model, dataloaders=model.val_dataloader(), ckpt_path=None)
    df = np.concatenate(gather)
    predicted_label = np.argmax(df, 1)
    score = np.max(df, 1)
    result_df = pd.DataFrame({"individual":model.val_ds.keys,"temporal_label_top1":predicted_label,"temporal_top1_score":score})
    
    return result_df
    