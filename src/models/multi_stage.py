#Multiple stage model
from src.models import Hang2020
from src.data import TreeDataset
from pytorch_lightning import LightningModule, Trainer
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch
import pandas as pd

class base_model(Module):
    def __init__(self, classes, config):
        super().__init__()
        #Load from state dict of previous run
        if config["pretrain_state_dict"]:
            self.model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=classes, bands=config["bands"])
        else:
            self.model = Hang2020.spectral_network(bands=config["bands"], classes=classes)
            
    def forward(self,x):
        x = self.model(x)
        # Last attention layer as score        
        score = x[-1]
        
        return score 
    
class MultiStage(LightningModule):
    def __init__(self, train_df,test_df, config):
        super().__init__()        
        # Generate each model
        self.loss_weights = []
        self.config = config
        self.models = []
        self.species_label_dict = train_df.taxonID.drop_duplicates().reset_index().to_dict()
        self.level_label_dicts = {}     
        self.label_to_taxonIDs = {}    
        self.train_df = train_df
        self.test_df = test_df
        self.train_datasets, self.test_datasets = self.create_datasets()
        
        for ds in self.train_datasets: 
            labels = [x[2] for x in self.train_datasets[ds]]
            base = base_model(classes=len(np.unique(labels)), config=config)
            loss_weight = []
            for x in np.unique(labels):
                loss_weight.append(1/np.sum(labels==x))
        
            loss_weight = np.array(loss_weight/np.max(loss_weight))
            
            if torch.cuda.is_available():
                loss_weight = torch.tensor(loss_weight, device="cuda", dtype=torch.float)
            else:
                loss_weight = torch.tensor(loss_weight, dtype=torch.float)
                
            self.loss_weights.append(torch.tensor(loss_weight))
            self.models.append(base)
            
        self.save_hyperparameters(ignore=["loss_weight"])
        
    def create_datasets(self):
        #Create levels
        ## Level 0        
        self.species_label_dict = self.train_df[["label","taxonID"]].drop_duplicates().set_index("taxonID").to_dict()["label"]
        self.level_label_dicts[0] = {"PIPA2":0,"OTHER":1}
        self.label_to_taxonIDs[0] = {v: k  for k, v in self.level_label_dicts[0].items()}
        
        level_0_train = self.train_df.copy()
        level_0_train.loc[~(level_0_train.taxonID == "PIPA2"),"taxonID"] = "OTHER"
        level_0_train.loc[(level_0_train.taxonID == "PIPA2"),"taxonID"] = "PIPA2"            
        level_0_train["label"] = [self.level_label_dicts[0][x] for x in level_0_train.taxonID]
        level_0_train = TreeDataset(df=level_0_train, config=self.config)
        
        level_0_test = self.test_df.copy()
        level_0_test.loc[~(level_0_test.taxonID == "PIPA2"),"taxonID"] = "OTHER"
        level_0_test.loc[(level_0_test.taxonID == "PIPA2"),"taxonID"] = "PIPA2"                        
        level_0_test["label"]= [self.level_label_dicts[0][x] for x in level_0_test.taxonID]            
        level_0_test = TreeDataset(df=level_0_test, config=self.config)
        
        ## Level 1
        self.level_label_dicts[1] =  {"CONIFER":0,"BROADLEAF":1}
        self.label_to_taxonIDs[1] = {v: k  for k, v in self.level_label_dicts[1].items()}
        level_1_train = self.train_df.copy()
        level_1_train = level_1_train[~(level_1_train.taxonID=="PIPA1")]    
        level_1_train.loc[~level_1_train.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"   
        level_1_train.loc[level_1_train.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER"            
        level_1_train["label"] = [self.level_label_dicts[1][x] for x in level_1_train.taxonID]
        level_1_train = TreeDataset(df=level_1_train, config=self.config)
        
        level_1_test = self.test_df.copy()
        level_1_test = level_1_test[~(level_1_test.taxonID=="PIPA1")]    
        level_1_test.loc[~level_1_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"   
        level_1_test.loc[level_1_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER"            
        level_1_test["label"] = [self.level_label_dicts[1][x] for x in level_1_test.taxonID]
        level_1_test = TreeDataset(df=level_1_test, config=self.config)
        
        ## Level 2
        broadleaf = [x for x in list(self.species_label_dict.keys()) if (not x in ["PICL","PIEL","PITA","PIPA2"]) & (not "QU" in x)]            
        self.level_label_dicts[2] =  {v:k for k, v in enumerate(broadleaf)}
        self.level_label_dicts[2]["OAK"] = len(self.level_label_dicts[2])
        self.label_to_taxonIDs[2] = {v: k  for k, v in self.level_label_dicts[2].items()}
                    
        level_2_train = self.train_df.copy()
        level_2_train = level_2_train[~level_2_train.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])]  
        level_2_train.loc[level_2_train.taxonID.str.contains("QU"),"taxonID"] = "OAK"
        level_2_train["label"] = [self.level_label_dicts[2][x] for x in level_2_train.taxonID]
        level_2_train = TreeDataset(df=level_2_train, config=self.config)
        
        level_2_test = self.test_df.copy()
        level_2_test = level_2_test[~level_2_test.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])]  
        level_2_test.loc[level_2_test.taxonID.str.contains("QU"),"taxonID"] = "OAK"
        level_2_test["label"] = [self.level_label_dicts[2][x] for x in level_2_test.taxonID]
        level_2_test = TreeDataset(df=level_2_test, config=self.config)
        
        ## Level 3
        evergreen = [x for x in list(self.species_label_dict.keys()) if x in ["PICL","PIEL","PITA"]]         
        self.level_label_dicts[3] =  {v:k for k, v in enumerate(evergreen)}
        self.label_to_taxonIDs[3] = {v: k  for k, v in self.level_label_dicts[3].items()}
                    
        level_3_train = self.train_df.copy()
        level_3_train = level_3_train[level_3_train.taxonID.isin(["PICL","PIEL","PITA"])]  
        level_3_train["label"] = [self.level_label_dicts[3][x] for x in level_3_train.taxonID]
        level_3_train = TreeDataset(df=level_3_train, config=self.config)
        
        level_3_test = self.test_df.copy()
        level_3_test = level_3_test[level_3_test.taxonID.isin(["PICL","PIEL","PITA"])]  
        level_3_test["label"] = [self.level_label_dicts[3][x] for x in level_3_test.taxonID]
        level_3_test = TreeDataset(df=level_3_test, config=self.config)
        
        ## Level 4
        oak = [x for x in list(self.species_label_dict.keys()) if "QU" in x]
        self.level_label_dicts[4] =  {v:k for k, v in enumerate(oak)}
        self.label_to_taxonIDs[4] = {v: k  for k, v in self.level_label_dicts[4].items()}
                    
        level_4_train = self.train_df.copy()
        level_4_train = level_4_train[level_4_train.taxonID.str.contains("QU")]
        level_4_train["label"] = [self.level_label_dicts[4][x] for x in level_4_train.taxonID]
        level_4_train = TreeDataset(df=level_4_train, config=self.config)
        
        level_4_test = self.test_df.copy()
        level_4_test = level_4_test[level_4_test.taxonID.str.contains("QU")]
        level_4_test["label"] = [self.level_label_dicts[4][x] for x in level_4_test.taxonID]
        level_4_test = TreeDataset(df=level_4_test, config=self.config)
        
        train_datasets = {0:level_0_train, 1:level_1_train,2:level_2_train,3:level_3_train,4:level_4_train}
        test_datasets = [level_0_test, level_1_test,level_2_test,level_3_test,level_4_test]
        
        return train_datasets, test_datasets
    
    def train_dataloader(self):
        data_loaders = {}
        for ds in self.train_datasets:
            data_loader = torch.utils.data.DataLoader(
                self.train_datasets[ds],
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
            )
            data_loaders[ds] = data_loader
        
        return data_loaders        

    def val_dataloader(self):
        ## Validation loaders are a list https://github.com/PyTorchLightning/pytorch-lightning/issues/10809
        data_loaders = []
        for ds in self.test_datasets:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
            )
            data_loaders.append(data_loader)
        
        return data_loaders 
      
        
    def configure_optimizers(self):
        """Create a optimizer for each level"""
        optimizers = []
        for x, ds in enumerate(self.train_datasets):
            optimizer = torch.optim.Adam(self.models[x].parameters(), lr=self.config["lr"], weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             factor=0.75,
                                                             patience=8,
                                                             verbose=True,
                                                             threshold=0.0001,
                                                             threshold_mode='rel',
                                                             cooldown=0,
                                                             min_lr=0.0000001,
                                                             eps=1e-08)
            
            optimizers.append({'optimizer':optimizer, 'lr_scheduler': {"scheduler":scheduler, "monitor":'val_loss/dataloader_idx_{}'.format(x)}})

        return optimizers     
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Calculate train_df loss
        """
        individual, inputs, y = batch[optimizer_idx]
        images = inputs["HSI"]  
        y_hat = self.models[optimizer_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights[optimizer_idx])    

        return loss        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss 
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights[dataloader_idx])   
        
        self.log("val_loss",loss)
        
        return loss    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Calculate predictions
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        
        return individual, y_hat
    
    def evaluate_crowns(self):
        """Post-process the predict method to create metrics"""
        output = Trainer.predict(dataloaders=self.val_dataloaders())
        
        if return_features: 
            features = []
            for level in output:
                features.append(np.vstack(level[1]))             
            return features
        
        for level in output:
            # Concat batches            
            individuals = np.concatenate([x[0] for x in level])
            predictions = np.vstack([x[1] for x in output])
            #Create dataframe
            predictions_top1 = np.argmax(predictions, 1)    
            predictions_top2 = pd.DataFrame(predictions).apply(lambda x: np.argsort(x.values)[-2], axis=1)
            top1_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[0], axis=1)
            top2_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[1], axis=1)
            
            # Construct a df of predictions
            df = pd.DataFrame({
                "pred_label_top1":predictions_top1,
                "pred_label_top2":predictions_top2,
                "top1_score":top1_score,
                "top2_score":top2_score,
                "individual":individuals
            })
            df["pred_taxa_top1"] = df["pred_label_top1"].apply(lambda x: self.index_to_label[x]) 
            df["pred_taxa_top2"] = df["pred_label_top2"].apply(lambda x: self.index_to_label[x])        
            if train:
                df["label"] = labels
                df["true_taxa"] = df["label"].apply(lambda x: self.index_to_label[x])            
    
     