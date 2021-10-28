#Autoencoder
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch import optim
from src.models.Hang2020 import conv_module
from src import visualize
from src import data
import numpy as np
from pytorch_lightning import LightningModule, Trainer
import pandas as pd

class encoder_block(nn.Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None, pool=False):
        super(encoder_block, self).__init__()
        self.conv = conv_module(in_channels, filters)
        self.bn = nn.BatchNorm2d(num_features=filters)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class decoder_block(nn.Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None, pool=False):
        super(decoder_block, self).__init__()
        self.conv = conv_module(in_channels, filters)
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=filters, kernel_size=(2,2))
        self.bn = nn.BatchNorm2d(num_features=filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class autoencoder(LightningModule):
    def __init__(self, csv_file, bands, config, data_dir):
        super(autoencoder, self).__init__()    
        
        self.config = config
        self.csv_file = csv_file
        self.data_dir = data_dir
        
        #Encoder
        self.encoder_block1 = encoder_block(in_channels=bands, filters=64, pool=True)
        self.encoder_block2 = encoder_block(in_channels=64, filters=32, pool=True)
        self.encoder_block3 = encoder_block(in_channels=32, filters=16, pool=True)
        
        #2D projection layer for visualization hook
        self.vis_layer = nn.Linear(in_features=1936, out_features=2)
        
        #Decoder
        self.decoder_block1 = decoder_block(in_channels=16, filters=32)
        self.decoder_block2 = decoder_block(in_channels=32, filters=64)
        self.decoder_block3 = decoder_block(in_channels=64, filters=bands)
        
        #Visualization
        # a dict to store the activations        
        self.vis_activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.vis_activation[name] = output.detach()
            return hook
        
        self.vis_layer.register_forward_hook(getActivation("vis_layer"))        
        self.encoder_block3.register_forward_hook(getActivation("encoder_block3"))        

    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        
        #vis layer projection
        y = x.view(-1, 16*11*11)
        y = self.vis_layer(y)

        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)

        return x

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, labels = batch
        images = inputs["HSI"]
        y_hat = self.forward(images)
        loss = F.mse_loss(y_hat, images)    

        return loss
    
    def predict_step(self, batch, batch_idx):
        individual, inputs, labels = batch
        images = inputs["HSI"]     
        losses = []
        for image in images:
            with torch.no_grad():
                y_hat = self.forward(image.unsqueeze(0)) 
            loss = F.mse_loss(y_hat, image.unsqueeze(0))
            losses.append(loss.cpu().numpy())
            
        return pd.DataFrame({"individual":individual, "loss":losses})
    
    def train_dataloader(self):
        ds = data.TreeDataset(csv_file = self.csv_file, config=self.config, HSI=True, metadata=False, train=True)
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
        )        
        
        return data_loader
    
    def predict_dataloader(self):
        ds = data.TreeDataset(csv_file = self.csv_file.format(self.data_dir), config=self.config, HSI=True, metadata=False, train=True)
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
        )        
        
        return data_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])

        return {'optimizer':optimizer,"monitor":'val_loss'}

    def on_epoch_end(self):
        """At the end of each epoch trigger the dataset to collect intermediate activation for plotting"""
        #plot 2d projection layer
        epoch_labels = []
        vis_epoch_activations = []
        encoder_epoch_activations = []
        
        for batch in self.train_dataloader():
            individual, inputs, label = batch
            epoch_labels.append(label)
            #trigger activation hook
            if next(self.parameters()).is_cuda:
                image = inputs["HSI"].cuda()
            else:
                image = inputs["HSI"]
            pred = self(image)
            vis_epoch_activations.append(self.vis_activation["vis_layer"].cpu())
            encoder_epoch_activations.append(self.vis_activation["encoder_block3"].cpu())

        #Create a single array
        epoch_labels = np.concatenate(epoch_labels)
        vis_epoch_activations = torch.tensor(np.concatenate(vis_epoch_activations))
        encoder_epoch_activations = torch.tensor(np.concatenate(encoder_epoch_activations))
        
        layerplot_vis = visualize.plot_2d_layer(vis_epoch_activations, epoch_labels)
        try:
            self.logger.experiment.log_figure(figure=layerplot_vis, figure_name="2d_vis_projection", step=self.current_epoch)
        except Exception as e:
            print("Comet logger failed: {}".format(e))
            
        layerplot_encoder = visualize.plot_2d_layer(encoder_epoch_activations, epoch_labels, use_pca=True)
        try:
            self.logger.experiment.log_figure(figure=layerplot_encoder, figure_name="2d_encoder_projection", step=self.current_epoch)
        except Exception as e:
            print("Comet logger failed: {}".format(e))
                
        #reset activations
        self.vis_epoch_activations = {}
        self.encoder_epoch_activations = {}
        
def find_outliers(annotations, config, data_dir, comet_logger=None):
    """Train a deep autoencoder and identify outliers based on a quantile threshold"""
    #For each species train and predict
    
    #Store class labels
    unique_species_labels = annotations.taxonID.unique()
    
    #Taxon to ID dict and the reverse    
    species_label_dict = {}
    for index, taxonID in enumerate(unique_species_labels):
        species_label_dict[taxonID] = index
    
    annotations["label"] = annotations.taxonID.apply(lambda x: species_label_dict[x])
    fname = "{}/interim/before_outlier.csv".format(data_dir)
    annotations.to_csv(fname)
    m = autoencoder(csv_file=fname, config=config, bands = config["bands"], data_dir=data_dir)
    
    trainer = Trainer(
        gpus=config["gpus"],
        fast_dev_run=config["fast_dev_run"],
        max_epochs=config["autoencoder_epochs"],
        accelerator=config["accelerator"],
        checkpoint_callback=False,
        logger=comet_logger)
    
    trainer.fit(model=m)
            
    prediction = trainer.predict(m)
    predictions = pd.concat(prediction)
            
    #remove lowest quantile    
    predictions.to_csv("{}/interim/reconstruction_error.csv".format(data_dir))
    threshold = predictions.loss.quantile(config["outlier_threshold"])
    print("Reconstruction threshold is {}".format(threshold))
    outliers = predictions[predictions.loss > threshold]
    
    #Upload loss image with the outliers
    #plot 2d projection layer
    epoch_labels = []
    epoch_activations = []
    individuals = []
    for batch in m.train_dataloader():
        individual, inputs, label = batch
        epoch_labels.append(label)
        individuals.append(individual)
        #trigger activation hook
        if next(m.parameters()).is_cuda:
            image = inputs["HSI"].cuda()
        else:
            image = inputs["HSI"]
        pred = m(image)
        epoch_activations.append(m.vis_activation["vis_layer"].cpu())

    #Create a single array
    epoch_activations = np.concatenate(epoch_activations) 
    individuals = np.concatenate(individuals) 
    
    #color by outlier status
    outlier_color = [(x in outliers.individual.values)*1 for x in individuals]
    layerplot = visualize.plot_2d_layer(epoch_activations, outlier_color)   
    
    if comet_logger:
        comet_logger.experiment.log_figure(figure = layerplot, figure_name = "outliers.png")
    
    return outliers
        
        
        
    
            
        