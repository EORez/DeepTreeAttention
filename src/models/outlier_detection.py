#Autoencoder
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch
from torch import optim
from src.models.Hang2020 import conv_module
from src import center_loss
from pytorch_lightning import LightningModule, Trainer
import pandas as pd
import torchmetrics

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
    
class classifier(nn.Module):
    def __init__(self, classes, image_size = 28, embedding_size = 2):
        super(classifier, self).__init__()
        self.image_size = image_size
        self.feature_length = 2 * self.image_size * image_size
        #Classification layer
        self.vis_conv1= encoder_block(in_channels=16, filters=2) 
        self.classfication_bottleneck = nn.Linear(in_features=self.feature_length, out_features=embedding_size)        
        self.classfication_layer = nn.Linear(in_features=embedding_size, out_features=classes)
        
        #Visualization
        # a dict to store the activations        
        self.vis_activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.vis_activation[name] = output.detach()
            return hook
        
        self.vis_conv1.register_forward_hook(getActivation("vis_conv1"))    
        self.classfication_bottleneck.register_forward_hook(getActivation("classification_bottleneck"))   
        
    def forward(self, x):
        y = self.vis_conv1(x)
        y = y.view(-1, self.feature_length)        
        y = self.classfication_bottleneck(y)
        y = self.classfication_layer(y)
        
        return y
    
class autoencoder(LightningModule):
    def __init__(self, bands, classes, config, comet_logger):
        super(autoencoder, self).__init__()    
        
        self.automatic_optimization = False        
        self.config = config
        self.comet_logger = comet_logger
        
        #Encoder
        self.encoder_block1 = encoder_block(in_channels=bands, filters=64, pool=True)
        self.encoder_block2 = encoder_block(in_channels=64, filters=32, pool=True)
        self.encoder_block3 = encoder_block(in_channels=32, filters=16, pool=True)
        
        self.alpha = nn.Parameter(torch.tensor(self.config["center_loss_weight"], dtype=float), requires_grad=False)
        self.classifier = classifier(classes, image_size=config["image_size"], embedding_size=config["embedding_size"])
        
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
             
        self.encoder_block3.register_forward_hook(getActivation("encoder_block3"))        

        #Metrics
        micro_recall = torchmetrics.Accuracy(average="micro", num_classes=10)
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall}, prefix="autoencoder_")
        
        #center loss
        use_gpu = self.config["gpus"] > 0
        self.closs = center_loss.CenterLoss(num_classes=classes, use_gpu=use_gpu, feat_dim=self.config["embedding_size"])
        
    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        
        #classification layer projection
        y = self.classifier(x)

        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)

        return x, y

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        
        #allow for empty data if data augmentation is generated
        index, images, observed_labels, true_labels = batch 
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        #Calculate losses
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)    
        classification_loss = F.cross_entropy(classification_yhat, observed_labels)
        
        features = self.classifier.vis_activation["classification_bottleneck"]            
        step_center_loss = self.closs(features, observed_labels)
        classification_loss = classification_loss + self.alpha * step_center_loss
        loss = self.config["autoencoder_loss_scalar"] * autoencoder_loss  + classification_loss * self.config["classification_loss_scalar"]

        ##Manual optimization
        opt_classifier, opt_center = self.optimizers()
        opt_classifier.zero_grad()
        opt_center.zero_grad()

        self.manual_backward(loss)
        opt_classifier.step()
        
        #Adjust center weights
        for param in self.closs.parameters():
            param.grad.data *= (1. / self.alpha)
        opt_center.step()
        
        ##Log
        self.log("center loss", step_center_loss,on_epoch=True,on_step=False)
        self.log("center alpha", self.alpha,on_epoch=True,on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        index, images, observed_labels, true_labels = batch 
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)  
        
        #ignore novel classes
        observed_labels[observed_labels==8] = -1
        observed_labels[observed_labels==9] = -1
        
        classification_loss = F.cross_entropy(classification_yhat, observed_labels, ignore_index=-1)
        features = self.classifier.vis_activation["classification_bottleneck"]        
        step_center_loss = self.closs(features, observed_labels)
        self.log("val_center_loss", step_center_loss, on_epoch=True, on_step=False)
        loss = self.config["autoencoder_loss_scalar"] * autoencoder_loss + classification_loss * self.config["classification_loss_scalar"]
        
        softmax_prob = F.softmax(classification_yhat, dim=1)
        softmax_prob = F.pad(input=softmax_prob, pad=(0, 2, 0, 0), mode='constant', value=0)
        
        output = self.metrics(softmax_prob, true_labels) 
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log_dict(output, on_epoch=True, on_step=False)
        
        return loss
        
    def configure_optimizers(self):
        classification_optimizer = optim.SGD(self.parameters(), lr=self.config["lr"], weight_decay=5e-04, momentum=0.9)
        center_loss_optimizer = optim.SGD(self.closs.parameters(), lr=self.config["center_loss_lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(classification_optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         eps=1e-08)
        
        center_scheduler = optim.lr_scheduler.ReduceLROnPlateau(classification_optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         eps=1e-08)
        
        return [classification_optimizer, center_loss_optimizer], [{"scheduler":scheduler,"monitor":'val_loss'}, {"scheduler":center_scheduler, "monitor":"val_center_loss"}]
        
    def predict(self, dataloader):
        """Generate labels and predictions for a data_loader"""
        observed_y = []
        yhat = []
        autoencoder_loss = []
        sample_ids = []
        vis_epoch_activations = []
        encoder_epoch_activations = []
        classification_bottleneck = []
        
        self.eval()
        
        for batch in dataloader:
            index, inputs, observed_labels = batch
            images = inputs["HSI"]
            observed_y.append(observed_labels.numpy())
            sample_ids.append(index)
            #trigger activation hook
            if next(self.parameters()).is_cuda:
                images = images.cuda()
            with torch.no_grad():
                for image in images:
                    image_yhat, classification_yhat = self(image.unsqueeze(0))
                    yhat.append(classification_yhat)
                    loss = F.mse_loss(image_yhat, image)    
                    autoencoder_loss.append(loss.numpy())
                    vis_epoch_activations.append(self.classifier.vis_activation["vis_conv1"].cpu().numpy())
                    encoder_epoch_activations.append(self.vis_activation["encoder_block3"].cpu().numpy())
                    classification_bottleneck.append(self.classifier.vis_activation["classification_bottleneck"].cpu().numpy())                    
           
        yhat = np.concatenate(yhat)
        yhat = np.argmax(yhat, 1)
        sample_ids = np.concatenate(sample_ids)
        observed_y = np.concatenate(observed_y)
        autoencoder_loss = np.asarray(autoencoder_loss)
        
        #Create a single array
        self.classification_conv_activations = np.concatenate(vis_epoch_activations)
        self.encoder_activations = np.concatenate(encoder_epoch_activations)
        self.classification_bottleneck = np.concatenate(classification_bottleneck)
        
        results = pd.DataFrame({"individual":sample_ids,"observed_label": observed_y,"predicted_label":yhat,"autoencoder_loss": autoencoder_loss})        
    
        return results

#Subclass model for trees

class tree_classifier(nn.Module):
    def __init__(self, classes, image_size = 28):
        super(tree_classifier, self).__init__()
        self.image_size = image_size
        self.feature_length = 2 * self.image_size * image_size
        
        #Classification layer
        self.vis_conv1= encoder_block(in_channels=16, filters=8) 
        self.vis_conv2= encoder_block(in_channels=8, filters=2)         
        self.classfication_bottleneck = nn.Linear(in_features=self.feature_length, out_features=2)        
        self.classfication_layer = nn.Linear(in_features=2, out_features=classes)
        
        #Visualization
        # a dict to store the activations        
        self.vis_activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.vis_activation[name] = output.detach()
            return hook
        
        self.vis_conv1.register_forward_hook(getActivation("vis_conv1"))    
        self.classfication_bottleneck.register_forward_hook(getActivation("classification_bottleneck"))   
        
    def forward(self, x):
        y = self.vis_conv1(x)
        y = self.vis_conv2(y)        
        y = y.view(-1, self.feature_length)        
        y = self.classfication_bottleneck(y)
        y = self.classfication_layer(y)
        
        return y
    
class tree_autoencoder(autoencoder):
    def __init__(self, bands, classes, config, comet_logger):
        super(tree_autoencoder, self).__init__(bands, classes, config, comet_logger) 
        
        #Deeper classification head
        self.classifier = tree_classifier(classes=classes, image_size=config["image_size"])

        #Metrics
        micro_recall = torchmetrics.Accuracy(average="micro", num_classes=classes)
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall}, prefix="autoencoder_")        
        
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        index, inputs, observed_labels = batch 
        images = inputs["HSI"]
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)    
        classification_loss = F.cross_entropy(classification_yhat, observed_labels)
        loss = self.config["autoencoder_loss_scalar"] * autoencoder_loss  + classification_loss * self.config["classification_loss_scalar"]
        self.log("loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        index, inputs, observed_labels = batch 
        images = inputs["HSI"]        
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)  
        classification_loss = F.cross_entropy(classification_yhat, observed_labels)
        loss = self.config["autoencoder_loss_scalar"] * autoencoder_loss + classification_loss * self.config["classification_loss_scalar"]
        
        softmax_prob = F.softmax(classification_yhat, dim=1)
        output = self.metrics(softmax_prob, observed_labels) 
        self.log("val_loss", loss, on_epoch=True)
        self.log_dict(output, on_epoch=True, on_step=False)
        
        return loss    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         eps=1e-08)
        
        return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'loss'} 
        
def train(model, dataloader, config, comet_logger):
    """Train a neural network arch"""
    #Create trainer
    config["classification_loss_scalar"] = 1
    config["autoencoder_loss_scalar"] = 0         
    if comet_logger:
        with comet_logger.experiment.context_manager("classification_only"):
            trainer = Trainer(
                gpus=config["gpus"],
                fast_dev_run=config["fast_dev_run"],
                max_epochs=config["classifier_epochs"],
                accelerator=config["accelerator"],
                checkpoint_callback=False,
                logger=comet_logger)
        
            trainer.fit(model, dataloader)
    else:
        trainer = Trainer(
            gpus=config["gpus"],
            fast_dev_run=config["fast_dev_run"],
            max_epochs=config["classifier_epochs"],
            accelerator=config["accelerator"],
            checkpoint_callback=False,
            logger=comet_logger)
    
        trainer.fit(model, dataloader)        
    
    #freeze classification and below layers
    for x in model.parameters():
        x.requires_grad = False
    
    for x in model.decoder_block1.parameters():
        x.requires_grad = True
    
    for x in model.decoder_block2.parameters():
        x.requires_grad = True
    
    for x in model.decoder_block3.parameters():
        x.requires_grad = True    
    if comet_logger:
        with comet_logger.experiment.context_manager("autoencoder_only"):  
            config["classification_loss_scalar"] = 0
            config["autoencoder_loss_scalar"] = 1
            trainer = Trainer(
                gpus=config["gpus"],
                fast_dev_run=config["fast_dev_run"],
                max_epochs=config["autoencoder_epochs"],
                accelerator=config["accelerator"],
                checkpoint_callback=False,
                logger=comet_logger)

                    
            trainer.fit(model, dataloader)
    else:
        config["classification_loss_scalar"] = 0
        config["autoencoder_loss_scalar"] = 1
        trainer = Trainer(
            gpus=config["gpus"],
            fast_dev_run=config["fast_dev_run"],
            max_epochs=config["autoencoder_epochs"],
            accelerator=config["accelerator"],
            checkpoint_callback=False,
            logger=comet_logger)

                
        trainer.fit(model, dataloader)        
        
    
            
