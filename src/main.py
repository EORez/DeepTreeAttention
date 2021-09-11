#Lightning Data Module
from . import __file__
import geopandas as gpd
import glob as glob
from deepforest.main import deepforest
import os
import numpy as np
from pytorch_lightning import LightningModule
import pandas as pd
from torch.nn import functional as F
from torch import optim
import torch
import torchmetrics
import tempfile
from src import data
from src import generate
from src import neon_paths
from src import patches
from shapely.geometry import Point, box

class TreeModel(LightningModule):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self,model, bands, classes, label_dict, config=None, *args, **kwargs):
        super().__init__()
    
        self.ROOT = os.path.dirname(os.path.dirname(__file__))    
        self.tmpdir = tempfile.gettempdir()
        if config is None:
            self.config = data.read_config("{}/config.yml".format(self.ROOT))   
        else:
            self.config = config
        
        self.classes = classes
        self.bands = bands
        self.label_to_index = label_dict
        self.index_to_label = {}
        for x in label_dict:
            self.index_to_label[label_dict[x]] = x 
        
        #Create model 
        self.model = model(bands, classes)
        
        #Metrics
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=classes)
        top_k_recall = torchmetrics.Accuracy(average="micro",top_k=self.config["top_k"])
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall,"Macro Accuracy":macro_recall,"Top {} Accuracy".format(self.config["top_k"]): top_k_recall})
        
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        images, y = batch
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        images, y = batch
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        
        output = self.metrics(y_hat, y) 
        self.log_dict(output)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config["lr"],
                                   momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         min_lr=0,
                                                         eps=1e-08)
        
        return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_loss'}
    
    def predict_image(self, img_path):
        """Given an image path, load image and predict"""
        self.model.eval()        
        image = data.load_image(img_path)
        batch = torch.unsqueeze(image, dim=0)
        y = self.model(batch).detach()
        index = np.argmax(y, 1).detach().numpy()[0]
        label = self.index_to_label[index]
        
        return label
        
    def predict_xy(self, coordinates, fixed_box=True):
        """Given an x,y location, find sensor data and predict tree crown class. If no predicted crown within 5m an error will be raised (fixed_box=False) or a 1m fixed box will created (fixed_box=True)
        Args:
            coordinates (tuple): x,y tuple in utm coordinates
            fixed_box (False): If no DeepForest tree is predicted within 5m of centroid, create a 1m fixed box. If false, raise ValueError
        Returns:
            label: species taxa label
        """
        #Predict crown
        gdf = gpd.GeoDataFrame(geometry=[Point(coordinates[0],coordinates[1])])
        img_pool = glob.glob(self.config["rgb_sensor_pool"], recursive=True)
        rgb_path = neon_paths.find_sensor_path(lookup_pool=img_pool, bounds=gdf.total_bounds)
        
        #DeepForest model to predict crowns
        deepforest_model = deepforest()
        deepforest_model.use_release(check_release=False)
        boxes = generate.predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_path, bounds=gdf.total_bounds, expand=40)
        boxes['geometry'] = boxes.apply(lambda x: box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
        if boxes.shape[0] > 1:
            centroid_distances = boxes.centroid.distance(Point(coordinates[0],coordinates[1])).sort_values()
            centroid_distances = centroid_distances[centroid_distances<5]
            boxes = boxes[boxes.index == centroid_distances.index[0]]
        if boxes.empty:
            if fixed_box:
                boxes = generate.create_boxes(boxes)
            else:
                raise ValueError("No predicted tree centroid within 5 m of point {}, to ignore this error and specify fixed_box=True".format(coordinates))
            
        #Create pixel crops
        img_pool = glob.glob(self.config["HSI_sensor_pool"], recursive=True)        
        sensor_path = neon_paths.find_sensor_path(lookup_pool=img_pool, bounds=gdf.total_bounds)        
        filenames = patches.crown_to_pixel(
            crown=boxes["geometry"].values[0],
            img_path=sensor_path,
            width=self.config["window_size"],
            height=self.config["window_size"],
            savedir=self.tmpdir,
            basename="prediction"
        )
       
        #Classify pixel crops
        self.model.eval()                
        predictions = []
        for x in filenames:
            #Preprocess starts from channels last
            image = data.load_image(x)
            #batch
            batch = torch.unsqueeze(image, dim=0)            
            class_probs = self.model(batch)
            predictions.append(class_probs.detach().numpy())
        
        predictions = np.vstack(predictions)
        majority_index = pd.Series(np.argmax(predictions, 1)).mode()[0]
        label = self.index_to_label[majority_index]
        
        #Average score for selected label
        average_score = np.mean(predictions[:,majority_index])
        
        return label, average_score

        
    def predict_crown(self, img):
        """Given an image, traverse the pixels and create crown prediction
        Args:
            img (np.array): a numpy array of image data
        Returns:
            label: species taxa label
        """
        pass
    
    def predict_file(self, csv_file):
        """Given a file with paths to image crops, create crown predictions 
        The format of image_path inform the crown membership, the files should be named crownid_counter.png where crownid is a
        unique identifier for each crown and counter is 0..n pixel crops that belong to that crown.
        
        Args: 
            csv_file: path to csv file
        Returns:
            results: pandas dataframe with columns crown and species label
        """
        df = pd.read_csv(csv_file)
        df["crown"] = df.image_path.apply(lambda x: os.path.basename(x).split("_")[0])
        df["label"] = df.image_path.apply(lambda x: self.predict_image(x))
        
        #Majority vote per crown
        results = df.groupby(["crown"]).agg(lambda x: x.mode()[0]).reset_index()
        results = results[["crown","label"]]
        
        return results
    
    def evaluate_crowns(self, csv_file):
        """Crown level measure of accuracy
        Args:
            csv_file: ground truth csv with image_path and label columns
        Returns:
            df: results dataframe
            metric_dict: metric -> value
        """
        ground_truth = pd.read_csv(csv_file)
        #convert to taxon label
        ground_truth["true_label"] = ground_truth.label
        ground_truth["crown"] = ground_truth.image_path.apply(lambda x: os.path.basename(x).split("_")[0])
        ground_truth = ground_truth.groupby(["crown"]).agg(lambda x: x.mode()[0]).reset_index()[["crown","true_label"]]
        results = self.predict_file(csv_file)
        results["label"] = results["label"].apply(lambda x: self.label_to_index[x])
        df = results.merge(ground_truth)
        crown_micro = torchmetrics.functional.accuracy(preds=torch.tensor(df.label.values, dtype=torch.long),target=torch.tensor(df.true_label.values, dtype=torch.long), average="micro")
        crown_macro = torchmetrics.functional.accuracy(preds=torch.tensor(df.label.values, dtype=torch.long),target=torch.tensor(df.true_label.values, dtype=torch.long), average="macro", num_classes=self.classes)
        
        return df, {"crown_micro":crown_micro,"crown_macro":crown_macro}
    
        
        
        
        
        

    