#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import geopandas as gpd
import rasterio
from src.main import TreeModel
from src.models import dead
from src import data 
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
import torch
from torch.utils.data.dataloader import default_collate

def RGB_transform(augment):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(transforms.Resize([224,224]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)

class on_the_fly_dataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       crowns: geodataframe of crown locations from a single rasterio src
       image_path: .tif file location
    """
    def __init__(self, crowns, image_path, data_type="HSI", config=None):
        self.config = config 
        self.crowns = crowns
        self.image_size = config["image_size"]
        self.data_type = data_type
        
        if data_type == "HSI":
            self.HSI_src = rasterio.open(image_path)
        elif data_type == "RGB":
            self.RGB_src = rasterio.open(image_path)
            self.transform = RGB_transform(augment=False)
        else:
            raise ValueError("data_type is {}, only HSI and RGB data types are currently allowed".format(data_type))
        
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        inputs = {}
        #Load crown and crop
        geom = self.crowns.iloc[index].geometry
        individual = self.crowns.iloc[index].individual
        left, bottom, right, top = geom.bounds
            
        #preprocess and batch
        if self.data_type =="HSI":
            crop = self.HSI_src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=self.HSI_src.transform))             
        
            if crop.size == 0:
                return individual, None
            
            image = data.preprocess_image(crop, channel_is_first=True)
            image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)

            inputs[self.data_type] = image
            
            return individual, inputs
        
        elif self.data_type=="RGB":
            #Expand RGB
            box = self.RGB_src.read(window=rasterio.windows.from_bounds(left-1, bottom-1, right+1, top+1, transform=self.RGB_src.transform))             
            #Channels last
            box = np.rollaxis(box,0,3)
            image = self.transform(box.astype(np.float32))
            image = image
            
            return image
        
def my_collate(batch):
    batch = [x for x in batch if x[1] is not None]
    
    return default_collate(batch)
    
def predict_tile(PATH, dead_model_path, species_model_path, config):
    #get rgb from HSI path
    HSI_basename = os.path.basename(PATH)
    if "hyperspectral" in HSI_basename:
        rgb_name = "{}.tif".format(HSI_basename.split("_hyperspectral")[0])    
    else:
        rgb_name = HSI_basename           
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
    rgb_path = [x for x in rgb_pool if rgb_name in x][0]
    crowns = predict_crowns(rgb_path)
    crowns["tile"] = PATH
    
    #Load Alive/Dead model
    dead_label, dead_score = predict_dead(crowns=crowns, dead_model_path=dead_model_path, rgb_tile=rgb_path, config=config)
    
    crowns["dead_label"] = dead_label
    crowns["dead_score"] = dead_score
    
    #Load species model
    m = TreeModel.load_from_checkpoint(species_model_path)
    trees, features = predict_species(HSI_path=PATH, crowns=crowns, m=m, config=config)
    
    #Spatial smooth
    trees = smooth(trees=trees, features=features, size=config["neighbor_buffer_size"], alpha=config["neighborhood_strength"])
    trees["spatial_taxonID"] = trees["spatial_label"]
    trees["spatial_taxonID"] = trees["spatial_label"].apply(lambda x: m.index_to_label[x]) 
    
    #Remove predictions for dead trees
    trees.loc[trees.dead_label==1,"spatial_taxonID"] = "DEAD"
    trees.loc[trees.dead_label==1,"spatial_label"] = None
    trees.loc[trees.dead_label==1,"spatial_score"] = None
    
    return trees

def predict_crowns(PATH):
    """Predict a set of tree crowns from RGB data"""
    m = main.deepforest()
    if torch.cuda.is_available():
        m.config["gpus"] = 1
    m.use_release(check_release=False)
    boxes = m.predict_tile(PATH)
    r = rasterio.open(PATH)
    transform = r.transform     
    crs = r.crs
    gdf = annotations_to_shapefile(boxes, transform=transform, crs=crs)
    
    #Dummy variables for schema
    basename = os.path.splitext(os.path.basename(PATH))[0]
    individual = ["{}_{}".format(x, basename) for x in range(gdf.shape[0])]
    gdf["individual"] = individual
    gdf["plotID"] = None
    gdf["siteID"] = None #TODO
    gdf["box_id"] = None
    gdf["plotID"] = None
    gdf["taxonID"] = None
    
    return gdf

def predict_species(crowns, HSI_path, m, config):
    ds = on_the_fly_dataset(crowns=crowns, image_path=HSI_path, config=config)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        collate_fn=my_collate
    )
    df, features = m.predict_dataloader(data_loader, train=False, return_features=True)
    crowns["bbox_score"] = crowns["score"]
    
    #If CHM exists TODO
    crowns = crowns.loc[:,crowns.columns.isin(["individual","geometry","bbox_score","tile","CHM_height","dead_label","dead_score"])]
    df = df.merge(crowns, on="individual")
    
    return df, features

def predict_dead(crowns, rgb_tile, dead_model_path, config):
    """Given a set of bounding boxes and an RGB tile, predict Alive/Dead binary model"""
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path)
    ds = on_the_fly_dataset(crowns=crowns, image_path=rgb_tile, config=config,data_type="RGB")
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        shuffle=False,
        num_workers=config["workers"],
    )
    if torch.cuda.is_available():
        dead_model = dead_model.to("cuda")
        dead_model.eval()
    
    gather_predictions = []
    for batch in data_loader:
        if torch.cuda.is_available():
            batch = batch.to("cuda")        
        with torch.no_grad():
            predictions = dead_model(batch)
            predictions = F.softmax(predictions, dim =1)
        gather_predictions.append(predictions.cpu())

    gather_predictions = np.concatenate(gather_predictions)
    
    label = np.argmax(gather_predictions,1)
    score = np.max(gather_predictions, 1)
    
    return label, score
    
def smooth(trees, features, size, alpha):
    """Given the results dataframe and feature labels, spatially smooth based on alpha value"""
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    sindex = trees.sindex
    tree_buffer = trees.buffer(size)
    smoothed_features = []
    for index, geom in enumerate(tree_buffer):
        intersects = sindex.query(geom)
        focal_feature = features[index,]
        neighbor_features = np.mean(features[intersects,], axis=0)
        smoothed_feature = focal_feature + alpha * neighbor_features
        smoothed_features.append(smoothed_feature)
    smoothed_features = np.vstack(smoothed_features)
    spatial_label = np.argmax(smoothed_features, axis=1)
    spatial_score = np.max(smoothed_features, axis=1)
    trees["spatial_label"] = spatial_label
    trees["spatial_score"] = spatial_score
    
    return trees
    
