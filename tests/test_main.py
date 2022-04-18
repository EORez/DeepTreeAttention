import geopandas as gpd
import os
import pandas as pd
from pytorch_lightning import Trainer
from src import data
from src import main
from src.models import Hang2020

def test_fit(config, m, dm, comet_logger):
    trainer = Trainer(fast_dev_run=True, logger=comet_logger)
    trainer.fit(m,datamodule=dm)
    
def test_predict_dataloader(config, m, dm, comet_logger, ROOT):
    if comet_logger:
        experiment = comet_logger.experiment
    else:
        experiment = None
    df = m.predict_dataloader(
        dm.val_dataloader())
    input_data = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))    
    
    assert df.shape[0] == len(input_data.image_path.apply(lambda x: os.path.basename(x).split("_")[0]).unique())
    
def test_evaluate_crowns(config, comet_logger, m, dm, ROOT):
    if comet_logger:
        experiment = comet_logger.experiment
    else:
        experiment = None    
    m.ROOT = "{}/tests".format(ROOT)
    df = m.evaluate_crowns(data_loader = dm.val_dataloader(), crowns=dm.crowns, experiment=experiment)
    assert all(["top{}_score".format(x) in df.columns for x in [1,2]]) 

def test_predict_xy(config, m, dm, ROOT):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    df = pd.read_csv(csv_file)
    label, score = m.predict_xy(coordinates=(df.itcEasting[0],df.itcNorthing[0]))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 

def test_predict_crown(config, m, dm, ROOT):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    label, score = m.predict_crown(geom = gdf.geometry[0], sensor_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 
    
def test_fit_new_dataloader(config, dm, ROOT):
    model = Hang2020.spectral_network(bands=config["bands"], classes=2)    
    trainer = Trainer(fast_dev_run=True)
    dm.train_ds = data.TreeDataset(
        csv_file="{}/tests/data/processed/train.csv".format(ROOT),
        config=config,
        taxonIDs=["PIST"],
        keep_others=True)    
    dm.val_ds = data.TreeDataset(
        csv_file="{}/tests/data/processed/test.csv".format(ROOT),
        config=config,
        taxonIDs=["PIST"],
        keep_others=True)    
    dm.species_label_dict = {"PIPA2":0,"OTHER":1}
    dm.label_to_taxonID = {v: k  for k, v in dm.species_label_dict.items()}
        #Load from state dict of previous run
    m = main.TreeModel(
        model=model, 
        loss_weight=[1,1],
        classes=2, 
        label_dict=dm.species_label_dict)
    trainer.fit(m,datamodule=dm)