#Train
import comet_ml
import glob
import geopandas as gpd
import os
import numpy as np
import torch
from src import main
from src import data
from src import start_cluster
from src.models import Hang2020
from src import visualize
from src import metrics
import sys
from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from pandas.util import hash_pandas_object

#Get branch name for the comet tag
git_branch=sys.argv[1]
git_commit=sys.argv[2]

NEON_micro = []
NEON_macro = []

IFAS_micro = []
IFAS_macro = []

client = start_cluster.start(cpus=5, mem_size="8GB")    
for x in range(5):
    #Create datamodule
    config = data.read_config("config.yml")
    comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=config["comet_workspace"], auto_output_logging="simple")    
    
    #Generate new data or use previous run
    if config["use_data_commit"]:
        config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
        client = None    
    else:
        crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
        os.mkdir(crop_dir)
        config["crop_dir"] = crop_dir
    
    comet_logger.experiment.log_parameter("git branch",git_branch)
    comet_logger.experiment.add_tag(git_branch)
    comet_logger.experiment.log_parameter("commit hash",git_commit)
    comet_logger.experiment.log_parameters(config)
    
    data_module = data.TreeData(
        csv_file="data/raw/neon_vst_data_2022.csv",
        data_dir=config["crop_dir"],
        config=config,
        client=client,
        metadata=True,
        comet_logger=comet_logger)
    
    data_module.setup()
    
    comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
    comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
    comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
    comet_logger.experiment.log_table("train.csv", data_module.train)
    comet_logger.experiment.log_table("test.csv", data_module.test)
    
    if not config["use_data_commit"]:
        comet_logger.experiment.log_table("novel_species.csv", data_module.novel)
    
    #Load from state dict of previous run
    if config["pretrain_state_dict"]:
        model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=data_module.num_classes, bands=config["bands"])
    else:
        model = Hang2020.spectral_network(bands=config["bands"], classes=data_module.num_classes)
        
    #Load from state dict of previous run
    
    #Loss weight, balanced
    loss_weight = []
    for x in data_module.species_label_dict:
        loss_weight.append(1/data_module.train[data_module.train.taxonID==x].shape[0])
        
    loss_weight = np.array(loss_weight/np.max(loss_weight))
    
    comet_logger.experiment.log_parameter("loss_weight", loss_weight)
    
    m = main.TreeModel(
        model=model, 
        classes=data_module.num_classes, 
        loss_weight=loss_weight,
        label_dict=data_module.species_label_dict)
    
    #Create trainer
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        gpus=data_module.config["gpus"],
        fast_dev_run=data_module.config["fast_dev_run"],
        max_epochs=data_module.config["epochs"],
        accelerator=data_module.config["accelerator"],
        checkpoint_callback=False,
        callbacks=[lr_monitor],
        logger=comet_logger)
    
    trainer.fit(m, datamodule=data_module)
    #Save model checkpoint
    trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}.pl".format(comet_logger.experiment.id))
    with comet_logger.experiment.context_manager("NEON"):
        results = m.evaluate_crowns(
            data_module.val_dataloader(),
            crowns = data_module.crowns,
            experiment=comet_logger.experiment,
        )
        
        
        #Get metric names
        print(comet_logger.experiment.metrics.keys())
              
        NEON_micro.append(comet_logger.experiment.get_metric("OSBS_micro"))
        NEON_macro.append(comet_logger.experiment.get_metric("OSBS_macro"))        
    
    #Get a test dataset for IFAS data
    with comet_logger.experiment.context_manager("IFAS"):
        ifas_dataset = data_module.annotations[data_module.annotations.plotID.str.contains("IFAS")].groupby("taxonID").apply(lambda x: x.sample(frac=1).head(20))
        ifas_dataset = ifas_dataset[ifas_dataset.taxonID.isin(data_module.train.taxonID)]
        ifas_dataset = ifas_dataset.reset_index(drop=True)
        ifas_dataset["label"] = ifas_dataset.taxonID.apply(lambda x: data_module.species_label_dict[x])
        #save
        ifas_dataset.to_csv(os.path.join(data_module.data_dir, "ifas_dataset.csv"))
        
        #Create dataloaders
        IFAS_ds = data.TreeDataset(
            df = ifas_dataset,
            config=data_module.config
        )
        
        data_loader = torch.utils.data.DataLoader(
            IFAS_ds,
            batch_size=data_module.config["batch_size"],
            shuffle=False,
            num_workers=data_module.config["workers"],
        )
        
        results = m.evaluate_crowns(
            data_loader=data_loader,
            crowns = data_module.crowns,
            experiment=comet_logger.experiment,
        )
        
        
        #Get metric names
        print(comet_logger.experiment.metrics.keys())
              
        IFAS_micro.append(comet_logger.experiment.get_metric("IFAS_OSBS_micro"))
        IFAS_macro.append(comet_logger.experiment.get_metric("IFAS_OSBS_macro"))  
        
    rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)
    
    #Visualizations
    visualize.plot_spectra(results, crop_dir=config["crop_dir"], experiment=comet_logger.experiment)
    visualize.rgb_plots(
        df=results,
        config=config,
        test_crowns=data_module.crowns,
        test_points=data_module.canopy_points,
        plot_n_individuals=config["plot_n_individuals"],
        experiment=comet_logger.experiment)
    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )
    
    #Log prediction
    comet_logger.experiment.log_table("test_predictions.csv", results)
    
df = pd.DataFrame({"NEON_micro":NEON_micro,"NEON_macro":NEON_macro,"IFAS_micro":IFAS_micro,"IFAS_macro":IFAS_macro})
df.to_csv("results/IFAS_predicts_NEON_OSBS.csv")