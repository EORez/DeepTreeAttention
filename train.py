#Train
import comet_ml
import glob
import geopandas as gpd
import os
import numpy as np
from src import main
from src import data
from src import start_cluster
from src.models import Hang2020
from src import visualize
from src import metrics
from src import utils
import sys
import torch
import torchmetrics
from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from pandas.util import hash_pandas_object
import copy

#Get branch name for the comet tag
git_branch=sys.argv[1]
git_commit=sys.argv[2]

#Create datamodule
config = data.read_config("config.yml")
comet_logger = CometLogger(project_name="DeepTreeAttention", workspace=config["comet_workspace"], auto_output_logging="simple")    

#Generate new data or use previous run
if config["use_data_commit"]:
    config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
    client = None    
    model_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key()) 
    os.mkdir(model_dir)
else:
    crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
    os.mkdir(crop_dir)
    client = start_cluster.start(cpus=100, mem_size="4GB")    
    config["crop_dir"] = crop_dir
    model_dir = crop_dir

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
if client:
    client.close()

comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
comet_logger.experiment.log_table("train.csv", data_module.train)
comet_logger.experiment.log_table("test.csv", data_module.test)

if not config["use_data_commit"]:
    comet_logger.experiment.log_table("novel_species.csv", data_module.novel)

#Load from state dict of previous run

#Load from state dict of previous run
if config["pretrain_state_dict"]:
    model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=data_module.num_classes, bands=config["bands"])
else:
    model = Hang2020.spectral_network(bands=config["bands"], classes=data_module.num_classes)

#Loss weight, balanced
loss_weight = []
for x in data_module.species_label_dict:
    loss_weight.append(1/data_module.train[data_module.train.taxonID==x].shape[0])
    
loss_weight = np.array(loss_weight/np.max(loss_weight))
#Provide min value
loss_weight[loss_weight < 0.5] = 0.5  

comet_logger.experiment.log_parameter("loss_weight", loss_weight)

year_model = {}
year_results = []
year_individuals = {}
train_year_individuals = {}
years = data_module.train.tile_year.unique()
for x in years:
    with comet_logger.experiment.context_manager(x):
        
        #Set the dataloaders by year
        data_module.train_ds = data.TreeDataset(
            csv_file = "{}/train.csv".format(data_module.data_dir),
            config=data_module.config,
            year=x,
        )
        
        data_module.val_ds = data.TreeDataset(
            csv_file = "{}/test.csv".format(data_module.data_dir),
            config=data_module.config,
            year=x
        )
        
        year_model[x] = main.TreeModel(
            model=copy.deepcopy(model), 
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
    
        trainer.fit(year_model[x], datamodule=data_module)
        
        #Save model checkpoint
        trainer.save_checkpoint(os.path.join(model_dir,"{}_{}.pl".format(comet_logger.experiment.id, x)))
        
        #Get train features
        train_results, train_features = year_model[x].predict_dataloader(data_module.train_dataloader(), return_features=True)   
        for index, row in enumerate(train_features):
            try:
                train_year_individuals[train_results.individual.iloc[index]].append(row)
            except:
                train_year_individuals[train_results.individual.iloc[index]] = [row]
        
        #Get test features
        results, features = year_model[x].predict_dataloader(
            data_loader=data_module.val_dataloader(),
            experiment=None,
            return_features=True
        )
        
        for index, row in enumerate(features):
            try:
                year_individuals[results.individual.iloc[index]].append(row)
            except :
                year_individuals[results.individual.iloc[index]] = [row]
                
        results = year_model[x].evaluate_crowns(
            data_module.val_dataloader(),
            crowns = data_module.crowns,
            experiment=comet_logger.experiment,
            points=data_module.canopy_points
        )
        rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)
        results["year"] = x
        year_results.append(results)
        
        #Confusion matrix all years
        visualize.confusion_matrix(
            comet_experiment=comet_logger.experiment,
            results=results,
            species_label_dict=data_module.species_label_dict,
            test_crowns=data_module.crowns,
            test=data_module.test,
            test_points=data_module.canopy_points,
            rgb_pool=rgb_pool
        )

results = pd.concat(year_results)

#Train meta-learner        
ensemble_model = year.year_ensemble(train_dict=train_year_individuals,
                           train_labels=train_results.label,
                           val_dict=year_individuals,
                           val_labels=results.label,
                           config=config,
                           classes=len(results.label.unique()),
                           years=years)

predicted_label, score = year.run_ensemble(ensemble_model)
results["temporal_label_top1"] = predicted_label
results["temporal_top1_score"] = score

#Log prediction
comet_logger.experiment.log_table("test_predictions.csv", results)

#Confusion matrix
temporal_only = results.groupby("individual").apply(lambda x: x.head(1)).reset_index(drop=True)
temporal_only["pred_taxa_top1"] = temporal_only["temporal_label_top1"].apply(lambda x: data_module.label_to_taxonID[x]) 
temporal_only["pred_label_top1"] = temporal_only["temporal_label_top1"]

#Temporal function
temporal_micro = torchmetrics.functional.accuracy(preds=torch.tensor(temporal_only.temporal_pred_label_top1.values),target=torch.tensor(temporal_only.label.values), num_classes=data_module.num_classes,average="micro")
temporal_macro = torchmetrics.functional.precision(preds=torch.tensor(temporal_only.temporal_pred_label_top1.values),target=torch.tensor(temporal_only.label.values), num_classes=data_module.num_classes, average="macro")

comet_logger.experiment.log_metric("temporal_micro",temporal_micro)
comet_logger.experiment.log_metric("temporal_macro",temporal_macro)

visualize.confusion_matrix(
    comet_experiment=comet_logger.experiment,
    results=temporal_only,
    species_label_dict=data_module.species_label_dict,
    test_crowns=data_module.crowns,
    test=data_module.test,
    test_points=data_module.canopy_points,
    rgb_pool=rgb_pool
)

#Within site confusion
site_lists = data_module.train.groupby("label").site.unique()
within_site_confusion = metrics.site_confusion(y_true = temporal_only.label, y_pred = temporal_only.pred_label_top1, site_lists=site_lists)
comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)

#Within plot confusion
plot_lists = data_module.train.groupby("label").plotID.unique()
within_plot_confusion = metrics.site_confusion(y_true = temporal_only.label, y_pred = temporal_only.pred_label_top1, site_lists=plot_lists)
comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)

