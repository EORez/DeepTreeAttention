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
import sys
import torchmetrics
import torch

from pytorch_lightning import Trainer
import subprocess
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
from pandas.util import hash_pandas_object

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
else:
    crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
    os.mkdir(crop_dir)
    client = start_cluster.start(cpus=50, mem_size="4GB")    
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
    comet_logger=comet_logger)

data_module.setup()
rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)

original_label_dict = data_module.species_label_dict.copy()
if client:
    client.close()

comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
comet_logger.experiment.log_table("train.csv", data_module.train)
comet_logger.experiment.log_table("test.csv", data_module.test)

if not config["use_data_commit"]:
    comet_logger.experiment.log_table("novel_species.csv", data_module.novel)

## Model 1 ##
#Create a list of dataloaders to traind
data_module.train_ds = data.TreeDataset(os.path.join(data_module.data_dir,"train.csv"), taxonIDs = ["PIPA2"], keep_others = True, config=config)
data_module.val_ds = data.TreeDataset(os.path.join(data_module.data_dir,"test.csv"), taxonIDs = ["PIPA2"], keep_others = True, config=config)
data_module.species_label_dict = {"PIPA2":0,"OTHER":1}
data_module.label_to_taxonID = {v: k  for k, v in data_module.species_label_dict.items()}

#Load from state dict of previous run
if config["pretrain_state_dict"]:
    model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=2, bands=config["bands"])
else:
    model = Hang2020.spectral_network(bands=config["bands"], classes=2)

#Load from state dict of previous run
m = main.TreeModel(
    model=model, 
    loss_weight=[1,1],
    classes=2, 
    label_dict=data_module.species_label_dict)

#Create trainer
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs_model1"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    callbacks=[lr_monitor],
    logger=comet_logger)

with comet_logger.experiment.context_manager("PIPA2"):
    trainer.fit(m, datamodule=data_module)

    #Save model checkpoint
    trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_PIPA.pl".format(comet_logger.experiment.id))
    results = m.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment,
    )
    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )

## MODEL 2 ##

all_but_PIPA = [x for x in list(original_label_dict.keys()) if not x == "PIPA2"]

#Create a list of dataloaders to traind
data_module.train_ds = data.TreeDataset(os.path.join(data_module.data_dir,"train.csv"), taxonIDs = all_but_PIPA, config=config)
data_module.val_ds = data.TreeDataset(os.path.join(data_module.data_dir,"test.csv"), taxonIDs = all_but_PIPA, config=config)
data_module.species_label_dict = {v:k for k, v in enumerate(all_but_PIPA)}
data_module.label_to_taxonID = {v:k for k, v in data_module.species_label_dict.items()}

#Load from state dict of previous run
if config["pretrain_state_dict"]:
    model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=len(all_but_PIPA), bands=config["bands"])
else:
    model = Hang2020.spectral_network(bands=config["bands"], classes=len(all_but_PIPA))
    
#Load from state dict of previous run
m2 = main.TreeModel(
    model=model, 
    loss_weight=[1 for x in range(len(all_but_PIPA))],
    classes=len(all_but_PIPA),
    label_dict=data_module.species_label_dict)

#Create trainer
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs_model2"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    callbacks=[lr_monitor],
    logger=comet_logger)

with comet_logger.experiment.context_manager("all_but_PIPA"):
    trainer.fit(m2, datamodule=data_module)

    #Save model checkpoint
    trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_not_PIPA.pl".format(comet_logger.experiment.id))
    results2 = m2.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment,
    )
    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )    

#Get joint scores
original_label_dict["OTHER"] = len(original_label_dict) + 1
results = results[results.taxonID=="PIPA2"]
joint_results = pd.concat([results, results2])
joint_results["joint_results.pred_label_top1"] = [original_label_dict[x] for x in joint_results.pred_taxa_top1]

final_micro = torchmetrics.functional.accuracy(
    preds=torch.tensor(joint_results.pred_label_top1.values),
    target=torch.tensor(joint_results.label.values),
    average="micro")

final_macro = torchmetrics.functional.accuracy(
    preds=torch.tensor(joint_results.pred_label_top1.values),
    target=torch.tensor(joint_results.label.values),
    average="macro",
    num_classes=data_module.num_classes+1)

comet_logger.experiment.log_metric("OSBS_micro",final_micro)
comet_logger.experiment.log_metric("OSBS_macro",final_macro)

# Log results by species
taxon_accuracy = torchmetrics.functional.accuracy(
    preds=torch.tensor(results.pred_label_top1.values),
    target=torch.tensor(results.label.values), 
    average="none", 
    num_classes=data_module.num_classes
)
taxon_precision = torchmetrics.functional.precision(
    preds=torch.tensor(joint_results.pred_label_top1.values),
    target=torch.tensor(joint_results.label.values),
    average="none",
    num_classes=data_module.num_classes
)
species_table = pd.DataFrame(
    {"taxonID":list(data_module.species_label_dict.keys()),
     "accuracy":taxon_accuracy,
     "precision":taxon_precision
     })

for key, value in species_table.set_index("taxonID").accuracy.to_dict().items():
    comet_logger.experiment.log_metric("{}_accuracy".format(key), value)

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

#Within site confusion
site_lists = data_module.train.groupby("label").site.unique()
within_site_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=site_lists)
comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)

#Within plot confusion
plot_lists = data_module.train.groupby("label").plotID.unique()
within_plot_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=plot_lists)
comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)

