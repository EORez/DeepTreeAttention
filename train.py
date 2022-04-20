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

train_df = data_module.train.copy()
test_df = data_module.test.copy()

## Model 1 ##
#Create a list of dataloaders to traind
data_module.species_label_dict = {"PIPA2":0,"OTHER":1}
data_module.label_to_taxonID = {v: k  for k, v in data_module.species_label_dict.items()}

PIPA_split_train = train_df.copy()
PIPA_split_train.loc[~(PIPA_split_train.taxonID == "PIPA2"),"taxonID"] = "OTHER"
PIPA_split_train.label = [data_module.species_label_dict[x] for x in PIPA_split_train.taxonID]
PIPA_split_train.to_csv(os.path.join(data_module.data_dir, "PIPA2_train.csv"))
PIPA_split_test = test_df.copy()
PIPA_split_test.loc[~(PIPA_split_test.taxonID == "PIPA2"),"taxonID"] = "OTHER"
PIPA_split_test.label = [data_module.species_label_dict[x] for x in PIPA_split_test.taxonID]
PIPA_split_test.to_csv(os.path.join(data_module.data_dir, "PIPA2_test.csv"))

data_module.train_ds = data.TreeDataset(os.path.join(data_module.data_dir, "PIPA2_train.csv"), config=config)
data_module.val_ds = data.TreeDataset(os.path.join(data_module.data_dir, "PIPA2_test.csv"), config=config)

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
    #trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_PIPA.pl".format(comet_logger.experiment.id))
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
data_module.species_label_dict = {"CONIFER":0,"BROADLEAF":1}
conifer_train = train_df.copy()
conifer_train = conifer_train[~(conifer_train.taxonID=="PIPA2")]
conifer_train.loc[~conifer_train.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"
conifer_train.loc[conifer_train.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER"
conifer_train.label = [data_module.species_label_dict[x] for x in conifer_train.taxonID]
conifer_train.to_csv(os.path.join(data_module.data_dir, "conifer_train.csv"))

conifer_test = test_df.copy()
conifer_test = conifer_test[~(conifer_test.taxonID=="PIPA2")]
conifer_test.loc[~conifer_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"
conifer_test.loc[conifer_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER"
conifer_test.label = [data_module.species_label_dict[x] for x in conifer_test.taxonID]
conifer_test.to_csv(os.path.join(data_module.data_dir, "conifer_test.csv"))

data_module.train_ds = data.TreeDataset(os.path.join(data_module.data_dir, "conifer_train.csv"), config=config)
data_module.val_ds = data.TreeDataset(os.path.join(data_module.data_dir, "conifer_test.csv"), config=config)

#Load from state dict of previous run
if config["pretrain_state_dict"]:
    model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=len(data_module.species_label_dict), bands=config["bands"])
else:
    model = Hang2020.spectral_network(bands=config["bands"], classes=len(data_module.species_label_dict))

#Loss weight, balanced
loss_weight = []
for x in data_module.species_label_dict:
    loss_weight.append(1/conifer_train[conifer_train.taxonID==x].shape[0])
loss_weight = np.array(loss_weight/np.max(loss_weight))

m2 = main.TreeModel(
    model=model, 
    loss_weight=loss_weight,
    classes=len(data_module.species_label_dict),
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

with comet_logger.experiment.context_manager("conifer"):
    trainer.fit(m2, datamodule=data_module)

    #Save model checkpoint
    #trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_not_OAK.pl".format(comet_logger.experiment.id))
    results2 = m2.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment,
    )
    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results2,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )    

## MODEL 3 ##
broadleaf = [x for x in list(original_label_dict.keys()) if not x in ["PICL","PIEL","PITA","PIPA2"]]
data_module.species_label_dict = {v:k for k, v in enumerate(broadleaf)}
data_module.label_to_taxonID = {v:k for k, v in data_module.species_label_dict.items()}

broadleaf_train = train_df.copy()
broadleaf_train = broadleaf_train[~broadleaf_train.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])]
broadleaf_train.label = [data_module.species_label_dict[x] for x in broadleaf_train.taxonID]
broadleaf_train.to_csv(os.path.join(data_module.data_dir, "broadleaf_train.csv"))

broadleaf_test = test_df.copy()
broadleaf_test = broadleaf_test[~broadleaf_test.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])]
broadleaf_test.label = [data_module.species_label_dict[x] for x in broadleaf_test.taxonID]
broadleaf_test.to_csv(os.path.join(data_module.data_dir, "broadleaf_test.csv"))

data_module.train_ds = data.TreeDataset(os.path.join(data_module.data_dir, "broadleaf_train.csv"), config=config)
data_module.val_ds = data.TreeDataset(os.path.join(data_module.data_dir, "broadleaf_test.csv"), config=config)

#Load from state dict of previous run
if config["pretrain_state_dict"]:
    model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=len(data_module.species_label_dict), bands=config["bands"])
else:
    model = Hang2020.spectral_network(bands=config["bands"], classes=len(data_module.species_label_dict))

#Loss weight, balanced
loss_weight = []
for x in data_module.species_label_dict:
    loss_weight.append(1/broadleaf_train[broadleaf_train.taxonID==x].shape[0])

loss_weight = np.array(loss_weight/np.max(loss_weight))

m3 = main.TreeModel(
    model=model, 
    loss_weight=loss_weight,
    classes=len(data_module.species_label_dict),
    label_dict=data_module.species_label_dict)

#Create trainer
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs_model3"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    callbacks=[lr_monitor],
    logger=comet_logger)

with comet_logger.experiment.context_manager("Broadleaf"):
    trainer.fit(m3, datamodule=data_module)

    #Save model checkpoint
    #trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_OAK.pl".format(comet_logger.experiment.id))
    results3 = m3.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment,
    )
    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results3,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )  

## MODEL 4 ##
evergreen = [x for x in list(original_label_dict.keys()) if x in ["PICL","PIEL","PITA"]]
data_module.species_label_dict = {v:k for k, v in enumerate(evergreen)}
data_module.label_to_taxonID = {v:k for k, v in data_module.species_label_dict.items()}

evergreen_train = train_df.copy()
evergreen_train = evergreen_train[evergreen_train.taxonID.isin(["PICL","PIEL","PITA"])]
evergreen_train.label = [data_module.species_label_dict[x] for x in evergreen_train.taxonID]
evergreen_train.to_csv(os.path.join(data_module.data_dir, "evergreen_train.csv"))

evergreen_test = test_df.copy()
evergreen_test = evergreen_test[evergreen_test.taxonID.isin(["PICL","PIEL","PITA"])]
evergreen_test.label = [data_module.species_label_dict[x] for x in evergreen_test.taxonID]
evergreen_test.to_csv(os.path.join(data_module.data_dir, "evergreen_test.csv"))

data_module.train_ds = data.TreeDataset(os.path.join(data_module.data_dir, "evergreen_train.csv"), config=config)
data_module.val_ds = data.TreeDataset(os.path.join(data_module.data_dir, "evergreen_test.csv"), config=config)

#Load from state dict of previous run
if config["pretrain_state_dict"]:
    model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=len(data_module.species_label_dict), bands=config["bands"])
else:
    model = Hang2020.spectral_network(bands=config["bands"], classes=len(data_module.species_label_dict))

#Loss weight, balanced
loss_weight = []
for x in data_module.species_label_dict:
    loss_weight.append(1/evergreen_train[evergreen_train.taxonID==x].shape[0])

loss_weight = np.array(loss_weight/np.max(loss_weight))
 
m4 = main.TreeModel(
    model=model, 
    loss_weight=loss_weight,
    classes=len(data_module.species_label_dict),
    label_dict=data_module.species_label_dict)

#Create trainer
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs_model3"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    callbacks=[lr_monitor],
    logger=comet_logger)

with comet_logger.experiment.context_manager("Broadleaf"):
    trainer.fit(m4, datamodule=data_module)

    #Save model checkpoint
    #trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_OAK.pl".format(comet_logger.experiment.id))
    results4 = m4.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment,
    )
    
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results4,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )  
        
#Get joint scores
new_label_dict = original_label_dict.copy()
new_label_dict["OTHER"] = len(new_label_dict)
results = results[results.taxonID=="PIPA2"]
joint_results = pd.concat([results, results3, results4])
joint_results["joint_results.pred_label_top1"] = [new_label_dict[x] for x in joint_results.pred_taxa_top1]

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
    {"taxonID":list(original_label_dict.keys()),
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

