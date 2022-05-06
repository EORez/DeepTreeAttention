#Train
import comet_ml
import glob
import geopandas as gpd
import os
import numpy as np
from src import main
from src import data
from src import start_cluster
from src.models import multi_stage
from src import visualize
from src import metrics
import subprocess
import sys
from pytorch_lightning import Trainer
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

m = multi_stage.MultiStage(data_module.train, data_module.test, config=data_module.config, crowns=data_module.crowns)

#Create trainer
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = Trainer(
    gpus=data_module.config["gpus"],
    fast_dev_run=data_module.config["fast_dev_run"],
    max_epochs=data_module.config["epochs"],
    accelerator=data_module.config["accelerator"],
    checkpoint_callback=False,
    num_sanity_val_steps=0,
    callbacks=[lr_monitor],
    logger=comet_logger)

trainer.fit(m)
#Save model checkpoint
trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}.pl".format(comet_logger.experiment.id))

output = trainer.predict(m, dataloaders=m.predict_dataloader(df=data_module.test))
results = m.gather_predictions(predict_df=output, crowns=data_module.crowns)
ensemble_df = m.ensemble(results)
ensemble_df = m.evaluation_scores(
    ensemble_df,
    experiment=comet_logger.experiment
)

#Log prediction
comet_logger.experiment.log_table("test_predictions.csv", results)
comet_logger.experiment.log_table("ensemble_df.csv", ensemble_df)

#Visualizations
ensemble_df["pred_taxa_top1"] = ensemble_df.ensembleTaxonID
ensemble_df["pred_label_top1"] = ensemble_df.ens_label
rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)
visualize.plot_spectra(ensemble_df, crop_dir=config["crop_dir"], experiment=comet_logger.experiment)
visualize.rgb_plots(
    df=ensemble_df,
    config=config,
    test_crowns=data_module.crowns,
    test_points=data_module.canopy_points,
    plot_n_individuals=config["plot_n_individuals"],
    experiment=comet_logger.experiment)

visualize.confusion_matrix(
    comet_experiment=comet_logger.experiment,
    results=ensemble_df,
    species_label_dict=data_module.species_label_dict,
    test_crowns=data_module.crowns,
    test=data_module.test,
    test_points=data_module.canopy_points,
    rgb_pool=rgb_pool
)

#Confusion matrix for each level
for x in range(len(m.models)):
    results["pred_taxa_top1"] = results["pred_taxa_top1_level_{}".format(x)]
    results["pred_label_top1"] = results["pred_label_top1_level_{}".format(x)]
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results,
        species_label_dict=m.level_label_dicts[x],
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )

#Within site confusion
site_lists = data_module.train.groupby("label").site.unique()
within_site_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.ens_label, site_lists=site_lists)
comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)

#Within plot confusion
plot_lists = data_module.train.groupby("label").plotID.unique()
within_plot_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.ens_label, site_lists=plot_lists)
comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)

