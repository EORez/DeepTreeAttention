#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
import os
import re
import glob
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.utils import multi_gpu_model
from sklearn.utils import class_weight

#Local Modules
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.models import Hang2020_geographic as Hang
from  DeepTreeAttention.models import layers
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.callbacks import callbacks


class AttentionModel():
    """The main class holding train, predict and evaluate methods"""

    def __init__(self, config="conf/config.yml", saved_model=None, log_dir=None):
        """
        Args:
            config: path to a config file, defaults to ../conf/config.yml
            saved_model: Optional, a previous saved AttentionModel .h5
        """
        self.config = parse_yaml(config)
        if saved_model:
            self.model = load_model(saved_model)

        #Holders
        self.testing_set = None
        self.training_set = None

        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = None

        #log some helpful data
        
        self.HSI_size = self.config["train"]["HSI"]["crop_size"]
        self.HSI_channels = self.config["train"]["HSI"]["sensor_channels"]
        self.HSI_weighted_sum = self.config["train"]["HSI"]["weighted_sum"]
        
        self.RGB_size= self.config["train"]["RGB"]["crop_size"]
        self.RGB_channels = self.config["train"]["RGB"]["sensor_channels"]
        self.RGB_weighted_sum = self.config["train"]["RGB"]["weighted_sum"]
        
        self.extend_box = self.config["train"]["extend_box"]
        self.classes_file = os.path.join(self.config["train"]["tfrecords"],
                                         "species_class_labels.csv")
        self.classes = self.config["train"]["classes"]
        self.sites = self.config["train"]["sites"]

    def generate(self, shapefile, HSI_sensor_path, RGB_sensor_path, elevation, site, species_label_dict=None, train=True, chunk_size=1000):
        """Predict species class for each DeepForest bounding box
            Args:
                shapefile: a DeepForest shapefile (see NeonCrownMaps) with a bounding box and utm projection
                train: generate a training record that yields, image, label, or a prediction record with metadata? Default True
                site: site metadata label
                sensor_path: supply a known path to a sensor geoTIFF tile. 
                chunk_size: number of crops per tfrecord
            """
        #set savedir
        if train:
            savedir = self.config["train"]["tfrecords"]
        else:
            savedir = self.config["predict"]["tfrecords"]

        created_records = boxes.generate_tfrecords(shapefile=shapefile,
                                                   HSI_sensor_path=HSI_sensor_path,
                                                   RGB_sensor_path=RGB_sensor_path,                                                   
                                                   site=site,
                                                   elevation=elevation,
                                                   species_label_dict=species_label_dict,
                                                   HSI_size=self.HSI_size,
                                                   RGB_size=self.RGB_size,                                                   
                                                   savedir=savedir,
                                                   train=train,
                                                   number_of_sites=self.sites,
                                                   classes=self.classes,
                                                   chunk_size=chunk_size,
                                                   extend_box=self.extend_box,
                                                   shuffle=True)

        return created_records

    def calc_class_weight(self):
        """Get class frequency of labels"""

        #Check if train_split has been create
        if not hasattr(self, "train_split"):
            raise ValueError(
                "No training split created, please call DeepTreeAttention.read_data()")

        labels = []
        for image, label in self.train_split:
            labels.append(label)

        #Convert from one_hot
        labels = np.vstack(labels)
        labels = np.argmax(labels, 1)

        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels),
                                                          labels)

        return class_weights

    def create(self, weights=None, submodel=None):
        """Load a model
            Args:
                weights: a saved model weights from previous run
                name: a model name from DeepTreeAttention.models
            """
        self.HSI_model, self.HSI_spatial, self.HSI_spectral = Hang.create_models(self.HSI_size, self.HSI_size, self.HSI_channels, self.classes, self.config["train"]["learning_rate"])
        self.RGB_model, self.RGB_spatial, self.RGB_spectral = Hang.create_models(self.RGB_size, self.RGB_size, self.RGB_channels, self.classes, self.config["train"]["learning_rate"])

    def read_data(self, mode="train", validation_split=False):
        """Read tfrecord into datasets from config
            Args:
                validation_split: True -> split tfrecords into train test. This overrides the evaluation config!
            """
        self.train_records = glob.glob(
            os.path.join(self.config["train"]["tfrecords"], "*.tfrecord"))

        if len(self.train_records) == 0:
            raise IOError("Cannot find .tfrecords at {}".format(
                self.config["train"]["tfrecords"]))

        if validation_split:
            print("Splitting training set into train-test")
            train_df = pd.Series(self.train_records)
            #Sample with set seed to make it the same between runs
            self.train_split_records = train_df.head(
                int(self.config["train"]["training_fraction"] * train_df.shape[0])).values
            self.test_split_records = train_df[~(
                train_df.isin(self.train_split_records))].values

            #Create training tf.data
            self.train_split = boxes.tf_dataset(
                tfrecords=self.train_split_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"],
                mode=mode,
                cores=self.config["cpu_workers"])

            #Create testing tf.data
            self.val_split = boxes.tf_dataset(
                tfrecords=self.test_split_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"],
                mode=mode,
                cores=self.config["cpu_workers"])
        else:
            #Create training tf.data
            self.train_split = boxes.tf_dataset(
                tfrecords=self.train_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"],
                mode=mode,
                cores=self.config["cpu_workers"])

            #honor config if validation not set
            self.val_split = None
            if self.config["evaluation"]["tfrecords"] is not None:
                self.test_records = glob.glob(
                    os.path.join(self.config["evaluation"]["tfrecords"], "*.tfrecord"))

                self.val_split = boxes.tf_dataset(
                    tfrecords=self.test_records,
                    batch_size=self.config["train"]["batch_size"],
                    shuffle=self.config["train"]["shuffle"],
                    mode=mode,
                    cores=self.config["cpu_workers"])

    def train(self, experiment=None, class_weight=None, submodel=None, sensor="hyperspectral"):
        """Train a model with callbacks"""

        if self.val_split is None:
            print("Cannot run callbacks without validation data, skipping...")
            callback_list = None
        elif experiment is None:
            print("Cannot run callbacks without comet experiment, skipping...")
            callback_list = None
        else:            
            if self.classes_file is not None:
                labeldf = pd.read_csv(self.classes_file)                
                label_names = list(labeldf.taxonID.values)
            else:
                label_names = None
                
            callback_list = callbacks.create(log_dir=self.log_dir,
                                             experiment=experiment,
                                             validation_data=self.val_split,
                                             train_data=self.train_split,
                                             label_names=label_names,
                                             submodel=submodel)
        
        if submodel == "spatial":
            if sensor == "hyperspectral":
                self.HSI_spatial.fit(self.train_split,
                                       epochs=int(self.config["train"]["epochs"]),
                                       validation_data=self.val_split,
                                       callbacks=callback_list,
                                       class_weight=class_weight)
            elif sensor == "RGB":
                self.RGB_spatial.fit(self.train_split,
                                                 epochs=int(self.config["train"]["epochs"]),
                                                   validation_data=self.val_split,
                                                   callbacks=callback_list,
                                                   class_weight=class_weight)                

        elif submodel == "spectral":
            if sensor == "hyperspectral":
                self.HSI_spectral.fit(self.train_split,
                                       epochs=int(self.config["train"]["epochs"]),
                                       validation_data=self.val_split,
                                       callbacks=callback_list,
                                       class_weight=class_weight)
            elif sensor == "RGB":
                self.RGB_spectral.fit(self.train_split,
                                                 epochs=int(self.config["train"]["epochs"]),
                                                   validation_data=self.val_split,
                                                   callbacks=callback_list,
                                                   class_weight=class_weight)      
        else:
            if sensor == "hyperspectral":
                self.HSI_model.fit(self.train_split,
                               epochs=self.config["train"]["epochs"],
                               validation_data=self.val_split,
                               callbacks=callback_list,
                               class_weight=class_weight)
            
            elif sensor == "RGB":
                self.RGB_model.fit(
                    self.train_split,
                    epochs=self.config["train"]["epochs"],
                    validation_data=self.val_split,
                    callbacks=callback_list,
                    class_weight=class_weight)
        
    def ensemble(self, freeze = True):
        self.read_data(mode="ensemble")
        self.ensemble = Hang.ensemble([self.RGB_model, self.HSI_model], freeze=freeze)
        
        self.ensemble.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(
            lr=float(self.config["train"]["learning_rate"])),
            metrics="acc")
            
        callback_list = callbacks.create(log_dir=self.log_dir,
                                         experiment=experiment,
                                         validation_data=self.val_split,
                                         train_data=self.train_split,
                                         label_names=label_names)
                
        #Train ensemble layer
        self.ensemble.fit(
            self.train_split,
            epochs=self.config["train"]["epochs"],
            validation_data=self.val_split,
            callbacks=callback_list,
            class_weight=class_weight)
        
    def predict(self, shapefile, savedir, create_records=True, sensor_path=None):
        """Predict species id for each box in a single shapefile
        Args:
            shapefile: path to a shapefile
            record_dirname: directory to save generated records
            create_records: overwrite previous records
        Returns:
            fname: path to predicted shapefile
        """
        if create_records:
            created_records = boxes.generate(shapefile,
                                             sensor_path=sensor_path,
                                             savedir=self.config["predict"]["savedir"],
                                             height=self.height,
                                             width=self.width,
                                             classes=self.classes,
                                             train=False)
        else:
            created_records = glob.glob(dirname + "*.tfrecord")

        #Merge with original box shapefile by index and write new shapefile to file
        results = self.predict_boxes(created_records)
        fname = self.merge_shapefile(shapefile, results, savedir=savedir)

        return fname

    def predict_boxes(self, tfrecords, batch_size=1):
        """Predicted a set of tfrecords and create a raster image"""
        prediction_set = boxes.tf_dataset(tfrecords=tfrecords,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          mode="predict",
                                          cores=self.config["cpu_workers"])

        predictions = []
        indices = []
        for image, box_index in prediction_set:
            try:
                softmax_batch = self.model.predict_on_batch(image)
                predictions.append(softmax_batch)
                indices.append(box_index)
            except tf.errors.OutOfRangeError:
                print("Completed {} predictions".format(len(predictions)))

        #stack
        predictions = np.vstack(predictions)
        predictions = np.argmax(predictions, 1)

        indices = np.concatenate(indices)

        #Read class labels
        labeldf = pd.read_csv(self.classes_file)
        labels = [
            labeldf.loc[labeldf.index == x, "taxonID"].values[0] for x in predictions
        ]
        results = pd.DataFrame({"label": labels, "box_index": indices})

        #decode results
        results["box_index"] = results["box_index"].apply(lambda x: x.decode()).astype(
            str)

        return results

    def merge_shapefile(self, shapefile, results, savedir):
        """Merge predicted species label with box id"""

        gdf = geopandas.read_file(shapefile)

        #Make sure there isn't a label column in merge data
        gdf = gdf.drop(columns="label")
        basename = os.path.splitext(os.path.basename(shapefile))[0]
        gdf["box_index"] = ["{}_{}".format(basename, x) for x in gdf.index.values]

        #Merge
        joined_gdf = gdf.merge(results, on="box_index")
        fname = "{}/{}.shp".format(savedir, basename)
        joined_gdf.to_file(fname)

        return fname

    def evaluate(self, tf_dataset):
        """Evaluate metrics on held out training data. Defaults to reading from config.yml evaluation sensor path
        Args: 
            tf_dataset: Optional a tf.dataset that yields data and labels, see make_dataset.py 
            steps: Optional, how many calls of the genertor to evaluate. None will evaluate until exhausted
        Returns:
            results: a dictionary of metrics
        """
        #gather y_true
        labels = []
        predictions = []
        for image, label in tf_dataset:
            try:
                softmax_batch = self.model.predict_on_batch(image)
                one_hot_label = label.numpy()
                predictions.append(softmax_batch)
                labels.append(label)
            except tf.errors.OutOfRangeError:
                print("Completed {} predictions".format(len(predictions)))

        #Create numpy arrays of batches
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)

        return predictions, labels
