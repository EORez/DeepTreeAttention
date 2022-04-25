#Test multi_stage
from pytorch_lightning import Trainer
from src.models import multi_stage
import torch
import pandas as pd
import numpy as np

def test_MultiStage(dm, config):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, config=config)
    image = torch.randn(20, 349, 110, 110)    
    for x in range(5):
        with torch.no_grad(): 
            output = m.models[x](image)
    
    train_dict = m.train_dataloader()
    assert len(train_dict) == 5
    
def test_fit(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m)
    
def test_predict(config, dm):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config)
    trainer = Trainer(fast_dev_run=True, limit_predict_batches=2)
    predictions = trainer.predict(m, dataloaders=m.val_dataloader())

def test_evaluate_crowns(config, dm, comet_logger):
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.train, config=config)
    trainer = Trainer(fast_dev_run=True)
    predictions = trainer.predict(m, dataloaders=m.val_dataloader())    
