#Test year model
from src.models import year
from pytorch_lightning import Trainer
def test_year_ensemble(m, dm, config):
    results, features = m.predict_dataloader(dm.val_dataloader(), return_features=True)   
    year_individuals = {}
    for index, row in enumerate(features):
        try:
            year_individuals[results.individual.iloc[index]].append(row)
        except:
            year_individuals[results.individual.iloc[index]] = [row]
            
    model = year.year_ensemble(train_dict=year_individuals,
                               train_labels=results.label,
                               val_dict=year_individuals,
                               val_labels=results.label,
                               config=config,
                               classes=len(results.label.unique()), years=2)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

def test_run_ensemble(m, config, dm):
    results, features = m.predict_dataloader(dm.val_dataloader(), return_features=True)   
    year_individuals = {}
    for index, row in enumerate(features):
        try:
            year_individuals[results.individual.iloc[index]].append(row)
        except:
            year_individuals[results.individual.iloc[index]] = [row]
            
    model = year.year_ensemble(train_dict=year_individuals,
                               train_labels=results.label,
                               val_dict=year_individuals,
                               val_labels=results.label,
                               config=config,
                               classes=len(results.label.unique()), years=2)    
    config["gpus"] = 0
    yeardf = year.run_ensemble(model, config)
    merged_df = results.merge(yeardf, on="individual")
    assert merged_df.shape[0] == results.shape[0]
    
