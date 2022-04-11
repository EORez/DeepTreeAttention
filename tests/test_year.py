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
                               classes=features.shape[0], years=1)
    trainer = Trainer()
    trainer.fit(model)