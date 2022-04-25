#Test data module
from src import data
import pandas as pd

def test_TreeData_setup(config, ROOT):
    #One site's worth of data
    config["use_data_commit"] = "2fc85646a96f4570942783aa8897bdae" 
    dm = data.TreeData(config=config, csv_file=None, data_dir="{}/tests/data/2fc85646a96f4570942783aa8897bdae".format(ROOT), debug=True) 
    dm.setup()  
    
    test = pd.read_csv("{}/tests/data/2fc85646a96f4570942783aa8897bdae/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/2fc85646a96f4570942783aa8897bdae/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
    assert not any([x in train.image_path.unique() for x in test.image_path.unique()])
    assert all([x in ["image_path","label",
                      "site","taxonID",
                      "siteID","plotID",
                      "individualID","point_id","box_id","RGB_tile",
                      "level_1","taxonID_level1",
                      "level_2","taxonID_level2",
                      "level_3","taxonID_level3",
                      "level_4","taxonID_level4",
                      "level_5","taxonID_level5"
                      ] for x in train.columns])
    
def test_TreeDataset(config, ROOT):
    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/2fc85646a96f4570942783aa8897bdae/train.csv".format(ROOT), config=config)
    individuals, inputs, label = data_loader[0]
    image = inputs["HSI"]
    assert image.shape == (3, config["image_size"], config["image_size"])
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/2fc85646a96f4570942783aa8897bdae/test.csv".format(ROOT), train=False, config=config)    
    annotations = pd.read_csv("{}/tests/data/2fc85646a96f4570942783aa8897bdae/test.csv".format(ROOT))
    
    assert len(data_loader) == annotations.shape[0]
