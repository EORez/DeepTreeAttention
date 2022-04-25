#Test data module
from src import data
import pandas as pd

def test_TreeData_setup(config, ROOT):
    #One site's worth of data
    config["use_data_commit"] = "2fc85646a96f4570942783aa8897bdae" 
    dm = data.TreeData(config=config, csv_file=None, data_dir="{}/tests/data/2fc85646a96f4570942783aa8897bdae".format(ROOT), debug=True) 
    dm.setup()  

    assert not dm.test.empty
    assert not dm.train.empty
    assert not any([x in dm.train.image_path.unique() for x in dm.test.image_path.unique()])
    assert all([x in ["image_path","label",
                      "site","taxonID",
                      "siteID","plotID",
                      "individualID","point_id","box_id","RGB_tile"
                      ] for x in dm.train.columns])
    
def test_TreeDataset(config, ROOT):
    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/2fc85646a96f4570942783aa8897bdae/train.csv".format(ROOT), config=config)
    individuals, inputs, label = data_loader[0]
    image = inputs["HSI"]
    assert image.shape == (349, config["image_size"], config["image_size"])
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/2fc85646a96f4570942783aa8897bdae/test.csv".format(ROOT), train=False, config=config)    
    annotations = pd.read_csv("{}/tests/data/2fc85646a96f4570942783aa8897bdae/test.csv".format(ROOT))
    
    assert len(data_loader) == annotations.shape[0]
