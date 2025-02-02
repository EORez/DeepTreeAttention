#crop random dataset
import glob
from src.data import read_config
import os
from src import neon_paths
from src.start_cluster import start
import rasterio
import random
import re
import numpy as np
from rasterio.windows import Window
from distributed import wait
import pandas as pd

client = start(cpus=200, mem_size = "20GB")
def crop(bounds, sensor_path, savedir = None, basename = None):
    """Given a 4 pointed bounding box, crop sensor data"""
    left, bottom, right, top = bounds 
    src = rasterio.open(sensor_path)        
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)) 
    res = src.res[0]
    height = (top - bottom)/res
    width = (right - left)/res      
    if savedir:
        profile = src.profile
        profile.update(height=height, width=width)
        filename = "{}/{}.tif".format(savedir, basename)
        with rasterio.open(filename, "w",**profile) as dst:
            dst.write(img)
    if savedir:
        return filename
    else:
        return img 
    
def random_crop(rgb_pool, hsi_pool, CHM_pool, config, iteration):  
    #Choose random tile
    geo_index = re.search("(\d+_\d+)_image", os.path.basename(random.choice(rgb_pool))).group(1)
    rgb_tiles = [x for x in rgb_pool if geo_index in x]
    if len(rgb_tiles) < 3:
        return None
    chm_tiles = [x for x in CHM_pool if geo_index in x]
    if len(chm_tiles) < 3:
        return None    
    if len([x for x in hsi_pool if geo_index in x]) < 3:
        return None
    #Get .tif from the .h5
    hsi_tifs = neon_paths.lookup_and_convert(rgb_pool=rgb_pool, hyperspectral_pool=hsi_pool, savedir=config["HSI_tif_dir"], geo_index=geo_index, all_years=True)           
    hsi_tifs = [x for x in hsi_tifs if not "neon-aop-products" in x]
    #year of each tile
    rgb_years = [neon_paths.year_from_tile(x) for x in rgb_tiles]
    hsi_years = [os.path.splitext(os.path.basename(x))[0].split("_")[-1] for x in hsi_tifs]
    chm_years = [neon_paths.year_from_tile(x) for x in chm_tiles]
    #Years in common
    selected_years = list(set(rgb_years) & set(hsi_years) & set(chm_years))
    selected_years = [x for x in selected_years if int(x) > 2017]
    selected_years.sort()
    selected_years = selected_years[-3:]
    if len(selected_years) < 3:
        print("not enough years")
        return None
    rgb_index = [index for index, value in enumerate(rgb_years) if value in selected_years]
    selected_rgb = [x for index, x in enumerate(rgb_tiles) if index in rgb_index]
    hsi_index = [index for index, value in enumerate(hsi_years) if value in selected_years]
    selected_hsi = [x for index, x in enumerate(hsi_tifs) if index in hsi_index]
    chm_index = [index for index, value in enumerate(chm_years) if value in selected_years]
    selected_chm = [x for index, x in enumerate(chm_tiles) if index in chm_index]
    if not all(np.array([len(selected_chm), len(hsi_tifs), len(selected_rgb)]) == [3,3,3]):
        print("Not enough years")
        return None
    #Get window, mask out black areas
    src = rasterio.open(selected_rgb[0])   
    mask = src.read_masks(1)
    coordx = np.argwhere(mask==255)
    #Get random coordiante
    xsize, ysize = 640, 640
    random_index = random.randint(0, coordx.shape[0])
    xmin, ymin = coordx[random_index,:]
    window = Window(xmin, ymin, xsize, ysize)
    bounds = rasterio.windows.bounds(window, src.transform)
    #crop rgb
    for tile in selected_rgb:
        crop(bounds=bounds, sensor_path= tile,
             savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/RGB/",
             basename="{}_{}".format(os.path.splitext(os.path.basename(tile))[0],iteration))
        
    for index, tile in enumerate(selected_chm):
        crop(bounds=bounds, sensor_path= tile,
             savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/CHM/",
             basename="{}_{}".format(os.path.splitext(os.path.basename(selected_rgb[index]))[0],iteration))
    #HSI
    for index, tile in enumerate(hsi_tifs):
        crop(bounds=bounds, sensor_path=tile,
             savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/HSI/",
             basename="{}_{}".format(os.path.splitext(os.path.basename(selected_rgb[index]))[0],iteration))

config = read_config("config.yml")    
rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
rgb_pool = [x for x in rgb_pool if not "classified" in x]
hsi_pool = glob.glob(config["HSI_sensor_pool"], recursive=True)
CHM_pool = glob.glob(config["CHM_pool"], recursive=True)
futures = []
for x in range(300):
    future = client.submit(random_crop, rgb_pool=rgb_pool, hsi_pool=hsi_pool, CHM_pool=CHM_pool, config=config, iteration=x)
    futures.append(future)

wait(futures)

# post process cleanup

files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/**/*.tif",recursive=True)
counts = pd.DataFrame({"basename":[os.path.basename(x) for x in files],"path":files}) 
less_than_3 = counts.basename.value_counts()
to_remove = less_than_3[less_than_3 < 3].index
for x in counts[counts.basename.isin(to_remove)].path:
    os.remove(x)
