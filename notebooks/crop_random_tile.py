#crop random dataset
import glob
from src.data import read_config
import os
from src.patches import crop
from src import neon_paths
from src.start_cluster import start
import rasterio
import random
import re
import numpy as np
from rasterio.windows import Window
from distributed import wait

client = start(cpus=100)

def random_crop(index):  
    config = read_config("config.yml")
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
    rgb_pool = [x for x in rgb_pool if not "classified" in x]
    hsi_pool = glob.glob(config["HSI_sensor_pool"], recursive=True)
    CHM_pool = glob.glob(config["CHM_pool"], recursive=True)
    #Choose random tile
    selected_rgb = random.choice(rgb_pool)
    geo_index = re.search("(\d+_\d+)_image", os.path.basename(selected_rgb)).group(1)
    rgb_tiles = [x for x in rgb_pool if geo_index in x]
    chm_tiles = [x for x in CHM_pool if geo_index in x]
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
        return None
    rgb_index = [index for index, value in enumerate(rgb_years) if value in selected_years]
    selected_rgb = [x for index, x in enumerate(rgb_tiles) if index in rgb_index]
    hsi_index = [index for index, value in enumerate(hsi_years) if value in selected_years]
    selected_hsi = [x for index, x in enumerate(hsi_tifs) if index in hsi_index]
    chm_index = [index for index, value in enumerate(chm_years) if value in selected_years]
    selected_chm = [x for index, x in enumerate(chm_tiles) if index in chm_index]
    if not all(np.array([len(selected_chm), len(hsi_tifs), len(selected_rgb)]) == [3,3,3]):
        return None
    #Get window
    with rasterio.open(selected_rgb[0]) as src:        
        # The size in pixels of your desired window
        xsize, ysize = 640, 640
        # Generate a random window location that doesn't go outside the image
        xmin, xmax = 0, src.width - xsize
        ymin, ymax = 0, src.height - ysize
        xoff, yoff = random.randint(xmin, xmax), random.randint(ymin, ymax)
        window = Window(xoff, yoff, xsize, ysize)
        transform = src.window_transform(window)
    bounds = rasterio.windows.bounds(window, transform)
    #crop rgb
    for tile in selected_rgb:
        crop(bounds=bounds, sensor_path= tile, savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/RGB/", basename=os.path.splitext(os.path.basename(tile))[0])
    
    for index, tile in enumerate(selected_chm):
        crop(bounds=bounds, sensor_path= tile, savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/CHM/", basename=os.path.splitext(os.path.basename(selected_rgb[index]))[0])
    #HSI
    for index, tile in enumerate(hsi_tifs):
        crop(bounds=bounds, sensor_path=tile, savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/selfsupervised/HSI/", basename=os.path.splitext(os.path.basename(selected_rgb[index]))[0])

futures = []
for x in range(100):
    future = client.submit(random_crop, index=x)
    futures.append(future)

wait(futures)
