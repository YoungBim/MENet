import numpy as np
import os
from PIL import Image


# This is given by the dataset
CityScapeClasses = \
    {
        'unlabeled'             :  0
        ,'ego vehicle'          :  1
        ,'rectification border' :  2
        ,'out of roi'           :  3
        ,'static'               :  4
        ,'dynamic'              :  5
        ,'ground'               :  6
        ,'road'                 :  7
        ,'sidewalk'             :  8
        ,'parking'              :  9
        ,'rail track'           : 10
        ,'building'             : 11
        ,'wall'                 : 12
        ,'fence'                : 13
        ,'guard rail'           : 14
        ,'bridge'               : 15
        ,'tunnel'               : 16
        ,'pole'                 : 17
        ,'polegroup'            : 18
        ,'traffic light'        : 19
        ,'traffic sign'         : 20
        ,'vegetation'           : 21
        ,'terrain'              : 22
        ,'sky'                  : 23
        ,'person'               : 24
        ,'rider'                : 25
        ,'car'                  : 26
        ,'truck'                : 27
        ,'bus'                  : 28
        ,'caravan'              : 29
        ,'trailer'              : 30
        ,'train'                : 31
        ,'motorcycle'           : 32
        ,'bicycle'              : 33
    }

# Manual mapping between cityscapes and camvid
CamVidClasses = \
    {
        'unlabeled'             : 11
        ,'ego vehicle'          : 11
        ,'rectification border' : 11
        ,'out of roi'           : 11
        ,'static'               : 11
        ,'dynamic'              : 11
        ,'ground'               : 11
        ,'road'                 : 3
        ,'sidewalk'             : 4
        ,'parking'              : 3
        ,'rail track'           : 11
        ,'building'             : 1
        ,'wall'                 : 1
        ,'fence'                : 7
        ,'guard rail'           : 7
        ,'bridge'               : 11
        ,'tunnel'               : 11
        ,'pole'                 : 2
        ,'polegroup'            : 2
        ,'traffic light'        : 6
        ,'traffic sign'         : 6
        ,'vegetation'           : 5
        ,'terrain'              : 5
        ,'sky'                  : 0
        ,'person'               : 9
        ,'rider'                : 10
        ,'car'                  : 8
        ,'truck'                : 8
        ,'bus'                  : 8
        ,'caravan'              : 8
        ,'trailer'              : 8
        ,'train'                : 11
        ,'motorcycle'           : 10
        ,'bicycle'              : 10
    }

# Build the LUT CityScapes -> CamVid
LUT_citySacpes2Camvid = np.zeros(shape=[len(CityScapeClasses.keys())])
for CityScapeKey in CityScapeClasses.keys():
    LUT_citySacpes2Camvid[CityScapeClasses[CityScapeKey]] = CamVidClasses[CityScapeKey]

# Convert an image and write it
def mapCityscapes2Camvid(filepath, convertPath):
    img = Image.open(filepath)
    img_array = np.array(img)
    img_array = LUT_citySacpes2Camvid[img_array]
    filename = os.path.basename(filepath)
    img = Image.fromarray(img_array)
    img.save(os.path.join(convertPath,filename),'PNG')

# Parse the image folder and run the conversion for every PNG file
def mapCityscapeDataset(path):
    pngfiles = np.array([os.path.join(root, name).replace('\\', '/')
                         for root, _, files in os.walk(path, followlinks=False)
                         for name in files
                         if name.endswith(".png")])

    convertPath = path+'_CamvidLabs'
    if not os.path.exists(convertPath):
        os.makedirs(convertPath)

    for filepath in pngfiles:
        mapCityscapes2Camvid(filepath, convertPath)
    pass

if __name__ == '__main__':
    CityScapePath = 'D:/Datasets/Cityscapes/segmentation/val'
    mapCityscapeDataset(CityScapePath)