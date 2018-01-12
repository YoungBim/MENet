import numpy as np
import os
from PIL import Image


# This is given by the dataset (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
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

# Groups
CityscapesGroups = \
    {
        'unlabeled'             : 7
        ,'ego vehicle'          : 7
        ,'rectification border' : 7
        ,'out of roi'           : 7
        ,'static'               : 7
        ,'dynamic'              : 7
        ,'ground'               : 7
        ,'road'                 : 4
        ,'sidewalk'             : 4
        ,'parking'              : 4
        ,'rail track'           : 4
        ,'building'             : 1
        ,'wall'                 : 1
        ,'fence'                : 1
        ,'guard rail'           : 1
        ,'bridge'               : 1
        ,'tunnel'               : 1
        ,'pole'                 : 2
        ,'polegroup'            : 2
        ,'traffic light'        : 2
        ,'traffic sign'         : 2
        ,'vegetation'           : 3
        ,'terrain'              : 3
        ,'sky'                  : 0
        ,'person'               : 5
        ,'rider'                : 5
        ,'car'                  : 6
        ,'truck'                : 6
        ,'bus'                  : 6
        ,'caravan'              : 6
        ,'trailer'              : 6
        ,'train'                : 6
        ,'motorcycle'           : 6
        ,'bicycle'              : 6
    }
def BuildLUT(LutName):
    # Build the LUT CityScapes -> CamVid
    LUT = np.zeros(shape=[len(CityScapeClasses.keys())])
    for CityScapeKey in CityScapeClasses.keys():
        if LutName == 'CamVid':
            LUT[CityScapeClasses[CityScapeKey]] = CamVidClasses[CityScapeKey]
        elif LutName == 'Groups':
            LUT[CityScapeClasses[CityScapeKey]] = CityscapesGroups[CityScapeKey]
    return LUT
# Convert an image and write it
def mapCityscapes2LUT(filepath, convertPath, LUT):
    img = Image.open(filepath)
    img_array = np.array(img)
    img_array = LUT[img_array]
    img_array = np.uint8(img_array)
    filename = os.path.basename(filepath)
    img = Image.fromarray(img_array)
    img.save(os.path.join(convertPath,filename),'PNG')

# Parse the image folder and run the conversion for every PNG file
def mapCityscapeDataset(path, LutName):
    pngfiles = np.array([os.path.join(root, name).replace('\\', '/')
                         for root, _, files in os.walk(path, followlinks=False)
                         for name in files
                         if name.endswith(".png")])

    convertPath = path+'_'+ LutName
    if not os.path.exists(convertPath):
        os.makedirs(convertPath)

    LUT = BuildLUT(LutName)
    for filepath in pngfiles:
        mapCityscapes2LUT(filepath, convertPath, LUT)
    pass

if __name__ == '__main__':
    CityScapePath = 'D:/Datasets/Cityscapes/segmentation/train'
    mapCityscapeDataset(CityScapePath,'Groups')