import numpy as np
import os
from PIL import Image



#img = np.asarray(Image.open(path))
#import re
#path_seg = [m for  m in re.finditer('inst_train_gt',path)]
#path_seg = [path_seg[0].start() , path_seg[0].end()]
#path_seg = path[:path_seg[0]] + 'seg_train_gt' + path[path_seg[1]:]
#img_seg = np.asarray(Image.open(path_seg))
#
#temp = np.zeros_like(img)
#for idx, val in enumerate(np.unique(img)):
#    temp[img==val]=idx
#if np.amax(temp)>255:
#    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#temp = temp.astype(np.uint8)
#im = Image.fromarray(temp)
#im.save('./temporary', 'PNG')



def filterCityscapeInstances(SegPath, InstPath, ClassesToInstanciate, Name):

    pngfiles = np.array([os.path.join(root, name).replace('\\', '/')
                             for root, _, files in os.walk(InstPath, followlinks=False)
                             for name in files
                             if name.endswith(".png")])
    convertPath = InstPath+'_'+ Name

    if not os.path.exists(convertPath):
        os.makedirs(convertPath)

    for filepath in pngfiles:
        img = np.asarray(Image.open(filepath))
        basename = os.path.basename(filepath)
        filepath_seg = os.path.join(SegPath, basename).replace('\\','/')
        if os.path.isfile(filepath_seg):
            pass
        else:
            print('error nofile found')
            continue
        temp = np.zeros_like(img)
        for idx, val in enumerate(np.unique(img)):
           temp[img==val]=idx+1

        img_seg = np.asarray(Image.open(filepath_seg))
        for cls in np.unique(img_seg):
            if not (cls in ClassesToInstanciate):
                temp[img_seg==cls] = 0

        img = np.zeros_like(temp)
        for idx, val in enumerate(np.unique(temp)):
            img[temp==val]=idx

        if np.amax(img) <= 255:
            img = 255*img/np.amax(img)
        else:
            print('error')
            continue

        img = img.astype(np.uint8)
        im = Image.fromarray(img)
        im.save(os.path.join(convertPath,basename), 'PNG')

    pass

if __name__ == '__main__':
    SegPath = 'C:/DL/MENet/dataset_overfit/seg_train_gt/cityscapes'
    InstPath = 'C:/DL/MENet/dataset_overfit/inst_train_gt/cityscapes'
    ClassesToInstanciate = [2, 6, 8, 9, 10]
    filterCityscapeInstances(SegPath, InstPath, ClassesToInstanciate, 'SegClasses')