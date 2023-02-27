from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances
import torch.nn.functional as F
import cv2 as cv

from random import choice, randint
from pycocotools.coco import COCO
import glob, tqdm, os

lvis_dir_path = "/home/hzf/data/lvis"
lvis_anno_path = os.path.join(lvis_dir_path, "annotations/lvisv0.5+coco_train.json")
resolve = COCO(lvis_anno_path)
lvis_img_list = glob.glob(os.path.join(lvis_dir_path, 'train2017') + "/*") 
lvis_anno_list = {}
for i in range(len(lvis_img_list)):
    name = lvis_img_list[i]
    img = cv.imread(name)
    if img.shape[2] != 3:
        print(img.shape)