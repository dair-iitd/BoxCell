# use this script to convert the predictions of yolo to sam format.

import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import os
import cv2
import argparse
import pickle as pkl
from tqdm import tqdm
import xml.etree.ElementTree as ET
from skimage.draw import polygon
import random
import shutil
import json

random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_json", type=str)
parser.add_argument("--saving_dir", type=str)
parser.add_argument("--score_thresh", type=int, default=0)
args = parser.parse_args()

all_predictions = json.load(open(args.predicted_json, 'r'))

all_images = [i['image_id'] for i in all_predictions]
all_images = list(set(all_images))
bboxes = {i : [] for i in all_images}

score_thresh = 0

for i in all_predictions:
    if i['score'] > score_thresh:
        bboxes[i['image_id']].append(i['bbox'])

os.makedirs(args.saving_dir, exist_ok = True)
for i in bboxes:
    y = []
    for k in bboxes[i]:
        y.append([0, -1, 1, k[0], k[1], k[0]+k[2], k[1]+k[3]])

    np.savetxt(os.path.join(args.saving_dir, i) + '.txt', y, delimiter=',', fmt = '%i')



