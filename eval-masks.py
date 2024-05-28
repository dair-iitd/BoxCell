import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import sys
import random
import argparse
from scipy import ndimage
from skimage import io
import shutil
import time
import gc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gt_masks_dir", type=str) # this should always be the numpy array of the image that has instance seg masks.
parser.add_argument("--pred_masks_dir", type=str)
parser.add_argument("--type", type=str, choices=['semantic', 'instance', 'all'])
args = parser.parse_args()

min_area_thresh = 100 if 'monuseg' in args.gt_masks_dir else 0 # using val set.

def find_contours(binary_mask):
    # Find objects and their boundaries using scipy
    # s = ndimage.generate_binary_structure(2,2)  # Connectivity structure
    labeled_array, _ = ndimage.label(binary_mask)
    return labeled_array

def iou_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection/union if union != 0 else 0
    dice = (2 * iou) / (iou + 1)
    return iou, dice

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA+1) * max(0, yB - yA+1)
    interArea2 = max((boxA[2] - boxA[0]+1) * (boxA[3] - boxA[1]+1), (boxB[2] - boxB[0]+1) * (boxB[3] - boxB[1]+1))
    boxAArea = (boxA[2] - boxA[0]+1) * (boxA[3] - boxA[1]+1)
    boxBArea = (boxB[2] - boxB[0]+1) * (boxB[3] - boxB[1]+1)
    iou = interArea / float(boxAArea + boxBArea - interArea2)
    return iou

def evaluate_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    true_positives = 0
    detected = []

    for pred_box in pred_boxes:
        for idx, gt_box in enumerate(gt_boxes):
            try:
                if idx not in detected and calculate_iou(pred_box, gt_box) >= iou_threshold:
                    true_positives += 1
                    detected.append(idx)
                    break
            except Exception as e:
                print(e, gt_box, pred_box)
                assert 1 == 0

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score
    
total_iou = 0.0
total_dice = 0.0
total_p = 0.0
total_r = 0.0
total_f = 0.0

image_ids = os.listdir(args.pred_masks_dir)

shape = (500,500)
if 'tnbc' in args.gt_masks_dir:
    shape == (512,512)

num_images = 0
for image_id in tqdm(image_ids):
    if not image_id.endswith('.png'):
        print(image_id)
        continue
    num_images += 1
    if args.type != 'instance':
        gt_mask = (np.load(os.path.join(args.gt_masks_dir, image_id.split('.')[0]), allow_pickle=True) > 0) * 1.0
        pred_mask = (cv2.imread(os.path.join(args.pred_masks_dir, image_id), cv2.IMREAD_GRAYSCALE) > 0) * 1.0

        if 'tnbc' in args.gt_masks_dir:
            gt_mask = cv2.resize(gt_mask, (500, 500)) > 0
            pred_mask = cv2.resize(pred_mask, (500, 500)) > 0
        
        contours = find_contours(pred_mask)
        unique_contours = np.unique(contours)
        for contour in unique_contours[1:]:
            if (contours==contour).sum() < min_area_thresh:
                contours[contours==contour] = 0
        pred_mask = np.where(contours > 0, pred_mask, 0)
    
        iou, dice = iou_score(pred_mask, gt_mask)
        total_iou += iou
        total_dice += dice
        
    if args.type != 'semantic':
        gt_mask = np.load(os.path.join(args.gt_masks_dir, image_id.split('.')[0]), allow_pickle=True)
        pred_mask = cv2.imread(os.path.join(args.pred_masks_dir, image_id), cv2.IMREAD_GRAYSCALE)
        contours = find_contours(pred_mask > 0)
        unique_contours = np.unique(contours)
        for contour in unique_contours[1:]:
            if (contours==contour).sum() < min_area_thresh:
                contours[contours==contour] = 0
        pred_mask = np.where(contours > 0, pred_mask, 0)
        
        # gt_boxes = 

        cells = np.unique(gt_mask)[1:]
        gt_boxes = []
        for i in cells:
            x,y = np.nonzero(gt_mask==i)
            gt_boxes.append([y.min(), x.min(), y.max(), x.max()])
        gt_boxes = np.array(gt_boxes)

        cells = np.unique(pred_mask)[1:]
        pred_boxes = []
        for i in cells:
            x,y = np.nonzero(pred_mask==i)
            pred_boxes.append([y.min(), x.min(), y.max(), x.max()])
        pred_boxes = np.array(pred_boxes)

        precision,recall,f1 = evaluate_boxes(gt_boxes, pred_boxes, iou_threshold=0.5)
        total_f += f1
        total_p += precision
        total_r += recall

print(num_images)
if args.type != 'instance':
    print("Dice:",total_dice/num_images)
    print("IOU:",total_iou/num_images)
if args.type != 'semantic':
    print("P:",total_p/num_images)
    print("R:",total_r/num_images)
    print("F1:",total_f/num_images)
