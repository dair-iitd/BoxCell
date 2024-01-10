from ultralytics import YOLO
import os
import torch
import numpy as np
import gc
import json
import argparse
gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument("--model_weight_path", type=str)
parser.add_argument("--image_dir", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--conf", type=float, default=0.31)
parser.add_argument("--iou", type=float, default=0.7)
parser.add_argument("--max_det", type=int, default=2000)
parser.add_argument("--imgsz", type=int, default=500)
args = parser.parse_args()

os.environ['WANDB_DISABLED'] = 'true'
model = YOLO(args.model_weight_path)

model.predict(args.image_dir, batch = args.batch_size, conf = args.conf, iou = args.iou, max_det = args.max_det, imgsz = args.imgsz, save_txt = True)

