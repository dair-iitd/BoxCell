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
parser.add_argument("--model_type", type=str, default='yolov8x')
parser.add_argument("--yaml_file", type=str)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--imgsz", type=int, default=500)
parser.add_argument("--verbose", type=bool, default=True)
args = parser.parse_args()

os.environ['WANDB_DISABLED'] = 'true'
if args.model_weight_path:
    model = YOLO(args.model_weight_path)
else:
    model = YOLO(args.model_type)

model.train(data = args.yaml_file, epochs = args.num_epochs, batch = args.batch_size, imgsz = args.imgsz, verbose = args.verbose, seed = 2023)

