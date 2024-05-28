import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import sys
import random
import argparse
import gurobipy as gp
from gurobipy import GRB
from scipy import ndimage
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
import torch.multiprocessing as mp
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
import monai
import pickle as pkl
import torch.nn.functional as F
import torch.nn as nn
import shutil
from monai.networks import one_hot
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
import time
import gc
from sklearn.cluster import KMeans


# best hyperparam by validation set : tnbc (2,6,25,5). monuseg(2,6,25,5). consep(2,5,20,5)

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir_path", type=str)
parser.add_argument("--box_dir_path", type=str)
parser.add_argument("--sam_s_path", type=str, default = None) # needed for sam-ilp
parser.add_argument("--gt_dir_path", type=str, default = None) # if you want to score the predictions as well.
parser.add_argument("--save_path", type=str)
parser.add_argument("--mu", type=float, default=2)
parser.add_argument("--nc", type=int, default=6)
parser.add_argument("--sigma", type=int, default=25)
parser.add_argument("--split", type=int, default=5)
parser.add_argument("--model_weights", type=str)
parser.add_argument("--model_type", type=str, choices = ['vit_b', 'vit_l', 'vit_h'], default='vit_b')
parser.add_argument("--gurobi_license", type=str, default = None)
parser.add_argument("--gurobi_license_file", type=str, default = None)
parser.add_argument("--mode", type=str, choices = ['d-sam', 'sam-ilp'], default = 'sam-ilp')
parser.add_argument("--type", type=str, choices = ['instance', 'semantic'], default = 'semantic')
args = parser.parse_args()

if args.mode == 'sam-ilp':
    env = gp.Env()
    os.environ['GRB_LICENSE_FILE']=args.gurobi_license_file
    env.setParam('LicenseKey', args.gurobi_license)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_point(points, ax):
    for point in points:
        ax.plot(point[0], point[1], '-ro')

def logit(x):
    a = torch.tensor(x)
    return torch.special.logit(a, eps=1e-7)

def track_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f't={t/1e9}, r={r/1e9}, a={a/1e9}, f={f/1e9}')

def free_mem():
    torch.cuda.empty_cache()
    gc.collect()
    
def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def largest_connected_component(binary_mask):
    labeled_mask, num_features = ndimage.label(binary_mask)
    fin = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
    fin = np.argsort(fin)
    if len(fin) == 0:
        return binary_mask
    return (labeled_mask == fin[-1]+1) * 1

def eval_image(sam_model, image, file, mode):
    free_mem()
    mask_save_path = args.save_path
    os.makedirs(mask_save_path, exist_ok = True)
    
    img = cv2.imread(image)
    predictor = SamPredictor(sam_model)
    predictor.set_image(img)
    img = img.astype(int)
    
    weak_supervision = np.loadtxt(file, delimiter = ',', dtype = int)

    gt_mask = None
    if args.gt_dir_path != None:
        gt_mask = np.load(os.path.join(args.gt_dir_path, image.split('/')[-1][:-4]), allow_pickle = True)
    
    d_sam_pred = np.zeros(gt_mask.shape)
        
    entity = 2
    for enu, i in enumerate(weak_supervision):
        box = i[3:]
        masks, conf, raw_logits = predictor.predict(box = box, multimask_output = False)

        if args.type == 'semantic':
            d_sam_pred += masks[0,:,:]
        elif args.type == 'instance':
            to_assign = masks[0,:,:] * entity
            d_sam_pred = np.where((to_assign > 0), to_assign, d_sam_pred)
        entity += 1
    
    if mode == 'd-sam':
        if args.type == 'semantic':
            cv2.imwrite(os.path.join(args.save_path, image.split('/')[-1].split('\\')[-1]), (d_sam_pred > 0) * 255)
        elif args.type == 'instance':
            cv2.imwrite(os.path.join(args.save_path, image.split('/')[-1].split('\\')[-1]), d_sam_pred)
        return
        
    elif mode == 'sam-ilp':

        sam_s_pred = cv2.imread(os.path.join(args.sam_s_path, image.split('/')[-1].split('\\')[-1]))
        
        if not np.all(sam_s_pred.shape == d_sam_pred.shape):
            sam_s_pred = (cv2.resize(sam_s_pred.astype(float), d_sam_pred.shape) > 0) * 1
            sam_s_pred = sam_s_pred[:,:,0]
        
        all_fg = img[(d_sam_pred != 0) & (sam_s_pred != 0)]
        all_bg = img[(d_sam_pred == 0) & (sam_s_pred == 0)]

        final_llh = np.zeros(gt_mask.shape)
        
        ln = img.shape[0]
        bt = img.shape[1]
        
        split = args.split
        
        for i in range(args.split):
            for j in range(args.split):
                
                pc = img[j*ln//split:j*ln//split+ln//split, i*bt//split:i*bt//split+bt//split]
                
                foreground_pixels = pc[(d_sam_pred[j*ln//split:j*ln//split+ln//split, i*bt//split:i*bt//split+bt//split] != 0) & (sam_s_pred[j*ln//split:j*ln//split+ln//split, i*bt//split:i*bt//split+bt//split] != 0)]
                
                background_pixels = pc[(d_sam_pred[j*ln//split:j*ln//split+ln//split, i*bt//split:i*bt//split+bt//split] == 0) & (sam_s_pred[j*ln//split:j*ln//split+ln//split, i*bt//split:i*bt//split+bt//split] == 0)]
                
                if len(foreground_pixels) <= args.nc:
                    foreground_pixels = all_fg
    
                if len(background_pixels) <= args.nc and len(foreground_pixels) > args.nc:
                    background_pixels = all_bg
                    
                fg_gmm = GaussianMixture(n_components=args.nc, random_state=0)
                fg_gmm.fit(foreground_pixels)
                bg_gmm = GaussianMixture(n_components=args.nc, random_state=0)
                bg_gmm.fit(background_pixels)
    
                llh_fg = fg_gmm.score_samples(pc.reshape((-1,pc.shape[2]))).reshape((ln//split, bt//split))
                llh_bg = bg_gmm.score_samples(pc.reshape((-1,pc.shape[2]))).reshape((ln//split, bt//split))
                
                llh_fg = np.clip(llh_fg, -100,100)
                llh_bg = np.clip(llh_bg, -100,100)
                llh_fg = np.exp(llh_fg)
                llh_bg = np.exp(llh_bg)
            
                llh = llh_fg/(llh_bg + llh_fg)
            
                final_llh[j*ln//split:j*ln//split+ln//split, i*bt//split:i*bt//split+bt//split] = llh
    
                # idx = np.s_[j*ln//args.split:j*ln//args.split+ln//args.split, i*bt//args.split:i*bt//args.split+bt//args.split]
                # pc = img[idx]
                # print(pc.shape)
                # foreground_pixels = pc[(d_sam_pred[idx] != 0) & (sam_s_pred[idx] != 0)]
                # background_pixels = pc[(d_sam_pred[idx] == 0) & (sam_s_pred[idx] == 0)]

                # if len(foreground_pixels) <= args.nc:
                #     foreground_pixels = pc[(sam_s_pred[idx] != 0)]
                    
                # if len(foreground_pixels) <= args.nc:
                #     foreground_pixels = pc[(d_sam_pred[idx] != 0)]

                # if len(background_pixels) <= args.nc:
                #     background_pixels = all_bg

                # fg_gmm = GaussianMixture(n_components=args.nc, random_state=0)
                # fg_gmm.fit(foreground_pixels)
                # bg_gmm = GaussianMixture(n_components=args.nc, random_state=0)
                # bg_gmm.fit(background_pixels)
                        
                # llh_fg = fg_gmm.score_samples(pc.reshape((-1,pc.shape[2]))).reshape((ln//args.split, bt//args.split))
                # llh_bg = bg_gmm.score_samples(pc.reshape((-1,pc.shape[2]))).reshape((ln//args.split, bt//args.split))
                
                # llh_fg = np.clip(llh_fg, -100,100)
                # llh_bg = np.clip(llh_bg, -100,100)
                # llh_fg = np.exp(llh_fg)
                # llh_bg = np.exp(llh_bg)
            
                # llh = llh_fg / (llh_bg + llh_fg)
                # final_llh[idx] = llh

        d_sam_component = (d_sam_pred > 0) * 1
        sam_ilp_pred = solve_ilp(img,sam_s_pred,d_sam_component,final_llh,gt_mask)

        if args.type == 'semantic':
            cv2.imwrite(os.path.join(args.save_path, image.split('/')[-1].split('\\')[-1]), (sam_ilp_pred > 0) * 255)
            
        elif args.type == 'instance':
            sam_ilp_pred = np.where((sam_ilp_pred > 0) & (d_sam_pred > 0), d_sam_pred, sam_ilp_pred)
            dsam_in_ilp = sam_ilp_pred.copy()
            mask2 = np.where((sam_ilp_pred > 0) & (d_sam_pred > 0), 0, sam_ilp_pred)
        
            curr_cells = np.unique(sam_ilp_pred)[-1]
            sem_map, num_sem_map = ndimage.label((mask2+sam_ilp_pred) > 0)
            
            for c in range(1,num_sem_map+1):
                pts = np.unique(sam_ilp_pred[(sem_map == c)])
                if len(pts) > 1 and pts[0] == 1:
                    num_cells = max(1, len(pts)-1)
                    kmeans = KMeans(n_clusters=num_cells, init=np.array([[j.mean() for j in np.nonzero(sam_ilp_pred == i)] for i in pts[1:]]), n_init=1)
                    x_ones, y_ones = np.nonzero((sem_map == c))
                    kmeans.fit([[x_ones[i], y_ones[i]] for i in range(len(x_ones))])
                    x_ones, y_ones = np.nonzero((sam_ilp_pred == 1) & (sem_map == c))
                    labels = [int(pts[i+1]) for i in kmeans.predict([[x_ones[i], y_ones[i]] for i in range(len(x_ones))])]
                    sam_ilp_pred[x_ones, y_ones] = labels
                elif pts[0] == 1 and len(pts) == 1:
                    curr_cells += 1
                    sam_ilp_pred[(sem_map == c)] = curr_cells
            cv2.imwrite(os.path.join(args.save_path, image.split('/')[-1].split('\\')[-1]), sam_ilp_pred)
        
        return
        
def eval_images(sam_model, list_of_images, mode):
    dices = {}
    for i, image in tqdm(enumerate(list_of_images)):
        dices[image] = eval_image(sam_model, image, os.path.join(args.box_dir_path, list_of_images[i].split('.')[0].split('/')[-1]+'.txt'), mode)

    # img_dice = sum(dices.values())
    # return dices, img_dice / len(dices)

def solve_ilp(img,sam_s_pred,d_sam_pred,llh,gt_mask):
    ln = img.shape[0]
    bt = img.shape[1]
    
    final_masks = np.zeros(img.shape)

    model = gp.Model("model", env=env)
    model.setParam("TimeLimit", 60)
    model.setParam("OutputFlag", 0)
    z = [[0 for j in range(bt)] for i in range(ln)]
    az_hori = [[0 for j in range(bt)] for i in range(ln)]
    az_verti = [[0 for j in range(bt)] for i in range(ln)]

    # creating the z_ij variables
    for i in range(ln):
        for j in range(bt):
            az_hori[i][j] = model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = f'azz_{i}_{j}')
            az_verti[i][j] = model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = f'azz_{i}_{j}')
            if sam_s_pred[i,j] > 0 and d_sam_pred[i,j] > 0:
                z[i][j] = model.addVar(lb = 1, ub = 1, vtype = GRB.BINARY, name = f'z_{i}_{j}')
                z[i][j].setAttr('Start', 1)
            elif sam_s_pred[i,j] == 0 and d_sam_pred[i,j] == 0:
                z[i][j] = model.addVar(lb = 0, ub = 0, vtype = GRB.BINARY, name = f'z_{i}_{j}')
                z[i][j].setAttr('Start', 0)
            else:
                z[i][j] = model.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name = f'z_{i}_{j}')
                z[i][j].setAttr('Start', d_sam_pred[i][j])

    for i in range(ln-1):
        for j in range(bt):
            model.addConstr(az_verti[i][j] <= z[i][j] + z[i+1][j], name=f"aabs1_{i}_{j}")
            model.addConstr(az_verti[i][j] <= 2 - z[i][j] - z[i+1][j], name=f"aabs2_{i}_{j}")
            model.addConstr(az_verti[i][j] >= z[i][j] - z[i+1][j], name=f"aabs3_{i}_{j}")
            model.addConstr(az_verti[i][j] >= - z[i][j] + z[i+1][j], name=f"aabs4_{i}_{j}")

    for i in range(ln):
        for j in range(bt-1):
            model.addConstr(az_hori[i][j] <= z[i][j] + z[i][j+1], name=f"aabs5_{i}_{j}")
            model.addConstr(az_hori[i][j] <= 2 - z[i][j] - z[i][j+1], name=f"aabs6_{i}_{j}")
            model.addConstr(az_hori[i][j] >= - z[i][j] + z[i][j+1], name=f"aabs7_{i}_{j}")
            model.addConstr(az_hori[i][j] >= z[i][j] - z[i][j+1], name=f"aabs8_{i}_{j}")
    
    model.setObjective(gp.quicksum(z[i][j]*llh[i][j] + (1-z[i][j])*(1-llh[i][j]) for i in range(ln) for j in range(bt)) / (ln * bt) - (args.mu / (ln * bt - ln - bt) ) * (gp.quicksum(np.exp(-(np.linalg.norm(img[i,j]-img[i,j+1])/args.sigma)**2) * az_hori[i][j] for j in range(bt-1) for i in range(ln)) + gp.quicksum(np.exp(-(np.linalg.norm(img[i+1,j]-img[i,j])/args.sigma)**2) * az_verti[i][j] for j in range(bt) for i in range(ln-1))), GRB.MAXIMIZE)
    model.update()
    model.optimize()
    soln = np.array([[j.X for j in i] for i in z])
    
    return soln

sam_model = sam_model_registry[args.model_type](checkpoint=args.model_weights).to(device)

list_of_images = os.listdir(args.img_dir_path)
list_of_images = [os.path.join(args.img_dir_path, i) for i in list_of_images]

# per_img_score, average_score = eval_images(sam_model, list_of_images, args.mode)
eval_images(sam_model, list_of_images, args.mode)