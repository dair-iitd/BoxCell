# use this script to finetune sam on weak supervision losses.
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
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
import torch.multiprocessing as mp
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
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

torch.manual_seed(2023)
np.random.seed(2023)

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str)
parser.add_argument("--val_path", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--model_weight_file", type=str)
parser.add_argument("--resume", type=str, default = None)
parser.add_argument("--model_type", type=str, choices = ['vit_b', 'vit_l', 'vit_h'])
parser.add_argument("--work_dir", type=str)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--mode", type=str, choices=["boxinst", "bbtp"])
# parser.add_argument("--save_pred_while_training", type=bool, default = True)
parser.add_argument("--lr", type=float, default=1e-7)

args = parser.parse_args()
    
os.makedirs(args.work_dir, exist_ok = True)
for i,j,k in os.walk(args.work_dir):
    j = [int(i) for i in j]
    j.sort()
    if len(j) == 0:
        run_id = '1'
    else:
        run_id = str(j[-1] + 1)
    break
    
run_path = os.path.join(temp, run_id)
os.makedirs(run_path, exist_ok = True)

print(f"The results of the run will be saved at : {run_path}")

with open(os.path.join(run_path, 'args.txt'), 'w') as f:
     f.write(json.dumps(args.__dict__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Consep2Sam(Dataset):
    def __init__(self, data_root, image_size = 500, weak_supervision_dir_name = 'gt_bboxes'):
        self.data_root = data_root
        self.files = os.listdir(os.path.join(data_root, 'image')) # this array only contains the image filename. eg. train_1_0.jpg
        self.image_size = image_size
        self.weak_supervision_dir_name = weak_supervision_dir_name

        self.num_images = len(self.files) 
        self.num_cells = 0

        for imgname in self.files:
            annotation_file = os.path.join(self.data_root, self.weak_supervision_dir_name, imgname.split('.')[0]+'.txt')
            with open(annotation_file,'r') as f:
                self.num_cells += len(f.readlines())

        self.img = None
        self.embedding = None # this holds the embedding of the current image
        self.gt2D = None # This holds the gt mask of the current image
        
        self.bboxes = None
        self.class_labels = None
        self.idx = None
        
        self.curr_file_index = -1 # this is the index of the current file as per the files list.
        self.curr_cell_index = None # this is the index of the current cell in the current image.

        self.transform = T.Resize(256)
        
    def __len__(self):
        return self.num_cells

    def __getitem__(self, index):
        if self.curr_file_index == -1 or self.curr_cell_index == len(self.bboxes) - 1:
            self.curr_file_index += 1
            self.curr_file_index %= len(self.files)
            
            if self.curr_file_index == 0:
                np.random.shuffle(self.files)
                
            self.curr_cell_index = -1
            self.embedding = torch.tensor(np.load(join(self.data_root, 'embeddings', self.files[self.curr_file_index].split('.')[0]+'.npy')), device = device).float()
            self.gt2D = torch.tensor(np.load(os.path.join(self.data_root, 'gt_masks', self.files[self.curr_file_index].split('.')[0]), allow_pickle = True), device = device)
            self.img = torch.tensor(cv2.imread(os.path.join(self.data_root, 'images', self.files[self.curr_file_index])), device = device)
            weak_supervision = np.loadtxt(os.path.join(self.data_root, self.weak_supervision_dir_name, self.files[self.curr_file_index].split('.')[0]+'.txt'), delimiter = ',', dtype = int)
            np.random.shuffle(weak_supervision)
            self.idx = torch.tensor(weak_supervision[:, :2], device = device)
            self.class_labels = torch.tensor(weak_supervision[:, 2], device = device)
            self.bboxes = torch.tensor(weak_supervision[:, 3:], device = device)

            img2 = cv2.resize(self.img.cpu().numpy().astype(float), (256,256))
            img3 = np.zeros(img2.shape)
            fin = np.zeros((8, img3.shape[0], img3.shape[1]))
            cnt = 0
            for i in range(1,-2,-1):
                for j in range(1,-2,-1):
                    if i == 0 and j == 0:
                        continue
                        
                    img3 = np.roll(np.roll(img2, shift=i, axis=0), shift = j, axis = 1)

                    if i > 0:
                        img3[:i, :] = np.nan
                    elif i < 0:
                        img3[i:, :] = np.nan
            
                    if j > 0:
                        img3[:, :j] = np.nan
                    elif j < 0:
                        img3[:, j:] = np.nan
            
                    cs = np.exp(-(np.linalg.norm(img3 - img2, axis = 2)/15))
                    fin[cnt,:,:] = cs
                    cnt += 1

            self.fin = torch.tensor(fin, device = device)
            
        self.curr_cell_index += 1
        
        cell_gt = ((self.gt2D == self.idx[self.curr_cell_index, 1]) * 1).double()
        cell_gt = self.transform(cell_gt[None, :, :]) # need this cuz sam's output of segmask is 256x256.
        
        cell_gt_bbox_mask = torch.zeros(256, 256, device = device).float()
        bbox = self.bboxes[self.curr_cell_index] * 256 // self.image_size
        cell_gt_bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        
        return self.files[self.curr_file_index], self.curr_cell_index, self.idx[self.curr_cell_index], self.class_labels[self.curr_cell_index], self.bboxes[self.curr_cell_index], self.embedding, cell_gt, self.bboxes[self.curr_cell_index][None, :] * 1024 // self.image_size, cell_gt_bbox_mask[None, :], self.img, self.fin

def mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0,45,5), mode='all', 
                                    focal_params={'alpha':0.9, 'gamma':2.0, 'sampling_prob':1.0}, 
                                    obj_size=0, epsilon=1e-6):
    """ Compute the mil unary loss from parallel transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        crop_boxes: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]
    Returns
        polar unary loss for each category (C,) if mode='balance'
        otherwise, the average polar unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob,-1]

        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1
        # print('-----',box_h,box_w, y1,y0)

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
        # print("#angles = {}".format(len(parallel_angle_params)))

        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            pred_parallel0 = pred_parallel*msk0
            pred_parallel1 = pred_parallel*msk1
            flag0 = torch.sum(msk0[0], dim=0)>0.5
            prob0 = torch.max(pred_parallel0[0], dim=0)[0]
            prob0 = prob0[flag0]
            flag1 = torch.sum(msk1[0], dim=1)>0.5
            prob1 = torch.max(pred_parallel1[0], dim=1)[0]
            prob1 = prob1[flag1]
            if len(prob0)>0:
                ypred_pos[c.item()].append(prob0)
            if len(prob1)>0:
                ypred_pos[c.item()].append(prob1)

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses

def mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode='mil_focal', 
                           focal_params={'alpha':0.7, 'gamma':2.0, 'sampling_prob':0.5}, 
                           epsilon=1e-6):
    """ Compute the mil unary loss.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]
    Returns
        unary loss for each category (C,) if mode='balance'
        otherwise, the average unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        # print('***',c,box, pred.shape, nb_img)
        if pred.numel() == 0:
            print("sad")
            continue
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                # loss = bce_pos.sum()/len(bce_pos)
                # loss = bce_neg.sum()/len(bce_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                print("sad2")
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))
        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses

def mil_pairwise_loss(ypred, mask, softmax=False, exp_coef=-1):
    """ Compute the pair-wise loss.

        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
    Returns
        pair-wise loss for each category (C,)
    """
    device = ypred.device
    center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),  
            torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),  
            torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])]
    ## pairwise loss for each col/row MIL
    num_classes = ypred.shape[1]
    if softmax:
        num_classes = num_classes - 1
    losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=device)

    for c in range(num_classes):
        pairwise_loss = []
        for w in pairwise_weights_list:
            weights = center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            aff_map = F.conv2d(ypred[:,c,:,:].unsqueeze(1), weights, padding=1)
            cur_loss = aff_map**2
            if exp_coef>0:
                cur_loss = torch.exp(exp_coef*cur_loss)-1
            cur_loss = torch.sum(cur_loss*mask[:,c,:,:].unsqueeze(1))/(torch.sum(mask[:,c,:,:]+1e-6))
            pairwise_loss.append(cur_loss)
        losses[c] = torch.mean(torch.stack(pairwise_loss))
    return losses

def mil_good_loss(ypred, mask, fin, softmax=False, exp_coef=-1):
    """ Compute the pair-wise loss.

        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
    Returns
        pair-wise loss for each category (C,)
    """

    pairwise_losses = compute_pairwise_term(ypred, 3, 2)
    weights = (fin >= 0.3) * mask.float()
    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)
    # print(pairwise_losses, loss_pairwise)
    return loss_pairwise

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    # from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        # print("hi")
        return torch.tensor(float("nan"))
    volume_intersect = (mask_gt & mask_pred).sum()
    # print("hi2")
    return 2*volume_intersect / volume_sum

# does not work with yolo bbox, use eval_images instead.
def eval_dataloader(sam_model, list_of_images, dataloader):
    sam_model.eval()
    dices = {i : ([], 0) for i in list_of_images}
    masks = {i : torch.zeros((256, 256), device = device) for i in list_of_images}
    gts = {i : torch.zeros((256, 256), device = device) for i in list_of_images}
    
    total_loss = 0
    
    for step, (filenames, cell_indices, cell_identities, cell_classes, boxes, image_embedding, gt2D, box_torch, gt_bbox_mask, img, fin) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
        for i in range(len(filenames)):
            masks[filenames[i]] += (mask_predictions[i, 0, :, :] > 0)
            gts[filenames[i]] += gt2D[i, 0, :, :]
            dices[filenames[i]][0].append([cell_identities[i, 0].item(), compute_dice_coefficient(gt2D[i, 0] > 0, mask_predictions[i, 0, :, :] > 0).item()])

        mask_predictions = torch.clamp(mask_predictions, -80, 80)
        sigmoid_mask_predictions = 1 / (1+torch.exp(-mask_predictions))

        bbox_list = torch.zeros(len(boxes), 6, device = device, dtype = torch.int64)
        bbox_list[:, 0] = torch.arange(0, len(boxes), 1)
        bbox_list[:, 2:] = boxes * 256 // 500

        if args.mode == "bbtp":
            loss = 1 * (mil_unary_sigmoid_loss(sigmoid_mask_predictions, gt_bbox_mask, bbox_list)) + 1 * (mil_pairwise_loss(sigmoid_mask_predictions, gt_bbox_mask))
        elif args.mode == "boxinst":
            loss = mil_unary_sigmoid_loss(sigmoid_mask_predictions, gt_bbox_mask, bbox_list) + (mil_good_loss(sigmoid_mask_predictions, gt_bbox_mask, fin))
            
        total_loss += loss.item()
     
    for i in dices:
        dices[i] = (np.array(dices[i][0]), compute_dice_coefficient(gts[i] > 0, masks[i] > 0).item())
            
        
    img_dice = np.array([v[1] for k,v in dices.items()]).mean()
    
    cell_dice = 0
    count = 0
    for k,v in dices.items():
        cell_dice += sum(v[0][:, 1])
        count += len(v[0])
    cell_dice /= count
    
    sam_model.train()
    
    return dices, img_dice, cell_dice, total_loss / (step + 1)

sam_model = sam_model_registry[args.model_type](checkpoint=args.model_weight_file).to(device)
sam_model.train()

train_dataset = Consep2Sam(args.train_path)
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size)#, pin_memory = True)

val_dataset = Consep2Sam(args.val_path)
val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size)#, pin_memory = True)

optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
num_epochs = args.num_epochs

losses = []
train_img_dices_list = []
train_cell_dices_list = []
val_img_dices_list = []
val_cell_dices_list = []
train_evals_list = []
val_evals_list = []
val_loss_list = []

best_dice = 0.0
best_loss = 100

start_epoch = 0
if args.resume is not None:
    if os.path.isfile(args.resume):

        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        sam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        losses = checkpoint['losses']
        train_img_dices_list = checkpoint['train_img_dices_list']
        train_cell_dices_list = checkpoint['train_cell_dices_list']
        val_img_dices_list = checkpoint['val_img_dices_list']
        val_cell_dices_list = checkpoint['val_cell_dices_list']
        
        train_evals_list = checkpoint['train_evals_list']
        val_evals_list = checkpoint['val_evals_list']
        val_loss_list = checkpoint['val_loss_list']

f = open(join(run_path, 'logs.txt'), 'w')

def log(message, in_terminal = True):
    line = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. {message}'
    f.write(line + '\n')
    if(args.verbose and in_terminal): 
        print(line)

log(f"Starting training at epoch {start_epoch}.")

epoch = start_epoch - 1

train_evals, train_img_dice, train_cell_dice, train_loss = eval_dataloader(sam_model, train_images, train_dataloader)
val_evals, val_img_dice, val_cell_dice, val_loss = eval_dataloader(sam_model, val_images, val_dataloader)

if start_epoch == 0: 
    train_evals_list.append(train_evals)
    val_evals_list.append(val_evals)
    
    train_img_dices_list.append(train_img_dice)
    val_img_dices_list.append(val_img_dice)

    train_cell_dices_list.append(train_cell_dice)
    val_cell_dices_list.append(val_cell_dice)
    
    val_loss_list.append(val_loss)

log(f'Epoch: {epoch}, Train Img Dice : {train_img_dice : .5f}, Val Img Dice : {val_img_dice : .5f}, Train Cell Dice : {train_cell_dice : .5f}, Val Cell Dice : {val_cell_dice : .5f}')

for epoch in range(start_epoch, 300):
    epoch_loss = 0

    __dices = {i : ([], 0) for i in train_images}
    __masks = {i : torch.zeros((256, 256), device = device) for i in train_images}
    __gts = {i : torch.zeros((256, 256), device = device) for i in train_images}

    # (B, 1), (B), (B, 2), (B), (B, 4), (B, 256, 64, 64), (B, 1, 500, 500), (B, 1, 4) {1024x1024 sized boxes}
    for step, (filenames, cell_indices, cell_identities, cell_classes, boxes, image_embedding, gt2D, box_torch, gt_bbox_mask, img, fin) in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        mask_predictions, _, hs = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        mask_predictions = torch.clamp(mask_predictions, -80, 80)
        sigmoid_mask_predictions = 1 / (1+torch.exp(-mask_predictions))
        
        bbox_list = torch.zeros(len(filenames), 6, device = device, dtype = torch.int64)
        bbox_list[:, 0] = torch.arange(0, len(filenames), 1)
        bbox_list[:, 2:] = boxes * 256 // 500

        if args.mode == "bbtp":
            loss = 1 * (mil_unary_sigmoid_loss(sigmoid_mask_predictions, gt_bbox_mask, bbox_list)) + 1 * (mil_pairwise_loss(sigmoid_mask_predictions, gt_bbox_mask))
        elif args.mode == "boxinst":
            loss = mil_unary_sigmoid_loss(sigmoid_mask_predictions, gt_bbox_mask, bbox_list) + (mil_good_loss(sigmoid_mask_predictions, gt_bbox_mask, fin))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        for i in range(len(filenames)):
            __masks[filenames[i]] += (mask_predictions[i, 0, :, :] > 0)
            __gts[filenames[i]] += gt2D[i, 0, :, :]
            __dices[filenames[i]][0].append([cell_identities[i, 0].item(), compute_dice_coefficient(gt2D[i, 0] > 0, mask_predictions[i, 0, :, :] > 0).item()])
    
        log(f'Iteration: {step}, Loss: {epoch_loss / (1 + step)}', in_terminal = False)

    for i in __dices:
        __dices[i] = (np.array(__dices[i][0]), compute_dice_coefficient(__gts[i] > 0, __masks[i] > 0).item())

    __img_dice = np.array([v[1] for k,v in __dices.items() if not np.isnan(v[1])]).mean()
    
    # plt.imshow(__masks['train_1_0.png'].cpu().numpy() > 0.5)
    # plt.show()
    
    __cell_dice = 0
    __count = 0
    # for k,v in __dices.items():
    #     __cell_dice += sum(v[0][:, 1])
    #     __count += len(v[0])
    # __cell_dice /= __count

    epoch_loss /= (step + 1)

    train_evals, train_img_dice, train_cell_dice = __dices, __img_dice, __cell_dice
    val_evals, val_img_dice, val_cell_dice, val_loss = 0,0,0,0
    # val_evals, val_img_dice, val_cell_dice, val_loss = eval_dataloader(sam_model, val_images, val_dataloader)
    # train_evals, train_img_dice, train_cell_dice = eval_images(sam_model, train_images, 'gt_bboxes', True, True)
    # train_evals, train_img_dice, train_cell_dice = eval_dataloader(sam_model, train_images, train_dataloader)

    losses.append(epoch_loss)
    train_img_dices_list.append(train_img_dice)
    train_cell_dices_list.append(train_cell_dice)
    train_evals_list.append(train_evals)
    
    val_img_dices_list.append(val_img_dice)
    val_cell_dices_list.append(val_cell_dice)
    val_evals_list.append(val_evals)
    val_loss_list.append(val_loss)

    log(f'Epoch: {epoch}, Loss: {epoch_loss : .5f}, Val Loss: {val_loss : .5f}, Train Img Dice : {train_img_dice : .5f}, Val Img Dice : {val_img_dice : .5f}, Train Cell Dice : {train_cell_dice : .5f}, Val Cell Dice : {val_cell_dice : .5f}')
    wandb.log({"Train Loss": epoch_loss, "Val Loss": val_loss, "Train Img Dice " : train_img_dice, "Val Img Dice" : val_img_dice, "Train Cell Dice" : train_cell_dice, "Val Cell Dice" : val_cell_dice})

    checkpoint = {
        "model": sam_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "losses": losses,
        
        "train_img_dices_list": train_img_dices_list,
        "val_img_dices_list": val_img_dices_list,
        
        "train_cell_dices_list": train_cell_dices_list,
        "val_cell_dices_list": val_cell_dices_list,
        
        "train_evals_list": train_evals_list,
        "val_evals_list": val_evals_list,
        
        "val_loss_list": val_loss_list,
    }
    torch.save(checkpoint, join(run_path, "latest.pth"))

    # if val_img_dice > best_dice:
        # best_dice = val_img_dice    
    if val_loss <= best_loss:
        best_loss = val_loss
        checkpoint = {
            "model": sam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "losses": losses,
            
            "train_img_dices_list": train_img_dices_list,
            "val_img_dices_list": val_img_dices_list,
            
            "train_cell_dices_list": train_cell_dices_list,
            "val_cell_dices_list": val_cell_dices_list,
            
            "train_evals_list": train_evals_list,
            "val_evals_list": val_evals_list,
            
            "val_loss_list": val_loss_list,
        }
        torch.save(checkpoint, join(run_path, "best.pth"))

    plt.plot(list(range(len(losses))), losses, marker = '.', label = 'training loss')
    plt.plot(list(range(len(losses) + 1)), val_loss_list, marker = '.', label = 'val loss')
    plt.legend()
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(run_path, "loss.png"))
    plt.close()

    plt.plot(list(range(len(train_img_dices_list))), train_img_dices_list, marker = '.', label = 'train_img_dices')
    plt.plot(list(range(len(val_img_dices_list))), val_img_dices_list, marker = '.', label = 'val_img_dices')
    plt.title("Val dice and Train dice averaged over each image")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice Scores avg'd over image")
    plt.savefig(join(run_path, "dice_image.png"))
    plt.close()
    
    plt.plot(list(range(len(train_cell_dices_list))), train_cell_dices_list, marker = '.', label = 'train_cell_dices')
    plt.plot(list(range(len(val_cell_dices_list))), val_cell_dices_list, marker = '.', label = 'val_cell_dices')
    plt.title("Val dice and Train dice averaged over each cell")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice Scores avg'd over cell")
    plt.savefig(join(run_path, "dice_cell.png"))
    plt.close()
    f.close()
    f = open(join(run_path, 'logs.txt'), 'a')
    
log(f'Training is complete.')