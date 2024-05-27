# use this script to preprocess the data for segment-anything and yolo.

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

random.seed(0)
np.random.seed(0)

val_img_consep = [(4,3),(6,2),(7,3),(12,0),(12,2),(17,2),(18,1),(19,3),(24,1),(25,2)]
val_img_monuseg = [(1,0),(3,2),(5,3),(11,3),(13,0),(15,3),(16,3),(17,3),(18,0),(20,1),(20,2),(25,1),(26,2),(33,0),(34,3)]
val_img_tnbc = [17,13,39,34,35]
test_img_tnbc = [18,10,7,8,6,41,29,28,26,36]

# change these two paths for the location of the original consep dataset and the saving directory.
parser = argparse.ArgumentParser()
parser.add_argument("--loading_dir", type=str)
parser.add_argument("--saving_dir", type=str)
parser.add_argument("--dataset", type=str, choices=['monuseg', 'consep', 'tnbc'])
parser.add_argument("--replicate", type=bool, default=True)
args = parser.parse_args()

load_path = args.loading_dir
save_path = args.saving_dir

train_dir = os.path.join(save_path, 'train')
val_dir = os.path.join(save_path, 'val')
test_dir = os.path.join(save_path, 'test')

yolo_dir = os.path.join(save_path, 'yolo')
yolo_img = os.path.join(yolo_dir, 'images')
yolo_lab = os.path.join(yolo_dir, 'labels')

os.makedirs(save_path, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

os.makedirs(yolo_dir, exist_ok=True)
os.makedirs(yolo_img, exist_ok=True)
os.makedirs(yolo_lab, exist_ok=True)

# os.makedirs(os.path.join(yolo_img, 'train'), exist_ok=True)
# os.makedirs(os.path.join(yolo_img, 'val'), exist_ok=True)
# os.makedirs(os.path.join(yolo_img, 'test'), exist_ok=True)

# os.makedirs(os.path.join(yolo_lab, 'train'), exist_ok=True)
# os.makedirs(os.path.join(yolo_lab, 'val'), exist_ok=True)
# os.makedirs(os.path.join(yolo_lab, 'test'), exist_ok=True)

# image, embedding, gt_mask, gt_image, gt_bbox
os.makedirs(os.path.join(train_dir, 'gt_image'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'gt_bbox'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'gt_mask'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'image'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'embeddings'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)

os.makedirs(os.path.join(val_dir, 'gt_image'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'gt_bbox'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'gt_mask'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'image'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'embeddings'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

os.makedirs(os.path.join(test_dir, 'gt_image'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'gt_bbox'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'gt_mask'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'image'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'embeddings'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

with open(os.path.join(save_path, 'train.yaml'), 'w') as f:
    f.write(f"train: {os.path.join(yolo_img, 'train')}\n")
    f.write(f"val: {os.path.join(yolo_img, 'val')}\n")
    f.write(f"test: {os.path.join(yolo_img, 'test')}\n\nnc: 1\n\nname: ['cell']\n")

if args.dataset == 'consep':
    train_files = {}
    print("Fetching Training Data.")

    for i in tqdm(range(27)):
        mat = scipy.io.loadmat(os.path.join(load_path, f'Train/Labels/train_{i+1}.mat'))
        img = Image.open(os.path.join(load_path, f'Train/Images/train_{i+1}.png'))

        for j in range(2):
            for k in range(2):
                minx = 500 * j
                maxx = 500 * j + 500
                miny = 500 * k
                maxy = 500 * k + 500
                img_ = img.crop((miny,minx,maxy,maxx))
                img_.save(os.path.join(train_dir, 'image', f'train_{i+1}_{2 * j + k}' + '.png'))
                instances = {}
                for xx in range(minx, maxx):
                    for yy in range(miny, maxy):
                        instance = mat['inst_map'][xx,yy]
                        if instance != 0:
                            if instance in instances:
                                instances[instance][1] = min(yy - miny, instances[instance][1])
                                instances[instance][2] = min(xx - minx, instances[instance][2])
                                instances[instance][3] = max(yy - miny, instances[instance][3])
                                instances[instance][4] = max(xx - minx, instances[instance][4])
                            else:
                                instances[instance] = [mat['type_map'][xx,yy], yy-miny,xx-minx,yy-miny,xx-minx]
                train_files[f'train_{i+1}_{2 * j + k}'] = (instances, mat['inst_map'][minx:maxx, miny:maxy])
    for i in train_files:
        mpl.image.imsave(os.path.join(train_dir, 'gt_image', i + '.png'), train_files[i][1])
        # np.save(os.path.join(train_dir, 'gt_mask', i), train_files[i][1])
        with open(os.path.join(train_dir, 'gt_mask', i), 'wb') as f:
            pkl.dump(train_files[i][1], f)
        with open(os.path.join(train_dir, 'gt_bbox', i + '.txt'), 'w') as f:
            cnt = 0
            for k,v in train_files[i][0].items():
                f.write(f'{cnt},{int(k)},{int(v[0])},{v[1]},{v[2]},{v[3]},{v[4]}\n') # label, label as in consep, class, x, y, x+w, y+h
                cnt += 1
        # yolo files.
        with open(os.path.join(train_dir, 'labels', i + '.txt'), 'w') as f:
            for k,v in train_files[i][0].items():
                f.write(f'{0} {((v[1] + v[3]) // 2) / 500} {((v[2] + v[4]) // 2) / 500} {(v[3] - v[1] + 1) / 500} {(v[4] - v[2] + 1) / 500}\n') # label, class, x_center, y_center, length, width

    print("Fetching Test Data.")
    test_files = {}
    for i in tqdm(range(14)):
        mat = scipy.io.loadmat(os.path.join(load_path, f'Test/Labels/test_{i+1}.mat'))
        img = Image.open(os.path.join(load_path, f'Test/Images/test_{i+1}.png'))

        for j in range(2):
            for k in range(2):
                minx = 500 * j
                maxx = 500 * j + 500
                miny = 500 * k
                maxy = 500 * k + 500
                img_ = img.crop((miny,minx,maxy,maxx))
                img_.save(os.path.join(test_dir, 'image', f'test_{i+1}_{2 * j + k}' + '.png'))
                instances = {}
                for xx in range(minx, maxx):
                    for yy in range(miny, maxy):
                        instance = mat['inst_map'][xx,yy]
                        if instance != 0:
                            if instance in instances:
                                instances[instance][1] = min(yy - miny, instances[instance][1])
                                instances[instance][2] = min(xx - minx, instances[instance][2])
                                instances[instance][3] = max(yy - miny, instances[instance][3])
                                instances[instance][4] = max(xx - minx, instances[instance][4])
                            else:
                                instances[instance] = [mat['type_map'][xx,yy], yy-miny,xx-minx,yy-miny,xx-minx]
                test_files[f'test_{i+1}_{2 * j + k}'] = (instances, mat['inst_map'][minx:maxx, miny:maxy])

    for i in test_files:
        mpl.image.imsave(os.path.join(test_dir, 'gt_image', i + '.png'), test_files[i][1])
        # np.save(os.path.join(test_dir, 'gt_mask', i), test_files[i][1])
        with open(os.path.join(test_dir, 'gt_mask', i), 'wb') as f:
            pkl.dump(test_files[i][1], f)
        with open(os.path.join(test_dir, 'gt_bbox', i + '.txt'), 'w') as f:
            cnt = 0
            for k,v in test_files[i][0].items():
                f.write(f'{cnt},{int(k)},{int(v[0])},{v[1]},{v[2]},{v[3]},{v[4]}\n') # label, label as in consep, class, x, y, x+w, y+h
                cnt += 1
        # yolo files.
        with open(os.path.join(test_dir, 'labels', i + '.txt'), 'w') as f:
            for k,v in test_files[i][0].items():
                f.write(f'{0} {((v[1] + v[3]) // 2) / 500} {((v[2] + v[4]) // 2) / 500} {(v[3] - v[1] + 1) / 500} {(v[4] - v[2] + 1) / 500}\n') # label, class, x_center, y_center, length, width


    list_of_images = os.listdir(os.path.join(train_dir, 'gt_mask'))
    list_of_images = [i.split('.')[0] for i in list_of_images]
    if not args.replicate:
        val_images = random.sample(list_of_images, int(0.2 * len(list_of_images)))
        train_images = [i for i in list_of_images if i not in val_images]
    else:
        val_images = [f'train_{i}_{j}' for i,j in val_img_consep]
        train_images = [i for i in list_of_images if i not in val_images]

    for dir in os.listdir(train_dir):
        if 'DS_Store' not in dir:
            for file in os.listdir(os.path.join(train_dir, dir)):
                if file.split('.')[0] in val_images:
                    shutil.move(os.path.join(train_dir, dir, file), os.path.join(val_dir, dir, file))

elif args.dataset == 'monuseg':
    train_files = {}
    print("Fetching Training Data.")

    for i,j,k in os.walk(os.path.join(load_path,'Train','Labels')):
        train_imgs = k

    for i in tqdm(range(len(train_imgs))):
        mat_file = os.path.join(load_path,'Train','Labels',train_imgs[i].split('.')[0]+'.xml')
        img = Image.open(os.path.join(load_path,'Train','Images',train_imgs[i].split('.')[0]+'.tif'))
        mat = {'type_map': np.zeros((1000,1000)), 'inst_map' : np.zeros((1000,1000))}
                
        tree = ET.parse(mat_file)
        root = tree.getroot()
        parent = root.find('Annotation')
        parent = parent.find('Regions')
        for _,element in enumerate(parent.findall('Region')):
            vertices = element.find('Vertices')
            vertex = vertices.findall('Vertex')
            x = np.array([float(i.attrib['X']) for i in vertex])
            y = np.array([float(i.attrib['Y']) for i in vertex])
            x = np.clip(x,0,999)
            y = np.clip(y,0,999)
            cc, rr = polygon(x, y)
            mat['inst_map'][rr,cc] = _+1
            mat['type_map'][rr,cc] = 1

        for j in range(2):
            for k in range(2):
                minx = 500 * j
                maxx = 500 * j + 500
                miny = 500 * k
                maxy = 500 * k + 500
                img_ = img.crop((miny,minx,maxy,maxx))
                img_.save(os.path.join(train_dir, 'image', f'train_{i+1}_{2 * j + k}' + '.png'))

                instances = {}
                for xx in range(minx, maxx):
                    for yy in range(miny, maxy):
                        instance = mat['inst_map'][xx,yy]
                        if instance != 0:
                            if instance in instances:
                                instances[instance][1] = min(yy - miny, instances[instance][1])
                                instances[instance][2] = min(xx - minx, instances[instance][2])
                                instances[instance][3] = max(yy - miny, instances[instance][3])
                                instances[instance][4] = max(xx - minx, instances[instance][4])
                            else:
                                instances[instance] = [mat['type_map'][xx,yy], yy-miny,xx-minx,yy-miny,xx-minx]
                train_files[f'train_{i+1}_{2 * j + k}'] = (instances, mat['inst_map'][minx:maxx, miny:maxy])
    
    for i in train_files:
        mpl.image.imsave(os.path.join(train_dir, 'gt_image', i + '.png'), train_files[i][1])
        # np.save(os.path.join(train_dir, 'gt_mask', i), train_files[i][1])
        with open(os.path.join(train_dir, 'gt_mask', i), 'wb') as f:
            pkl.dump(train_files[i][1], f)
        with open(os.path.join(train_dir, 'gt_bbox', i + '.txt'), 'w') as f:
            cnt = 0
            for k,v in train_files[i][0].items():
                f.write(f'{cnt},{int(k)},{int(v[0])},{v[1]},{v[2]},{v[3]},{v[4]}\n') # label, label as in consep, class, x, y, x+w, y+h
                cnt += 1
        # yolo files.
        with open(os.path.join(train_dir, 'labels', i + '.txt'), 'w') as f:
            for k,v in train_files[i][0].items():
                f.write(f'{0} {((v[1] + v[3]) // 2) / 500} {((v[2] + v[4]) // 2) / 500} {(v[3] - v[1] + 1) / 500} {(v[4] - v[2] + 1) / 500}\n') # label, class, x_center, y_center, length, width

    test_files = {}
    print("Fetching Test Data.")

    for i,j,k in os.walk(os.path.join(load_path,'Test','Labels')):
        test_imgs = k

    for i in tqdm(range(len(test_imgs))):
        mat_file = os.path.join(load_path,'Test','Labels',test_imgs[i].split('.')[0]+'.xml')
        img = Image.open(os.path.join(load_path,'Test','Images',test_imgs[i].split('.')[0]+'.tif'))
        mat = {'type_map': np.zeros((1000,1000)), 'inst_map' : np.zeros((1000,1000))}
                
        tree = ET.parse(mat_file)
        root = tree.getroot()
        parent = root.find('Annotation')
        parent = parent.find('Regions')
        for _,element in enumerate(parent.findall('Region')):
            vertices = element.find('Vertices')
            vertex = vertices.findall('Vertex')
            x = np.array([float(i.attrib['X']) for i in vertex])
            y = np.array([float(i.attrib['Y']) for i in vertex])
            x = np.clip(x,0,999)
            y = np.clip(y,0,999)
            cc, rr = polygon(x, y)
            mat['inst_map'][rr,cc] = _+1
            mat['type_map'][rr,cc] = 1

        for j in range(2):
            for k in range(2):
                minx = 500 * j
                maxx = 500 * j + 500
                miny = 500 * k
                maxy = 500 * k + 500
                img_ = img.crop((miny,minx,maxy,maxx))
                img_.save(os.path.join(test_dir, 'image', f'test_{i+1}_{2 * j + k}' + '.png'))

                instances = {}
                for xx in range(minx, maxx):
                    for yy in range(miny, maxy):
                        instance = mat['inst_map'][xx,yy]
                        if instance != 0:
                            if instance in instances:
                                instances[instance][1] = min(yy - miny, instances[instance][1])
                                instances[instance][2] = min(xx - minx, instances[instance][2])
                                instances[instance][3] = max(yy - miny, instances[instance][3])
                                instances[instance][4] = max(xx - minx, instances[instance][4])
                            else:
                                instances[instance] = [mat['type_map'][xx,yy], yy-miny,xx-minx,yy-miny,xx-minx]
                test_files[f'test_{i+1}_{2 * j + k}'] = (instances, mat['inst_map'][minx:maxx, miny:maxy])
    
    for i in test_files:
        mpl.image.imsave(os.path.join(test_dir, 'gt_image', i + '.png'), test_files[i][1])
        # np.save(os.path.join(test_dir, 'gt_mask', i), test_files[i][1])
        with open(os.path.join(test_dir, 'gt_mask', i), 'wb') as f:
            pkl.dump(test_files[i][1], f)
        with open(os.path.join(test_dir, 'gt_bbox', i + '.txt'), 'w') as f:
            cnt = 0
            for k,v in test_files[i][0].items():
                f.write(f'{cnt},{int(k)},{int(v[0])},{v[1]},{v[2]},{v[3]},{v[4]}\n') # label, label as in consep, class, x, y, x+w, y+h
                cnt += 1
        # yolo files.
        with open(os.path.join(test_dir, 'labels', i + '.txt'), 'w') as f:
            for k,v in test_files[i][0].items():
                f.write(f'{0} {((v[1] + v[3]) // 2) / 500} {((v[2] + v[4]) // 2) / 500} {(v[3] - v[1] + 1) / 500} {(v[4] - v[2] + 1) / 500}\n') # label, class, x_center, y_center, length, width

    list_of_images = os.listdir(os.path.join(train_dir, 'gt_mask'))
    list_of_images = [i.split('.')[0] for i in list_of_images]
    if not args.replicate:
        val_images = random.sample(list_of_images, int(0.2 * len(list_of_images)))
        train_images = [i for i in list_of_images if i not in val_images]
    else:
        val_images = [f'train_{i}_{j}' for i,j in val_img_monuseg]
        train_images = [i for i in list_of_images if i not in val_images]

    for dir in os.listdir(train_dir):
        if 'DS_Store' not in dir:
            for file in os.listdir(os.path.join(train_dir, dir)):
                if file.split('.')[0] in val_images:
                    shutil.move(os.path.join(train_dir, dir, file), os.path.join(val_dir, dir, file))

elif args.dataset == 'tnbc':
    train_files = {}
    print("Fetching Training Data.")

    for i,j,k in os.walk(os.path.join(load_path,'Labels')):
        train_imgs = k

    for i in tqdm(range(len(train_imgs))):
        mat = scipy.io.loadmat(os.path.join(load_path, f'Labels/{train_imgs[i]}'))
        img = Image.open(os.path.join(load_path, f'Images/{train_imgs[i].split(".")[0]}.png'))

        img.save(os.path.join(train_dir, 'image', f'train_{i+1}' + '.png'))
        instances = {}
        for xx in range(0, 512):
            for yy in range(0, 512):
                instance = mat['inst_map'][xx,yy]
                if instance != 0:
                    if instance in instances:
                        instances[instance][1] = min(yy - 0, instances[instance][1])
                        instances[instance][2] = min(xx - 0, instances[instance][2])
                        instances[instance][3] = max(yy - 0, instances[instance][3])
                        instances[instance][4] = max(xx - 0, instances[instance][4])
                    else:
                        instances[instance] = [1, yy-0,xx-0,yy-0,xx-0]
        train_files[f'train_{i+1}'] = (instances, mat['inst_map'])

    for i in train_files:
        mpl.image.imsave(os.path.join(train_dir, 'gt_image', i + '.png'), train_files[i][1])
        # np.save(os.path.join(train_dir, 'gt_mask', i), train_files[i][1])
        with open(os.path.join(train_dir, 'gt_mask', i), 'wb') as f:
            pkl.dump(train_files[i][1], f)
        with open(os.path.join(train_dir, 'gt_bbox', i + '.txt'), 'w') as f:
            cnt = 0
            for k,v in train_files[i][0].items():
                f.write(f'{cnt},{int(k)},{int(v[0])},{v[1]},{v[2]},{v[3]},{v[4]}\n') # label, label as in consep, class, x, y, x+w, y+h
                cnt += 1
        # yolo files.
        with open(os.path.join(train_dir, 'labels', i + '.txt'), 'w') as f:
            for k,v in train_files[i][0].items():
                f.write(f'{0} {((v[1] + v[3]) // 2) / 512} {((v[2] + v[4]) // 2) / 512} {(v[3] - v[1] + 1) / 512} {(v[4] - v[2] + 1) / 512}\n') # label, class, x_center, y_center, length, width


    list_of_images = os.listdir(os.path.join(train_dir, 'gt_mask'))
    list_of_images = [i.split('.')[0] for i in list_of_images]
    if not args.replicate:
        val_images = random.sample(list_of_images, int(0.1 * len(list_of_images)))
        test_images = random.sample([i for i in list_of_images if i not in val_images], int(0.2 * len(list_of_images)))
        train_images = [i for i in list_of_images if (i not in val_images and i not in test_images)]
    else:
        val_images = [f'train_{i}' for i in val_img_tnbc]
        test_images = [f'train_{i}' for i in test_img_tnbc]
        train_images = [i for i in list_of_images if i not in val_images and i not in test_images]

    for dir in os.listdir(train_dir):
        if 'DS_Store' not in dir:
            for file in os.listdir(os.path.join(train_dir, dir)):
                if file.split('.')[0] in val_images:
                    shutil.move(os.path.join(train_dir, dir, file), os.path.join(val_dir, dir, file))
                if file.split('.')[0] in test_images:
                    shutil.move(os.path.join(train_dir, dir, file), os.path.join(test_dir, dir, file))

shutil.move(os.path.join(train_dir, 'labels'), os.path.join(yolo_lab))
os.rename(os.path.join(yolo_lab, 'labels'), os.path.join(yolo_lab, 'train'))
shutil.move(os.path.join(val_dir, 'labels'), os.path.join(yolo_lab))
os.rename(os.path.join(yolo_lab, 'labels'), os.path.join(yolo_lab, 'val'))
shutil.move(os.path.join(test_dir, 'labels'), os.path.join(yolo_lab))
os.rename(os.path.join(yolo_lab, 'labels'), os.path.join(yolo_lab, 'test'))

shutil.copytree(os.path.join(train_dir, 'image'), os.path.join(yolo_img, 'train'))
shutil.copytree(os.path.join(val_dir, 'image'), os.path.join(yolo_img, 'val'))
shutil.copytree(os.path.join(test_dir, 'image'), os.path.join(yolo_img, 'test'))