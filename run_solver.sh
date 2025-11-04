#!/bin/bash
# MoNuSeg dataset with CBC solver
python sam-ilp_opensource.py --img_dir_path /path/to/monuseg/test/images --box_dir_path /path/to/monuseg/test/yolo_bboxes --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/monuseg/test/gt_masks --save_path /path/to/output/CBC/monuseg/ --sam_s_path /path/to/monuseg/caranet_generated_mask/ --mode sam-ilp --type instance --solver cbc
# MoNuSeg dataset with Sparse solver
python -u sam-ilp_opensource.py --img_dir_path /path/to/monuseg/test/gt_images/ --box_dir_path /path/to/monuseg/test/yolo_bboxes/ --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/monuseg/test/gt_masks/ --save_path /path/to/output/Sparse/monuseg/ --sam_s_path /path/to/monuseg/caranet_generated_mask/test/ --mode sam-ilp --type instance --solver sparse --downsample_factor 1

# MoNuSeg dataset with OR-Tools solver
python -u sam-ilp_opensource.py --img_dir_path /path/to/monuseg/test/images/ --box_dir_path /path/to/monuseg/test/yolo_bboxes/ --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/monuseg/test/gt_masks/ --save_path /path/to/output/ORTools/monuseg/ --sam_s_path /path/to/monuseg/caranet_generated_mask/ --mode sam-ilp --type instance --solver ortools --downsample_factor 1


# CoNSeP dataset with CBC solver
python sam-ilp_opensource.py --img_dir_path /path/to/consep/test/images --box_dir_path /path/to/consep/test/yolo_bboxes --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/consep/test/gt_masks --save_path /path/to/output/CBC/consep/ --sam_s_path /path/to/consep/caranet_generated_mask/ --mode sam-ilp --type instance --solver cbc

# CoNSeP dataset with Sparse solver
python -u sam-ilp_opensource.py --img_dir_path /path/to/consep/test/gt_images/ --box_dir_path /path/to/consep/test/yolo_bboxes/ --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/consep/test/gt_masks/ --save_path /path/to/output/Sparse/consep/ --sam_s_path /path/to/consep/caranet_generated_mask/test/ --mode sam-ilp --type instance --solver sparse --downsample_factor 1

# CoNSeP dataset with OR-Tools solver
python -u sam-ilp_opensource.py --img_dir_path /path/to/consep/test/images/ --box_dir_path /path/to/consep/test/yolo_bboxes/ --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/consep/test/gt_masks/ --save_path /path/to/output/ORTools/consep/ --sam_s_path /path/to/consep/caranet_generated_mask/ --mode sam-ilp --type instance --solver ortools --downsample_factor 1


# TNBC dataset with Sparse solver
python -u sam-ilp_opensource.py --img_dir_path /path/to/tnbc/test/gt_images/ --box_dir_path /path/to/tnbc/test/yolo_bboxes/ --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/tnbc/test/gt_masks/ --save_path /path/to/output/Sparse/tnbc/ --sam_s_path /path/to/tnbc/caranet_generated_mask_latest/ --mode sam-ilp --type instance --solver sparse --downsample_factor 2

# TNBC dataset with OR-Tools solver
python -u sam-ilp_opensource.py --img_dir_path /path/to/tnbc/test/gt_images/ --box_dir_path /path/to/tnbc/test/yolo_bboxes/ --model_weights /path/to/sam_vit_b_01ec64.pth --gt_dir_path /path/to/tnbc/test/gt_masks/ --save_path /path/to/output/ORTools/tnbc/ --sam_s_path /path/to/tnbc/caranet_generated_mask_latest/ --mode sam-ilp --type instance --solver ortools --downsample_factor 2