# an example run over the consep dataset. you would need to setup the environment and add gurobi and enable it on your system.

# A. preprocess the datasets

python3 data-preprocessing.py --loading_dir /home/cse/btech/cs1200448/guided-prompting/consep --saving_dir /home/cse/btech/cs1200448/guided-prompting/processed-datasets/consep --dataset consep --replicate True

python3 data-preprocessing.py --loading_dir /home/cse/btech/cs1200448/guided-prompting/monuseg --saving_dir /home/cse/btech/cs1200448/guided-prompting/processed-datasets/monuseg --dataset monuseg --replicate True

python3 data-preprocessing.py --loading_dir /home/cse/btech/cs1200448/guided-prompting/tnbc --saving_dir /home/cse/btech/cs1200448/guided-prompting/processed-datasets/tnbc --dataset tnbc --replicate True

# after this you would need to populate the embeddings folder in the three datasets with the output of the SAM encoder.

# B. train caranet on the same split and get the ITS masks. @aayush has this code.

# C. Running yolo.
python3 training-yolo.py --yaml_file /home/cse/btech/cs1200448/guided-prompting/processed-datasets/consep/train.yaml --model_weight_path /home/cse/btech/cs1200448/guided-prompting/btp/HistSAM/runs/detect/tnbc-final-trained/weights/best.pt

# eval yolo to get the bounding boxes. the default conf and iou values are set for the optimal preds on the val of consep. need to change for monu and tnbc
python3 infering-yolo.py --model_weight_path /home/cse/btech/cs1200448/guided-prompting/btp/HistSAM/runs/detect/the-final-trained/weights/best.pt --image_dir /home/cse/btech/cs1200448/guided-prompting/processed-datasets/consep/yolo/images/test

# convert these to the format that SAM takes the boxes in.

# for D-sam / ITD, run the following command.
python3 sam-ilp.py --img_dir_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/images --box_dir_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/yolo_bboxes --model_weights /scratch/cse/btech/cs1200448/sam-weights/sam_vit_b_01ec64.pth --gt_dir_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/gt_masks --save_path /home/cse/btech/cs1200448/guided-prompting/predictions --sam_s_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/caranet_masks --mode d-sam

# for BoxCell, you would need to pass your gurobi license path as well
python3 sam-ilp.py --img_dir_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/images --box_dir_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/yolo_bboxes --model_weights /scratch/cse/btech/cs1200448/sam-weights/sam_vit_b_01ec64.pth --gt_dir_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/gt_masks --save_path /home/cse/btech/cs1200448/guided-prompting/predictions --sam_s_path /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/caranet_masks --mode sam-ilp --gurobi_license_file /home/cse/btech/cs1200448/gurobi.lic --gurobi_license ed42fa2d-e085-4572-b299-87dc9e40a8ab

# evaluate semantic segmentation
python3 eval-masks.py --type semantic --gt_masks_dir /scratch/cse/btech/cs1200448/Projects/BBSeg/consep2/sam/test/gt_masks --pred_masks_dir /scratch/cse/btech/cs1200448/Projects/BBSeg/checkpoints/default/consep-ilp/test_inference/img