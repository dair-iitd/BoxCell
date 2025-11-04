# an example run over the consep dataset. you would need to setup the environment and add gurobi and enable it on your system.

# A. preprocess the datasets

python3 data-preprocessing.py --loading_dir path_to_consep/consep --saving_dir save_path/consep --dataset consep --replicate True

python3 data-preprocessing.py --loading_dir path_to_monuseg/monuseg --saving_dir save_path/monuseg --dataset monuseg --replicate True

python3 data-preprocessing.py --loading_dir path_to_tnbc/tnbc --saving_dir save_path/tnbc --dataset tnbc --replicate True

# after this you would need to populate the embeddings folder in the three datasets with the output of the SAM encoder.


# C. Running yolo.
python3 training-yolo.py --yaml_file path/consep/train.yaml --model_weight_path path/runs/detect/tnbc-final-trained/weights/best.pt

# eval yolo to get the bounding boxes. the default conf and iou values are set for the optimal preds on the val of consep. need to change for monu and tnbc
python3 infering-yolo.py --model_weight_path path/runs/detect/the-final-trained/weights/best.pt --image_dir path/consep/yolo/images/test

# convert these to the format that SAM takes the boxes in.

# for D-sam / ITD, run the following command.
python3 sam-ilp.py --img_dir_path path_to/test/images --box_dir_path path_to/test/yolo_bboxes --model_weights path_to/sam-weights/sam_vit_b_01ec64.pth --gt_dir_path path_to/sam/test/gt_masks --save_path path_to/predictions/consep-dsam --sam_s_path path_to/sam/test/caranet_masks --mode d-sam --gurobi_license_file paht_to/gurobi.lic --gurobi_license cdf89c70-6f2f-4589-b577-8b3355c44b1c --type instance

# for BoxCell, you would need to pass your gurobi license path as well
python3 sam-ilp.py --img_dir_path path_to/sam/test/images --box_dir_path path_to/sam/test/yolo_bboxes --model_weights path_to/sam-weights/sam_vit_b_01ec64.pth --gt_dir_path path_to/sam/test/gt_masks --save_path path_to/guided-prompting/predictions --sam_s_path path_to/sam/test/caranet_masks --mode sam-ilp --gurobi_license_file path_to/gurobi.lic --gurobi_license cdf89c70-6f2f-4589-b577-8b3355c44b1c

# evaluate semantic segmentation
python3 eval-masks.py --type all --gt_masks_dir path_to/sam/test/gt_masks --pred_masks_dir path_to/tnbc-instance/test_inference/d-sam
