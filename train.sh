#PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
CUDA_VISIBLE_DEVICES="3" \
python train.py \
--hiera_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/sam2_hiera_tiny.pt" \
--train_image_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/data/dataset/input/" \
--train_mask_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/data/dataset/groundtruth/" \
--save_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/checkpoint" \
--epoch 10 \
--lr 0.001 \
--batch_size 8