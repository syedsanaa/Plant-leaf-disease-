CUDA_VISIBLE_DEVICES="3" \
python test.py \
--checkpoint "/mnt/hdd2/anshul5/imseg-SAM2-UNet/checkpoint/SAM2-UNet-10.pth" \
--test_image_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/testing/input/" \
--test_gt_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/testing/groundtruth/" \
--save_path "/mnt/hdd2/anshul5/imseg-SAM2-UNet/testing/prediction"