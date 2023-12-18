# ICCV2984

Requirements: pytorch 1.7.1 cudatoolkit 10.1

Test DeiT-Tiny achieved by Bi-ViT  (66.9 Top-1 accuracy):

> python -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 --use_env main_1bit.py --model bi_deit_tiny_patch16_224 --data-path /your/path/to/ImageNet/ --output_dir ./test --distillation-type hard --teacher-model deit_tiny_patch16_224  --resume best_checkpoint_tiny.pth --eval 


Test DeiT-Small achieved by Bi-ViT  (66.9 Top-1 accuracy):

> python -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 --use_env main_1bit.py --model bi_deit_small_patch16_224 --data-path /your/path/to/ImageNet/ --output_dir ./test --distillation-type hard --teacher-model deit_small_patch16_224  --resume best_checkpoint_tiny.pth --eval 

checkpoints can be fetched in:

[https://drive.google.com/drive/folders/1StJBp_-aQqOe2S5YWo5P8HWPjhGEORKG?usp=sharing](https://drive.google.com/drive/folders/11vdNMsx-O0KxfqZ8cziRe7D6q_MGWMtv?usp=sharing)
