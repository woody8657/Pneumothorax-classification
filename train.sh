clear

CUDA_VISIBLE_DEVICES=0 python train.py \
    --img 512 \
    --batch 24 \
    --epochs 100 \
    --lr 0.001 \
    --tensorboard /home/u/woody8657/runs/2cls/INCEPTIONV3_SGD
    
