CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 13345 main_pretrain.py \
    --num_workers 10 \
    --accum_iter 1 \
    --batch_size 2 \
    --save_freq 5 \
    --model alta \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 100 \
    --max_epochs 30 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ./MIMIC-CXR_dataset/ \
    --resume ./vision_encoder_weights/MRM.pth \
    --from_begin \
    --script $0 \
    --zeroshot_valid \
    --output_dir ./ALTA_results/ \
    --note alta \


# -m debugpy --listen 5678 --wait-for-client
