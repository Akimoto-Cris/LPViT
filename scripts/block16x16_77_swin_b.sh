python prune_vit.py --model swin_base_patch4_window7_224 --blocksize 16 16 --data_path /imagenet --amount 0.77 --lambda_power 1 --flop_budget --second_order --smooth_curve

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env finetune_vit.py \
    --model swin_base_patch4_window7_224 \
    --batch-size 64 \
    --data-path /imagenet \
    --init_mask rd_curves/0/sp0.77_swin_base_patch4_window7_224_ndz_0100_rdcurves_block16x16_ranking_taylor_secondorderapprox_opt_dist_mask_.pt \
    --output_dir experiment/block16x16_dp_4gpus_swin_base_flops77_power \
    --dist_url 'tcp://127.0.0.1:29400' --distillation-type none 