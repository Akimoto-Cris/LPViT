python prune_vit.py --model deit_base_distilled_patch16_224 --blocksize 32 64 --data_path /imagenet --amount 0.5 --smooth 0.9 --lambda_power 1 --flop_budget --second_order --smooth_curve

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env finetune_vit.py \
    --model deit_base_distilled_patch16_224 \
    --batch-size 128 \
    --data-path /imagenet \
    --init_mask rd_curves/0/sp0.50_deit_base_distilled_patch16_224_ndz_0100_rdcurves_block32x64_ranking_taylor_secondorderapprox_opt_dist_mask_.pt \
    --init_weight rd_curves/0/sp0.50_deit_base_distilled_patch16_224_ndz_0100_rdcurves_block32x64_ranking_taylor_secondorderapprox_opt_dist_mask_.pt \
    --output_dir experiment/taylor1score_4gpus_deit_base_flops50_block32x64_approx_taylorrank_derivative_secondorder_power_smooth0.9h4_distilled \
    --dist_url 'tcp://127.0.0.1:33251' --num_workers 16 --distillation-type soft 
