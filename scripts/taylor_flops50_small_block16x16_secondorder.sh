python prune_vit.py --model deit_small_distilled_patch16_224 --blocksize 16 16 --data_path /imagenet --amount 0.5 --smooth 0.1 --flop_budget --lambda_power 1 --second_order --smooth_curve

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env finetune_vit.py \
    --model deit_small_distilled_patch16_224 \
    --batch-size 256 \
    --data-path /imagenet \
    --init_mask rd_curves/0/sp0.50_deit_small_distilled_patch16_224_ndz_0100_rdcurves_block16x16_ranking_taylor_secondorderapprox_opt_dist_mask_.pt \
    --init_weight rd_curves/0/sp0.50_deit_small_distilled_patch16_224_ndz_0100_rdcurves_block16x16_ranking_taylor_secondorderapprox_opt_dist_mask_.pt \
    --output_dir experiment/taylor1score_4gpus_deit_small_flops5188_block16x16_approx_taylorrank_derivative_secondorder_smooth0.1h4_min_sp0.05_distilled_power \
    --dist_url 'tcp://127.0.0.1:33251' --distillation-type soft 
