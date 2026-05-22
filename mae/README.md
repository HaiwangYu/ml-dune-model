
For may15 branch
```bash
python mae/scripts/train_mae.py \
    --ssl_subset_frac=0.01 --sft_subset_frac=0.001 \
    --epochs=2 --vicreg_lambda_v=25 --vicreg_lambda_c=1 \
    --checkpoints_dir=./checkpoints
```

For mar26 branch
```bash
source setup.sh
python -u mae/scripts/train_mae.py --true_mae=True --ssl_subset_frac=0.01 --sft_subset_frac=0.001 --epochs=3 --n_sft_epochs_per_ssl_epoch=5 | tee log
python -u mae/scripts/train_mae.py --ssl_subset_frac=0.1 --sft_subset_frac=0.001 --epochs=10 --n_sft_epochs_per_ssl_epoch=10 | tee log
python -u mae/scripts/train_mae.py \
    --ssl_subset_frac=0.001 --sft_subset_frac=0.0001 --n_sft_epochs_per_ssl_epoch=1 \
    --resume=run_2026-03-06_05/checkpoints/mae_epoch10.pt \
    --epochs=15 \
    | tee log
python scripts/eval_mae.py --checkpoint=checkpoints/mae_epoch10.pt
python mae/scripts/inspect-train_mae-log.py --log_path=log --out_dir=viz_eval
python mae/diagnostics/plot_histories.py debug/histories.json
python mae/diagnostics/extract_features.py checkpoints/mae_epoch10.pt --max_images=2000 --output=./feats.npz
python mae/diagnostics/plot_knn.py feats.npz
```