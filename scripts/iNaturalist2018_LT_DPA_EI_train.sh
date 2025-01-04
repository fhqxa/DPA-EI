# Running the Python script with arguments
python train.py \
  --dataset iNaturalist18 \
  --data_root /iNaturalist18 \
  --num_classes 8142 \
  --num_meta 2 \
  --batch-size 64 \
  --imb_factor 0.01 \
  --test-batch-size 100 \
  --epochs 180 \
  --lr 0.0003 \
  --momentum 0.9 \
  --nesterov True \
  --weight-decay 5e-4 \
  --no-cuda False \
  --split 1000 \
  --seed 42 \
  --print-freq 100 \
  --lam 0.5 \
  --gpu 0 \
  --meta_lr 0.1 \
  --save_name 'name' \
  --idx '0'


