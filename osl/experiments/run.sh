uv run osl/experiments/train_regressive.py \
  --config config/test.yml --device cuda:0 --seq_length 8 --batch_size 4 --tracker wandb \
  --config config/regressive.yml --model ForeViT/M --augment --augment_noise_schedule cosine --augment_noise_start 0.9 --augment_noise_end 0.2 \
  --name test_regressive 

uv run osl/experiments/train_progressive.py \
  --config config/test.yml --device cuda:0 --seq_length 8 --batch_size 4 --tracker wandb \
  --config config/progressive.yml  --model unet-time --prediction_type epsilon \
  --name test_progressive 

uv run osl/experiments/train_forecaster.py \
  --config config/test.yml --device cuda:0 --seq_length 8 --batch_size 4 --tracker wandb \
  --config config/forecaster.yml --model dyffusion/forecaster/unet --interpolator dyffusion/interpolator/unet --interpolator_weights runs/dyffusion/forecaster_unet_008/weights/best.pth \
  --name test_dyffusion
