python3 train.py \
  --train cleaned_data/train_dedup.txt \
  --test cleaned_data/test_dedup.txt \
  --score-threshold 0. \
  --output-dir experiments/exp_m2m100_065_kz_rus \
  --model facebook/m2m100_418M \
  --epochs 3 \
  --batch-size 56 \
  --max-len 112

python3 train.py \
  --train cleaned_data/train_dedup.txt \
  --test cleaned_data/test_dedup.txt \
  --score-threshold 0.65 \
  --output-dir experiments/exp_m2m100_filtered065_kz_rus \
  --model facebook/m2m100_418M \
  --epochs 3 \
  --batch-size 56 \
  --max-len 112 \
  --cache-dir './cachef'
  

# python3 train_rus_kz.py \
#   --train cleaned_data/train_dedup.txt \
#   --test cleaned_data/test_dedup.txt \
#   --score-threshold 0. \
#   --output-dir experiments/exp_m2m100_065_rus_kz \
#   --model facebook/m2m100_418M \
#   --epochs 3 \
#   --batch-size 56 \
#   --max-len 112

# python3 train_rus_kz.py \
#   --train cleaned_data/train_dedup.txt \
#   --test cleaned_data/test_dedup.txt \
#   --score-threshold 0.65 \
#   --output-dir experiments/exp_m2m100_filtered065_ruz_kz \
#   --model facebook/m2m100_418M \
#   --epochs 3 \
#   --batch-size 56 \
#   --max-len 112


  python3 train_scratch.py \
  --train cleaned_data/train_dedup.txt \
  --test cleaned_data/test_dedup.txt \
  --score-threshold 0. \
  --output-dir experiments/t5small_kaz_rus \
  --model t5-small \
  --epochs 3 \
  --batch-size 256 \
  --max-len 112



  python3 train_scratch.py \
  --train cleaned_data/train_dedup.txt \
  --test cleaned_data/test_dedup.txt \
  --score-threshold 0.7 \
  --output-dir experiments/t5small_kaz_rus_filtered \
  --model t5-small \
  --epochs 3 \
  --batch-size 256 \
  --max-len 112
