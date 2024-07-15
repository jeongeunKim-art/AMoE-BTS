#!/bin/bash

#SBATCH -J Eoformer
#SBATCH -o record.txt

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#실행 명령어
python main.py --distributed  --logdir=log_train_nestedformer --fold=0 --json_list=brats2020_datajson.json --max_epochs=230 --lrschedule=warmup_cosine --val_every=10 --data_dir=MICCAI_BraTS2020_TrainingData/  --out_channels=3 --batch_size=1 --infer_overlap=0.5


echo "###"
echo "### END DATE=$(date)"
