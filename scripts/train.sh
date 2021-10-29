#!/bin/bash

source activate text2brain

PROJECT_DIR=/home/<CHANGETHIS>/text2brain
cd $PROJECT_DIR

BASE_DIR=/share/sablab/nfs04/data/text2brain
DATA_DIR=$BASE_DIR/data/processed
TRAIN_CSV=$DATA_DIR/filtered_train.csv
VAL_CSV=$DATA_DIR/filtered_val.csv
ARTICLE_IMAGES_DIR=$DATA_DIR/images

PRETRAINED_BERT_DIR=/share/sablab/nfs04/data/text2brain/models/pretrained/scibert_scivocab_uncased
MASK_FILE=$ARTICLE_IMAGES_DIR/mask.npy

# OUTPUTS_DIR=$BASE_DIR/outputs
OUTPUTS_DIR=/share/sablab/nfs04/<CHANGETHIS>/outputs

python -u train.py \
       --ver debug \
       --gpus 0 \
       --train_csv $TRAIN_CSV \
       --val_csv $VAL_CSV\
       --images_dir $ARTICLE_IMAGES_DIR \
       --pretrained_bert_dir $PRETRAINED_BERT_DIR \
       --mask_file $MASK_FILE \
       --save_dir $OUTPUTS_DIR \
       --lr 3e-2 \
       --batch_size 18 \
       --n_fc_channels 1024 \
       --n_decoder_channels 512 \
       --weight_decay 1e-6 \
       --drop_p 0.5


