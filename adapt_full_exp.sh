#!/usr/bin/env sh

MODEL_NAME=$1
CONFIG=$2
OVERWRITE=$3
DECODE_MODE=$4
ALGO=$5
PRETRAIN_SETTING=$6
PRETRAIN_SUFFIX=$7
PRETRAIN_STEP=$8
ACCENT=$9

DECODE_BATCH_SIZE=32


name=$(hostname)
if [[ $name == *.speech ]];then
  if [[ $name == "login.speech" ]];then
    echo "[ERR] CANNOT run on login.speech!!"
    exit 1
  fi
  echo "Running on battleship ($name)"
else
  echo "Running on $name"
fi

## train
cmd="python train.py --config $CONFIG --pretrain --pretrain_setting $PRETRAIN_SETTING --pretrain_suffix $PRETRAIN_SUFFIX --pretrain_step $PRETRAIN_STEP --eval_suffix step$PRETRAIN_STEP --accent $ACCENT --algo $ALGO --model_name $MODEL_NAME --njobs 4 --eval_every_epoch"
if [[ $OVERWRITE == "overwrite" ]];then
  cmd="$cmd --overwrite"
fi
echo $cmd
eval $cmd

## decode
cmd="python train.py --config $CONFIG --pretrain --pretrain_setting $PRETRAIN_SETTING --pretrain_suffix $PRETRAIN_SUFFIX --pretrain_step $PRETRAIN_STEP --eval_suffix step$PRETRAIN_STEP --accent $ACCENT --algo $ALGO --model_name $MODEL_NAME --test --decode_mode $DECODE_MODE --njobs 4 --decode_batch_size $DECODE_BATCH_SIZE"
if [[ $OVERWRITE == "overwrite" ]];then
  cmd="$cmd --overwrite"
fi
echo $cmd
eval $cmd

## score
cmd="./score.sh --config $CONFIG --pretrain_suffix $PRETRAIN_SUFFIX --eval_suffix step$PRETRAIN_STEP --accent $ACCENT --algo $ALGO --model_name $MODEL_NAME --decode_mode $DECODE_MODE"
echo $cmd
eval $cmd
