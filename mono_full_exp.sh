#!/usr/bin/env sh

MODEL_NAME=$1
CONFIG=$2
OVERWRITE=$3
DECODE_MODE=$4
EVAL_SUFFIX=$5
ACCENT=$6

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
cmd="python train.py --config $CONFIG --eval_suffix $EVAL_SUFFIX --accent $ACCENT --algo no --model_name $MODEL_NAME --njobs 4"
if [[ $OVERWRITE == "overwrite" ]];then
  cmd="$cmd --overwrite"
fi
echo $cmd
eval $cmd

## decode
cmd="python train.py --config $CONFIG --eval_suffix $EVAL_SUFFIX --accent $ACCENT --algo no --model_name $MODEL_NAME --test --decode_mode $DECODE_MODE --njobs 4 --decode_batch_size $DECODE_BATCH_SIZE"
if [[ $OVERWRITE == "overwrite" ]];then
  cmd="$cmd --overwrite"
fi
echo $cmd
eval $cmd

## score
cmd="./score.sh --config $CONFIG --eval_suffix $EVAL_SUFFIX --accent $ACCENT --algo $ALGO --model_name $MODEL_NAME --decode_mode $DECODE_MODE"
echo $cmd
eval $cmd
