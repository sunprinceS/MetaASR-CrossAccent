#!/usr/bin/env sh

# e.g ./run_adapt.sh blstm config non-spot no-overwrite greedy ALGO PRETRAIN_SUFFIX PRETRAIN_MODEL_PATH EVAL_SUFFIX ca en hk us wa 
MODEL_NAME=$1
CONFIG_NAME=$2
SPOT=$3
OVERWRITE=$4
DECODE_MODE=$5
ALGO=$6
PRETRAIN_SUFFIX=$7
PRETRAIN_MODEL_PATH=$8
EVAL_SUFFIX=$9
DECODE_BATCH_SIZE=32

shift 9

EVAL_ACCENTS=$@
echo $EVAL_ACCENTS


for accent in $EVAL_ACCENTS
do
  CONFIG="config/$MODEL_NAME/$CONFIG_NAME.yaml"

  hrun_prefix="hrun -X s09 -G -d -c 6 -m 6 -t 3-0 -n \"$MODEL_NAME/$CONFIG_NAME ft on $PRETRAIN_MODEL_PATH ($DECODE_MODE-decode)\""
  cmd="./adapt_model_full_exp.sh $MODEL_NAME $CONFIG $OVERWRITE $DECODE_MODE $ALGO $PRETRAIN_SUFFIX $PRETRAIN_MODEL_PATH $EVAL_SUFFIX $accent"

  if [[ $SPOT == "spot" ]];then
    hrun_prefix="$hrun_prefix -s"
  fi

  name=$(hostname)
  if [[ $name == "login.speech" ]];then
    echo "On battleship login"
    echo "$hrun_prefix $cmd"
    eval "$hrun_prefix $cmd"
  else
    echo "On $name"
    echo "$cmd"
    eval "$cmd"
  fi
done
