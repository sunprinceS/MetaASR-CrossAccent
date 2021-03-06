#!/usr/bin/env sh

# e.g ./run_adapt.sh blstm config non-spot no-overwrite greedy ALGO PRETRAIN_SETTING PRETRAIN_SUFFIX ca en hk us wa 
# cross-region: af hk in ph sg
# mixed-region: be ph
# val are ca sc sa

MODEL_NAME=$1
CONFIG_NAME=$2
SPOT=$3
OVERWRITE=$4
DECODE_MODE=$5
ALGO=$6
PRETRAIN_SETTING=$7
PRETRAIN_SUFFIX=$8
DECODE_BATCH_SIZE=32

shift 8

EVAL_ACCENTS=$@


for pretrain_step in $(seq 20000 20000 200000) # for metabz5, multi
do
  for accent in $EVAL_ACCENTS
  do
    CONFIG="config/$MODEL_NAME/adapt/$CONFIG_NAME.yaml"

    hrun_prefix="hrun -X s09 -G -d -c 6 -m 6 -t 3-0 -n \"$MODEL_NAME/$CONFIG_NAME $ALGO-transfer($PRETRAIN_SUFFIX) at step $pretrain_step ($DECODE_MODE-decode)\""
    cmd="./adapt_full_exp.sh $MODEL_NAME $CONFIG $OVERWRITE $DECODE_MODE $ALGO $PRETRAIN_SETTING $PRETRAIN_SUFFIX $pretrain_step $accent"

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
done
