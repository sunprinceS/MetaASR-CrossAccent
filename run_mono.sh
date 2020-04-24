#!/usr/bin/env sh 

# e.g ./run_mono.sh blstm config non-spot no-overwrite greedy eval_suffix ca us

MODEL_NAME=$1
CONFIG_NAME=$2
SPOT=$3
OVERWRITE=$4
DECODE_MODE=$5
EVAL_SUFFIX=$6

shift 6

EVAL_ACCENTS=$@

for accent in $EVAL_ACCENTS
do
  CONFIG="config/$MODEL_NAME/$CONFIG_NAME.yaml"

  hrun_prefix="hrun -G -d -c 6 -m 6 -t 3-0 -n \"$MODEL_NAME/$CONFIG_NAME ($DECODE_MODE-decode)\""
  cmd="./mono_full_exp.sh $MODEL_NAME $CONFIG $OVERWRITE $DECODE_MODE $EVAL_SUFFIX $accent"

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
