#!/usr/bin/env sh
set -e

function usage
{
    echo "usage: ./score.sh --config CONFIG --accent accent --algo ALGO --model_name MODEL_NAME --eval_suffix EVAL_SUFFIX --decode_mode DECODE_MODE [--pretrain_suffix PRETRAIN_SUFFIX --runs RUNS || -h]"
    echo "   ";
    echo "  --config            : Path to config file";
    echo "  --accent              : accent";
    echo "  --algo              : algo";
    echo "  --model_name        : model_name";
    echo "  --eval_suffix       : Eval suffix";
    echo "  --decode_mode       : decode_mode";
    echo "  --pretrain_suffix   : Pretrain suffix (not specified in MonoASR)";
    echo "  --runs              : runs";
    echo "  -h | --help         : This message";
}

function parse_args
{
  # positional args
  args=()

  # named args
  while [ "$1" != "" ]; do
      case "$1" in
          --config )           config="$2";             shift;;
          --accent   )           accent="$2";               shift;;
          --algo)              algo="$2";               shift;;
          --model_name )       model_name="$2";         shift;;
          --eval_suffix )      eval_suffix="$2";        shift;;
          --decode_mode )      decode_mode="$2";        shift;;
          --pretrain_suffix )  pretrain_suffix="$2";    shift;;
          --runs  )            runs="$2";               shift;;
          -h | --help )        usage;                   exit;; # quit and show usage
          * )                  args+=("$1")             # if no match, add it to the positional args
      esac
      shift # move to next kv pair
  done

  # restore positional args
  #set -- "${args[@]}"

  # validate required args
  if [[ -z "${config}" || -z "${accent}" || -z "${algo}"  ||  -z "${model_name}" ||  -z "${eval_suffix}" || -z "${decode_mode}" ]]; then
      show_args
      usage
      exit;
  fi

}


function show_args
{
  echo "config: $config"
  echo "accent: $accent"
  echo "algo: $runs"
  echo "model_name: $model_name"
  echo "eval_suffix: $eval_suffix"
  echo "decode_mode: $decode_mode"
  echo "pretrain_suffix: $pretrain_suffix"
  echo "runs: $runs"
}

function run
{
  parse_args "$@"
  show_args


  name=$(hostname)
  if [[ $name == "JYH-Speech" ]];then
    echo "On local machine"
    export PATH=/home/sunprince/local/kaldi/tools/sctk/bin/:$PATH
  elif [[ $name == *.speech ]];then
    echo "On battleship"
    export PATH=/opt/kaldi/tools/sctk/bin:$PATH
  else
    echo "[ERR] unknown hostname $name"
    exit
  fi

  python translate.py --config ${config} --accent ${accent} --algo ${algo} --model_name ${model_name} --pretrain_suffix ${pretrain_suffix} --eval_suffix ${eval_suffix} --decode_mode ${decode_mode} --runs ${runs}
}

run "$@";
