#!/usr/bin/env sh

for data_type in "train" "dev" "test"
#for data_type in "train"
do
  for dir in $(ls -d */ | cut -f1 -d '/')
  do
    if [[ $dir != "all" ]];then
    echo $dir
    rm $dir/$data_type
    cmd="ln -s /new-data/local/espnet/egs/commonvoice/asr1/mydata-separate/$data_type/$dir $dir/$data_type"
    echo $cmd
    eval $cmd
    fi
  done
done
