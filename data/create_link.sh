#!/usr/bin/env sh
mkdir african all australia bermuda canada england hongkong indian ireland malaysia newzealand philippines scotland singapore southatlandtic us wales
for data_type in "train" "dev" "test"
#for data_type in "train"
do
  for dir in $(ls -d */ | cut -f1 -d '/')
  do
    if [[ $dir != "all" ]];then
    echo $dir
    cmd="ln -s /new-data/local/espnet/egs/commonvoice/asr1/mydata-separate/$data_type/$dir $dir/$data_type"

    #rm $dir/$data_type
    echo $cmd
    eval $cmd
    fi
  done
done
