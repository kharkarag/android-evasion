#!/bin/bash

if [ $# -eq 0 ]
then
    echo "Specify 'L1' or 'L2'"
    exit
fi

percentages=(5.0 10.0 25.0 50.0 100.0)

for p in "${percentages[@]}"
do
    echo $p
    training_set_path=output/train/$p%_evasive/$p%.train
    model_path=output/train/$p%_evasive/$p%.model.$1
    results_path=output/train/$p%_evasive/$p%.results

    if [ $1 == "L1" ]
    then
        ./output/train/train -s 6 -n 8 $training_set_path $model_path
    elif [ $1 == "L2" ]
    then
        ./output/train/train -s 0 -n 8 $training_set_path $model_path
    fi
    ./output/train/predict -b 1 $training_set_path $model_path $results_path
    ./util/clean_logs.sh
    python mutate_test_opt.py $model_path "seeds/training_500.seeds" "/dev/null"
    mv output/logs/master.log output/train/$p%_evasive/$p%.$1.log
done