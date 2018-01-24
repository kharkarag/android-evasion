#!/bin/bash

if [ $# -eq 0 ]
then
    echo "Specify 'L1' or 'L2'"
    exit
fi

percentages=(1.0 5.0 10.0 25.0 50.0)

for p in "${percentages[@]}"
do
    echo $p
    training_set_path=train/$p%_evasive/$p%.train
    model_path=train/$p%_evasive/$p%.model.$1
    results_path=train/$p%_evasive/$p%.results

    if [ $1 == "L1" ]
    then
        ./train/train -s 6 -n 8 $training_set_path $model_path
    elif [ $1 == "L2" ]
    then
        ./train/train -s 0 -n 8 $training_set_path $model_path
    fi
    ./train/predict -b 1 $training_set_path $model_path $results_path
    rm assisted_mutation/master.log
    rm assisted_mutation/features.log
    python mutate_test_opt.py $model_path "seeds/training100.seeds" "/dev/null"
    mv assisted_mutation/master.log train/$p%_evasive/$p%.$1.log
done