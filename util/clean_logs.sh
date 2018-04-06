#!/bin/bash
if [ $# -lt 1 ];
then
    echo "Missing log sub-path | Marvin: gp/weighted_ | CNN: gp_cnn/"\
    exit
fi

rm output/logs/$1master.log
rm output/logs/$1features.log
rm output/logs/$1evasive.log
rm output/logs/$1mutations.log

if [ $# -gt 1 ] && [ $2 == '-cnn' ];
then
    echo "Cleaning CNN eval"
    rm deep-android/eval/*;
fi
