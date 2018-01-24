#!/bin/bash

cd assisted_mutation

for i in `seq 0 7`;
do
    cat master_$i.log >> master.log
    cat features_$i.log >> features.log
    rm master_$i.log
    rm features_$i.log
done

cd ../evasive

for i in `seq 0 7`;
do
    cat $i.evasive >> latest.evasive
    rm $i.evasive
done