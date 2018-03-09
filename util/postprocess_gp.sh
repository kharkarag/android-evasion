#!/bin/bash

str=""
for i in `sort -u output/logs/gp_features.log`;
do
    str+=$((i + 1))"p;"
done

sed -n "$str" Marvin/features/featurenames > output/logs/gp_mutations.log
cat output/logs/gp_mutations.log

#success=`grep "Success" output/logs/gp.log | wc -l`
#echo "------------------------------"
#echo "Successful:" $success
